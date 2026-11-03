//! A fully wired [`Session`]: footage decoding through `effectcraft-media` (FilmCraft's codecs),
//! the media importer, the expression engine, JavaScript scripting (`effectcraft-script`) and
//! Render Queue export through `effectcraft-export` (FilmCraft's encoders). Frontends (desktop,
//! CLI, MCP, web) start here.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::sync::Arc;

use effectcraft_engine::project::render_queue::OutputFormat;
use effectcraft_engine::{ExportJob, ExportResult, Exporter, Importer, Session};
use effectcraft_project::Footage;
pub use effectcraft_script as script;

pub mod app_dirs;

struct MediaImporter;

impl Importer for MediaImporter {
    fn probe(&self, path: &str) -> Result<Footage, String> {
        effectcraft_media::probe(path).map_err(|e| e.to_string())
    }
}

/// Render Queue export via `effectcraft-export`: to the file system, or to `sink` (the web app
/// turns written files into downloads).
#[derive(Default)]
pub struct FileExporter {
    pub sink: Option<Arc<effectcraft_export::Sink>>,
    /// The session only exports ([`bound_export_memory`]): each job's footage frame cache and
    /// layer cache get [`export_cache_budget`] for its frames instead of the viewer's defaults.
    pub bounded: bool,
}

impl FileExporter {
    /// The cache budget of `job` when [`FileExporter::bounded`]: [`export_cache_budget`] for its
    /// frames as rendered (the composition at the job's resolution).
    fn cache_budget(&self, job: &ExportJob) -> Option<usize> {
        let comp = job.project.comp(job.item.comp)?;
        let scale = job.item.settings.resolution.clamp(0.01, 4.0);
        let side = |px: u32| (f64::from(px) * scale).ceil() as u32;
        self.bounded.then(|| export_cache_budget(side(comp.width), side(comp.height)))
    }

    /// The export of `job` as `effectcraft-export` runs it.
    pub fn job<'a>(&'a self, job: &'a ExportJob<'a>) -> effectcraft_export::Job<'a> {
        effectcraft_export::Job {
            project: job.project,
            footage: job.footage,
            expr: job.expr,
            accel: job.accel,
            comp: job.item.comp,
            settings: &job.item.settings,
            output: &job.item.output,
            path: job.path,
            sink: self.sink.as_deref(),
            options: effectcraft_export::JobOptions {
                log: job.item.log,
                label: job.label.clone(),
                storage: job.storage,
                overflow: job.project.render_prefs.overflow_folders.clone(),
                layer_cache_budget: self.cache_budget(job),
            },
            nested_switches: job.nested_switches,
        }
    }
}

impl Exporter for FileExporter {
    fn formats(&self) -> Vec<OutputFormat> {
        effectcraft_export::available_formats()
    }
    fn export(&self, job: &ExportJob, progress: &mut dyn FnMut(u64, u64) -> bool) -> Result<ExportResult, String> {
        effectcraft_engine::render::passes::block_on(self.export_async(job, progress))
    }

    fn export_async<'a>(
        &'a self,
        job: &'a ExportJob<'a>,
        progress: &'a mut dyn FnMut(u64, u64) -> bool,
    ) -> effectcraft_engine::render_queue::LocalFuture<'a, Result<ExportResult, String>> {
        Box::pin(async move {
            if let Some(bytes) = self.cache_budget(job) {
                job.footage.set_cache_budget(bytes);
            }
            let j = self.job(job);
            match effectcraft_export::export_async(&j, &mut |p| progress(p.done, p.total)).await {
                Ok(r) => Ok(ExportResult {
                    path: r.path,
                    frames: r.frames,
                    width: r.width,
                    height: r.height,
                    bytes: r.bytes,
                    seconds: r.seconds,
                    audio: r.audio,
                    log: r.log,
                    overflow: r.overflow,
                }),
                Err(effectcraft_export::ExportError::Cancelled) => Err(effectcraft_engine::render_queue::CANCELLED.into()),
                Err(e) => Err(e.to_string()),
            }
        })
    }
}

/// A new session with media, import, expressions, scripting and export enabled.
pub fn session() -> Session {
    let mut s = Session {
        // The desktop app, the CLI and the MCP server share installed Roto Brush models.
        models_dir: config_dir().map(|d| d.join("models")),
        exporter: Some(Arc::new(FileExporter::default())),
        footage: Arc::new(effectcraft_media::MediaPool::new()),
        importer: Some(Arc::new(MediaImporter)),
        expr: Some(Arc::new(effectcraft_expr::Expressions)),
        expr_check: Some(effectcraft_expr::check_syntax),
        plugin_loader: effectcraft_plugin::wasm_available().then_some(effectcraft_plugin::loader as effectcraft_engine::PluginLoader),
        ..Default::default()
    };
    effectcraft_script::install(&mut s);
    s
}

/// Frames an export renders after a movie clip's last read before its picture decoder closes, in
/// a session set up with [`bound_export_memory`]: two seconds at 24 fps, so a quick cut away and
/// back keeps the clip's decoder.
pub const EXPORT_IDLE_FRAMES: u32 = 48;

/// Frames an export's caches have room for in a session set up with [`bound_export_memory`]
/// ([`export_cache_budget`]).
pub const EXPORT_CACHE_FRAMES: usize = 16;

/// The smallest cache budget [`export_cache_budget`] gives: 512 MiB.
pub const EXPORT_CACHE_MIN: usize = 512 << 20;

/// The largest cache budget [`export_cache_budget`] gives: the viewer's default, 1 GiB.
pub const EXPORT_CACHE_MAX: usize = 1 << 30;

/// The footage frame cache and layer cache budget, in bytes, of an export whose frames render at
/// `width` × `height` in a session set up with [`bound_export_memory`]: room for
/// [`EXPORT_CACHE_FRAMES`] frames of four 32-bit floats a pixel, within [`EXPORT_CACHE_MIN`] and
/// [`EXPORT_CACHE_MAX`]. A render that goes forward once needs a few frames' worth for the frames
/// in flight, not the viewer's 1 GiB: 512 MiB up to 1080p, the viewer's default from 4K.
pub fn export_cache_budget(width: u32, height: u32) -> usize {
    let frame = (width as usize).saturating_mul(height as usize).saturating_mul(16);
    frame.saturating_mul(EXPORT_CACHE_FRAMES).clamp(EXPORT_CACHE_MIN, EXPORT_CACHE_MAX)
}

/// Set up a session that only exports (the command-line renderer) to keep its memory bounded
/// however many clips the composition uses. A movie clip's picture decoder, with its cache of
/// decoded frames, closes once the render has gone [`EXPORT_IDLE_FRAMES`] frames without reading
/// it, and opens again (giving the same frames) if the composition comes back to the clip
/// (audio readers stay open; see `MediaPool::set_idle_release`). Each
/// job's footage frame cache and layer cache get [`export_cache_budget`] for its frames; this
/// replaces the exporter with a [`FileExporter`] writing files. A session with a viewer keeps
/// [`session`]'s defaults: scrubbing back to a clip must not decode it from its keyframe again.
pub fn bound_export_memory(s: &mut Session) {
    s.footage.set_idle_release(Some(EXPORT_IDLE_FRAMES));
    s.exporter = Some(Arc::new(FileExporter { bounded: true, ..Default::default() }));
}

/// The platform config directory for EffectCraft (`EFFECTCRAFT_CONFIG_DIR` overrides):
/// `~/Library/Application Support/EffectCraft` (macOS), `%APPDATA%\EffectCraft` (Windows),
/// `$XDG_CONFIG_HOME/effectcraft` or `~/.config/effectcraft` (Linux and others). A `portable.txt`
/// beside the executable puts it in `EffectCraftData` there instead ([`app_dirs`]).
pub fn config_dir() -> Option<std::path::PathBuf> {
    app_dirs::current().dir.clone()
}

#[cfg(test)]
mod tests;
