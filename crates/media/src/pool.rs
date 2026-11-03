//! [`MediaPool`]: decoded footage frames for the compositor, with a memory-budgeted LRU cache.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use effectcraft_project::{AlphaMode, Footage, FootageKind, ItemId};
use effectcraft_raster::{Image, Px};
use effectcraft_render::{FloatCurve, FootageSource};
use effectcraft_time::{FrameRate, TICKS_PER_SECOND, Tick};
use filmcraft_media::{FrameRequest, SharedSource};
use image::metadata::Orientation;

use crate::convert::{AlphaOp, dynamic_to_image, frame_to_image_in, frame_to_image_scaled, turned};
use crate::gif_anim::{self, Anim};
use crate::{MediaError, Result};

/// Default frame-cache budget: 1 GiB of decoded frames.
pub const DEFAULT_BUDGET: usize = 1 << 30;

/// The cache path of a missing image-sequence frame (no file can have this name).
const PLACEHOLDER: &str = "\0missing frame";

/// What a missing image-sequence frame shows: 75 % colour bars (white, yellow, cyan, green,
/// magenta, red, blue), the usual "no picture here" frame.
fn colour_bars(w: u32, h: u32) -> Image {
    const BARS: [[f32; 3]; 7] = [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let (w, h) = (w.clamp(1, 16_384), h.clamp(1, 16_384));
    let row: Vec<Px> = (0..w)
        .map(|x| {
            let c = BARS.get((x as usize * BARS.len()) / w as usize).copied().unwrap_or([0.0; 3]);
            [c[0] * 0.75, c[1] * 0.75, c[2] * 0.75, 1.0]
        })
        .collect();
    Image { width: w, height: h, data: row.repeat(h as usize) }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    path: Arc<str>,
    frame: i64,
    /// The alpha interpretation (bits 0–1) and how float values are encoded (bits 2–3: sRGB,
    /// as stored, a 2.4 power curve): they change the decoded pixels.
    interp: u8,
    matte: [u32; 3],
    /// A movie frame made at this size for a reduced-resolution render ((0, 0): as decoded).
    size: (u32, u32),
    /// A still at a reduced `size`: the footage's nominal size, which decides whether it is
    /// resampled (see `Inner::still_at_size`); (0, 0) otherwise.
    nominal: (u32, u32),
}

struct Entry {
    img: Arc<Image>,
    stamp: u64,
    bytes: usize,
}

/// Conformed audio files written by builds before this version are ignored (and written again):
/// version 1 files of sources longer than about 12:36 at 48 kHz were silent from there on (#172);
/// version 2 files were written silent where the source failed to decode, and kept the footage
/// silent from then on (#324).
const CONFORM_VERSION: u32 = 3;

/// Footage longer than this (24 hours) is read from its decoder rather than conformed: its
/// duration comes from the file's header, and a damaged one claiming days of audio had the
/// conform thread allocate all of it at once (#408).
const MAX_CONFORM_SECONDS: i64 = 24 * 3600;

/// The sample frames of `footage`'s audio at `rate` Hz that a conformed file holds, `None` past
/// [`MAX_CONFORM_SECONDS`].
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn conform_frames(footage: &Footage, rate: u32) -> Option<usize> {
    let frames = footage.duration.to_units_floor(i64::from(rate)).max(0);
    if frames > MAX_CONFORM_SECONDS.saturating_mul(i64::from(rate)) {
        return None;
    }
    usize::try_from(frames).ok()
}

/// The source time of sample frame `at` at `rate` Hz. (`at` × ticks per second overflows `i64`
/// past about 12:36 at 48 kHz.)
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn sample_time(at: usize, rate: u32) -> Tick {
    Tick::from_units(i64::try_from(at).unwrap_or(i64::MAX), i64::from(rate))
}

/// At most this many recycled pixel buffers are kept (outside the budget).
const MAX_SPARE: usize = 4;

/// Least-recently-used frames, bounded by bytes.
#[derive(Default)]
struct Lru {
    map: HashMap<Key, Entry>,
    order: BTreeMap<u64, Key>,
    clock: u64,
    bytes: usize,
    /// Pixel buffers of evicted frames nobody else holds, reused for new frames: playback then
    /// writes into memory that is already mapped instead of faulting in fresh pages every frame.
    spare: Vec<Vec<Px>>,
}

impl Lru {
    fn get(&mut self, k: &Key) -> Option<Arc<Image>> {
        let e = self.map.get_mut(k)?;
        self.order.remove(&e.stamp);
        self.clock += 1;
        e.stamp = self.clock;
        self.order.insert(e.stamp, k.clone());
        Some(e.img.clone())
    }
    fn insert(&mut self, k: Key, img: Arc<Image>, budget: usize) {
        let bytes = img.data.len() * std::mem::size_of::<Px>() + 64;
        self.clock += 1;
        if let Some(old) = self.map.insert(k.clone(), Entry { img, stamp: self.clock, bytes }) {
            self.order.remove(&old.stamp);
            self.bytes -= old.bytes;
        }
        self.order.insert(self.clock, k);
        self.bytes += bytes;
        self.evict(budget);
    }
    fn remove(&mut self, k: &Key) {
        if let Some(e) = self.map.remove(k) {
            self.order.remove(&e.stamp);
            self.bytes -= e.bytes;
        }
    }
    /// Evict least-recently-used frames until within `budget` (the newest frame always stays).
    fn evict(&mut self, budget: usize) {
        while self.bytes > budget && self.map.len() > 1 {
            let Some((_, k)) = self.order.pop_first() else { break };
            if let Some(e) = self.map.remove(&k) {
                self.bytes -= e.bytes;
                if self.spare.len() < MAX_SPARE
                    && let Ok(img) = Arc::try_unwrap(e.img)
                {
                    self.spare.push(img.data);
                }
            }
        }
    }
    /// A recycled buffer of exactly `len` pixels, or an empty Vec.
    fn take_spare(&mut self, len: usize) -> Vec<Px> {
        match self.spare.iter().position(|b| b.len() == len) {
            Some(i) => self.spare.swap_remove(i),
            None => Vec::new(),
        }
    }
}

/// Cache counters (see [`MediaPool::stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Requests answered from the frame cache.
    pub hits: u64,
    /// Requests that had to decode (or wait for a read-ahead decode).
    pub misses: u64,
    /// Frames decoded ahead of sequential playback.
    pub prefetched: u64,
    /// Frames currently cached and their total size in bytes.
    pub frames: usize,
    pub bytes: usize,
    /// Movie and audio files opened (each path once, however many callers ask for it at once;
    /// again after it was closed).
    pub opened: u64,
    /// Movie and audio sources open now.
    pub open: usize,
    /// Picture sources closed because a sequential render stopped reading them
    /// ([`MediaPool::set_idle_release`]).
    pub released: u64,
}

#[derive(Default)]
struct CacheState {
    lru: Lru,
    /// Frames being decoded right now (requests for them wait instead of decoding again).
    inflight: HashSet<Key>,
    /// Last frame index requested per movie (sequential-playback detection).
    last: HashMap<Arc<str>, i64>,
}

/// A movie/audio source by path and the turn its picture is shown with
/// ([`crate::probe::video_turn`]), opened by the first caller while the others wait for it
/// (`None`: failed to open; not retried until `forget`).
type SourceSlot = Arc<OnceLock<Option<(SharedSource, Orientation)>>>;

/// What a source in the pool is opened for. A movie's audio, and its pictures too by default,
/// come from one source per file. With idle release on ([`MediaPool::set_idle_release`]),
/// pictures come from a source of their own: those keep the decoded frames, and are the ones
/// closed once unread.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Reader {
    Shared,
    Pictures,
}

/// A source in the pool and when it was last asked for: the frames a sequential render had
/// finished then ([`MediaPool::release_idle`]).
struct Open {
    slot: SourceSlot,
    step: u64,
}

impl Open {
    /// Whether nobody but the pool holds it: it has finished opening, opened (a file that failed
    /// to open holds nothing and is not retried), and no reader holds its source.
    fn unheld(&self) -> bool {
        Arc::strong_count(&self.slot) == 1 && matches!(self.slot.get(), Some(Some((src, _))) if Arc::strong_count(src) == 1)
    }
}

/// An animated GIF by path, as [`SourceSlot`] (the error message when it failed to load).
type GifSlot = Arc<OnceLock<std::result::Result<Arc<Anim>, String>>>;

struct Inner {
    /// Opened movie/audio sources by path and what they were opened for.
    sources: Mutex<HashMap<(Arc<str>, Reader), Open>>,
    /// Files opened into `sources` (see [`PoolStats::opened`]).
    opened: AtomicU64,
    /// Frames of a sequential render a picture source may go unread before
    /// [`MediaPool::release_idle`] closes it (0: never, and pictures share their file's source).
    idle_steps: AtomicU64,
    /// Frames the render has finished ([`MediaPool::release_idle`]).
    step: AtomicU64,
    /// Sources it closed (see [`PoolStats::released`]).
    released: AtomicU64,
    /// In-memory files registered with `add_bytes` (web builds, tests).
    files: Mutex<HashMap<String, Arc<[u8]>>>,
    /// Parsed 3D models by path (`None`: failed to load; not retried until `forget`).
    models: Mutex<HashMap<String, Option<Arc<effectcraft_model::Model>>>>,
    /// Decoded animated GIFs and WebPs by path, loaded by the first caller while the others wait (an
    /// `Err` is not retried until `forget`).
    gifs: Mutex<HashMap<Arc<str>, GifSlot>>,
    cache: Mutex<CacheState>,
    done: Condvar,
    budget: AtomicU64,
    read_ahead: AtomicBool,
    hits: AtomicU64,
    misses: AtomicU64,
    prefetched: AtomicU64,
    /// Settings ▸ Disk ▸ Conformed Audio Folder: decoded audio tracks written once per file and
    /// rate, read back instead of decoding again (`None` = off).
    conform: Mutex<Option<std::path::PathBuf>>,
    /// Conformed files being written right now.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    conforming: Mutex<HashSet<std::path::PathBuf>>,
    /// Conformed files that failed to write (the source didn't decode) this session: their
    /// audio is read from the decoder instead of trying again on every read.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    conform_failed: Mutex<HashSet<std::path::PathBuf>>,
    /// Movies that failed to open or decode (path to a readable reason; see
    /// [`FootageSource::problem`]). The first failure of a file is logged, later ones are not.
    problems: Mutex<HashMap<Arc<str>, String>>,
}

/// Decodes footage for the renderer: stills, image sequences and movies (FilmCraft codecs), plus
/// audio. Thread-safe and cheap to clone (clones share decoders and cache).
///
/// - One decoder per movie: FilmCraft's GOP cache keeps it positioned, so sequential frames
///   decode without re-seeking, and a seek decodes forward from the preceding sync sample.
/// - Converted frames live in an LRU cache bounded by a byte budget ([`DEFAULT_BUDGET`]).
/// - Sequential playback reads one frame ahead on a worker thread (native builds), so decoding
///   and conversion of the next frame overlap with the caller's use of the current one.
/// - Concurrent requests for the same frame decode it once.
#[derive(Clone)]
pub struct MediaPool {
    inner: Arc<Inner>,
}

impl Default for MediaPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Where a frame comes from.
struct Loc {
    key: Key,
    /// Media time for movies (`None`: an image file, `key.path`).
    media_t: Option<Tick>,
    /// How float values are encoded (also in `key.interp`).
    float: FloatCurve,
}

/// Removes an in-flight key and wakes waiters, also when decoding panics.
struct Inflight<'a> {
    inner: &'a Inner,
    key: Key,
}

impl Drop for Inflight<'_> {
    fn drop(&mut self) {
        lock(&self.inner.cache).inflight.remove(&self.key);
        self.inner.done.notify_all();
    }
}

impl MediaPool {
    /// A pool with the [`DEFAULT_BUDGET`] frame cache.
    pub fn new() -> Self {
        Self::with_budget(DEFAULT_BUDGET)
    }

    /// A pool whose frame cache holds at most `bytes` of decoded frames.
    pub fn with_budget(bytes: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                sources: Mutex::default(),
                files: Mutex::default(),
                models: Mutex::default(),
                gifs: Mutex::default(),
                cache: Mutex::default(),
                done: Condvar::new(),
                budget: AtomicU64::new(bytes as u64),
                read_ahead: AtomicBool::new(cfg!(not(target_arch = "wasm32"))),
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                prefetched: AtomicU64::new(0),
                opened: AtomicU64::new(0),
                idle_steps: AtomicU64::new(0),
                step: AtomicU64::new(0),
                released: AtomicU64::new(0),
                conform: Mutex::default(),
                conforming: Mutex::default(),
                conform_failed: Mutex::default(),
                problems: Mutex::default(),
            }),
        }
    }

    pub fn budget(&self) -> usize {
        self.inner.budget()
    }

    /// Change the frame-cache budget (evicts immediately when shrinking).
    pub fn set_budget(&self, bytes: usize) {
        self.inner.budget.store(bytes as u64, Ordering::Relaxed);
        lock(&self.inner.cache).lru.evict(bytes);
    }

    /// Enable or disable reading one frame ahead during sequential playback (on by default on
    /// native builds; ignored on wasm32).
    pub fn set_read_ahead(&self, on: bool) {
        self.inner.read_ahead.store(on && cfg!(not(target_arch = "wasm32")), Ordering::Relaxed);
    }

    pub fn stats(&self) -> PoolStats {
        let i = &self.inner;
        let open = lock(&i.sources).values().filter(|o| matches!(o.slot.get(), Some(Some(_)))).count();
        let c = lock(&i.cache);
        PoolStats {
            hits: i.hits.load(Ordering::Relaxed),
            misses: i.misses.load(Ordering::Relaxed),
            prefetched: i.prefetched.load(Ordering::Relaxed),
            frames: c.lru.map.len(),
            bytes: c.lru.bytes,
            opened: i.opened.load(Ordering::Relaxed),
            open,
            released: i.released.load(Ordering::Relaxed),
        }
    }

    /// Close a movie's picture decoder once a sequential render has finished `frames` frames
    /// without reading it ([`MediaPool::release_idle`]); `None` (the default) keeps every decoder
    /// open until [`MediaPool::forget`] or [`MediaPool::clear`]. Set it before reading.
    ///
    /// Interactive playback wants decoders open: a clip scrubbed back to would otherwise decode
    /// from its keyframe again. A sequential render (an export) never comes back to most clips,
    /// and each open picture decoder keeps up to a few hundred megabytes of decoded frames, so a
    /// render that turns this on keeps only the picture decoders of the clips it is showing open.
    /// A closed decoder opens again when the clip is read, decodes from its keyframe and gives the
    /// same frames; if it can't open again (the file went away), the clip's pictures come from
    /// its shared source. With this on, a movie's pictures come from a source of their own;
    /// audio keeps its file's source open, as a track's decoded audio depends on what its decoder
    /// decoded before (AAC is decoded on from one packet of pre-roll after a seek), so a reopened
    /// one could sound different where a render plays a passage again. An open audio reader keeps
    /// no decoded pictures, but does keep the audio it decoded (FilmCraft keeps up to 4096 packets
    /// a file, about 32 MB of stereo AAC; a few MB for a clip of a few seconds). `Some(0)` counts
    /// as `Some(1)`.
    pub fn set_idle_release(&self, frames: Option<u32>) {
        self.inner.idle_steps.store(frames.map_or(0, |f| u64::from(f.max(1))), Ordering::Relaxed);
    }

    /// The frames set with [`MediaPool::set_idle_release`].
    pub fn idle_release(&self) -> Option<u32> {
        let steps = self.inner.idle_steps.load(Ordering::Relaxed);
        (steps > 0).then(|| u32::try_from(steps).unwrap_or(u32::MAX))
    }

    /// A sequential render (an export) finished `frames` more frames, a batch: with
    /// [`MediaPool::set_idle_release`] on, close the picture decoders not read since that many
    /// frames before this batch (so at least that many frames after the batch they were last read
    /// in, for batches of one size), and return how many closed. A source a reader holds, or one
    /// still opening, stays open.
    pub fn release_idle(&self, frames: u64) -> usize {
        let i = &self.inner;
        let idle = i.idle_steps.load(Ordering::Relaxed);
        if idle == 0 {
            return 0;
        }
        let closed: Vec<Open> = {
            let mut sources = lock(&i.sources);
            let now = i.step.fetch_add(frames, Ordering::Relaxed).saturating_add(frames);
            let after = idle.saturating_add(frames);
            sources.extract_if(|(_, r), o| *r == Reader::Pictures && now.saturating_sub(o.step) >= after && o.unheld()).map(|(_, o)| o).collect()
        };
        i.released.fetch_add(closed.len() as u64, Ordering::Relaxed);
        // (the decoders and their frames are freed here, outside the lock)
        closed.len()
    }

    /// Drop every cached frame (decoders stay open).
    pub fn clear_frames(&self) {
        let mut c = lock(&self.inner.cache);
        c.lru = Lru::default();
        c.last.clear();
        drop(c);
        // Decoded GIFs sit outside the frame budget; Purge frees them too.
        lock(&self.inner.gifs).clear();
    }

    /// Drop every cached frame and decoder.
    pub fn clear(&self) {
        self.clear_frames();
        lock(&self.inner.sources).clear();
        lock(&self.inner.problems).clear();
    }

    /// Forget a file (after it changed on disk, or to retry a file that failed to open).
    pub fn forget(&self, path: &str) {
        lock(&self.inner.sources).retain(|(p, _), _| &**p != path);
        lock(&self.inner.models).remove(path);
        lock(&self.inner.gifs).remove(path);
        lock(&self.inner.problems).remove(path);
        let mut c = lock(&self.inner.cache);
        let keys: Vec<Key> = c.lru.map.keys().filter(|k| &*k.path == path).cloned().collect();
        for k in keys {
            c.lru.remove(&k);
        }
        c.last.remove(path);
        drop(c);
        crate::exr_channels::forget(path);
    }

    /// Provide the contents of `path` from memory (used instead of the file system; web builds).
    pub fn add_bytes(&self, path: &str, bytes: Arc<[u8]>) {
        lock(&self.inner.files).insert(path.to_string(), bytes);
        self.forget(path);
    }

    /// Decode (or fetch from cache) the frame of `footage` at source time `t`.
    pub fn frame_at(&self, footage: &Footage, t: Tick) -> Result<Arc<Image>> {
        Inner::frame_at(&self.inner, footage, t, true, None, FloatCurve::Srgb)
    }

    /// [`FootageSource::frame`] / [`FootageSource::frame_at_size`]: `None` (logged) on errors.
    fn frame_sized(&self, _item: ItemId, footage: &Footage, t: Tick, size: Option<(u32, u32)>, float: FloatCurve) -> Option<Arc<Image>> {
        if footage.missing {
            return None;
        }
        match Inner::frame_at(&self.inner, footage, t, true, size, float) {
            Ok(img) => Some(img),
            // nobody wants the frame any more (`cancellable`): not an error
            Err(MediaError::Cancelled) => None,
            // (a movie's failure is logged once per file, when it is recorded)
            Err(e) => {
                if lock(&self.inner.problems).get(footage.path.as_str()).is_none() {
                    log::warn!("media: frame of {} at {:?}: {e}", footage.path, t);
                }
                None
            }
        }
    }

    /// Settings ▸ Disk ▸ Conformed Audio Folder: write each footage file's decoded audio there
    /// once per sample rate (raw interleaved stereo `f32`) and read it back instead of decoding
    /// again. `None` turns it off.
    pub fn set_conform_folder(&self, folder: Option<std::path::PathBuf>) {
        *lock(&self.inner.conform) = folder;
    }

    /// The conformed-audio file of `path` at `rate` (named by a hash of the path, size and
    /// modification time, so an edited file conforms again, and of `CONFORM_VERSION`).
    pub fn conformed_path(&self, path: &str, rate: u32) -> Option<std::path::PathBuf> {
        use std::hash::{Hash, Hasher};
        let folder = lock(&self.inner.conform).clone()?;
        let meta = std::fs::metadata(path).ok()?;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (CONFORM_VERSION, path, meta.len(), meta.modified().ok()).hash(&mut h);
        Some(folder.join(format!("{:016x}_{rate}.ecaf", h.finish())))
    }

    /// Samples from a conformed file: `Some` when it exists and covers the request.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn read_conformed(file: &std::path::Path, s0: usize, frames: usize) -> Option<Vec<f32>> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(file).ok()?;
        let len = f.metadata().ok()?.len() as usize / 8;
        let mut out = vec![0.0f32; frames * 2];
        if s0 >= len {
            return Some(out);
        }
        let n = frames.min(len - s0);
        f.seek(SeekFrom::Start(s0 as u64 * 8)).ok()?;
        let mut bytes = vec![0u8; n * 8];
        f.read_exact(&mut bytes).ok()?;
        for (o, c) in out.iter_mut().zip(bytes.as_chunks::<4>().0.iter()) {
            *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        Some(out)
    }

    /// Conform `frames` sample frames of `footage`'s audio at `rate` into `file` on a worker
    /// thread (once).
    #[cfg(not(target_arch = "wasm32"))]
    fn conform(&self, footage: &Footage, file: std::path::PathBuf, frames: usize, rate: u32) {
        if lock(&self.inner.conform_failed).contains(&file) || !lock(&self.inner.conforming).insert(file.clone()) {
            return;
        }
        let (pool, f) = (self.clone(), footage.clone());
        let spawned = std::thread::Builder::new().name("ec-conform-audio".into()).spawn(move || {
            let tmp = file.with_extension("tmp");
            let written = match file.parent().map(std::fs::create_dir_all) {
                Some(Ok(())) => pool.write_conformed(&f, &tmp, frames, rate).and_then(|()| std::fs::rename(&tmp, &file)),
                Some(Err(e)) => Err(e),
                None => Err(std::io::Error::other("no folder")),
            };
            // A failed decode leaves no file: one written silent would keep the footage silent
            // in every later session (#324).
            if let Err(e) = written {
                let _ = std::fs::remove_file(&tmp);
                log::warn!("media: could not write conformed audio {} of {}: {e}", file.display(), f.path);
                lock(&pool.inner.conform_failed).insert(file.clone());
            }
            lock(&pool.inner.conforming).remove(&file);
        });
        if spawned.is_err() {
            lock(&self.inner.conforming).clear();
        }
    }

    /// Write `frames` sample frames of `footage`'s audio at `rate` to `path` (raw interleaved
    /// stereo `f32`) a chunk at a time: the whole track was held in memory before it was written.
    /// Fails when any chunk doesn't decode.
    #[cfg(not(target_arch = "wasm32"))]
    fn write_conformed(&self, footage: &Footage, path: &std::path::Path, frames: usize, rate: u32) -> std::io::Result<()> {
        use std::io::Write;
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut at = 0usize;
        while at < frames {
            let n = (frames - at).min(1 << 16);
            for v in self.try_decode_audio(footage, sample_time(at, rate), n, rate).map_err(std::io::Error::other)? {
                out.write_all(&v.to_le_bytes())?;
            }
            at += n;
        }
        out.flush()
    }

    /// `frames` stereo sample frames of `footage`'s audio starting at source time `start`, at
    /// `rate` Hz, interleaved (L R L R …). Mono is duplicated to both channels; multichannel
    /// sources are downmixed to stereo ([`crate::downmix`]). Silence where there is no audio (or before the media starts).
    /// With a conformed audio folder, reads come from the conformed file once it is written.
    pub fn audio_samples(&self, footage: &Footage, start: Tick, frames: usize, rate: u32) -> Vec<f32> {
        if !footage.has_audio || footage.missing || rate == 0 || frames == 0 {
            return vec![0.0; frames * 2];
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(conform) = conform_frames(footage, rate)
            && let Some(file) = self.conformed_path(&footage.path, rate)
        {
            let s0 = start.to_units_floor(rate as i64);
            if file.exists() && !lock(&self.inner.conforming).contains(&file) {
                let skip = (-s0).max(0) as usize;
                if skip >= frames {
                    return vec![0.0; frames * 2];
                }
                if let Some(v) = Self::read_conformed(&file, s0.max(0) as usize, frames - skip) {
                    let mut out = vec![0.0; skip * 2];
                    out.extend(v);
                    return out;
                }
            } else {
                self.conform(footage, file, conform, rate);
            }
        }
        self.decode_audio(footage, start, frames, rate)
    }

    /// [`MediaPool::audio_samples`] straight from the decoder. The source is read at its own
    /// rate and resampled here (linear, by absolute sample position, so a range comes out the
    /// same however it is split into reads). FilmCraft's MP4 reader, asked for another rate,
    /// applied the edit list's priming offset twice and ramped the start of every read: AAC
    /// footage not at the mix rate stuttered in previews and exports (#274).
    fn decode_audio(&self, footage: &Footage, start: Tick, frames: usize, rate: u32) -> Vec<f32> {
        self.try_decode_audio(footage, start, frames, rate).unwrap_or_else(|e| {
            log::warn!("media: audio of {}: {e}", footage.path);
            vec![0.0; frames * 2]
        })
    }

    /// [`MediaPool::decode_audio`], failing when the source can't be opened or read.
    fn try_decode_audio(&self, footage: &Footage, start: Tick, frames: usize, rate: u32) -> std::result::Result<Vec<f32>, String> {
        let path: Arc<str> = footage.path.as_str().into();
        let src = self.inner.source(&path).ok_or("the file can't be opened")?;
        let s0 = start.to_units_floor(rate as i64);
        let native = src.info().audio.as_ref().map_or(rate, |a| a.sample_rate);
        if native == 0 || rate == 0 || native == rate {
            return read_stereo(&src, s0, frames, rate);
        }
        // Output sample `n` sits at source position n × native / rate.
        let (native_i, rate_i) = (i128::from(native), i128::from(rate));
        let first = (i128::from(s0) * native_i).div_euclid(rate_i);
        let last = ((i128::from(s0) + frames as i128) * native_i).div_euclid(rate_i) + 1;
        let (Ok(first64), Ok(count)) = (i64::try_from(first), usize::try_from(last - first + 1)) else { return Ok(vec![0.0; frames * 2]) };
        let buf = read_stereo(&src, first64, count, native)?;
        let mut out = Vec::with_capacity(frames * 2);
        for k in 0..frames as i128 {
            let pos = (i128::from(s0) + k) * native_i;
            let i = usize::try_from(pos.div_euclid(rate_i) - first).unwrap_or(0) * 2;
            let f = (pos.rem_euclid(rate_i) as f64 / rate as f64) as f32;
            for c in 0..2 {
                let a = buf.get(i + c).copied().unwrap_or(0.0);
                let b = buf.get(i + 2 + c).copied().unwrap_or(a);
                out.push(a + (b - a) * f);
            }
        }
        Ok(out)
    }

    /// A thumbnail of `footage` (its first frame) fitting in `max_side` × `max_side`, aspect kept.
    pub fn thumbnail(&self, footage: &Footage, max_side: u32) -> Result<Image> {
        let img = Inner::frame_at(&self.inner, footage, Tick::ZERO, false, None, FloatCurve::Srgb)?;
        let (w, h) = (img.width.max(1), img.height.max(1));
        let s = (max_side.max(1) as f64 / w.max(h) as f64).min(1.0);
        if s >= 1.0 {
            return Ok((*img).clone());
        }
        let (tw, th) = (((w as f64 * s).round() as u32).max(1), ((h as f64 * s).round() as u32).max(1));
        Ok(effectcraft_raster::resample(&img, tw, th))
    }
}

impl Inner {
    fn budget(&self) -> usize {
        self.budget.load(Ordering::Relaxed) as usize
    }

    /// A 3D model file and its sibling resources (buffers, textures, MTL files), parsed once.
    fn model(&self, path: &str) -> Option<Arc<effectcraft_model::Model>> {
        if let Some(m) = lock(&self.models).get(path) {
            return m.clone();
        }
        let loaded = self.read(path).map_err(|e| e.to_string()).and_then(|bytes| {
            // Files the model names are read only when local or on the model's own network share.
            let resolve = |uri: &str| -> Option<Vec<u8>> {
                let p = craft_docpath::resolve_absolute(uri, Some(std::path::Path::new(path))).ok()?;
                self.read(&p.to_string_lossy()).ok().map(|b| b.to_vec())
            };
            effectcraft_model::load(path, &bytes, &resolve).map_err(|e| e.to_string())
        });
        let m = match loaded {
            Ok(m) => Some(Arc::new(m)),
            Err(e) => {
                log::warn!("media: cannot load model {path}: {e}");
                None
            }
        };
        lock(&self.models).entry(path.to_string()).or_insert(m).clone()
    }

    fn read(&self, path: &str) -> Result<Arc<[u8]>> {
        if let Some(b) = lock(&self.files).get(path) {
            return Ok(b.clone());
        }
        std::fs::read(path).map(Into::into).map_err(|e| MediaError::Io(format!("{path}: {e}")))
    }

    /// The opened movie/audio source for `path`. The first caller opens it, outside the map's
    /// lock (reading and parsing a large file takes a moment; FLAC, MP3, Ogg and AIFF files are
    /// decoded whole); callers asking for the same file meanwhile wait for that instead of opening
    /// it again: a preview's frame workers, audio feeder, conform and waveform threads each
    /// decoded a long FLAC in full at once (#408).
    fn source(&self, path: &Arc<str>) -> Option<SharedSource> {
        self.opened(path, Reader::Shared).map(|(src, _)| src)
    }

    /// The source movie frames are read from: their file's shared one, or with idle release on
    /// one of their own ([`MediaPool::set_idle_release`]).
    fn picture_reader(&self) -> Reader {
        if self.idle_steps.load(Ordering::Relaxed) > 0 { Reader::Pictures } else { Reader::Shared }
    }

    /// The source movie frames of `path` are read from ([`Self::picture_reader`]), and the turn
    /// the picture is shown with. A picture source of its own that can't open (the file went away
    /// after the render closed it) falls back to the file's shared source, which its audio may
    /// still hold open.
    fn picture_source(&self, path: &Arc<str>) -> Option<(SharedSource, Orientation)> {
        match self.picture_reader() {
            Reader::Pictures => self.opened(path, Reader::Pictures).or_else(|| self.opened(path, Reader::Shared)),
            Reader::Shared => self.opened(path, Reader::Shared),
        }
    }

    /// The slot of the source for `path` opened for `reader` (made empty when there is none),
    /// marked as asked for now ([`MediaPool::release_idle`]).
    fn slot(&self, path: &Arc<str>, reader: Reader) -> SourceSlot {
        let mut sources = lock(&self.sources);
        let step = self.step.load(Ordering::Relaxed);
        let open = sources.entry((path.clone(), reader)).or_insert_with(|| Open { slot: SourceSlot::default(), step });
        open.step = step;
        open.slot.clone()
    }

    /// The source for `path` opened for `reader`, and the turn the movie's picture is shown with.
    fn opened(&self, path: &Arc<str>, reader: Reader) -> Option<(SharedSource, Orientation)> {
        let slot = self.slot(path, reader);
        slot.get_or_init(|| {
            self.opened.fetch_add(1, Ordering::Relaxed);
            // Files on disk are read in place (the index, then samples on demand) rather than
            // whole; in-memory files (`add_bytes`) and wasm32 builds open from bytes.
            let in_memory = lock(&self.files).contains_key(&**path);
            let file = std::path::Path::new(&**path);
            let in_place = if in_memory { None } else { crate::stream::open_source(file) };
            let in_place = in_place.map(|src| {
                let turn = crate::stream::movie_header(file).map_or(Orientation::NoTransforms, |h| crate::probe::video_turn(&h));
                src.map(|src| (src, turn))
            });
            let opened = in_place.unwrap_or_else(|| {
                self.read(path).and_then(|bytes| {
                    let name = std::path::Path::new(&**path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                    let turn = crate::probe::video_turn(&bytes);
                    filmcraft_codecs::open_bytes(&name, bytes).map(|src| (src, turn)).map_err(MediaError::from)
                })
            });
            match opened {
                Ok(s) => Some(s),
                Err(e) => {
                    self.note_problem(path, "", &e);
                    None
                }
            }
        })
        .clone()
    }

    /// Remember why the movie at `path` cannot be shown (`codec` is the footage's, when known)
    /// and log it the first time. I/O errors (missing or unreadable file) are not decoder
    /// problems and are left to the missing-footage handling.
    fn note_problem(&self, path: &Arc<str>, codec: &str, e: &MediaError) {
        // A decode called off for a newer frame (#329) says nothing about the file either.
        if matches!(e, MediaError::Io(_) | MediaError::Cancelled) {
            return;
        }
        let reason = describe_problem(codec, e);
        if lock(&self.problems).insert(path.clone(), reason.clone()).is_none() {
            log::warn!("media: {path}: {reason}");
        }
    }

    /// The decoded frames of the animated GIF or WebP at `path`. The first caller decodes it, outside the
    /// map's lock; callers asking for the same file meanwhile wait for that.
    fn gif(&self, path: &Arc<str>) -> Result<Arc<Anim>> {
        let slot = lock(&self.gifs).entry(path.clone()).or_default().clone();
        let loaded = slot.get_or_init(|| {
            let loaded = self.read(path).and_then(|bytes| Anim::load(&bytes));
            loaded.map(Arc::new).map_err(|e| {
                log::warn!("media: cannot decode {path}: {e}");
                e.to_string()
            })
        });
        loaded.clone().map_err(|e| MediaError::Decode(format!("{path}: {e}")))
    }

    /// The frame index (in the footage's interpreted rate) shown at source time `t`, after
    /// looping (`loop_count`) and clamping to the footage's frames; `n` = frames per loop.
    fn frame_index(rate: FrameRate, t: Tick, n: i64, loops: u32) -> i64 {
        let n = n.max(1);
        let total = n.saturating_mul(loops.max(1) as i64);
        rate.frame_at(t).clamp(0, total - 1) % n
    }

    /// Number of frames in `d` at `rate` (rounded).
    fn frame_count(rate: FrameRate, d: Tick) -> i64 {
        let den = TICKS_PER_SECOND as i128 * rate.den as i128;
        ((d.0 as i128 * rate.num as i128 + den / 2) / den) as i64
    }

    fn locate(footage: &Footage, t: Tick, float: FloatCurve) -> Loc {
        let alpha = match footage.alpha {
            AlphaMode::Straight => 0,
            AlphaMode::Premultiplied => 1,
            AlphaMode::Ignore => 2,
        };
        let float = AlphaOp::of(footage, float).float_curve();
        let interp = alpha
            | (match float {
                FloatCurve::Srgb => 0,
                FloatCurve::Linear => 1,
                FloatCurve::Gamma24 => 2,
            } << 2);
        let matte = if footage.alpha == AlphaMode::Premultiplied { footage.premul_color.map(f32::to_bits) } else { [0; 3] };
        let key = |path: &str, frame| Key { path: path.into(), frame, interp, matte, size: (0, 0), nominal: (0, 0) };
        match footage.kind {
            FootageKind::Sequence if !footage.sequence.is_empty() => {
                let i = Self::frame_index(footage.frame_rate, t, footage.sequence_frames(), footage.loop_count);
                match footage.sequence_file(i) {
                    Some(path) => Loc { key: key(path, 0), media_t: None, float },
                    // A gap in the numbering: colour bars, as in After Effects.
                    None => Loc { key: key(PLACEHOLDER, ((footage.width as i64) << 32) | footage.height as i64), media_t: None, float },
                }
            }
            // A layer of a layered still is keyed by its layer index and size mode (smart objects
            // by their embedded file; PDF pages by their number).
            FootageKind::Still if footage.layer.is_some() || footage.page != 0 => {
                let l = footage.layer.as_ref().map_or(0, |l| 1 + l.index as i64 * 2 + l.layer_size as i64 + if l.embedded.is_some() { 1 << 40 } else { 0 })
                    + ((footage.page as i64) << 42);
                Loc { key: key(&footage.path, l), media_t: None, float }
            }
            FootageKind::Still | FootageKind::Sequence | FootageKind::Model | FootageKind::Data => Loc { key: key(&footage.path, 0), media_t: None, float },
            FootageKind::Video | FootageKind::Audio => {
                let rate = footage.frame_rate;
                let n = Self::frame_count(rate, footage.duration);
                let i = Self::frame_index(rate, t, n, footage.loop_count);
                // Middle of the frame in the file's own timing (robust to timestamp rounding).
                let native = footage.native_rate.unwrap_or(rate);
                let mt = native.tick_of(i) + Tick(native.frame_duration().0 / 2);
                Loc { key: key(&footage.path, i), media_t: Some(mt), float }
            }
        }
    }

    /// The frame of `footage` at `t`, resampled to `size` (see [`FootageSource::frame_at_size`]):
    /// a movie's made at that size directly, a still's resampled once and kept at that size.
    fn frame_at(self: &Arc<Self>, footage: &Footage, t: Tick, playback: bool, size: Option<(u32, u32)>, float: FloatCurve) -> Result<Arc<Image>> {
        if !footage.has_video {
            return Err(MediaError::Unsupported(format!("{}: no video", footage.path)));
        }
        let mut loc = Self::locate(footage, t, float);
        let movie = loc.media_t.is_some();
        // A still asked for at its own size is the unsized frame (one decode, one entry).
        if let Some(size) = size.filter(|s| movie || *s != (footage.width, footage.height)) {
            loc.key.size = size;
            if !movie {
                loc.key.nominal = (footage.width, footage.height);
            }
        }
        let mut c = lock(&self.cache);
        // sequential playback of a movie: read the next frame ahead
        if playback && movie {
            let prev = c.last.insert(loc.key.path.clone(), loc.key.frame);
            if prev == Some(loc.key.frame - 1) && self.read_ahead.load(Ordering::Relaxed) {
                let next_t = t + footage.frame_rate.frame_duration();
                let mut next = Self::locate(footage, next_t, float);
                next.key.size = loc.key.size;
                if next.key != loc.key && !c.lru.map.contains_key(&next.key) && !c.inflight.contains(&next.key) {
                    self.prefetch(footage, next_t, size, float);
                }
            }
        }
        loop {
            if let Some(img) = c.lru.get(&loc.key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(img);
            }
            if !c.inflight.contains(&loc.key) {
                break;
            }
            // Another thread (or the read-ahead) is decoding this frame. Never block a rayon
            // worker on that: while the decoder waits on its own parallel join, the worker can
            // steal this very job, and waiting here would then wait on a decode further down
            // its own stack (deadlock). Decode a private copy instead.
            if rayon::current_thread_index().is_some() {
                self.misses.fetch_add(1, Ordering::Relaxed);
                drop(c);
                return Ok(Arc::new(self.decode(&loc, footage)?));
            }
            c = self.done.wait(c).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        c.inflight.insert(loc.key.clone());
        drop(c);
        let _guard = Inflight { inner: self, key: loc.key.clone() };
        let img = Arc::new(self.decode(&loc, footage)?);
        lock(&self.cache).lru.insert(loc.key, img.clone(), self.budget());
        Ok(img)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn prefetch(self: &Arc<Self>, footage: &Footage, t: Tick, size: Option<(u32, u32)>, float: FloatCurve) {
        let (me, f) = (self.clone(), footage.clone());
        self.prefetched.fetch_add(1, Ordering::Relaxed);
        // A plain thread, not a rayon job: the decode below parallelises with rayon itself.
        std::thread::spawn(move || {
            let _ = me.frame_at(&f, t, false, size, float);
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn prefetch(self: &Arc<Self>, _footage: &Footage, _t: Tick, _size: Option<(u32, u32)>, _float: FloatCurve) {}

    /// A still (image file or sequence frame) at `loc.key.size` when one is set: resampled as the
    /// renderer would ([`FootageSource::frame_at_size`]), so the pool keeps the reduced frame and a
    /// reduced-resolution render neither decodes nor resamples a large still again for every frame
    /// (#1812: 21120-px plates at Quarter). Like the renderer's `footage_buf`, an image decoded
    /// larger than the footage's nominal size (a smart object's pixels) is resampled too, and one
    /// decoded smaller is not (the renderer scales the layer instead).
    fn still_at_size(img: Image, loc: &Loc, footage: &Footage) -> Image {
        let (w, h) = loc.key.size;
        let smaller = img.width == 0 || footage.width as f64 / img.width as f64 > 1.0 + 1e-9;
        if w == 0 || h == 0 || smaller || (w, h) == (img.width, img.height) {
            return img;
        }
        effectcraft_raster::resample(&img, w, h)
    }

    fn decode(&self, loc: &Loc, footage: &Footage) -> Result<Image> {
        match loc.media_t {
            None => self.decode_still(loc, footage).map(|img| Self::still_at_size(img, loc, footage)),
            Some(_) => self.decode_media(loc, footage),
        }
    }

    fn decode_still(&self, loc: &Loc, footage: &Footage) -> Result<Image> {
        let op = AlphaOp::of(footage, loc.float);
        let path = &loc.key.path;
        if &**path == PLACEHOLDER {
            return Ok(colour_bars(footage.width, footage.height));
        }
        let bytes = self.read(path)?;
        if let Some(img) = crate::dpx::decode(path, &bytes)? {
            return Ok(dynamic_to_image(&img, op));
        }
        if let Some(img) = crate::layered::decode(path, &bytes, footage, op)? {
            return Ok(img);
        }
        // (an OpenEXR compression the decoder lacks came out transparent or as a missing
        // channel error)
        if path.to_ascii_lowercase().ends_with(".exr")
            && let Some(why) = crate::exr_channels::unsupported_compression(&bytes)
        {
            return Err(MediaError::Decode(format!("{path}: {why}")));
        }
        let loaded = crate::probe::load_oriented(&bytes, crate::still_format(path, &bytes));
        let img = match loaded {
            Ok(img) => img,
            // A multi-layer OpenEXR file without an unnamed RGB layer: its colour layer.
            Err(e) => crate::exr_channels::cached_layered_image(path, bytes).ok_or_else(|| MediaError::Decode(format!("{path}: {e}")))?,
        };
        Ok(dynamic_to_image(&img, op))
    }

    fn decode_media(&self, loc: &Loc, footage: &Footage) -> Result<Image> {
        let op = AlphaOp::of(footage, loc.float);
        let path = &loc.key.path;
        match loc.media_t {
            None => self.decode_still(loc, footage),
            Some(mt) if gif_anim::is_animated(footage) => {
                let anim = self.gif(path)?;
                let frame = anim.frame_at(Tick(mt.0)).ok_or_else(|| MediaError::Decode(format!("{path}: no frames")))?;
                Ok(dynamic_to_image(frame, op))
            }
            Some(mt) => {
                let (src, turn) = self.picture_source(path).ok_or_else(|| MediaError::Io(format!("{path}: cannot open")))?;
                let vf = src
                    .video_frame(FrameRequest::full(filmcraft_time::Tick(mt.0)))
                    .map_err(MediaError::from)
                    .inspect_err(|e| self.note_problem(path, &footage.codec, e))?;
                // A movie shown turned a quarter (a phone held upright): the footage's size and
                // the render's are the turned picture's, the decoded frame's are the file's.
                let side = |(w, h): (u32, u32)| if crate::probe::swaps_sides(turn) { (h, w) } else { (w, h) };
                // A reduced-resolution render's frame, made at its size (a frame of the footage's
                // own size only, as the renderer resamples only those).
                let (w, h) = side(loc.key.size);
                if w > 0
                    && (vf.width, vf.height) == side((footage.width, footage.height))
                    && let Some(img) = frame_to_image_scaled(&vf, w, h)
                {
                    return Ok(turned(img, turn));
                }
                let buf = lock(&self.cache).lru.take_spare(vf.width as usize * vf.height as usize);
                Ok(turned(frame_to_image_in(&vf, op, buf), turn))
            }
        }
    }
}

impl FootageSource for MediaPool {
    fn set_conform_folder(&self, folder: Option<std::path::PathBuf>) {
        MediaPool::set_conform_folder(self, folder);
    }
    fn model(&self, _item: ItemId, footage: &Footage) -> Option<Arc<effectcraft_model::Model>> {
        if footage.missing || footage.kind != FootageKind::Model {
            return None;
        }
        self.inner.model(&footage.path)
    }
    fn set_cache_budget(&self, bytes: usize) {
        self.set_budget(bytes);
    }
    fn purge(&self) {
        self.clear_frames();
    }
    fn forget(&self, path: &str) {
        MediaPool::forget(self, path);
    }
    fn set_idle_release(&self, frames: Option<u32>) {
        MediaPool::set_idle_release(self, frames);
    }
    fn idle_release(&self) -> Option<u32> {
        MediaPool::idle_release(self)
    }
    fn release_idle(&self, frames: u64) {
        MediaPool::release_idle(self, frames);
    }
    fn cache_budget(&self) -> Option<usize> {
        Some(self.budget())
    }
    /// A movie's decoder gives up on a seek (before it, and between samples while it decodes
    /// forward from the keyframe) once `cancel` is set: a scrub's superseded frames no longer
    /// hold the decoder the frame under the pointer waits for (#329).
    fn cancellable(&self, cancel: &Arc<AtomicBool>, f: &mut dyn FnMut()) {
        filmcraft_media::cancel::with_cancel(cancel, f)
    }
    fn frame(&self, item: ItemId, footage: &Footage, t: Tick) -> Option<Arc<Image>> {
        self.frame_sized(item, footage, t, None, FloatCurve::Srgb)
    }
    fn frame_at_size(&self, item: ItemId, footage: &Footage, t: Tick, width: u32, height: u32) -> Option<Arc<Image>> {
        self.frame_sized(item, footage, t, Some((width, height)), FloatCurve::Srgb)
    }
    fn frame_with(&self, item: ItemId, footage: &Footage, t: Tick, size: Option<(u32, u32)>, float: FloatCurve) -> Option<Arc<Image>> {
        self.frame_sized(item, footage, t, size, float)
    }

    fn problem(&self, footage: &Footage) -> Option<String> {
        if footage.missing || footage.kind != FootageKind::Video {
            return None;
        }
        lock(&self.inner.problems).get(footage.path.as_str()).cloned()
    }

    fn vector_frame(&self, _item: ItemId, footage: &Footage, scale: f64) -> Option<Arc<Image>> {
        if footage.missing || !effectcraft_render::is_vector_footage(footage) {
            return None;
        }
        let bytes = self.inner.read(&footage.path).ok()?;
        crate::layered::rasterize_vector(&footage.path, &bytes, footage.layer.as_ref(), footage.page, scale).map(Arc::new)
    }

    fn aux(&self, _item: ItemId, footage: &Footage, t: Tick) -> Option<Arc<effectcraft_raster::AuxChannels>> {
        if footage.missing || !footage.has_video {
            return None;
        }
        let loc = Inner::locate(footage, t, FloatCurve::Srgb);
        if loc.media_t.is_some() || !loc.key.path.to_ascii_lowercase().ends_with(".exr") {
            return None;
        }
        let path = loc.key.path.clone();
        crate::exr_channels::cached(&path, || self.inner.read(&path).ok())
    }

    fn audio(&self, _item: ItemId, footage: &Footage, t: Tick, frames: usize, rate: u32) -> Option<Vec<f32>> {
        if !footage.has_audio || footage.missing {
            return None;
        }
        Some(self.audio_samples(footage, t, frames, rate))
    }
}

/// `frames` stereo sample frames (interleaved) of `src` from sample `s0` at `rate` Hz, which
/// FilmCraft serves as decoded when it is the source's own rate. Mono is duplicated to both
/// channels, multichannel sources are downmixed ([`crate::downmix`]); silence before the start. An error where decoding fails.
fn read_stereo(src: &SharedSource, s0: i64, frames: usize, rate: u32) -> std::result::Result<Vec<f32>, String> {
    let mut out = vec![0.0; frames * 2];
    let skip = usize::try_from(s0.saturating_neg()).unwrap_or(0);
    if skip >= frames {
        return Ok(out);
    }
    let buf = src.audio(s0.max(0), frames - skip, rate).map_err(|e| e.to_string())?;
    let Some(dst) = skip.checked_mul(2).and_then(|i| out.get_mut(i..)) else { return Ok(out) };
    let aac = src.info().audio.as_ref().is_some_and(|a| a.codec.starts_with("AAC"));
    crate::downmix::downmix_into(&buf.channels, aac, dst);
    Ok(out)
}

/// The one-line reason a movie shows no picture: its `codec` (the probe's, e.g. "H.264 (High
/// 4:2:2)") and what the decoder said, without the error wrappers it passed through.
fn describe_problem(codec: &str, e: &MediaError) -> String {
    let mut detail = match e {
        MediaError::Io(s) | MediaError::Unsupported(s) | MediaError::Decode(s) => s.as_str(),
        MediaError::Cancelled => "the decode was called off",
    };
    let mut unsupported = matches!(e, MediaError::Unsupported(_));
    while let Some(rest) = ["decode error: ", "unsupported media: ", "unsupported: ", "I/O error: "].iter().find_map(|p| detail.strip_prefix(p)) {
        unsupported |= detail.starts_with("unsupported");
        detail = rest;
    }
    let what = if codec.is_empty() { "This video" } else { codec };
    if unsupported { format!("{what} isn't supported by the built-in decoder: {detail}") } else { format!("{what} can't be decoded: {detail}") }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh, empty folder for a test under `target/`.
    fn test_dir(name: &str) -> std::path::PathBuf {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target").join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// One second of a 440 Hz tone, mono 16-bit PCM at `rate` Hz, written as `dir/name`.
    fn tone_wav(dir: &std::path::Path, name: &str, rate: u32) -> Footage {
        let pcm: Vec<u8> = (0..rate).flat_map(|n| (((n as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 8000.0) as i16).to_le_bytes()).collect();
        let mut wav = b"RIFF".to_vec();
        wav.extend((36 + pcm.len() as u32).to_le_bytes());
        wav.extend(b"WAVEfmt ");
        wav.extend(16u32.to_le_bytes());
        wav.extend([1u16, 1].iter().flat_map(|v| v.to_le_bytes()));
        wav.extend(rate.to_le_bytes());
        wav.extend((rate * 2).to_le_bytes());
        wav.extend([2u16, 16].iter().flat_map(|v| v.to_le_bytes()));
        wav.extend(b"data");
        wav.extend((pcm.len() as u32).to_le_bytes());
        wav.extend(&pcm);
        let path = dir.join(name);
        std::fs::write(&path, &wav).unwrap();
        Footage { path: path.to_string_lossy().into(), kind: FootageKind::Audio, duration: Tick::from_seconds_f64(1.0), has_audio: true, ..Default::default() }
    }

    /// Read a little of `footage`'s audio (which opens its source).
    fn read_audio(pool: &MediaPool, footage: &Footage) {
        assert_eq!(pool.audio_samples(footage, Tick::ZERO, 256, 48_000).len(), 512);
    }

    /// Open `footage` for pictures as a movie frame read does (any file opens; these tests use
    /// short WAV files).
    fn read_pictures(pool: &MediaPool, footage: &Footage) {
        assert!(pool.inner.picture_source(&footage.path.as_str().into()).is_some());
    }

    /// The paths of the sources `pool` holds open for pictures of their own, sorted.
    fn picture_sources(pool: &MediaPool) -> Vec<String> {
        let mut open: Vec<String> = lock(&pool.inner.sources).keys().filter(|(_, r)| *r == Reader::Pictures).map(|(k, _)| k.to_string()).collect();
        open.sort();
        open
    }

    /// An export moving through a composition: with idle release on, each movie's picture source
    /// closes once it has gone that many steps without a read, the one read longest ago first,
    /// and a closed source opens again when it is read.
    #[test]
    fn idle_release_closes_picture_sources_in_the_order_they_went_unused() {
        let dir = test_dir("test-idle-release-order");
        let [a, b, c] = ["a.wav", "b.wav", "c.wav"].map(|n| tone_wav(&dir, n, 48_000));
        let pool = MediaPool::new();
        pool.set_idle_release(Some(1));
        assert_eq!(pool.idle_release(), Some(1));
        for f in [&a, &b, &c] {
            read_pictures(&pool, f);
        }
        assert_eq!(pool.release_idle(1), 0, "every source was read this step");
        assert_eq!(pool.stats().open, 3);
        for f in [&b, &c] {
            read_pictures(&pool, f);
        }
        assert_eq!(pool.release_idle(1), 1);
        assert_eq!(picture_sources(&pool), [b.path.clone(), c.path.clone()]);
        read_pictures(&pool, &c);
        assert_eq!(pool.release_idle(1), 1);
        assert_eq!(picture_sources(&pool), std::slice::from_ref(&c.path));
        assert_eq!(pool.release_idle(1), 1);
        assert!(picture_sources(&pool).is_empty());
        assert_eq!((pool.stats().opened, pool.stats().released, pool.stats().open), (3, 3, 0));
        read_pictures(&pool, &a);
        assert_eq!(picture_sources(&pool), std::slice::from_ref(&a.path));
        assert_eq!(pool.stats().opened, 4, "a closed source opens again");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A source a reader still holds, or one still being opened, is never closed however long
    /// ago it was asked for; it closes at the first step after the reader lets go.
    #[test]
    fn a_source_being_read_or_opened_is_never_released() {
        let dir = test_dir("test-idle-release-in-use");
        let a = tone_wav(&dir, "a.wav", 48_000);
        let pool = MediaPool::new();
        pool.set_idle_release(Some(1));
        let reading = pool.inner.opened(&a.path.as_str().into(), Reader::Pictures).unwrap();
        let opening: Arc<str> = dir.join("opening.mp4").to_string_lossy().into();
        let slot = pool.inner.slot(&opening, Reader::Pictures);
        for _ in 0..5 {
            assert_eq!(pool.release_idle(1), 0);
        }
        assert_eq!(picture_sources(&pool).len(), 2);
        drop(reading);
        assert_eq!(pool.release_idle(1), 1, "the reader is done");
        assert_eq!(picture_sources(&pool), [opening.to_string()]);
        assert!(slot.get().is_none(), "still opening");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Audio readers stay open with idle release on: a track's decoded audio depends on what its
    /// decoder decoded before (AAC is decoded on from a packet of pre-roll after a seek, and keeps
    /// the packets it decoded), so a reopened one could sound different where the render plays a
    /// passage again. They keep no decoded pictures.
    #[test]
    fn audio_readers_are_not_released() {
        let dir = test_dir("test-idle-release-audio");
        let a = tone_wav(&dir, "a.wav", 48_000);
        let pool = MediaPool::new();
        pool.set_idle_release(Some(1));
        read_audio(&pool, &a);
        for _ in 0..5 {
            assert_eq!(pool.release_idle(1), 0);
        }
        assert_eq!((pool.stats().open, pool.stats().released), (1, 0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A clip whose picture source closed and can't open again (here its file went away during
    /// the render) shows its pictures from the file's shared source, which its audio holds open,
    /// rather than nothing for the rest of the render.
    #[test]
    fn a_picture_source_that_cannot_reopen_falls_back_to_the_shared_one() {
        let dir = test_dir("test-idle-release-reopen-fails");
        let a = tone_wav(&dir, "a.wav", 48_000);
        let pool = MediaPool::new();
        pool.set_idle_release(Some(1));
        read_audio(&pool, &a);
        read_pictures(&pool, &a);
        for _ in 0..2 {
            pool.release_idle(1);
        }
        assert!(picture_sources(&pool).is_empty(), "closed");
        std::fs::remove_file(&a.path).unwrap();
        read_pictures(&pool, &a);
        read_pictures(&pool, &a);
        assert_eq!(pool.stats().open, 1, "the shared source");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Steps count the frames a render finished: a source closes once that many frames have gone
    /// by since the batch it was last read in, however many frames a batch has.
    #[test]
    fn idle_release_counts_frames() {
        let dir = test_dir("test-idle-release-frames");
        let a = tone_wav(&dir, "a.wav", 48_000);
        for batch in [1u64, 4, 16] {
            let pool = MediaPool::new();
            pool.set_idle_release(Some(48));
            read_pictures(&pool, &a);
            let mut done = batch;
            pool.release_idle(batch);
            while picture_sources(&pool).len() == 1 {
                pool.release_idle(batch);
                done += batch;
            }
            // Read in the first batch; closed after the first batch that ends 48 or more frames later.
            assert_eq!(done, batch + 48_u64.div_ceil(batch) * batch, "batches of {batch}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Interactive playback keeps every decoder open (scrubbing back to a clip must not decode it
    /// from its keyframe again), and a movie's pictures and audio share one: without idle
    /// release, steps close nothing.
    #[test]
    fn without_idle_release_sources_stay_open_and_shared() {
        let dir = test_dir("test-idle-release-off");
        let [a, b] = ["a.wav", "b.wav"].map(|n| tone_wav(&dir, n, 48_000));
        let pool = MediaPool::new();
        assert_eq!(pool.idle_release(), None);
        for f in [&a, &b] {
            read_pictures(&pool, f);
            read_audio(&pool, f);
        }
        for _ in 0..10 {
            assert_eq!(pool.release_idle(1), 0);
        }
        assert_eq!((pool.stats().opened, pool.stats().open, pool.stats().released), (2, 2, 0), "one source per file");
        pool.set_idle_release(Some(1));
        pool.set_idle_release(None);
        assert_eq!(pool.release_idle(1), 0, "turned off again");
        assert_eq!(pool.stats().open, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Conforming a long file's audio reads every chunk at its own time: the sample index times
    /// ticks per second overflowed `i64` past 12:36 at 48 kHz, and later chunks came out silent
    /// (#172).
    #[test]
    fn conform_chunks_keep_their_times_on_long_files() {
        for rate in [24_000u32, 44_100, 48_000, 96_000] {
            let mut last = Tick(-1);
            // Every chunk start of a 2-hour file.
            for at in (0..rate as usize * 7200).step_by(1 << 16) {
                let t = sample_time(at, rate);
                assert!(t > last, "{rate} Hz, frame {at}: {t:?} after {last:?}");
                assert!((t.seconds() - at as f64 / rate as f64).abs() < 1e-6, "{rate} Hz, frame {at}");
                last = t;
            }
        }
        assert!((sample_time(36_372_480, 48_000).seconds() - 757.76).abs() < 1e-9);
    }

    /// #408: footage whose header claims a month of audio isn't conformed: the conform thread
    /// allocated the whole conformed track up front (a terabyte here), and the failed allocation
    /// aborted the app. Its audio is read from the decoder instead (silence: this file doesn't
    /// decode).
    #[test]
    fn a_header_claiming_a_month_of_audio_is_not_conformed() {
        let dir = test_dir("test-conform-month");
        let path = dir.join("month.flac");
        std::fs::write(&path, b"fLaC, not really").unwrap();
        let footage = Footage {
            path: path.to_string_lossy().into(),
            kind: FootageKind::Audio,
            duration: Tick::from_seconds_f64(30.0 * 86_400.0),
            has_audio: true,
            ..Default::default()
        };
        assert_eq!(conform_frames(&footage, 48_000), None);
        assert_eq!(conform_frames(&Footage { duration: Tick::from_seconds_f64(60.0), ..footage.clone() }, 48_000), Some(2_880_000));
        let pool = MediaPool::new();
        let conformed = dir.join("conformed");
        pool.set_conform_folder(Some(conformed.clone()));
        let s = pool.audio_samples(&footage, Tick::from_seconds_f64(3600.0), 1024, 48_000);
        assert_eq!(s.len(), 2048);
        assert!(s.iter().all(|v| *v == 0.0));
        assert!(lock(&pool.inner.conforming).is_empty(), "no conform thread started");
        assert!(!conformed.exists(), "nothing was written");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// #324: audio that fails to decode isn't conformed. A conformed file of silence was written
    /// instead, and as it is found by the file's path, size and date the footage stayed silent
    /// in every later session (a renamed copy played).
    #[test]
    fn audio_that_fails_to_decode_is_not_conformed() {
        let dir = test_dir("test-conform-fail");
        let path = dir.join("song.mp3");
        std::fs::write(&path, b"ID3, not really").unwrap();
        let footage = Footage {
            path: path.to_string_lossy().into(),
            kind: FootageKind::Audio,
            duration: Tick::from_seconds_f64(2.0),
            has_audio: true,
            ..Default::default()
        };
        let pool = MediaPool::new();
        let conformed = dir.join("conformed");
        pool.set_conform_folder(Some(conformed.clone()));
        let file = pool.conformed_path(&footage.path, 48_000).unwrap();
        assert!(pool.audio_samples(&footage, Tick::ZERO, 1024, 48_000).iter().all(|v| *v == 0.0));
        for _ in 0..500 {
            if lock(&pool.inner.conforming).is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(lock(&pool.inner.conforming).is_empty(), "the conform thread finished");
        assert!(!file.exists() && !file.with_extension("tmp").exists(), "no conformed file was left");
        assert!(lock(&pool.inner.conform_failed).contains(&file));
        // Later reads go to the decoder without trying again.
        pool.audio_samples(&footage, Tick::ZERO, 1024, 48_000);
        assert!(lock(&pool.inner.conforming).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Regression test for #382 (#274): resampling footage audio divided by the requested mix
    // rate, so a rate of 0 panicked (division by zero); the read must come back instead.
    #[test]
    fn decoding_audio_at_a_zero_mix_rate_does_not_divide_by_zero() {
        let dir = test_dir("test-zero-rate");
        // 44.1 kHz: not the 48 kHz mix rate.
        let footage = tone_wav(&dir, "tone.wav", 44_100);
        let pool = MediaPool::new();
        // The file really is read, and resampled, at a normal rate...
        let tone = pool.decode_audio(&footage, Tick::from_seconds_f64(0.1), 1024, 48_000);
        assert!(tone.iter().any(|v| v.abs() > 0.05), "the tone is audible at 48 kHz");
        // ...and a mix rate of 0 (which `audio_samples` itself turns away) returns a read of the
        // requested length instead of dividing by it.
        let degenerate = pool.try_decode_audio(&footage, Tick::from_seconds_f64(0.1), 64, 0).unwrap();
        assert_eq!(degenerate.len(), 128);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn frame_index_loops_and_clamps() {
        let r = FrameRate::FPS_30;
        assert_eq!(Inner::frame_index(r, r.tick_of(-3), 10, 1), 0);
        assert_eq!(Inner::frame_index(r, r.tick_of(4), 10, 1), 4);
        assert_eq!(Inner::frame_index(r, r.tick_of(25), 10, 1), 9);
        assert_eq!(Inner::frame_index(r, r.tick_of(25), 10, 3), 5);
        assert_eq!(Inner::frame_index(r, r.tick_of(45), 10, 3), 9);
        assert_eq!(Inner::frame_count(r, Tick(2 * TICKS_PER_SECOND)), 60);
    }

    #[test]
    fn lru_respects_budget_and_recycles() {
        let mut c = Lru::default();
        let img = || Arc::new(Image::new(16, 16)); // 4 KiB + overhead
        let k = |i| Key { path: "a".into(), frame: i, interp: 0, matte: [0; 3], size: (0, 0), nominal: (0, 0) };
        let budget = 3 * (4096 + 64);
        for i in 0..10 {
            c.insert(k(i), img(), budget);
        }
        assert_eq!(c.map.len(), 3);
        assert!(c.get(&k(9)).is_some() && c.get(&k(6)).is_none());
        // touching 7 makes 8 the oldest
        c.get(&k(7));
        c.insert(k(10), img(), budget);
        assert!(c.get(&k(7)).is_some() && c.get(&k(8)).is_none());
        assert_eq!(c.spare.len(), MAX_SPARE);
        assert_eq!(c.take_spare(256).len(), 256);
        assert!(c.take_spare(17).is_empty());
    }

    /// #482: an OpenEXR file with HTJ2K compression (which the decoder lacks) says so on import
    /// and when its frame is read, instead of "no non-deep rgb channels" or a transparent frame.
    #[test]
    fn htj2k_exr_is_reported_as_unsupported() {
        use exr::prelude::*;
        let channels =
            AnyChannels::sort(vec![AnyChannel::new("R", FlatSamples::F32(vec![0.5; 4])), AnyChannel::new("G", FlatSamples::F32(vec![0.5; 4]))].into());
        let image = exr::image::Image::from_layer(Layer::new((2, 2), LayerAttributes::default(), Encoding::UNCOMPRESSED, channels));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write().to_buffered(&mut bytes).unwrap();
        let mut bytes = bytes.into_inner();
        assert_eq!(crate::exr_channels::unsupported_compression(&bytes), None);
        // The header's compression attribute (name, type, size 1, value) says HTJ2K32 (11).
        let tag = b"compression\0compression\0\x01\0\0\0";
        let at = bytes.windows(tag.len()).position(|w| w == tag).unwrap() + tag.len();
        bytes[at] = 11;
        let b: Arc<[u8]> = bytes.into();
        let e = crate::probe_bytes("/shot.exr", b.clone()).unwrap_err().to_string();
        assert!(e.contains("/shot.exr: OpenEXR HTJ2K compression isn't supported yet"), "{e}");
        let pool = MediaPool::new();
        pool.add_bytes("/shot.exr", b);
        let f = Footage { path: "/shot.exr".into(), kind: FootageKind::Still, width: 2, height: 2, has_video: true, ..Default::default() };
        let e = pool.frame_at(&f, Tick::ZERO).unwrap_err().to_string();
        assert!(e.contains("HTJ2K compression isn't supported yet"), "{e}");
    }

    /// #411: Interpret Footage ▸ Preserve RGB reads a float OpenEXR's values as the file stores
    /// them (no automatic linear → sRGB encode), so an OCIO effect is the only conversion; the
    /// default interpretation still encodes. Both interpretations of one file decode separately.
    #[test]
    fn preserve_rgb_reads_float_exr_values_unconverted() {
        use exr::prelude::*;
        let (w, h) = (2usize, 2usize);
        let ch = |n: &str, v: f32| AnyChannel::new(n, FlatSamples::F32(vec![v; w * h]));
        let channels = AnyChannels::sort(vec![ch("R", 0.18), ch("G", 0.5), ch("B", 2.0), ch("A", 1.0)].into());
        let image = exr::image::Image::from_layer(Layer::new((w, h), LayerAttributes::default(), Encoding::FAST_LOSSLESS, channels));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write().to_buffered(&mut bytes).unwrap();
        let b: Arc<[u8]> = bytes.into_inner().into();
        let pool = MediaPool::new();
        pool.add_bytes("/render.exr", b.clone());
        let f = crate::probe_bytes("/render.exr", b).unwrap();
        let encoded = pool.frame_at(&f, Tick::ZERO).unwrap().get(0, 0);
        assert!((encoded[0] - 0.4613).abs() < 1e-3 && (encoded[1] - 0.7354).abs() < 1e-3 && encoded[2] > 1.3, "sRGB-encoded: {encoded:?}");
        let raw = pool.frame_at(&Footage { preserve_rgb: true, ..f.clone() }, Tick::ZERO).unwrap().get(0, 0);
        assert_eq!(raw, [0.18, 0.5, 2.0, 1.0]);
        assert_eq!(pool.frame_at(&f, Tick::ZERO).unwrap().get(0, 0), encoded, "the default interpretation is cached apart");
    }

    /// #750: float footage is handed over as the renderer asks: sRGB-encoded, with Working
    /// Gamma 2.4's pure power curve, or as stored (Interpret As Linear Light encodes it then);
    /// Preserve RGB reads it as stored whatever is asked. Each decodes and caches apart.
    #[test]
    fn float_exr_is_encoded_with_the_requested_curve() {
        use exr::prelude::*;
        let (w, h) = (2usize, 2usize);
        let ch = |n: &str, v: f32| AnyChannel::new(n, FlatSamples::F32(vec![v; w * h]));
        let channels = AnyChannels::sort(vec![ch("R", 0.051), ch("G", 0.18), ch("B", 2.0), ch("A", 1.0)].into());
        let image = exr::image::Image::from_layer(Layer::new((w, h), LayerAttributes::default(), Encoding::FAST_LOSSLESS, channels));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write().to_buffered(&mut bytes).unwrap();
        let b: Arc<[u8]> = bytes.into_inner().into();
        let pool = MediaPool::new();
        pool.add_bytes("/linear.exr", b.clone());
        let f = crate::probe_bytes("/linear.exr", b).unwrap();
        let px = |f: &Footage, float| pool.frame_with(ItemId(1), f, Tick::ZERO, None, float).unwrap().get(0, 0);
        let near = |o: [f32; 4], want: [f32; 3]| (0..3).all(|i| (o[i] - want[i]).abs() < 1e-3);
        let srgb = px(&f, FloatCurve::Srgb);
        assert!(near(srgb, [0.2503, 0.4614, 1.3532]), "sRGB: {srgb:?}");
        assert_eq!(pool.frame(ItemId(1), &f, Tick::ZERO).unwrap().get(0, 0), srgb, "frame() is sRGB");
        let g24 = px(&f, FloatCurve::Gamma24);
        assert!(near(g24, [0.2898, 0.4898, 2.0f32.powf(1.0 / 2.4)]), "2.4: {g24:?}");
        assert_eq!(px(&f, FloatCurve::Linear), [0.051, 0.18, 2.0, 1.0]);
        let preserve = Footage { preserve_rgb: true, ..f.clone() };
        assert_eq!(px(&preserve, FloatCurve::Gamma24), [0.051, 0.18, 2.0, 1.0]);
        // Each curve is cached apart: asking again returns the same pixels.
        assert_eq!(px(&f, FloatCurve::Srgb), srgb);
        assert_eq!(px(&f, FloatCurve::Gamma24), g24);
    }

    /// A model's material library is read beside it; one named by a path to another computer
    /// is not read.
    #[test]
    fn model_files_on_other_computers_are_not_read() {
        let dir = std::env::temp_dir().join(format!("ec-model-files-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("red.mtl"), "newmtl red\nKd 1 0 0\n").unwrap();
        let pool = MediaPool::new();
        let red = |name: &str, mtl: &str| {
            let path = dir.join(name);
            std::fs::write(&path, format!("mtllib {mtl}\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl red\nf 1 2 3\n")).unwrap();
            let model = pool.inner.model(&path.to_string_lossy()).unwrap();
            model.materials.iter().any(|m| m.base_color[..3] == [1.0, 0.0, 0.0])
        };
        assert!(red("beside.obj", "red.mtl"));
        assert!(!red("unc.obj", r"\\192.0.2.1\share\red.mtl"));
        assert!(!red("url.obj", "file://192.0.2.1/s/red.mtl"));
        // Linux reads `//dir/red.mtl` as the local file.
        #[cfg(unix)]
        assert!(!red("slashes.obj", &format!("/{}", dir.join("red.mtl").display())));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
