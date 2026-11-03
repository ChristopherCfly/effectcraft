//! Footage whose frame times are rounded to whole milliseconds (#1442), as phones and messaging
//! apps write MP4s: a 30 fps clip's frames last 33 or 34 ms, a 60 fps clip's 16 or 17 ms. It
//! imports at its true rate, a comp made from it (New Comp from Selection takes the footage's
//! rate) renders every source frame once, and the H.264 output has that rate and frame count.
//!
//! The fixtures are written with FilmCraft's own encoder and muxer ([`crate::binary_movie`]).

use crate::out::out_dir;
use std::path::Path;
use std::sync::Arc;

use crate::binary_movie::{H, W, shown, write_movie};

use effectcraft_color::Label;
use effectcraft_export::render_queue::{OutputFormat, OutputModule, RenderSettings, TimeSpan};
use effectcraft_export::{Job, export};
use effectcraft_media::MediaPool;
use effectcraft_project::{Comp, Footage, ItemKind, LayerSource, Project, build};
use effectcraft_time::{FrameRate, Tick};

/// Durations of `n` frames at `rate` with their times rounded to whole milliseconds.
fn ms_rounded(rate: FrameRate, n: i64) -> Vec<u32> {
    let at = |k: i64| (2 * k * 1000 * rate.den + rate.num) / (2 * rate.num);
    (0..n).map(|k| u32::try_from(at(k + 1) - at(k)).expect("duration")).collect()
}

fn frame_count(f: &Footage) -> i64 {
    f.frame_rate.frame_at(f.duration - Tick(1)) + 1
}

/// Every frame of a movie in order, as indices.
fn frames_of(f: &Footage) -> Vec<u32> {
    let pool = MediaPool::new();
    (0..frame_count(f)).map(|k| shown(&pool.frame_at(f, f.frame_rate.tick_of(k)).expect("decode"))).collect()
}

/// Import `path`, make a comp from it as New Comp from Selection does (the footage's size, rate
/// and duration), render it to H.264 and return the output's footage.
fn comp_from_footage(path: &Path, out: &Path) -> Footage {
    let footage = effectcraft_media::probe(path).expect("probe");
    let mut p = Project::default();
    let fid = p.add_item("clip.mp4", Label::Aqua, None, ItemKind::Footage(footage.clone()));
    let rate = p.item(fid).and_then(|it| it.frame_rate()).expect("a movie has a rate");
    let mut comp = Comp::new(W, H, rate, footage.duration);
    comp.layers.push(build::layer(&mut p, &comp, "clip.mp4", LayerSource::Footage { item: fid }, (W, H), None));
    let cid = p.add_item("clip", Label::Sandstone, None, ItemKind::Comp(comp.into()));
    let pool = MediaPool::new();
    let settings = RenderSettings { time_span: TimeSpan::LengthOfComp, ..Default::default() };
    let om = OutputModule { bitrate_kbps: 4000, ..OutputModule::for_format(OutputFormat::H264) };
    let out_path = out.to_string_lossy().to_string();
    let job = Job {
        project: &p,
        footage: &pool,
        expr: None,
        accel: None,
        comp: cid,
        settings: &settings,
        output: &om,
        path: &out_path,
        sink: None,
        options: Default::default(),
        nested_switches: true,
    };
    let r = export(&job, &mut |_| true).expect("export");
    assert_eq!(r.frames as i64, frame_count(&footage), "the render has the footage's frames");
    effectcraft_media::probe(out).expect("probe the render")
}

#[test]
fn millisecond_rounded_footage_keeps_its_rate_through_comp_and_render() {
    let d = out_dir("ms_rounded");
    for (rate, frames) in [(FrameRate::FPS_30, 120), (FrameRate::FPS_60, 240)] {
        let path = d.join(format!("ms-{}.mp4", rate.num));
        write_movie(&path, &ms_rounded(rate, frames), 0);
        let f = effectcraft_media::probe(&path).expect("probe");
        assert_eq!((f.frame_rate, frame_count(&f)), (rate, frames), "{rate:?} footage");
        let bytes: Arc<[u8]> = std::fs::read(&path).expect("read").into();
        let from_bytes = effectcraft_media::probe_bytes(&path.to_string_lossy(), bytes).expect("probe bytes");
        assert_eq!(from_bytes.frame_rate, rate, "{rate:?} footage read from memory");
        let all: Vec<u32> = (0..frames as u32).collect();
        assert_eq!(frames_of(&f), all, "{rate:?} footage: each source frame once");

        let out = comp_from_footage(&path, &d.join(format!("render-{}.mp4", rate.num)));
        assert_eq!((out.frame_rate, frame_count(&out)), (rate, frames), "{rate:?} render");
        assert_eq!(out.duration, Tick::from_seconds_f64(4.0), "{rate:?} render duration");
        assert_eq!(frames_of(&out), all, "{rate:?} render: no frame skipped or repeated");
    }
}

#[test]
fn ntsc_and_variable_rate_footage_is_not_pulled_to_a_round_rate() {
    let d = out_dir("ms_rounded_controls");
    for rate in [FrameRate::FPS_23_976, FrameRate::FPS_29_97, FrameRate::FPS_59_94] {
        let path = d.join(format!("ms-{}.mp4", rate.num));
        write_movie(&path, &ms_rounded(rate, 60), 0);
        assert_eq!(effectcraft_media::probe(&path).expect("probe").frame_rate, rate);
    }
    // Frames of 40, 45 and 50 ms: the reader's median (45 ms, 22.2 fps) stands.
    let path = d.join("variable.mp4");
    write_movie(&path, &[40, 45, 50].repeat(20), 0);
    let f = effectcraft_media::probe(&path).expect("probe");
    assert!(!FrameRate::COMMON.contains(&f.frame_rate) && (f.frame_rate.as_f64() - 1000.0 / 45.0).abs() < 0.01, "{:?}", f.frame_rate);
}
