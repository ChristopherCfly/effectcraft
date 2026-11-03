//! An export that moves past a movie clip lets its footage source close that clip's decoder
//! (`MediaPool::set_idle_release`), and a composition that comes back to the clip later renders
//! exactly as with every decoder kept open.

use std::path::{Path, PathBuf};

use crate::binary_movie::{H, W, shown, write_movie};
use crate::out::out_dir;

use effectcraft_color::Label;
use effectcraft_export::render_queue::{OutputFormat, OutputModule, RenderSettings, TimeSpan};
use effectcraft_export::{Job, export};
use effectcraft_media::MediaPool;
use effectcraft_project::{Comp, ItemId, ItemKind, LayerSource, Project, build};
use effectcraft_time::{FrameRate, Tick};

const RATE: FrameRate = FrameRate { num: 10, den: 1 };

/// An 8 s clip at 10 fps whose frame k shows `first + k`.
fn clip(dir: &Path, name: &str, first: usize) -> effectcraft_project::Footage {
    let path = dir.join(name);
    write_movie(&path, &[100; 80], first);
    effectcraft_media::probe(&path).expect("probe")
}

/// A 15 s comp that cuts from clip A (0–4 s) to clip B (4–12 s) and back to A (12–15 s, A's 3–6 s:
/// a second it already showed, then ones it hadn't).
fn cut_back(dir: &Path) -> (Project, ItemId) {
    let mut p = Project::default();
    let mut comp = Comp::new(W, H, RATE, Tick::from_seconds_f64(15.0));
    let a = p.add_item("a.mp4", Label::Aqua, None, ItemKind::Footage(clip(dir, "a.mp4", 0)));
    let b = p.add_item("b.mp4", Label::Aqua, None, ItemKind::Footage(clip(dir, "b.mp4", 100)));
    for (item, start, (from, to)) in [(a, 0.0, (0.0, 4.0)), (b, 4.0, (4.0, 12.0)), (a, 9.0, (12.0, 15.0))] {
        let mut l = build::layer(&mut p, &comp, "clip", LayerSource::Footage { item }, (W, H), None);
        (l.start_time, l.in_point, l.out_point) = (Tick::from_seconds_f64(start), Tick::from_seconds_f64(from), Tick::from_seconds_f64(to));
        comp.layers.push(l);
    }
    let cid = p.add_item("cut back", Label::Sandstone, None, ItemKind::Comp(comp.into()));
    (p, cid)
}

/// The comp rendered losslessly (a PNG sequence) into the folder `out` with `pool`: the files,
/// in frame order.
fn render(p: &Project, cid: ItemId, pool: &MediaPool, out: &Path) -> Vec<PathBuf> {
    std::fs::create_dir_all(out).expect("mkdir");
    let settings = RenderSettings { time_span: TimeSpan::LengthOfComp, ..Default::default() };
    let om = OutputModule::for_format(OutputFormat::PngSequence);
    let path = out.join("frame_[#####].png").to_string_lossy().to_string();
    let job = Job {
        project: p,
        footage: pool,
        expr: None,
        accel: None,
        comp: cid,
        settings: &settings,
        output: &om,
        path: &path,
        sink: None,
        options: Default::default(),
        nested_switches: true,
    };
    let r = export(&job, &mut |_| true).expect("export");
    assert_eq!(r.frames, 150);
    let mut files: Vec<PathBuf> = std::fs::read_dir(out).expect("list").map(|e| e.expect("entry").path()).collect();
    files.sort();
    assert_eq!(files.len(), 150);
    files
}

#[test]
fn a_clip_the_export_comes_back_to_renders_as_with_every_decoder_open() {
    let d = out_dir("released-decoders");
    let (p, cid) = cut_back(&d);
    let kept = MediaPool::new();
    let released = MediaPool::new();
    released.set_idle_release(Some(16));
    let a = render(&p, cid, &kept, &d.join("kept"));
    let b = render(&p, cid, &released, &d.join("released"));
    for (k, (a, b)) in a.iter().zip(&b).enumerate() {
        assert!(std::fs::read(a).expect("read") == std::fs::read(b).expect("read"), "frame {k} differs");
    }
    assert_eq!((kept.stats().opened, kept.stats().released), (2, 0));
    // A closed while B played, and opened again for the cut back.
    assert_eq!(released.stats().opened, 3);
    assert!(released.stats().released >= 1);
    // The render shows the right frames: A 0–39, B 100–179, A 30–59.
    let pool = MediaPool::new();
    for k in (0..150usize).step_by(7).chain([39, 40, 119, 120, 149]) {
        let want = match k {
            0..40 => k,
            40..120 => 100 + k - 40,
            _ => 30 + k - 120,
        };
        let png = effectcraft_media::probe(&b[k]).expect("probe");
        let img = pool.frame_at(&png, Tick::ZERO).expect("decode");
        assert_eq!(shown(&img), want as u32, "frame {k}");
    }
}
