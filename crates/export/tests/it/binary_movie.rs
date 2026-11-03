//! Test movies written with FilmCraft's own encoder and muxer: 128×32 H.264 frames showing a
//! number in binary (eight bars, white for a set bit), so a decoded frame says which it is.

use std::path::Path;

use effectcraft_raster::Image;
use filmcraft_isobmff::{AvcConfig, Brand, Mp4Writer, SampleEntry, TrackConfig, WriteSample, WriterOptions};

pub const W: u32 = 128;
pub const H: u32 = 32;
const BARS: u32 = 8;

/// An H.264 MP4 (video track timescale 1000, a sync sample every 30 frames) with one frame per
/// entry of `durations` (ms); frame k shows `first + k` in binary (below 256).
pub fn write_movie(path: &Path, durations: &[u32], first: usize) {
    let mut cfg = filmcraft_h264enc::EncoderConfig::new(W, H, 30, 1);
    cfg.format = filmcraft_h264enc::PacketFormat::LengthPrefixed;
    cfg.aud = false;
    cfg.bframes = 0;
    cfg.keyint = 30;
    cfg.scenecut = false;
    let mut enc = filmcraft_h264enc::Encoder::new(cfg).expect("encoder");
    let (u, v) = (vec![128u8; (W * H / 4) as usize], vec![128u8; (W * H / 4) as usize]);
    let mut packets = Vec::new();
    for k in 0..durations.len() {
        let n = first + k;
        let y: Vec<u8> = (0..W * H).map(|i| if (n >> (i % W * BARS / W)) & 1 == 1 { 235 } else { 16 }).collect();
        let frame = filmcraft_h264enc::YuvFrame { y: &y, u: &u, v: &v, y_stride: W as usize, uv_stride: W as usize / 2 };
        packets.extend(enc.try_encode(&frame, k as i64).expect("encode"));
    }
    packets.extend(enc.flush());
    assert_eq!(packets.len(), durations.len(), "one access unit per frame");
    let avcc = AvcConfig::parse(&enc.avcc()).expect("avcC");
    let file = std::fs::File::create(path).expect("create");
    let mut mux = Mp4Writer::new(file, WriterOptions::new(Brand::Mp4)).expect("mux");
    let t = mux.add_track(TrackConfig::new(SampleEntry::avc(avcc, W as u16, H as u16), 1000)).expect("track");
    for (p, &duration) in packets.iter().zip(durations) {
        mux.write_sample(t, WriteSample { data: &p.data, duration, composition_offset: 0, is_sync: p.keyframe }).expect("sample");
    }
    mux.finish().expect("finish");
}

/// The number a decoded frame shows.
pub fn shown(img: &Image) -> u32 {
    (0..BARS).filter(|b| img.data[img.idx(b * W / BARS + W / BARS / 2, H / 2)][1] > 0.5).map(|b| 1 << b).sum()
}
