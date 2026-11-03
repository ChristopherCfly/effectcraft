//! Where the integration tests write their output.

use std::path::{Path, PathBuf};

/// A fresh, empty folder `target/test-out/export/<name>`.
pub fn out_dir(name: &str) -> PathBuf {
    out_dir_in("export", name)
}

/// A fresh, empty folder `target/test-out/<group>/<name>`.
pub fn out_dir_in(group: &str, name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/test-out").join(group).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("mkdir");
    d
}
