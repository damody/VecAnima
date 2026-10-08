//! Content-verified local stage cache; truncated or corrupted entries are misses.
use anyhow::{Context, Result};
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn file(path: &Path) -> Result<String> {
    let mut input = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub fn algorithm() -> String {
    digest(
        concat!(
            include_str!("vectorize.rs"),
            include_str!("strokes.rs"),
            include_str!("lineart.rs"),
            include_str!("fill_recovery.rs"),
            include_str!("curves.rs"),
            include_str!("spline.rs"),
            include_str!("shared.rs"),
            include_str!("geometry.rs"),
            include_str!("mesh_fill.rs"),
            include_str!("temporal.rs"),
            include_str!("pipeline.rs"),
            include_str!("main.rs"),
            include_str!("viewer.html"),
            include_str!("cache.rs"),
            include_str!("../Cargo.toml"),
            include_str!("../Cargo.lock")
        )
        .as_bytes(),
    )
}
pub fn atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let name = path
        .file_name()
        .context("Invalid atomic output path")?
        .to_string_lossy();
    let temp = path.with_file_name(format!(
        ".{name}.{}-{}.tmp",
        std::process::id(),
        SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    drop(output);
    std::fs::rename(temp, path)?;
    Ok(())
}
pub fn load<T: DeserializeOwned>(root: &Path, key: &str) -> Option<T> {
    let bytes = std::fs::read(root.join(format!("{key}.json"))).ok()?;
    let expected = std::fs::read_to_string(root.join(format!("{key}.sha256"))).ok()?;
    if digest(&bytes) != expected {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}
pub fn save<T: Serialize>(root: &Path, key: &str, value: &T) -> Result<()> {
    std::fs::create_dir_all(root)?;
    let bytes = serde_json::to_vec(value)?;
    atomic(&root.join(format!("{key}.json")), &bytes)?;
    atomic(
        &root.join(format!("{key}.sha256")),
        digest(&bytes).as_bytes(),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_hash_and_truncated_cache_are_rejected() -> Result<()> {
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let dir = std::env::temp_dir().join(format!("vecanima-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        save(&dir, "test", &vec![1, 2, 3])?;
        assert_eq!(load::<Vec<i32>>(&dir, "test"), Some(vec![1, 2, 3]));
        atomic(&dir.join("test.json"), b"[")?;
        assert!(load::<Vec<i32>>(&dir, "test").is_none());
        Ok(())
    }
}
