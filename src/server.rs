//! Read-only loopback artifact server with bounded workers and byte ranges.
use anyhow::{Context, Result, ensure};
use std::{
    io::{Read, Seek, SeekFrom, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    time::Duration,
};
fn resolve(root: &Path, url: &str) -> Result<PathBuf> {
    let raw = url.split('?').next().unwrap_or("/");
    let mut bytes = Vec::new();
    let mut i = 0;
    let input = raw.as_bytes();
    while i < input.len() {
        if input[i] == b'%' {
            ensure!(i + 2 < input.len(), "Invalid URL escape");
            bytes.push(u8::from_str_radix(
                std::str::from_utf8(&input[i + 1..i + 3])?,
                16,
            )?);
            i += 3;
        } else {
            bytes.push(input[i]);
            i += 1;
        }
    }
    let name = String::from_utf8(bytes)?;
    let relative = name.trim_start_matches('/');
    ensure!(
        !relative.contains('\\') && !relative.contains(':') && !relative.contains('\0'),
        "Invalid URL path"
    );
    ensure!(
        Path::new(relative)
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_))),
        "Invalid path components"
    );
    let target = root
        .join(if relative.is_empty() {
            "index.html"
        } else {
            relative
        })
        .canonicalize()?;
    ensure!(
        target.starts_with(root) && target.is_file(),
        "Path outside artifact root"
    );
    Ok(target)
}
fn range(text: Option<&str>, len: u64) -> Result<(u64, u64, bool)> {
    if let Some(text) = text {
        let text = text
            .strip_prefix("bytes=")
            .context("Unsupported byte range")?;
        ensure!(!text.contains(','), "Multiple ranges unsupported");
        let (a, b) = text.split_once('-').context("Invalid range")?;
        ensure!(len > 0, "Empty file has no byte range");
        let (start, end) = if a.is_empty() {
            let suffix = b.parse::<u64>()?;
            ensure!(suffix > 0, "Empty suffix range");
            (len.saturating_sub(suffix), len - 1)
        } else {
            (
                a.parse::<u64>()?,
                if b.is_empty() {
                    len - 1
                } else {
                    b.parse::<u64>()?.min(len - 1)
                },
            )
        };
        ensure!(start <= end && start < len, "Unsatisfiable range");
        Ok((start, end - start + 1, true))
    } else {
        Ok((0, len, false))
    }
}
fn reply(stream: &mut TcpStream, status: &str, body: &str) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    Ok(())
}
fn handle(mut stream: TcpStream, root: &Path) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(15)))?;
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 1024];
    while !bytes.windows(4).any(|v| v == b"\r\n\r\n") {
        let n = stream.read(&mut buffer)?;
        if n == 0 {
            return Ok(());
        }
        bytes.extend_from_slice(&buffer[..n]);
        if bytes.len() > 16384 {
            return reply(&mut stream, "431 Request Header Fields Too Large", "");
        }
    }
    let request = std::str::from_utf8(&bytes)?;
    let mut lines = request.split("\r\n");
    let first = lines.next().context("Missing request line")?;
    let fields: Vec<_> = first.split_whitespace().collect();
    if fields.len() != 3 || !matches!(fields[0], "GET" | "HEAD") {
        return reply(&mut stream, "405 Method Not Allowed", "");
    }
    let path = match resolve(root, fields[1]) {
        Ok(path) => path,
        Err(_) => return reply(&mut stream, "404 Not Found", ""),
    };
    let mut file = std::fs::File::open(&path)?;
    let len = file.metadata()?.len();
    let header = lines
        .filter_map(|line| line.split_once(':'))
        .find(|(k, _)| k.eq_ignore_ascii_case("range"))
        .map(|(_, v)| v.trim());
    let (start, count, partial) = match range(header, len) {
        Ok(range) => range,
        Err(_) => {
            write!(
                stream,
                "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{len}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )?;
            return Ok(());
        }
    };
    let mime = match path.extension().and_then(|p| p.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "mp4" => "video/mp4",
        _ => "application/octet-stream",
    };
    write!(
        stream,
        "HTTP/1.1 {}\r\nContent-Type: {mime}\r\nContent-Length: {count}\r\nAccept-Ranges: bytes\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n",
        if partial {
            "206 Partial Content"
        } else {
            "200 OK"
        }
    )?;
    if partial {
        write!(
            stream,
            "Content-Range: bytes {}-{}/{len}\r\n",
            start,
            start + count - 1
        )?;
    }
    write!(stream, "\r\n")?;
    if fields[0] == "GET" {
        file.seek(SeekFrom::Start(start))?;
        std::io::copy(&mut file.take(count), &mut stream)?;
    }
    Ok(())
}
pub fn serve(root: &Path, port: u16) -> Result<()> {
    let root = Arc::new(root.canonicalize()?);
    ensure!(
        root.join("project.json").is_file(),
        "Artifact root requires project.json"
    );
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))?;
    eprintln!("VecAnima viewer: http://{}/", listener.local_addr()?);
    let (sender, receiver) = mpsc::sync_channel::<TcpStream>(16);
    let receiver = Arc::new(Mutex::new(receiver));
    for _ in 0..4 {
        let receiver = receiver.clone();
        let root = root.clone();
        std::thread::spawn(move || {
            loop {
                let stream = receiver.lock().unwrap().recv();
                let Ok(stream) = stream else {
                    break;
                };
                if let Err(error) = handle(stream, &root) {
                    eprintln!("Viewer request: {error}");
                }
            }
        });
    }
    for stream in listener.incoming() {
        let stream = stream?;
        if let Err(mpsc::TrySendError::Full(mut stream)) = sender.try_send(stream) {
            reply(&mut stream, "503 Service Unavailable", "")?;
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_ranges_are_bounded_and_traversal_is_rejected() -> Result<()> {
        assert_eq!(range(Some("bytes=3-6"), 10)?, (3, 4, true));
        assert_eq!(range(Some("bytes=-4"), 10)?, (6, 4, true));
        assert_eq!(range(Some("bytes=8-99"), 10)?, (8, 2, true));
        assert!(range(Some("bytes=10-"), 10).is_err());
        let root = std::env::current_dir()?.canonicalize()?;
        assert!(resolve(&root, "/%2e%2e/Cargo.toml").is_err());
        assert!(resolve(&root, "/C:%5cWindows").is_err());
        assert!(resolve(&root, "/Cargo.toml")?.starts_with(&root));
        Ok(())
    }
}
