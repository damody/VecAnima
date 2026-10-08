use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
#[cfg(feature = "opencv-backend")]
use opencv::{core, imgcodecs, imgproc, prelude::*};
use serde_json::{Value, json};
mod cache;
mod server;
use std::{
    path::{Path, PathBuf},
    process::Command,
};
#[cfg(feature = "opencv-backend")]
mod curves;
#[cfg(feature = "opencv-backend")]
mod fill_recovery;
mod geometry;
#[cfg(feature = "opencv-backend")]
mod lineart;
#[cfg(feature = "opencv-backend")]
mod mesh_fill;
#[cfg(feature = "opencv-backend")]
mod pipeline;
#[cfg(feature = "opencv-backend")]
mod shared;
#[cfg(feature = "opencv-backend")]
mod spline;
#[cfg(feature = "opencv-backend")]
mod strokes;
#[cfg(feature = "opencv-backend")]
mod temporal;
#[cfg(feature = "opencv-backend")]
mod vectorize;

#[derive(Parser)]
#[command(
    name = "vecanima",
    version,
    about = "Animation vectorization experiments (native Rust CLI)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Render the native fill mesh from a geometry JSON, without stroke overlays.
    #[cfg(feature = "opencv-backend")]
    RenderFills {
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Render an exported pure-vector SVG to PNG for inspection.
    #[cfg(feature = "opencv-backend")]
    RenderSvg {
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Run the local AniLines RGB ONNX detector and export its boundary evidence.
    #[cfg(feature = "opencv-backend")]
    Lineart {
        source: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Serve an output project read-only on loopback, with video byte ranges.
    Serve {
        source: PathBuf,
        #[arg(long, default_value_t = 51284)]
        port: u16,
    },
    /// Triangulate a JSON PSLG with exact predicates, constraints and evenodd holes.
    Triangulate {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Verify FFmpeg, ffprobe and an actual OpenCV image operation.
    Doctor,
    /// Inspect media streams and timestamps through ffprobe.
    Probe { source: PathBuf },
    /// Decode a timestamped PNG sequence without using the Python prototype.
    Extract {
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 0.0)]
        start: f64,
        #[arg(long, default_value_t = 5.0)]
        duration: f64,
        #[arg(long, default_value_t = 640)]
        width: u32,
    },
    /// Build pure-vector SVG frames and a rendered preview (flat-color baseline).
    #[cfg(feature = "opencv-backend")]
    Vectorize {
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 0.0)]
        start: f64,
        #[arg(long, default_value_t = 5.0)]
        duration: f64,
        #[arg(long, default_value_t = 640)]
        width: u32,
        #[command(flatten)]
        settings: vectorize::Settings,
        #[arg(long)]
        no_preview: bool,
        #[arg(long)]
        resume: bool,
    },
    /// Vectorize a still image through the same native stroke/mesh core.
    #[cfg(feature = "opencv-backend")]
    Image {
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        width: Option<u32>,
        #[command(flatten)]
        settings: vectorize::Settings,
        #[arg(long)]
        resume: bool,
    },
    /// Vectorize a JSON sequence manifest with explicit per-frame durations.
    #[cfg(feature = "opencv-backend")]
    Sequence {
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        width: Option<u32>,
        #[command(flatten)]
        settings: vectorize::Settings,
        #[arg(long)]
        no_preview: bool,
        #[arg(long)]
        resume: bool,
    },
    /// Run a native OpenCV grayscale + Canny smoke test on an image.
    #[cfg(feature = "opencv-backend")]
    Edges {
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn tool(name: &str) -> PathBuf {
    let key = format!("VECANIMA_{}", name.to_uppercase());
    std::env::var_os(key).map(PathBuf::from).unwrap_or_else(|| {
        let local = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".tools")
            .join(format!("{name}.exe"));
        if local.is_file() {
            local
        } else {
            PathBuf::from(name)
        }
    })
}

fn execute(name: &str, args: &[&std::ffi::OsStr]) -> Result<std::process::Output> {
    let output = Command::new(tool(name))
        .args(args)
        .output()
        .with_context(|| format!("Cannot start {name}"))?;
    if !output.status.success() {
        bail!(
            "{name} failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output)
}

fn probe(source: &Path) -> Result<Value> {
    if !source.is_file() {
        bail!("Source does not exist: {}", source.display());
    }
    let output = execute(
        "ffprobe",
        &[
            "-v".as_ref(),
            "error".as_ref(),
            "-show_format".as_ref(),
            "-show_streams".as_ref(),
            "-of".as_ref(),
            "json".as_ref(),
            source.as_os_str(),
        ],
    )?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    cache::atomic(path, &serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn fresh_directory(path: &Path) -> Result<()> {
    if path.exists() && std::fs::read_dir(path)?.next().is_some() {
        bail!("Output must be empty: {}", path.display());
    }
    std::fs::create_dir_all(path)?;
    Ok(())
}

fn parse_frame(line: &str) -> Option<Value> {
    if !line.contains("showinfo") || !line.contains("pts_time:") {
        return None;
    }
    fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
        line.split_once(key)?.1.split_whitespace().next()
    }
    Some(json!({
        "pts": field(line, " pts:")?.parse::<i64>().ok()?,
        "seek_relative_time": field(line, "pts_time:")?.parse::<f64>().ok()?,
        "decoded_duration": field(line, "duration_time:").and_then(|s| s.parse::<f64>().ok()),
    }))
}

fn extract(
    source: &Path,
    out: &Path,
    start: f64,
    duration: f64,
    width: u32,
    finalize: bool,
) -> Result<Value> {
    if !start.is_finite()
        || start < 0.
        || !duration.is_finite()
        || duration <= 0.
        || width < 32
        || !width.is_multiple_of(2)
    {
        bail!("Require finite start >= 0, duration > 0 and even width >= 32");
    }
    let metadata = probe(source)?;
    let video = metadata["streams"]
        .as_array()
        .context("Missing streams")?
        .iter()
        .find(|s| s["codec_type"] == "video" && s["disposition"]["attached_pic"] != 1)
        .context("No video stream")?;
    fresh_directory(out)?;
    write_json(
        &out.join("status.json"),
        &json!({"status":"incomplete","implementation":"rust"}),
    )?;
    let result = (|| -> Result<Value> {
        let images = out.join("original");
        std::fs::create_dir(&images)?;
        let absolute_source = source.canonicalize()?;
        let pattern = images.canonicalize()?.join("%06d.png");
        let start_arg = start.to_string();
        let duration_arg = duration.to_string();
        let map = format!(
            "0:{}",
            video["index"].as_u64().context("Invalid stream index")?
        );
        let filter = format!("scale={width}:trunc(ow/dar/2)*2,setsar=1,showinfo");
        let args: Vec<&std::ffi::OsStr> = vec![
            "-hide_banner".as_ref(),
            "-nostdin".as_ref(),
            "-ss".as_ref(),
            start_arg.as_ref(),
            "-i".as_ref(),
            absolute_source.as_os_str(),
            "-t".as_ref(),
            duration_arg.as_ref(),
            "-map".as_ref(),
            map.as_ref(),
            "-an".as_ref(),
            "-sn".as_ref(),
            "-vf".as_ref(),
            filter.as_ref(),
            "-fps_mode".as_ref(),
            "passthrough".as_ref(),
            "-enc_time_base".as_ref(),
            "1/1000000".as_ref(),
            pattern.as_os_str(),
        ];
        let decoded = execute("ffmpeg", &args)?;
        std::fs::write(out.join("decode.log"), &decoded.stderr)?;
        let mut frames: Vec<Value> = String::from_utf8_lossy(&decoded.stderr)
            .lines()
            .filter_map(parse_frame)
            .collect();
        let count = std::fs::read_dir(&images)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
            .count();
        if count == 0 || frames.len() < count {
            bail!("No frames or PTS/image count mismatch");
        }
        let lookahead = frames
            .get(count)
            .and_then(|f| f["seek_relative_time"].as_f64());
        frames.truncate(count);
        let first = frames[0]["seek_relative_time"]
            .as_f64()
            .context("Invalid first PTS")?;
        let origin = metadata["format"]["start_time"]
            .as_str()
            .unwrap_or("0")
            .parse::<f64>()?;
        for i in 0..count {
            let stamp = frames[i]["seek_relative_time"].as_f64().unwrap();
            let next = frames
                .get(i + 1)
                .and_then(|f| f["seek_relative_time"].as_f64())
                .or(lookahead);
            let raw_duration = next
                .map(|n| n - stamp)
                .or_else(|| frames[i]["decoded_duration"].as_f64())
                .filter(|d| *d > 0.)
                .unwrap_or(duration - stamp);
            let dt = raw_duration.min(duration - stamp);
            if dt <= 0. {
                bail!("Non-increasing PTS");
            }
            frames[i]["file"] = json!(format!("{:06}.png", i + 1));
            frames[i]["source_time"] = json!(origin + start + stamp);
            frames[i]["time"] = json!(stamp - first);
            frames[i]["duration"] = json!(dt);
        }
        let last = frames.last().unwrap();
        let actual_duration = last["time"].as_f64().unwrap() + last["duration"].as_f64().unwrap();
        let timeline = json!({"implementation":"rust", "source":absolute_source,
            "source_time_base":video["time_base"],"requested_start":start,"requested_duration":duration,
            "first_frame_offset":first,"duration":actual_duration,"frames":frames});
        write_json(&out.join("source.json"), &metadata)?;
        write_json(&out.join("timeline.json"), &timeline)?;
        Ok(json!({"frame_count":count,"duration":actual_duration,"implementation":"rust"}))
    })();
    match &result {
        Ok(_) => write_json(
            &out.join("status.json"),
            &json!({"status":if finalize {"complete"} else {"incomplete"},"implementation":"rust","phase":"decoded"}),
        )?,
        Err(e) => write_json(
            &out.join("status.json"),
            &json!({"status":"incomplete","error":e.to_string()}),
        )?,
    }
    result
}

fn main() -> Result<()> {
    let result = match Cli::parse().command {
        Commands::Serve { source, port } => {
            server::serve(&source, port)?;
            json!({"status":"stopped"})
        }
        Commands::Triangulate { input, out } => {
            if out.exists() {
                bail!("Output already exists: {}", out.display());
            }
            let input: geometry::Input = serde_json::from_slice(&std::fs::read(input)?)?;
            let mesh = geometry::triangulate(input)?;
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            write_json(&out, &serde_json::to_value(&mesh)?)?;
            json!({"vertices":mesh.points.len(),"triangles":mesh.triangles.len(),"constraints":mesh.constraints.len(),"output":out})
        }
        Commands::Doctor => {
            let mut versions =
                json!({"implementation":"rust", "opencv_enabled":cfg!(feature="opencv-backend")});
            #[cfg(feature = "opencv-backend")]
            {
                let image = core::Mat::new_rows_cols_with_default(
                    16,
                    16,
                    core::CV_8UC3,
                    core::Scalar::all(128.),
                )?;
                let mut gray = core::Mat::default();
                imgproc::cvt_color_def(&image, &mut gray, imgproc::COLOR_BGR2GRAY)?;
                versions["opencv"] = json!(core::get_version_string()?);
                versions["opencv_smoke"] =
                    json!({"rows":gray.rows(),"cols":gray.cols(),"channels":gray.channels()});
            }
            for name in ["ffmpeg", "ffprobe"] {
                let output = execute(name, &["-version".as_ref()])?;
                versions[name] = json!(String::from_utf8_lossy(&output.stdout).lines().next());
            }
            versions
        }
        Commands::Probe { source } => probe(&source)?,
        Commands::Extract {
            source,
            out,
            start,
            duration,
            width,
        } => extract(&source, &out, start, duration, width, true)?,
        #[cfg(feature = "opencv-backend")]
        Commands::Vectorize {
            source,
            out,
            start,
            duration,
            width,
            settings,
            no_preview,
            resume,
        } => pipeline::build(
            &source,
            &out,
            (start, duration, width),
            &settings,
            !no_preview,
            resume,
        )?,
        #[cfg(feature = "opencv-backend")]
        Commands::Image {
            source,
            out,
            width,
            settings,
            resume,
        } => pipeline::build_images(&source, &out, width, &settings, false, resume, false)?,
        #[cfg(feature = "opencv-backend")]
        Commands::Sequence {
            source,
            out,
            width,
            settings,
            no_preview,
            resume,
        } => pipeline::build_images(&source, &out, width, &settings, !no_preview, resume, true)?,
        #[cfg(feature = "opencv-backend")]
        Commands::RenderFills { source, out } => {
            if out.exists() {
                bail!("Output already exists: {}", out.display());
            }
            let frame: vectorize::Frame = serde_json::from_slice(&std::fs::read(&source)?)?;
            anyhow::ensure!(
                frame.version == 3 && frame.width > 0 && frame.height > 0,
                "Invalid native frame dimensions/version"
            );
            let rendered = if let Some(mesh) = &frame.mesh {
                anyhow::ensure!(
                    mesh.colors_linear.len() == mesh.geometry.triangles.len()
                        && mesh
                            .geometry
                            .triangles
                            .iter()
                            .all(|t| t.iter().all(|i| *i < mesh.geometry.points.len())),
                    "Invalid native mesh indices/colors"
                );
                mesh_fill::raster(mesh, frame.width, frame.height)?
            } else {
                vectorize::rasterize(&vectorize::fills_svg(&frame))?
            };
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            rendered.save_png(&out)?;
            json!({"implementation":"rust","layer":"fills","output":out})
        }
        #[cfg(feature = "opencv-backend")]
        Commands::RenderSvg { source, out } => {
            if out.exists() {
                bail!("Output already exists: {}", out.display());
            }
            let rendered = vectorize::rasterize(&std::fs::read_to_string(&source)?)?;
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            rendered.save_png(&out)?;
            json!({"implementation":"rust","output":out})
        }
        #[cfg(feature = "opencv-backend")]
        Commands::Lineart { source, model, out } => {
            if out.exists() {
                bail!("Output already exists: {}", out.display());
            }
            let image = vectorize::read(&source)?;
            let hint = lineart::predict(&image, &model)?;
            let bytes: Vec<_> = hint
                .iter()
                .map(|v| ((1. - v) * 255.).round() as u8)
                .collect();
            let data = core::Mat::from_slice(&bytes)?;
            let mat = data.reshape(1, image.rows())?;
            let mut encoded = core::Vector::<u8>::new();
            anyhow::ensure!(
                imgcodecs::imencode(".png", &mat, &mut encoded, &core::Vector::new())?,
                "Cannot encode line map"
            );
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, encoded.as_slice())?;
            json!({"implementation":"rust","model":lineart::identity(Some(&model))?,"output":out})
        }
        #[cfg(feature = "opencv-backend")]
        Commands::Edges { source, out } => {
            if out.exists() {
                bail!("Output already exists: {}", out.display());
            }
            let image = vectorize::read(&source)?;
            if image.empty() {
                bail!("Cannot read image: {}", source.display());
            }
            let mut gray = core::Mat::default();
            let mut edges = core::Mat::default();
            imgproc::cvt_color_def(&image, &mut gray, imgproc::COLOR_BGR2GRAY)?;
            imgproc::canny(&gray, &mut edges, 50., 150., 3, false)?;
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            let mut encoded = core::Vector::<u8>::new();
            if !imgcodecs::imencode(".png", &edges, &mut encoded, &core::Vector::new())? {
                bail!("OpenCV failed to write edge image");
            }
            std::fs::write(&out, encoded.as_slice())?;
            json!({"implementation":"rust","edge_pixels":core::count_non_zero(&edges)?,"output":out})
        }
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_negative_and_fractional_timestamps() {
        let f = parse_frame(
            "[Parsed_showinfo_2] n: 0 pts: -21 pts_time:-0.021 duration: 41 duration_time:0.041",
        )
        .unwrap();
        assert_eq!(f["pts"], -21);
        assert_eq!(f["seek_relative_time"], -0.021);
        assert_eq!(f["decoded_duration"], 0.041);
    }
    #[test]
    fn rejects_nonframe_logging() {
        assert!(parse_frame("[Parsed_showinfo_2] config in time_base: 1/1000").is_none());
    }
}
