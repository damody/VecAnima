//! Rust-only decode -> SVG -> resvg -> preview baseline pipeline.
use crate::{cache, execute, extract, probe, vectorize as v, write_json};
use anyhow::{Context, Result, ensure};
use opencv::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone)]
struct Asset {
    id: usize,
    original: PathBuf,
    svg: String,
    geometry: String,
    raster: String,
    paths: usize,
    vertices: usize,
    bytes: usize,
    mesh_buffer: Option<String>,
}

fn record_error(out: &Path, error: &anyhow::Error) -> Result<()> {
    use std::io::Write;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join("ERRORS.md"))?;
    writeln!(
        log,
        "\n## Runtime failure · Unix {stamp}\n\n{}\n\n原始資料與已校驗快取保留。修正原因後使用相同設定 --resume；不能忽略約束或降低校驗要求。",
        format!("{error:#}")
            .lines()
            .map(|line| format!("    {line}"))
            .collect::<Vec<_>>()
            .join("\n")
    )?;
    Ok(())
}

#[cfg(windows)]
fn peak_memory() -> Option<usize> {
    #[repr(C)]
    struct Counters {
        size: u32,
        faults: u32,
        peak_working_set: usize,
        working_set: usize,
        peak_paged_pool: usize,
        paged_pool: usize,
        peak_nonpaged_pool: usize,
        nonpaged_pool: usize,
        pagefile: usize,
        peak_pagefile: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn K32GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut Counters,
            size: u32,
        ) -> i32;
    }
    let mut counters = Counters {
        size: std::mem::size_of::<Counters>() as u32,
        faults: 0,
        peak_working_set: 0,
        working_set: 0,
        peak_paged_pool: 0,
        paged_pool: 0,
        peak_nonpaged_pool: 0,
        nonpaged_pool: 0,
        pagefile: 0,
        peak_pagefile: 0,
    };
    // GetCurrentProcess's documented pseudo-handle, with a correctly sized C
    // structure owned exclusively by this call. No process permissions changed.
    let result = unsafe {
        K32GetProcessMemoryInfo(
            (-1isize) as *mut _,
            &mut counters,
            std::mem::size_of::<Counters>() as u32,
        )
    };
    (result != 0).then_some(counters.peak_working_set)
}
#[cfg(not(windows))]
fn peak_memory() -> Option<usize> {
    None
}

pub fn build(
    source: &Path,
    out: &Path,
    selection: (f64, f64, u32),
    settings: &v::Settings,
    preview: bool,
    resume: bool,
) -> Result<Value> {
    settings.validate()?;
    let began = Instant::now();
    let (start, duration, width) = selection;
    ensure!(
        start.is_finite()
            && start >= 0.
            && duration.is_finite()
            && duration > 0.
            && width >= 32
            && width.is_multiple_of(2),
        "Invalid video selection"
    );
    let identity = json!({"mode":"video","source":source.canonicalize()?,"sha256":cache::file(source)?,"selection":selection,"settings":settings,"line_model":crate::lineart::identity(settings.strokes.line_model.as_deref())?,"algorithm":cache::algorithm(),"opencv":opencv::core::get_version_string()?,"ffmpeg":String::from_utf8_lossy(&execute("ffmpeg",&["-version".as_ref()])?.stdout).lines().next()});
    reserve(out, &identity, resume)?;
    let result = (|| {
        if !prepared(out)? {
            let attempt = attempt_dir(out)?;
            extract(source, &attempt, start, duration, width, false)?;
            adopt(out, &attempt)?;
        }
        build_decoded(source, out, settings, preview, began)
    })();
    match &result {
        Ok(_) => write_json(
            &out.join("status.json"),
            &json!({"status":"complete","implementation":"rust","phase":"vectorized","settings":settings}),
        )?,
        Err(error) => write_json(
            &out.join("status.json"),
            &json!({"status":"incomplete","implementation":"rust","error":format!("{error:#}")}),
        )?,
    }
    if let Err(error) = &result {
        record_error(out, error)?;
    }
    result
}

fn reserve(out: &Path, identity: &Value, resume: bool) -> Result<()> {
    if resume {
        let existing: Value = serde_json::from_slice(
            &std::fs::read(out.join("run.json")).context("Resume requires a VecAnima run.json")?,
        )?;
        ensure!(
            existing == *identity,
            "Resume source, selection, settings or implementation fingerprint changed"
        );
    } else {
        crate::fresh_directory(out)?;
        write_json(&out.join("run.json"), identity)?;
    }
    write_json(
        &out.join("status.json"),
        &json!({"status":"incomplete","phase":"preparing","implementation":"rust"}),
    )
}
fn attempt_dir(out: &Path) -> Result<PathBuf> {
    for i in 1.. {
        let dir = out.join(format!("decode-attempt-{i:06}"));
        if !dir.exists() {
            return Ok(dir);
        }
    }
    unreachable!()
}
fn prepared(out: &Path) -> Result<bool> {
    let path = out.join("prepared.json");
    if !path.is_file() {
        return Ok(false);
    }
    let manifest: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let frames = manifest["frames"]
        .as_array()
        .context("Invalid prepared manifest")?;
    for frame in frames {
        let file = frame["file"].as_str().context("Invalid cached frame")?;
        ensure!(
            Path::new(file).components().count() == 1,
            "Unsafe cached frame name"
        );
        ensure!(
            cache::file(&out.join("original").join(file))?
                == frame["sha256"].as_str().context("Missing frame checksum")?,
            "Decoded frame checksum mismatch: {file}"
        );
    }
    write_json(&out.join("timeline.json"), &manifest["timeline"])?;
    Ok(true)
}
fn adopt(out: &Path, attempt: &Path) -> Result<()> {
    if out.join("original").exists() {
        let backup = out.join(format!(
            "original-abandoned-{}",
            attempt.file_name().unwrap().to_string_lossy()
        ));
        std::fs::rename(out.join("original"), backup)?;
    }
    std::fs::rename(attempt.join("original"), out.join("original"))?;
    for name in ["timeline.json", "source.json"] {
        cache::atomic(&out.join(name), &std::fs::read(attempt.join(name))?)?;
    }
    let timeline: Value = serde_json::from_slice(&std::fs::read(out.join("timeline.json"))?)?;
    let mut frames = Vec::new();
    for f in timeline["frames"]
        .as_array()
        .context("Missing prepared frames")?
    {
        let name = f["file"].as_str().context("Invalid decoded file")?;
        frames.push(json!({"file":name,"sha256":cache::file(&out.join("original").join(name))?}));
    }
    write_json(
        &out.join("prepared.json"),
        &json!({"timeline":timeline,"frames":frames}),
    )
}

pub fn build_images(
    source: &Path,
    out: &Path,
    width: Option<u32>,
    settings: &v::Settings,
    preview: bool,
    resume: bool,
    sequence: bool,
) -> Result<Value> {
    use opencv::{core, imgcodecs, imgproc};
    settings.validate()?;
    ensure!(width.is_none_or(|v| v > 0), "Image width must be positive");
    let began = Instant::now();
    let absolute = source.canonicalize()?;
    let inputs: Vec<(PathBuf, f64)> = if sequence {
        let manifest: Value = serde_json::from_slice(&std::fs::read(source)?)?;
        manifest["frames"]
            .as_array()
            .context("Sequence requires frames array")?
            .iter()
            .map(|f| {
                let path = absolute
                    .parent()
                    .unwrap()
                    .join(f["file"].as_str().context("Frame requires file")?);
                let duration = f["duration"].as_f64().context("Frame requires duration")?;
                ensure!(
                    duration.is_finite() && duration > 0.,
                    "Frame duration must be positive and finite"
                );
                Ok((path, duration))
            })
            .collect::<Result<_>>()?
    } else {
        vec![(absolute.clone(), 1.)]
    };
    ensure!(!inputs.is_empty(), "Empty image sequence");
    let identities: Vec<_> = inputs
        .iter()
        .map(|(p, d)| Ok(json!({"source":p.canonicalize()?,"sha256":cache::file(p)?,"duration":d})))
        .collect::<Result<_>>()?;
    reserve(
        out,
        &json!({"mode":if sequence{"sequence"}else{"image"},"inputs":identities,"width":width,"settings":settings,"line_model":crate::lineart::identity(settings.strokes.line_model.as_deref())?,"algorithm":cache::algorithm(),"opencv":opencv::core::get_version_string()?}),
        resume,
    )?;
    let result = (|| {
        if !prepared(out)? {
            let attempt = attempt_dir(out)?;
            std::fs::create_dir_all(attempt.join("original"))?;
            let (mut time, mut size) = (0., None);
            let mut frames = Vec::new();
            for (i, (path, duration)) in inputs.iter().enumerate() {
                let mut image = v::read(path)?;
                if let Some(width) = width {
                    let height = (image.rows() as f64 * width as f64 / image.cols() as f64)
                        .round()
                        .max(1.) as i32;
                    let mut resized = core::Mat::default();
                    imgproc::resize(
                        &image,
                        &mut resized,
                        core::Size::new(width as i32, height),
                        0.,
                        0.,
                        imgproc::INTER_AREA,
                    )?;
                    image = resized;
                }
                ensure!(
                    size.is_none_or(|s| s == (image.cols(), image.rows())),
                    "Sequence dimensions changed"
                );
                size = Some((image.cols(), image.rows()));
                let name = format!("{:06}.png", i + 1);
                let mut bytes = core::Vector::<u8>::new();
                ensure!(
                    imgcodecs::imencode(".png", &image, &mut bytes, &core::Vector::new())?,
                    "Cannot encode normalized frame"
                );
                cache::atomic(&attempt.join("original").join(&name), bytes.as_slice())?;
                frames.push(json!({"file":name,"time":time,"source_time":time,"duration":duration,"source":path}));
                time += duration;
            }
            write_json(
                &attempt.join("timeline.json"),
                &json!({"implementation":"rust","source":absolute,"input_kind":if sequence{"sequence"}else{"image"},"requested_start":0.,"first_frame_offset":0.,"duration":time,"frames":frames}),
            )?;
            write_json(
                &attempt.join("source.json"),
                &json!({"streams":[],"inputs":identities}),
            )?;
            adopt(out, &attempt)?;
        }
        build_decoded(source, out, settings, preview, began)
    })();
    match &result {
        Ok(_) => write_json(
            &out.join("status.json"),
            &json!({"status":"complete","phase":"vectorized","implementation":"rust"}),
        )?,
        Err(e) => write_json(
            &out.join("status.json"),
            &json!({"status":"incomplete","error":format!("{e:#}")}),
        )?,
    };
    if let Err(error) = &result {
        record_error(out, error)?;
    }
    result
}

fn build_decoded(
    source: &Path,
    out: &Path,
    settings: &v::Settings,
    encode_preview: bool,
    began: Instant,
) -> Result<Value> {
    let mut timeline: Value = serde_json::from_slice(&std::fs::read(out.join("timeline.json"))?)?;
    let frames = timeline["frames"]
        .as_array_mut()
        .context("Missing decoded frames")?;
    ensure!(!frames.is_empty(), "No decoded frames");
    let count = frames.len();
    for dir in ["frames", "geometry", "raster", "tracking"] {
        std::fs::create_dir_all(out.join(dir))?;
    }
    let indices = sample_indices(count, 12);
    let paths: Vec<_> = indices
        .iter()
        .map(|i| {
            out.join("original")
                .join(frames[*i]["file"].as_str().unwrap())
        })
        .collect();
    eprintln!("Fitting shared palette from {} frames", paths.len());
    let prefix = cache::digest(&serde_json::to_vec(
        &json!({"algorithm":cache::algorithm(),"settings":settings,"line_model":crate::lineart::identity(settings.strokes.line_model.as_deref())?}),
    )?);
    let palette_root = out.join("cache/palette");
    let palette = if let Some(palette) = cache::load::<Vec<[f32; 3]>>(&palette_root, &prefix) {
        palette
    } else {
        let palette = v::fit_palette(&paths, settings)?;
        cache::save(&palette_root, &prefix, &palette)?;
        palette
    };
    let frame_root = out.join("cache/vectorized");
    let mut tracker = crate::temporal::Tracker::default();
    let mut cache_hits = 0usize;
    let mut cache: HashMap<u64, Vec<Asset>> = HashMap::new();
    let mut assets = Vec::new();
    let mut metrics = Vec::new();
    let mut dimensions = None;
    for (i, frame) in frames.iter_mut().enumerate() {
        let tick = Instant::now();
        let name = frame["file"]
            .as_str()
            .context("Invalid frame file")?
            .to_owned();
        let image_path = out.join("original").join(&name);
        let original = v::read(&image_path)?;
        let size = (original.cols(), original.rows());
        ensure!(
            dimensions.is_none_or(|d| d == size),
            "Frame dimensions changed within clip"
        );
        dimensions = Some(size);
        let mut hash = DefaultHasher::new();
        original.data_bytes()?.hash(&mut hash);
        let key = hash.finish();
        // Hash is only an accelerator: equality is always checked before reuse.
        let mut found = None;
        if let Some(candidates) = cache.get(&key).filter(|_| !settings.temporal.temporal) {
            for asset in candidates {
                if v::read(&asset.original)?.data_bytes()? == original.data_bytes()? {
                    found = Some(asset.clone());
                    break;
                }
            }
        }
        let reused = found.is_some();
        let raster_path = out.join("raster").join(&name);
        let asset = if let Some(asset) = found {
            std::fs::copy(out.join(&asset.raster), &raster_path)?;
            asset
        } else {
            let frame_key = cache::digest(
                format!("{prefix}:{}", cache::digest(original.data_bytes()?)).as_bytes(),
            );
            let mut project = if let Some(frame) = cache::load::<v::Frame>(&frame_root, &frame_key)
            {
                ensure!(frame.version == 3, "Unsupported vector cache format");
                cache_hits += 1;
                frame
            } else {
                let frame = v::vectorize(&original, &palette, settings)?;
                cache::save(&frame_root, &frame_key, &frame)?;
                frame
            };
            if settings.temporal.temporal {
                let record = tracker.process(&original, &mut project, &settings.temporal)?;
                let path = format!("tracking/{:06}.json", i + 1);
                write_json(&out.join(&path), &serde_json::to_value(record)?)?;
                frame["tracking"] = json!(path);
            }
            let text = v::svg(&project);
            let raster = v::render(&project, &text)?;
            cache::atomic(&raster_path, &raster.encode_png()?)?;
            let svg = format!("frames/{:06}.svg", i + 1);
            let geometry = format!("geometry/{:06}.json", i + 1);
            cache::atomic(&out.join(&svg), text.as_bytes())?;
            cache::atomic(
                &out.join(format!("frames/{:06}.strokes.svg", i + 1)),
                v::strokes_svg(&project).as_bytes(),
            )?;
            cache::atomic(
                &out.join(format!("frames/{:06}.fills.svg", i + 1)),
                v::fills_svg(&project).as_bytes(),
            )?;
            write_json(&out.join(&geometry), &serde_json::to_value(&project)?)?;
            let mesh_buffer = if let Some(mesh) = &project.mesh {
                let path = format!("geometry/{:06}.mesh.bin", i + 1);
                cache::atomic(
                    &out.join(&path),
                    &crate::mesh_fill::gpu_buffer(mesh, project.width, project.height),
                )?;
                Some(path)
            } else {
                None
            };
            let asset = Asset {
                id: assets.len(),
                original: image_path,
                svg,
                geometry,
                raster: format!("raster/{name}"),
                paths: project.layers.len()
                    + project.strokes.len()
                    + project
                        .mesh
                        .as_ref()
                        .map_or(0, |m| m.geometry.triangles.len()),
                vertices: project
                    .layers
                    .iter()
                    .flat_map(|l| &l.rings)
                    .map(Vec::len)
                    .sum::<usize>()
                    + project
                        .strokes
                        .iter()
                        .map(|s| s.outlines.iter().map(|o| o.cubics.len() * 4).sum::<usize>())
                        .sum::<usize>()
                    + project.mesh.as_ref().map_or(0, |m| m.geometry.points.len()),
                bytes: text.len(),
                mesh_buffer,
            };
            cache.entry(key).or_default().push(asset.clone());
            assets.push(asset.clone());
            asset
        };
        frame["asset"] = json!(asset.id);
        frame["svg"] = json!(asset.svg);
        let rendered = v::read(&raster_path)?;
        let (mae, mse) = error(original.data_bytes()?, rendered.data_bytes()?)?;
        metrics.push(json!({"index":i,"mae":mae,"mse":mse,"psnr_db":if mse==0. {None} else {Some(10.*(255.*255./mse).log10())},
            "paths":asset.paths,"vertices":asset.vertices,"svg_bytes":asset.bytes,"reused_exact_frame":reused,"seconds":tick.elapsed().as_secs_f64()}));
        if i % 24 == 0 || i + 1 == count {
            eprintln!("Vectorized {}/{count} frames", i + 1);
        }
    }
    let (width, height) = dimensions.unwrap();
    timeline["fill_model"] = serde_json::to_value(settings.fill.fill_model)?;
    if settings.temporal.temporal {
        write_json(
            &out.join("tracking.json"),
            &json!({"format":"vecanima-tracking","version":1,"records":"tracking/%06d.json","count":count}),
        )?;
    }
    timeline["stroke_model"] = serde_json::to_value(settings.strokes.stroke_model)?;
    write_json(&out.join("timeline.json"), &timeline)?;
    let assets_json: Vec<_> = assets
        .iter()
        .map(|a| json!({"id":a.id,"svg":a.svg,"geometry":a.geometry,"mesh_buffer":a.mesh_buffer,"mesh_buffer_format":"le-f32-xy-linear-rgb","strokes_svg":a.svg.replace(".svg",".strokes.svg"),"fills_svg":a.svg.replace(".svg",".fills.svg")}))
        .collect();
    write_json(
        &out.join("project.json"),
        &json!({"format":"vecanima","format_version":3,"implementation":"rust",
        "model":if settings.fill.fill_model==crate::mesh_fill::Model::Mesh {"profile-strokes-and-linear-light-mesh"} else if settings.strokes.stroke_model==crate::strokes::Model::Baseline {"flat-regions-and-dark-mask-baseline"} else {"profile-strokes-and-flat-regions"},"width":width,"height":height,"timeline":"timeline.json","assets":assets_json,
        "palette_lab8":palette,"settings":settings,"tools":{"opencv":opencv::core::get_version_string()?,"resvg":"0.48.1"},
        "tracking":if settings.temporal.temporal {Some("tracking.json")}else{None},
        "limitations":["SVG gradients are a bounded piecewise-constant approximation of native mesh colors","Tracking split/merge and occlusion records are hypotheses, not semantic ground truth"],"stroke_model":settings.strokes.stroke_model}),
    )?;
    if encode_preview {
        preview(source, out, &timeline)?;
    }
    write_viewer(out, &timeline, encode_preview)?;
    comparison(out, &timeline)?;
    let psnr: Vec<_> = metrics
        .iter()
        .filter_map(|m| m["psnr_db"].as_f64())
        .collect();
    let report = json!({"implementation":"rust","frame_count":count,"unique_assets":assets.len(),"duration":timeline["duration"],
        "elapsed_seconds":began.elapsed().as_secs_f64(),"mean_mae":metrics.iter().map(|m|m["mae"].as_f64().unwrap()).sum::<f64>()/count as f64,
        "mean_psnr_db":if psnr.is_empty(){None}else{Some(psnr.iter().sum::<f64>()/psnr.len() as f64)},
        "unique_svg_bytes":assets.iter().map(|a|a.bytes).sum::<usize>(),"preview_encoded":encode_preview,"verified_vector_cache_hits":cache_hits,"process_peak_working_set_bytes":peak_memory(),
        "metric_notes":"Compared with decoded working-resolution originals; PSNR is the arithmetic mean of finite per-frame dB values. No temporal-stability claim.","frames":metrics});
    write_json(&out.join("report.json"), &report)?;
    let mut summary = report;
    summary.as_object_mut().unwrap().remove("frames");
    Ok(summary)
}

fn sample_indices(count: usize, limit: usize) -> Vec<usize> {
    let n = count.min(limit);
    if n <= 1 {
        return vec![0];
    }
    (0..n).map(|i| i * (count - 1) / (n - 1)).collect()
}

fn error(a: &[u8], b: &[u8]) -> Result<(f64, f64)> {
    ensure!(!a.is_empty() && a.len() == b.len(), "Raster size mismatch");
    let (mut absolute, mut square) = (0u64, 0u64);
    for (a, b) in a.iter().zip(b) {
        let d = (*a as i64 - *b as i64).unsigned_abs();
        absolute += d;
        square += d * d;
    }
    Ok((
        absolute as f64 / a.len() as f64,
        square as f64 / a.len() as f64,
    ))
}

fn run_ffmpeg(args: Vec<String>) -> Result<()> {
    let args: Vec<_> = args.iter().map(|s| s.as_ref()).collect();
    execute("ffmpeg", &args)?;
    Ok(())
}

fn ffmpeg_path(path: &Path) -> Result<String> {
    let text = path.to_str().context("FFmpeg path must be UTF-8")?;
    // Rust canonicalize uses extended Windows paths. FFmpeg's concat demuxer
    // cannot resolve relative entries against that prefix.
    let text = if let Some(unc) = text.strip_prefix("\\\\?\\UNC\\") {
        format!("//{unc}")
    } else {
        text.strip_prefix("\\\\?\\").unwrap_or(text).to_owned()
    };
    Ok(text.replace('\\', "/"))
}

fn preview(source: &Path, out: &Path, timeline: &Value) -> Result<()> {
    let frames = timeline["frames"].as_array().unwrap();
    let duration = timeline["duration"].as_f64().unwrap();
    let mut listing = String::from("ffconcat version 1.0\n");
    for frame in frames {
        listing.push_str(&format!(
            "file 'raster/{}'\noption framerate 1000000\nduration {:.9}\n",
            frame["file"].as_str().unwrap(),
            frame["duration"].as_f64().unwrap()
        ));
    }
    listing.push_str(&format!(
        "file 'raster/{}'\noption framerate 1000000\n",
        frames.last().unwrap()["file"].as_str().unwrap()
    ));
    let list = out.join("preview.ffconcat");
    std::fs::write(&list, listing)?;
    let audio_start = timeline["requested_start"].as_f64().unwrap()
        + timeline["first_frame_offset"].as_f64().unwrap();
    let last_duration = frames.last().unwrap()["duration"].as_f64().unwrap();
    let video_path = out.join("preview.mp4");
    let mut args = vec![
        "-hide_banner".into(),
        "-nostdin".into(),
        "-y".into(),
        "-f".into(),
        "concat".into(),
        "-safe".into(),
        "0".into(),
        "-i".into(),
        ffmpeg_path(&list.canonicalize()?)?,
    ];
    if timeline["input_kind"].is_null() {
        args.extend([
            "-ss".into(),
            audio_start.to_string(),
            "-i".into(),
            ffmpeg_path(&source.canonicalize()?)?,
        ]);
    }
    args.extend(["-map".into(), "0:v:0".into()]);
    if timeline["input_kind"].is_null() {
        args.extend(["-map".into(), "1:a:0?".into()]);
    }
    args.extend([
        "-sn".into(),
        "-dn".into(),
        "-t".into(),
        duration.to_string(),
        "-c:v".into(),
        "libx264".into(),
        "-preset".into(),
        "fast".into(),
        "-crf".into(),
        "18".into(),
        "-pix_fmt".into(),
        "yuv420p".into(),
        "-vf".into(),
        "pad=ceil(iw/2)*2:ceil(ih/2)*2".into(),
        "-fps_mode".into(),
        "vfr".into(),
        "-enc_time_base".into(),
        "1/1000000".into(),
        "-bf".into(),
        "0".into(),
        "-bsf:v".into(),
        format!(
            "setts=duration=if(eq(N\\,{})\\,{last_duration:.9}/TB\\,DURATION)",
            frames.len() - 1
        ),
        "-video_track_timescale".into(),
        "1000000".into(),
        "-c:a".into(),
        "aac".into(),
        "-b:a".into(),
        "160k".into(),
        "-movflags".into(),
        "+faststart".into(),
        video_path.to_string_lossy().into_owned(),
    ]);
    run_ffmpeg(args)?;
    let metadata = probe(&video_path)?;
    write_json(&out.join("preview-probe.json"), &metadata)?;
    let video = metadata["streams"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["codec_type"] == "video")
        .context("Missing preview video")?;
    let output_count = video["nb_frames"]
        .as_str()
        .context("Missing preview frame count")?
        .parse::<usize>()?;
    let output_duration = video["duration"]
        .as_str()
        .context("Missing preview duration")?
        .parse::<f64>()?;
    ensure!(output_count == frames.len(), "Preview frame count mismatch");
    ensure!(
        (output_duration - duration).abs() <= 0.002,
        "Preview duration mismatch"
    );
    Ok(())
}

fn comparison(out: &Path, timeline: &Value) -> Result<()> {
    let frames = timeline["frames"].as_array().unwrap();
    let selected = sample_indices(frames.len(), 4);
    let first = v::read(
        &out.join("original")
            .join(frames[0]["file"].as_str().unwrap()),
    )?;
    let (w, h) = (first.cols() as usize, first.rows() as usize);
    let mut pixmap = resvg::tiny_skia::Pixmap::new((w * 2) as u32, (h * selected.len()) as u32)
        .context("Cannot allocate comparison")?;
    for (row, index) in selected.iter().enumerate() {
        let file = frames[*index]["file"].as_str().unwrap();
        for (col, dir) in ["original", "raster"].iter().enumerate() {
            let image = v::read(&out.join(dir).join(file))?;
            for (i, p) in image.data_bytes()?.chunks_exact(3).enumerate() {
                let offset = ((row * h + i / w) * w * 2 + col * w + i % w) * 4;
                pixmap.data_mut()[offset..offset + 4].copy_from_slice(&[p[2], p[1], p[0], 255]);
            }
        }
    }
    pixmap.save_png(out.join("comparison.png"))?;
    Ok(())
}

fn write_viewer(out: &Path, timeline: &Value, preview: bool) -> Result<()> {
    let frames: Vec<_> = timeline["frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| json!({"time":f["time"],"svg":f["svg"],"asset":f["asset"],"original":format!("original/{}",f["file"].as_str().unwrap()),"tracking":f["tracking"]}))
        .collect();
    let html = include_str!("viewer.html")
        .replace(
            "__MODEL_DESCRIPTION__",
            if timeline["fill_model"] == "Mesh" {
                "原生網格以線性光內插顏色，筆觸使用 Bézier 外輪廓；SVG 以可量測的純向量近似匯出。"
            } else if timeline["stroke_model"] == "Profile" {
                "筆觸使用中心線與雙側線寬曲線，填色為純色色塊。可使用 mesh 模式重跑漸層。"
            } else {
                "純色色塊與暗線遮罩基準；尚未套用正式筆觸或網格漸層。"
            },
        )
        .replace("__TIMELINE__", &serde_json::to_string(&frames)?)
        .replace("__DURATION__", &timeline["duration"].to_string())
        .replace(
            "__VIDEO__",
            if preview {
                "<video id=\"video\" controls src=\"preview.mp4\"></video>"
            } else {
                "<p>此輸出未編碼 MP4；可直接播放右側 SVG。</p>"
            },
        );
    std::fs::write(out.join("index.html"), html)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metrics_measure_actual_channels() -> Result<()> {
        assert_eq!(error(&[0, 100, 255], &[0, 90, 245])?, (20. / 3., 200. / 3.));
        assert!(error(&[0], &[]).is_err());
        assert_eq!(sample_indices(1, 12), vec![0]);
        assert_eq!(sample_indices(120, 4), vec![0, 39, 79, 119]);
        Ok(())
    }
    #[test]
    fn concat_paths_strip_extended_windows_prefix() -> Result<()> {
        assert_eq!(
            ffmpeg_path(Path::new(r"\\?\E:\clip's folder\preview.ffconcat"))?,
            "E:/clip's folder/preview.ffconcat"
        );
        assert_eq!(
            ffmpeg_path(Path::new(r"\\?\UNC\server\clips\preview.ffconcat"))?,
            "//server/clips/preview.ffconcat"
        );
        Ok(())
    }
}
