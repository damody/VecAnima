//! Flat-color reference model. Dark masks are not fitted stroke centerlines.
use anyhow::{Context, Result, ensure};
use clap::Args;
use opencv::{core, imgcodecs, imgproc, prelude::*};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fmt::Write, path::Path};

#[derive(Args, Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    #[arg(long, default_value_t = 32)]
    pub colors: usize,
    #[arg(long, default_value_t = 0.65)]
    pub epsilon: f64,
    #[arg(long, default_value_t = 65)]
    pub stroke_threshold: u8,
    #[arg(long, default_value_t = 7)]
    pub seed: i32,
    #[command(flatten)]
    pub strokes: crate::strokes::Options,
    #[command(flatten)]
    pub fill: crate::mesh_fill::Options,
    #[command(flatten)]
    pub temporal: crate::temporal::Options,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            colors: 32,
            epsilon: 0.65,
            stroke_threshold: 65,
            seed: 7,
            strokes: Default::default(),
            fill: Default::default(),
            temporal: Default::default(),
        }
    }
}

impl Settings {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=256).contains(&self.colors),
            "colors must be in 1..=256"
        );
        ensure!(
            self.epsilon.is_finite() && self.epsilon >= 0.,
            "epsilon must be finite and nonnegative"
        );
        self.strokes.validate()?;
        self.fill.validate()?;
        self.temporal.validate()?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct Layer {
    pub kind: String,
    pub fill: String,
    pub rings: Vec<Vec<[i32; 2]>>,
    pub palette: Option<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct Frame {
    pub version: u32,
    pub model: String,
    pub width: usize,
    pub height: usize,
    pub background: String,
    pub settings: Settings,
    pub layers: Vec<Layer>,
    pub strokes: Vec<crate::strokes::Stroke>,
    pub mesh: Option<crate::mesh_fill::Fill>,
    #[serde(default)]
    pub regions: Vec<Region>,
    pub boundaries: Option<crate::shared::Graph>,
    #[serde(default)]
    pub fill_recovery: Option<crate::fill_recovery::Report>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Region {
    pub id: usize,
    pub palette: usize,
    pub area: usize,
    pub center: [f64; 2],
    pub bounds: [usize; 4],
    pub color: [u8; 3],
}
fn components(labels: &[usize], rgb: &[[u8; 3]], w: usize, h: usize) -> Vec<Region> {
    let mut visited = vec![false; labels.len()];
    let mut regions = Vec::new();
    for start in 0..labels.len() {
        if visited[start] {
            continue;
        }
        let label = labels[start];
        let mut stack = vec![start];
        visited[start] = true;
        let (mut area, mut center, mut color, mut bounds) =
            (0usize, [0.; 2], [0u64; 3], [w, h, 0, 0]);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            area += 1;
            center[0] += x as f64 + 0.5;
            center[1] += y as f64 + 0.5;
            for j in 0..3 {
                color[j] += rgb[i][j] as u64;
            }
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x + 1);
            bounds[3] = bounds[3].max(y + 1);
            for n in [
                if x > 0 { Some(i - 1) } else { None },
                if x + 1 < w { Some(i + 1) } else { None },
                if y > 0 { Some(i - w) } else { None },
                if y + 1 < h { Some(i + w) } else { None },
            ]
            .into_iter()
            .flatten()
            {
                if !visited[n] && labels[n] == label {
                    visited[n] = true;
                    stack.push(n);
                }
            }
        }
        regions.push(Region {
            id: regions.len(),
            palette: label,
            area,
            center: center.map(|v| v / area as f64),
            bounds,
            color: color.map(|v| (v / area as u64) as u8),
        });
    }
    regions
}

pub fn read(path: &Path) -> Result<core::Mat> {
    // OpenCV's Windows narrow-path fopen cannot reliably open Unicode names.
    let bytes = std::fs::read(path).with_context(|| format!("Cannot read {}", path.display()))?;
    let data = core::Mat::from_slice(&bytes)?;
    let mat = imgcodecs::imdecode(&data, imgcodecs::IMREAD_COLOR)?;
    ensure!(!mat.empty(), "Cannot read image: {}", path.display());
    Ok(mat)
}

fn smooth_lab(image: &core::Mat) -> Result<(core::Mat, core::Mat)> {
    let mut smooth = core::Mat::default();
    let mut lab = core::Mat::default();
    imgproc::bilateral_filter_def(image, &mut smooth, 5, 25., 25.)?;
    imgproc::cvt_color_def(&smooth, &mut lab, imgproc::COLOR_BGR2Lab)?;
    Ok((smooth, lab))
}

pub fn fit_palette(paths: &[std::path::PathBuf], settings: &Settings) -> Result<Vec<[f32; 3]>> {
    settings.validate()?;
    ensure!(!paths.is_empty(), "No palette samples");
    let mut samples: Vec<[f32; 3]> = Vec::new();
    let mut unique = BTreeSet::new();
    for path in paths {
        let (_, lab) = smooth_lab(&read(path)?)?;
        let pixels = lab.data_typed::<core::Vec3b>()?;
        for p in pixels.iter().step_by((pixels.len() / 4000).max(1)) {
            unique.insert([p[0], p[1], p[2]]);
            samples.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }
    }
    let k = settings.colors.min(unique.len());
    ensure!(k > 0, "Empty palette");
    let data = core::Mat::from_slice_2d(&samples)?;
    let mut labels = core::Mat::default();
    let mut centers = core::Mat::default();
    core::set_rng_seed(settings.seed)?;
    core::kmeans(
        &data,
        k as i32,
        &mut labels,
        core::TermCriteria::new(
            core::TermCriteria_Type::COUNT as i32 | core::TermCriteria_Type::EPS as i32,
            50,
            0.1,
        )?,
        1,
        core::KMEANS_PP_CENTERS,
        &mut centers,
    )?;
    Ok(centers
        .data_typed::<f32>()?
        .chunks_exact(3)
        .map(|p| [p[0], p[1], p[2]])
        .collect())
}

fn area2(ring: &[[i32; 2]]) -> i64 {
    ring.iter()
        .zip(ring.iter().cycle().skip(1))
        .take(ring.len())
        .map(|(a, b)| a[0] as i64 * b[1] as i64 - b[0] as i64 * a[1] as i64)
        .sum()
}

/// Trace foreground pixel cells clockwise, with holes counterclockwise.
/// Direction bits and a sorted start list give deterministic four-connected rings.
pub fn mask_rings(
    mask: &[u8],
    width: usize,
    height: usize,
    epsilon: f64,
) -> Result<Vec<Vec<[i32; 2]>>> {
    ensure!(
        width > 0 && height > 0 && width.checked_mul(height) == Some(mask.len()),
        "Invalid mask dimensions"
    );
    ensure!(
        epsilon.is_finite() && epsilon >= 0.,
        "Invalid contour epsilon"
    );
    let stride = width + 1;
    let mut edges = vec![0u8; stride * (height + 1)];
    let mut active = Vec::new();
    let mut add = |x: usize, y: usize, direction: u8| {
        let i = y * stride + x;
        if edges[i] == 0 {
            active.push(i);
        }
        edges[i] |= 1 << direction;
    };
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            if mask[i] == 0 {
                continue;
            }
            if y == 0 || mask[i - width] == 0 {
                add(x, y, 0);
            }
            if x + 1 == width || mask[i + 1] == 0 {
                add(x + 1, y, 1);
            }
            if y + 1 == height || mask[i + width] == 0 {
                add(x + 1, y + 1, 2);
            }
            if x == 0 || mask[i - 1] == 0 {
                add(x, y + 1, 3);
            }
        }
    }
    active.sort_unstable();
    let offsets = [1isize, stride as isize, -1, -(stride as isize)];
    let mut rings = Vec::new();
    for start in active {
        while edges[start] != 0 {
            let mut current = start;
            let mut previous = edges[start].trailing_zeros() as usize;
            let mut ring = Vec::new();
            loop {
                ring.push([(current % stride) as i32, (current / stride) as i32]);
                let direction = [1, 0, 3, 2]
                    .iter()
                    .map(|turn| (previous + turn) % 4)
                    .find(|d| edges[current] & (1 << d) != 0)
                    .context("Broken boundary graph")?;
                edges[current] &= !(1 << direction);
                current = current
                    .checked_add_signed(offsets[direction])
                    .context("Invalid boundary offset")?;
                previous = direction;
                if current == start {
                    break;
                }
            }
            let contour = core::Vector::<core::Point>::from_iter(
                ring.iter().map(|p| core::Point::new(p[0], p[1])),
            );
            let mut reduced = core::Vector::<core::Point>::new();
            imgproc::approx_poly_dp(&contour, &mut reduced, epsilon, true)?;
            let simplified: Vec<_> = reduced.iter().map(|p| [p.x, p.y]).collect();
            if simplified.len() >= 3 && area2(&simplified).abs() > 0 {
                rings.push(simplified);
            } else {
                rings.push(ring);
            }
        }
    }
    Ok(rings)
}

fn hex(p: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", p[2], p[1], p[0])
}

pub fn vectorize(image: &core::Mat, palette: &[[f32; 3]], settings: &Settings) -> Result<Frame> {
    settings.validate()?;
    ensure!(!palette.is_empty(), "Empty palette");
    let (filtered, lab) = smooth_lab(image)?;
    let width = image.cols() as usize;
    let height = image.rows() as usize;
    let mut labels: Vec<usize> = lab
        .data_typed::<core::Vec3b>()?
        .iter()
        .map(|p| {
            palette
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let distance =
                        |c: &[f32; 3]| (0..3).map(|j| (p[j] as f32 - c[j]).powi(2)).sum::<f32>();
                    distance(a).total_cmp(&distance(b))
                })
                .unwrap()
                .0
        })
        .collect();
    let bytes: Vec<core::Vec3b> = palette
        .iter()
        .map(|c| {
            core::Vec3b::from([
                c[0].clamp(0., 255.) as u8,
                c[1].clamp(0., 255.) as u8,
                c[2].clamp(0., 255.) as u8,
            ])
        })
        .collect();
    let lab_colors = core::Mat::from_slice(&bytes)?;
    let mut bgr_colors = core::Mat::default();
    imgproc::cvt_color_def(&lab_colors, &mut bgr_colors, imgproc::COLOR_Lab2BGR)?;
    let colors: Vec<[u8; 3]> = bgr_colors
        .data_typed::<core::Vec3b>()?
        .iter()
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    let mut gray = core::Mat::default();
    imgproc::cvt_color_def(&filtered, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    let mut dark: Vec<u8> = gray
        .data_typed::<u8>()?
        .iter()
        .map(|v| {
            if *v < settings.stroke_threshold {
                255
            } else {
                0
            }
        })
        .collect();
    let mut strokes = Vec::new();
    let mut line_hint = None;
    if settings.strokes.stroke_model == crate::strokes::Model::Profile {
        let detection = crate::strokes::detect(image, &settings.strokes)?;
        dark = detection.mask;
        strokes = detection.strokes;
        line_hint = detection.line_hint;
    }
    let mut fill_rgb: Vec<[u8; 3]> = image
        .data_typed::<core::Vec3b>()?
        .iter()
        .map(|p| [p[2], p[1], p[0]])
        .collect();
    let fill_recovery = if settings.strokes.stroke_model == crate::strokes::Model::Profile {
        Some(crate::fill_recovery::restore(
            &mut fill_rgb,
            &mut labels,
            (width, height),
            &dark,
            &strokes,
            line_hint.as_deref(),
            &settings.strokes,
        )?)
    } else {
        None
    };
    if settings.strokes.stroke_model == crate::strokes::Model::Baseline
        && dark.contains(&255)
        && dark.contains(&0)
    {
        let dark_mat = core::Mat::from_slice(&dark)?;
        let dark_mat = dark_mat.reshape(1, height as i32)?;
        let mut distances = core::Mat::default();
        let mut nearest = core::Mat::default();
        imgproc::distance_transform_with_labels(
            &dark_mat,
            &mut distances,
            &mut nearest,
            imgproc::DIST_L2,
            5,
            imgproc::DIST_LABEL_PIXEL,
        )?;
        let lookup: Vec<usize> = labels
            .iter()
            .zip(&dark)
            .filter(|(_, d)| **d == 0)
            .map(|(l, _)| *l)
            .collect();
        let lookup_rgb: Vec<_> = fill_rgb
            .iter()
            .zip(&dark)
            .filter(|(_, d)| **d == 0)
            .map(|(p, _)| *p)
            .collect();
        for (i, near) in nearest.data_typed::<i32>()?.iter().enumerate() {
            if dark[i] != 0 {
                fill_rgb[i] = *lookup_rgb
                    .get((*near - 1) as usize)
                    .context("Invalid nearest fill color")?;
            }
        }
        for ((label, d), near) in labels
            .iter_mut()
            .zip(&dark)
            .zip(nearest.data_typed::<i32>()?)
        {
            if *d != 0 {
                *label = *lookup
                    .get((*near - 1) as usize)
                    .context("Invalid nearest region")?;
            }
        }
    }
    let mut counts = vec![0usize; palette.len()];
    for l in &labels {
        counts[*l] += 1;
    }
    let dominant = counts
        .iter()
        .enumerate()
        .max_by_key(|(i, n)| (**n, std::cmp::Reverse(*i)))
        .unwrap()
        .0;
    let mut layers = Vec::new();
    for (i, color) in colors.iter().enumerate() {
        if counts[i] == 0 {
            continue;
        }
        let mask: Vec<u8> = labels.iter().map(|l| u8::from(*l == i)).collect();
        let rings = mask_rings(&mask, width, height, settings.epsilon)?;
        if !rings.is_empty() {
            layers.push(Layer {
                kind: "region".into(),
                fill: hex(*color),
                rings,
                palette: Some(i),
            });
        }
    }
    if dark.contains(&255) && settings.strokes.stroke_model == crate::strokes::Model::Baseline {
        let rings = mask_rings(&dark, width, height, settings.epsilon / 2.)?;
        let pixels = image.data_typed::<core::Vec3b>()?;
        let mut median = [0u8; 3];
        for j in 0..3 {
            let mut values: Vec<u8> = pixels
                .iter()
                .zip(&dark)
                .filter(|(_, d)| **d != 0)
                .map(|(p, _)| p[j])
                .collect();
            values.sort_unstable();
            let n = values.len();
            median[j] = ((values[(n - 1) / 2] as u16 + values[n / 2] as u16) / 2) as u8;
        }
        layers.push(Layer {
            kind: "dark-mask-baseline".into(),
            fill: hex(median),
            rings,
            palette: None,
        });
    }
    let boundaries = if settings.strokes.stroke_model == crate::strokes::Model::Profile {
        Some(crate::shared::build(&labels, width, height, 0.65)?)
    } else {
        None
    };
    let mesh = if settings.fill.fill_model == crate::mesh_fill::Model::Mesh {
        Some(crate::mesh_fill::build(
            &fill_rgb,
            &labels,
            width,
            height,
            &settings.fill,
            palette,
            boundaries.as_ref(),
        )?)
    } else {
        None
    };
    let regions = components(&labels, &fill_rgb, width, height);
    Ok(Frame {
        version: 3,
        model: if mesh.is_some() {
            "profile-strokes-and-linear-light-mesh"
        } else if settings.strokes.stroke_model == crate::strokes::Model::Baseline {
            "flat-regions-and-dark-mask-baseline"
        } else {
            "profile-strokes-and-flat-regions"
        }
        .into(),
        width,
        height,
        background: hex(colors[dominant]),
        settings: settings.clone(),
        layers,
        strokes,
        mesh,
        regions,
        boundaries,
        fill_recovery,
    })
}

pub fn svg(frame: &Frame) -> String {
    svg_parts(frame, true, true)
}
pub fn strokes_svg(frame: &Frame) -> String {
    svg_parts(frame, false, true)
}
pub fn fills_svg(frame: &Frame) -> String {
    svg_parts(frame, true, false)
}
fn svg_parts(frame: &Frame, include_fill: bool, include_strokes: bool) -> String {
    let mut text = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
        frame.width, frame.height, frame.width, frame.height
    );
    if include_fill {
        writeln!(
            text,
            "<rect width=\"100%\" height=\"100%\" fill=\"{}\"/>",
            frame.background
        )
        .unwrap();
        if let Some(mesh) = &frame.mesh {
            crate::mesh_fill::svg(mesh, &mut text);
        }
    }
    for kind in ["region", "dark-mask-baseline"] {
        writeln!(text, "<g id=\"{kind}\">").unwrap();
        for layer in frame.layers.iter().filter(|l| {
            l.kind == kind
                && (if kind == "region" {
                    include_fill && frame.mesh.is_none()
                } else {
                    include_strokes
                })
        }) {
            write!(
                text,
                "<path fill=\"{}\" fill-rule=\"evenodd\" d=\"",
                layer.fill
            )
            .unwrap();
            if kind == "region"
                && let Some(graph) = &frame.boundaries
            {
                let index = frame
                    .layers
                    .iter()
                    .filter(|l| l.kind == "region")
                    .position(|l| std::ptr::eq(l, layer))
                    .unwrap();
                // Layers retain palette identity independently of missing colors.
                let palette = layer.palette.unwrap_or(index);
                if let Some(rings) = graph.loops.get(&palette) {
                    for ring in rings {
                        curve_path(&mut text, &[graph.ring(ring)]);
                    }
                }
            } else {
                for ring in &layer.rings {
                    for (i, p) in ring.iter().enumerate() {
                        write!(text, "{}{},{} ", if i == 0 { "M" } else { "L" }, p[0], p[1])
                            .unwrap();
                    }
                    text.push_str("Z ");
                }
            }
            text.push_str("\"/>\n");
        }
        text.push_str("</g>\n");
    }
    text.push_str("<g id=\"profile-strokes\">\n");
    for stroke in frame.strokes.iter().filter(|_| include_strokes) {
        if let Some(clips) = &stroke.clip_curves {
            write!(
                text,
                "<defs><clipPath id=\"stroke-clip-{}\"><path clip-rule=\"evenodd\" d=\"",
                stroke.id
            )
            .unwrap();
            curve_path(&mut text, clips);
            write!(
                text,
                "\"/></clipPath></defs><g clip-path=\"url(#stroke-clip-{})\">",
                stroke.id
            )
            .unwrap();
        }
        write!(
            text,
            "<path data-stroke=\"{}\" fill=\"{}\" fill-rule=\"evenodd\" d=\"",
            stroke.id, stroke.fill
        )
        .unwrap();
        for outline in &stroke.outlines {
            if let Some(first) = outline.cubics.first() {
                write!(text, "M{:.3},{:.3} ", first[0][0], first[0][1]).unwrap();
                for c in &outline.cubics {
                    write!(
                        text,
                        "C{:.3},{:.3} {:.3},{:.3} {:.3},{:.3} ",
                        c[1][0], c[1][1], c[2][0], c[2][1], c[3][0], c[3][1]
                    )
                    .unwrap();
                }
                text.push_str("Z ");
            }
        }
        text.push_str("\"/>\n");
        if let Some(surface) = &stroke.surface {
            write!(
                text,
                "<defs><clipPath id=\"stroke-shape-{}\"><path clip-rule=\"evenodd\" d=\"",
                stroke.id
            )
            .unwrap();
            curve_path(&mut text, &stroke.outlines);
            write!(
                text,
                "\"/></clipPath></defs><g clip-path=\"url(#stroke-shape-{})\">",
                stroke.id
            )
            .unwrap();
            crate::mesh_fill::svg_surface(
                &surface.geometry,
                &surface.colors_linear,
                1.,
                &format!("stroke-{}", stroke.id),
                &mut text,
            );
            text.push_str("</g>\n");
        }
        if stroke.clip_curves.is_some() {
            text.push_str("</g>\n");
        }
    }
    text.push_str("</g>\n</svg>\n");
    text
}

fn curve_path(text: &mut String, curves: &[crate::curves::Fit]) {
    for curve in curves {
        if let Some(first) = curve.cubics.first() {
            write!(text, "M{:.6},{:.6} ", first[0][0], first[0][1]).unwrap();
            for c in &curve.cubics {
                write!(
                    text,
                    "C{:.6},{:.6} {:.6},{:.6} {:.6},{:.6} ",
                    c[1][0], c[1][1], c[2][0], c[2][1], c[3][0], c[3][1]
                )
                .unwrap();
            }
            text.push_str("Z ");
        }
    }
}
pub fn render(frame: &Frame, text: &str) -> Result<resvg::tiny_skia::Pixmap> {
    if let Some(mesh) = &frame.mesh {
        let mut image = crate::mesh_fill::raster(mesh, frame.width, frame.height)?;
        let overlay = rasterize_alpha(&svg_parts(frame, false, true))?;
        image.draw_pixmap(
            0,
            0,
            overlay.as_ref(),
            &resvg::tiny_skia::PixmapPaint::default(),
            resvg::tiny_skia::Transform::identity(),
            None,
        );
        Ok(image)
    } else {
        rasterize(text)
    }
}
pub fn rasterize(text: &str) -> Result<resvg::tiny_skia::Pixmap> {
    let pixmap = rasterize_alpha(text)?;
    ensure!(
        pixmap.data().chunks_exact(4).all(|p| p[3] == 255),
        "SVG raster must be opaque"
    );
    Ok(pixmap)
}
fn rasterize_alpha(text: &str) -> Result<resvg::tiny_skia::Pixmap> {
    let tree = resvg::usvg::Tree::from_str(text, &resvg::usvg::Options::default())?;
    let size = tree.size().to_int_size();
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size.width(), size.height())
        .context("Cannot allocate SVG raster")?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::identity(),
        &mut pixmap.as_mut(),
    );
    Ok(pixmap)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cell_boundaries_preserve_holes_diagonals_and_single_pixels() -> Result<()> {
        let (w, h) = (16, 16);
        let mut mask = vec![0u8; w * h];
        for y in 0..12 {
            for x in 0..12 {
                mask[y * w + x] = 1;
            }
        }
        for y in 2..10 {
            for x in 2..10 {
                mask[y * w + x] = 0;
            }
        }
        for y in 4..8 {
            for x in 4..8 {
                mask[y * w + x] = 1;
            }
        }
        mask[12 * w + 12] = 1;
        mask[15 * w + 15] = 1;
        let rings = mask_rings(&mask, w, h, 0.)?;
        assert_eq!(rings.len(), 5);
        let frame = Frame {
            version: 1,
            model: "test".into(),
            fill_recovery: None,
            width: w,
            height: h,
            background: "#ffffff".into(),
            settings: Settings {
                colors: 2,
                epsilon: 0.,
                stroke_threshold: 65,
                seed: 7,
                strokes: Default::default(),
                fill: Default::default(),
                temporal: Default::default(),
            },
            layers: vec![Layer {
                kind: "region".into(),
                fill: "#000000".into(),
                rings,
                palette: None,
            }],
            strokes: Vec::new(),
            mesh: None,
            regions: Vec::new(),
            boundaries: None,
        };
        let text = svg(&frame);
        assert!(!text.contains("<image"));
        let raster = rasterize(&text)?;
        for (p, m) in raster.data().chunks_exact(4).zip(&mask) {
            assert_eq!(p[0], if *m == 0 { 255 } else { 0 });
        }
        Ok(())
    }
    #[test]
    fn fills_svg_removes_ink_across_unequal_backgrounds_but_preserves_dark_regions() -> Result<()> {
        let (w, h) = (128, 96);
        let mut image =
            core::Mat::new_rows_cols_with_default(h, w, core::CV_8UC3, core::Scalar::all(210.))?;
        imgproc::rectangle(
            &mut image,
            core::Rect::new(64, 0, 64, h),
            core::Scalar::all(130.),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::line(
            &mut image,
            core::Point::new(8, 32),
            core::Point::new(120, 32),
            core::Scalar::all(20.),
            3,
            imgproc::LINE_AA,
            0,
        )?;
        imgproc::line(
            &mut image,
            core::Point::new(64, 8),
            core::Point::new(64, 54),
            core::Scalar::all(20.),
            3,
            imgproc::LINE_AA,
            0,
        )?;
        imgproc::rectangle(
            &mut image,
            core::Rect::new(20, 64, 36, 25),
            core::Scalar::all(20.),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        let palette_bytes = [
            core::Vec3b::from([20; 3]),
            core::Vec3b::from([130; 3]),
            core::Vec3b::from([210; 3]),
        ];
        let input = core::Mat::from_slice(&palette_bytes)?;
        let mut lab = core::Mat::default();
        imgproc::cvt_color_def(&input, &mut lab, imgproc::COLOR_BGR2Lab)?;
        let palette: Vec<_> = lab
            .data_typed::<core::Vec3b>()?
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        for model in [crate::mesh_fill::Model::Flat, crate::mesh_fill::Model::Mesh] {
            let mut settings = Settings::default();
            settings.fill.fill_model = model;
            let frame = vectorize(&image, &palette, &settings)?;
            assert!(!frame.strokes.is_empty());
            let fills = rasterize(&fills_svg(&frame))?;
            let full = rasterize(&svg(&frame))?;
            for (x, background) in [(24usize, 210u8), (96, 130)] {
                for y in 30..=34 {
                    let c = fills.data()[(y * w as usize + x) * 4];
                    assert!(
                        c.abs_diff(background) <= 12,
                        "{model:?}: fill retained ink at {x},{y}: {c}"
                    );
                }
                assert!(
                    full.data()[(32 * w as usize + x) * 4] < 65,
                    "stroke missing after separation"
                );
            }
            for y in [16usize, 44] {
                for x in 62..=66 {
                    assert!(
                        fills.data()[(y * w as usize + x) * 4] >= 115,
                        "{model:?}: asymmetric boundary retained ink at {x},{y}"
                    );
                }
                assert!(
                    full.data()[(y * w as usize + 64) * 4] < 65,
                    "asymmetric boundary stroke not reconstructed"
                );
            }
            assert!(
                fills.data()[(76 * w as usize + 36) * 4] < 40,
                "genuine dark region was removed"
            );
        }
        Ok(())
    }
    #[test]
    fn very_large_epsilon_keeps_isolated_pixels() -> Result<()> {
        let rings = mask_rings(&[1], 1, 1, 100.)?;
        assert_eq!(rings.len(), 1);
        assert_eq!(area2(&rings[0]).abs(), 2);
        Ok(())
    }
    #[test]
    fn wide_faint_penumbra_does_not_become_a_fill_shadow() -> Result<()> {
        let (w, h) = (128, 80);
        let mut image =
            core::Mat::new_rows_cols_with_default(h, w, core::CV_8UC3, core::Scalar::all(220.))?;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let background = if x < 64 { 220. } else { 130. };
                let alpha = if (8..120).contains(&x) {
                    (-(y as f64 - 26.).powi(2) / 10.).exp()
                } else {
                    0.
                };
                let value = (background * (1. - alpha) + 20. * alpha).round() as u8;
                image.data_typed_mut::<core::Vec3b>()?[y * w as usize + x] =
                    core::Vec3b::from([value; 3]);
            }
        }
        imgproc::rectangle(
            &mut image,
            core::Rect::new(18, 52, 30, 20),
            core::Scalar::all(60.),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        let palette_bytes = [
            core::Vec3b::from([20; 3]),
            core::Vec3b::from([60; 3]),
            core::Vec3b::from([130; 3]),
            core::Vec3b::from([220; 3]),
        ];
        let input = core::Mat::from_slice(&palette_bytes)?;
        let mut lab = core::Mat::default();
        imgproc::cvt_color_def(&input, &mut lab, imgproc::COLOR_BGR2Lab)?;
        let palette: Vec<_> = lab
            .data_typed::<core::Vec3b>()?
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        for model in [crate::mesh_fill::Model::Flat, crate::mesh_fill::Model::Mesh] {
            let mut settings = Settings::default();
            settings.fill.fill_model = model;
            let frame = vectorize(&image, &palette, &settings)?;
            assert!(
                frame
                    .strokes
                    .iter()
                    .flat_map(|s| &s.samples)
                    .any(|s| s.support_left > s.left + 1.)
            );
            let fills = rasterize(&fills_svg(&frame))?;
            for (x, bg) in [(28usize, 220u8), (96, 130)] {
                for y in 18..=34 {
                    let color = fills.data()[(y * w as usize + x) * 4];
                    assert!(
                        color.abs_diff(bg) <= 4,
                        "{model:?}: penumbra copied into fill at {x},{y}: {color}, expected {bg}"
                    );
                }
            }
            assert!(
                fills.data()[(62 * w as usize + 32) * 4].abs_diff(60) <= 4,
                "real shadow lost"
            );
            let full = rasterize(&svg(&frame))?;
            assert!(
                full.data()[(26 * w as usize + 28) * 4] < 65,
                "ink reconstruction missing"
            );
        }
        Ok(())
    }
    #[test]
    fn invalid_settings_and_masks_are_rejected() {
        assert!(mask_rings(&[1], 2, 2, 0.).is_err());
        assert!(mask_rings(&[1], 1, 1, f64::NAN).is_err());
        assert!(
            Settings {
                colors: 0,
                epsilon: 0.,
                stroke_threshold: 65,
                seed: 7,
                strokes: Default::default(),
                fill: Default::default(),
                temporal: Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
