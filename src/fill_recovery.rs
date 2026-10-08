//! Recover clean color surfaces from the full ink footprint, not FWHM interiors.
use anyhow::{Result, ensure};
use opencv::{core, imgproc, prelude::*};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub restored_pixels: usize,
    pub clean_donor_pixels: usize,
    pub donor_core_fallback: bool,
    pub method: String,
    pub line_model: Option<serde_json::Value>,
    #[serde(default)]
    pub harmonic_iterations: usize,
    #[serde(default)]
    pub harmonic_residual: f64,
    #[serde(default)]
    pub harmonic_converged: bool,
    #[serde(default)]
    pub source_guided_pixels: usize,
    #[serde(default)]
    pub preserved_observations: usize,
}

/// Screened harmonic extension in linear light. Actual color discontinuities
/// are no-flux barriers; nearest donors initialize labels but do not flatten
/// every unknown color to a Voronoi patch. The positive screen also anchors
/// disconnected components and preserves the discrete maximum principle.
fn harmonic(
    colors: &mut [[u8; 3]],
    size: (usize, usize),
    domain: &[u8],
    labels: &[usize],
) -> (usize, f64, bool) {
    let (w, h) = size;
    let pixels: Vec<_> = domain
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (*m != 0).then_some(i))
        .collect();
    let n = pixels.len();
    if n == 0 {
        return (0, 0., true);
    }
    let mut ids = vec![usize::MAX; w * h];
    for (j, &i) in pixels.iter().enumerate() {
        ids[i] = j;
    }
    let linear: Vec<_> = colors
        .iter()
        .map(|c| c.map(|v| crate::mesh_fill::linear(v as f64)))
        .collect();
    let mut neighbors = vec![Vec::<usize>::new(); n];
    let mut diagonal = vec![1e-4; n];
    let mut rhs = vec![[0.; 3]; n];
    for (j, &i) in pixels.iter().enumerate() {
        rhs[j] = linear[i].map(|v| v * 1e-4);
        let (x, y) = (i % w, i / w);
        for other in [
            x.checked_sub(1).map(|_| i - 1),
            (x + 1 < w).then_some(i + 1),
            y.checked_sub(1).map(|_| i - w),
            (y + 1 < h).then_some(i + w),
        ]
        .into_iter()
        .flatten()
        {
            if labels[i] != labels[other]
                && (0..3).any(|c| colors[i][c].abs_diff(colors[other][c]) > 24)
            {
                continue;
            }
            diagonal[j] += 1.;
            if ids[other] != usize::MAX {
                neighbors[j].push(ids[other]);
            } else {
                for c in 0..3 {
                    rhs[j][c] += linear[other][c];
                }
            }
        }
    }
    let apply = |v: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| diagonal[i] * v[i] - neighbors[i].iter().map(|j| v[*j]).sum::<f64>())
            .collect()
    };
    let (mut iterations, mut residual, mut converged) = (0, 0f64, true);
    let mut output = vec![[0.; 3]; n];
    for c in 0..3 {
        let mut x: Vec<_> = pixels.iter().map(|i| linear[*i][c]).collect();
        let ax = apply(&x);
        let mut r: Vec<_> = (0..n).map(|i| rhs[i][c] - ax[i]).collect();
        let mut z: Vec<_> = (0..n).map(|i| r[i] / diagonal[i]).collect();
        let mut p = z.clone();
        let mut rz = r.iter().zip(&z).map(|(r, z)| r * z).sum::<f64>();
        let mut steps = 0;
        while steps < 512 && r.iter().any(|r| r.abs() > 1e-6) {
            let ap = apply(&p);
            let denominator = p.iter().zip(&ap).map(|(p, a)| p * a).sum::<f64>();
            if denominator <= 1e-30 || rz <= 1e-30 {
                break;
            }
            let alpha = rz / denominator;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
                z[i] = r[i] / diagonal[i];
            }
            let next = r.iter().zip(&z).map(|(r, z)| r * z).sum::<f64>();
            for i in 0..n {
                p[i] = z[i] + (next / rz) * p[i];
            }
            rz = next;
            steps += 1;
        }
        iterations = iterations.max(steps);
        let error = r.iter().map(|r| r.abs()).fold(0., f64::max);
        residual = residual.max(error);
        converged &= error <= 1e-6;
        for i in 0..n {
            output[i][c] = x[i].clamp(0., 1.);
        }
    }
    for (j, &i) in pixels.iter().enumerate() {
        colors[i] = output[j].map(|v| crate::mesh_fill::srgb(v).round().clamp(0., 255.) as u8);
    }
    (iterations, residual, converged)
}

/// The hidden boundary is not the bisector of two clean-core distances.
/// Compare local donors against the observed ink/color mixture, with the fitted
/// centerline's side only a prior when the source has lost color information.
fn source_guided_donors(
    source: &[[u8; 3]],
    recovered: &mut [[u8; 3]],
    labels: &mut [usize],
    domain: &mut [u8],
    size: (usize, usize),
    strokes: &[crate::strokes::Stroke],
    radius: f64,
) -> Result<(usize, usize)> {
    let (w, h) = size;
    let mut guide = vec![None; w * h];
    let mut proximity = vec![f64::INFINITY; w * h];
    for stroke in strokes {
        for sample in &stroke.samples {
            let sides: [Option<usize>; 2] = std::array::from_fn(|side| {
                let sign = if side == 0 { -1. } else { 1. };
                (1..=(radius * 2.) as usize).find_map(|step| {
                    let p = crate::curves::add(
                        sample.center,
                        crate::curves::mul(sample.normal, sign * step as f64 * 0.5),
                    );
                    let x = p[0].floor().clamp(0., (w - 1) as f64) as usize;
                    let y = p[1].floor().clamp(0., (h - 1) as f64) as usize;
                    let i = y * w + x;
                    (domain[i] == 0).then_some(labels[i])
                })
            });
            let reach = (sample.support_left.max(sample.support_right) + 4.)
                .max(4.)
                .min(radius);
            let lo = sample.center.map(|v| (v - reach).floor().max(0.) as usize);
            let hi = [
                (sample.center[0] + reach).ceil().min(w as f64) as usize,
                (sample.center[1] + reach).ceil().min(h as f64) as usize,
            ];
            for y in lo[1]..hi[1] {
                for x in lo[0]..hi[0] {
                    let i = y * w + x;
                    if domain[i] == 0 {
                        continue;
                    }
                    let delta = [
                        x as f64 + 0.5 - sample.center[0],
                        y as f64 + 0.5 - sample.center[1],
                    ];
                    let distance = delta[0] * delta[0] + delta[1] * delta[1];
                    if distance < proximity[i] {
                        proximity[i] = distance;
                        guide[i] = Some((sample, sides));
                    }
                }
            }
        }
    }
    let donor_labels: std::collections::BTreeSet<_> = labels
        .iter()
        .zip(domain.iter())
        .filter_map(|(label, m)| (*m == 0).then_some(*label))
        .collect();
    let mut best = vec![f64::INFINITY; w * h];
    #[derive(Clone, Copy)]
    struct Candidate {
        label: usize,
        color: [u8; 3],
        cost: f64,
        observed: bool,
    }
    let mut candidates = vec![Vec::<Candidate>::new(); w * h];
    let mut observed = vec![false; w * h];
    let opacity = |i: usize, g: &crate::strokes::Sample| {
        let side = ((i % w) as f64 + 0.5 - g.center[0]) * g.normal[0]
            + ((i / w) as f64 + 0.5 - g.center[1]) * g.normal[1];
        let (half, support) = if side < 0. {
            (g.left, g.support_left)
        } else {
            (g.right, g.support_right)
        };
        let t = side.abs();
        if t <= half + 0.25 {
            1.
        } else {
            0.6 * ((support + 0.5 - t) / (support + 0.5 - half - 0.25).max(0.5)).clamp(0., 1.)
        }
    };
    for label in donor_labels {
        let donors: Vec<_> = (0..w * h)
            .filter(|i| domain[*i] == 0 && labels[*i] == label)
            .collect();
        let mut mask = vec![255u8; w * h];
        for &i in &donors {
            mask[i] = 0;
        }
        let data = core::Mat::from_slice(&mask)?;
        let mat = data.reshape(1, h as i32)?;
        let (mut distance, mut nearest) = (core::Mat::default(), core::Mat::default());
        imgproc::distance_transform_with_labels(
            &mat,
            &mut distance,
            &mut nearest,
            imgproc::DIST_L2,
            5,
            imgproc::DIST_LABEL_PIXEL,
        )?;
        for (i, (&d, &near)) in distance
            .data_typed::<f32>()?
            .iter()
            .zip(nearest.data_typed::<i32>()?)
            .enumerate()
        {
            if domain[i] == 0 || d as f64 > radius || near <= 0 {
                continue;
            }
            let donor = donors[near as usize - 1];
            let color = source[donor];
            let clean = (0..3).all(|c| source[i][c].abs_diff(color[c]) <= 1);
            let mut cost = 0.05 * (d as f64).powi(2);
            if let Some((g, sides)) = guide[i] {
                if (0..3).all(|j| color[j].abs_diff(g.color[j]) <= 8)
                    && proximity[i] <= (g.left.max(g.right) + 0.75).powi(2)
                {
                    continue;
                }
                let c = color.map(|v| crate::mesh_fill::linear(v as f64));
                let k = g.color.map(|v| crate::mesh_fill::linear(v as f64));
                let o = source[i].map(|v| crate::mesh_fill::linear(v as f64));
                let kk: f64 = (0..3).map(|j| (k[j] - c[j]).powi(2)).sum();
                let bound = opacity(i, g);
                let alpha = if kk > 1e-12 {
                    ((0..3).map(|j| (o[j] - c[j]) * (k[j] - c[j])).sum::<f64>() / kk)
                        .clamp(0., 1. - (1. - bound).powf(2.4))
                } else {
                    0.
                };
                let linear_error = (0..3)
                    .map(|j| {
                        let predicted = crate::mesh_fill::srgb(c[j] * (1. - alpha) + k[j] * alpha);
                        (predicted - source[i][j] as f64).powi(2)
                    })
                    .sum::<f64>();
                let direction =
                    std::array::from_fn::<_, 3, _>(|j| g.color[j] as f64 - color[j] as f64);
                let length = direction.iter().map(|v| v * v).sum::<f64>();
                let alpha = if length > 1e-9 {
                    ((0..3)
                        .map(|j| (source[i][j] as f64 - color[j] as f64) * direction[j])
                        .sum::<f64>()
                        / length)
                        .clamp(0., bound)
                } else {
                    0.
                };
                let encoded_error = (0..3)
                    .map(|j| (color[j] as f64 + alpha * direction[j] - source[i][j] as f64).powi(2))
                    .sum::<f64>();
                cost += linear_error.min(encoded_error) / 3.;
                let signed = |p: [f64; 2]| {
                    (p[0] - g.center[0]) * g.normal[0] + (p[1] - g.center[1]) * g.normal[1]
                };
                let side = signed([(i % w) as f64 + 0.5, (i / w) as f64 + 0.5]);
                let target = if side.abs() > 0.35 {
                    sides[usize::from(side >= 0.)]
                } else if sides[0] == sides[1] {
                    sides[0]
                } else {
                    None
                };
                if target.is_some_and(|target| target != label) {
                    cost += 256.;
                }
            }
            // A source pixel already matching a clean local surface is direct
            // evidence, even when it is too narrow to have a 3x3 donor core.
            if clean {
                cost -= 4096.;
            }
            candidates[i].push(Candidate {
                label,
                color: if clean { source[i] } else { color },
                cost,
                observed: clean,
            });
            if cost < best[i] {
                best[i] = cost;
                recovered[i] = if clean { source[i] } else { color };
                labels[i] = label;
                observed[i] = clean;
            }
        }
    }
    let neighbors = |i: usize| {
        [
            (i % w > 0).then(|| i - 1),
            (i % w + 1 < w).then_some(i + 1),
            (i >= w).then(|| i - w),
            (i + w < w * h).then_some(i + w),
        ]
    };
    let edge_weight = |i: usize, j: usize| {
        let mut ink = false;
        for at in [i, j] {
            if let Some((g, sides)) = guide[at] {
                let signed = |v: usize| {
                    ((v % w) as f64 + 0.5 - g.center[0]) * g.normal[0]
                        + ((v / w) as f64 + 0.5 - g.center[1]) * g.normal[1]
                };
                if sides[0].is_some()
                    && sides[1].is_some()
                    && sides[0] != sides[1]
                    && signed(i) * signed(j) < 0.
                {
                    return 0.;
                }
                ink |= opacity(at, g) > 0.2;
            }
        }
        let jump = (0..3)
            .map(|c| source[i][c].abs_diff(source[j][c]) as f64)
            .fold(0., f64::max);
        if ink {
            128.
        } else {
            128. / (1. + (jump / 8.).powi(2))
        }
    };
    // Minimize an explicit local Potts energy, preserving source/centerline
    // barriers. Strict energy decrease and checkerboard updates are deterministic.
    // This is a local optimum, not a claim of globally optimal segmentation.
    for _ in 0..32 {
        let mut changed = 0;
        for parity in 0..2 {
            for i in 0..w * h {
                if candidates[i].is_empty() || ((i % w + i / w) % 2) != parity {
                    continue;
                }
                let energy = |c: &Candidate| {
                    c.cost
                        + neighbors(i)
                            .into_iter()
                            .flatten()
                            .filter(|j| labels[*j] != c.label)
                            .map(|j| edge_weight(i, j))
                            .sum::<f64>()
                };
                let candidate = candidates[i]
                    .iter()
                    .min_by(|a, b| energy(a).total_cmp(&energy(b)))
                    .unwrap();
                if candidate.label != labels[i] {
                    let old = candidates[i].iter().find(|c| c.label == labels[i]).unwrap();
                    if energy(candidate) >= energy(old) - 1e-9 {
                        continue;
                    }
                    changed += 1;
                }
                labels[i] = candidate.label;
                recovered[i] = candidate.color;
                observed[i] = candidate.observed;
            }
        }
        if changed == 0 {
            break;
        }
    }
    let mut guided = 0;
    let mut preserved = 0;
    for i in 0..w * h {
        if domain[i] != 0 && best[i].is_finite() {
            guided += usize::from(guide[i].is_some());
            if observed[i] {
                domain[i] = 0;
                preserved += 1;
            }
        }
    }
    Ok((guided, preserved))
}

pub fn restore(
    rgb: &mut [[u8; 3]],
    labels: &mut [usize],
    size: (usize, usize),
    mask: &[u8],
    strokes: &[crate::strokes::Stroke],
    hint: Option<&[f64]>,
    options: &crate::strokes::Options,
) -> Result<Report> {
    let (w, h) = size;
    ensure!(
        rgb.len() == w * h && labels.len() == w * h && mask.len() == w * h,
        "Invalid recovery dimensions"
    );
    let mut footprint = mask.to_vec();
    if let Some(hint) = hint {
        ensure!(hint.len() == w * h, "Invalid line-hint dimensions");
        // Rebuild uncertain interfaces from clean colors on both sides. Never
        // paint the prediction or classify every dark region as ink.
        for (pixel, strength) in footprint.iter_mut().zip(hint) {
            if *strength >= 0.25 {
                *pixel = 255;
            }
        }
    }
    let mut mat = core::Mat::from_slice(&footprint)?
        .reshape(1, h as i32)?
        .try_clone()?;
    for stroke in strokes {
        let mut ring = core::Vector::<core::Point>::new();
        let point = |sample: &crate::strokes::Sample, sign: f64| {
            let width = if sign < 0. {
                sample.support_left.max(sample.left)
            } else {
                sample.support_right.max(sample.right)
            };
            let p = crate::curves::add(
                sample.center,
                crate::curves::mul(sample.normal, sign * width),
            );
            core::Point::new((p[0] * 16.).round() as i32, (p[1] * 16.).round() as i32)
        };
        for sample in &stroke.samples {
            ring.push(point(sample, -1.));
        }
        for sample in stroke.samples.iter().rev() {
            ring.push(point(sample, 1.));
        }
        let rings = core::Vector::<core::Vector<core::Point>>::from_iter([ring]);
        imgproc::fill_poly(
            &mut mat,
            &rings,
            core::Scalar::all(255.),
            imgproc::LINE_8,
            4,
            core::Point::default(),
        )?;
    }
    if hint.is_some() {
        // Learned boundaries also include shading; removal needs source ink.
        let source: Vec<core::Vec3b> = rgb.iter().map(|p| core::Vec3b::from(*p)).collect();
        let source_data = core::Mat::from_slice(&source)?;
        let source_mat = source_data.reshape(3, h as i32)?;
        let mut gray = core::Mat::default();
        imgproc::cvt_color_def(&source_mat, &mut gray, imgproc::COLOR_RGB2GRAY)?;
        let radius = (options.stroke_max_width * 0.5).ceil() as i32;
        let kernel = imgproc::get_structuring_element_def(
            imgproc::MORPH_ELLIPSE,
            core::Size::new(2 * radius + 1, 2 * radius + 1),
        )?;
        let mut closed = core::Mat::default();
        imgproc::morphology_ex_def(&gray, &mut closed, imgproc::MORPH_CLOSE, &kernel)?;
        let source_gray = gray.data_typed::<u8>()?;
        let background = closed.data_typed::<u8>()?;
        for (i, m) in mat.data_typed_mut::<u8>()?.iter_mut().enumerate() {
            if (background[i].saturating_sub(source_gray[i]) as f64)
                < (options.stroke_contrast * 0.15).max(1.)
            {
                *m = 0;
            }
        }
    }
    if options.fill_edge_padding > 0. {
        let radius = options.fill_edge_padding.ceil() as i32;
        let kernel = imgproc::get_structuring_element_def(
            imgproc::MORPH_ELLIPSE,
            core::Size::new(2 * radius + 1, 2 * radius + 1),
        )?;
        let mut padded = core::Mat::default();
        imgproc::dilate_def(&mat, &mut padded, &kernel)?;
        mat = padded;
    }
    let mut unknown = mat.data_typed::<u8>()?.to_vec();
    // Complete connected mixed-color collars up to the configured line scale.
    // Stable region interiors are barriers, so genuine broad shadows survive.
    let mut coherent = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let same_label = (y.saturating_sub(1)..=(y + 1).min(h - 1)).all(|yy| {
                (x.saturating_sub(1)..=(x + 1).min(w - 1))
                    .all(|xx| labels[yy * w + xx] == labels[i])
            });
            let linear_field =
                [(1isize, 0isize), (0, 1), (1, 1), (1, -1)]
                    .iter()
                    .all(|&(dx, dy)| {
                        let at = |sign: isize| {
                            let xx = (x as isize + sign * dx).clamp(0, w as isize - 1) as usize;
                            let yy = (y as isize + sign * dy).clamp(0, h as isize - 1) as usize;
                            rgb[yy * w + xx]
                        };
                        let (a, b) = (at(-1), at(1));
                        (0..3)
                            .all(|c| (a[c] as i32 + b[c] as i32 - 2 * rgb[i][c] as i32).abs() <= 3)
                    });
            coherent[i] = (same_label || linear_field)
                && (y.saturating_sub(1)..=(y + 1).min(h - 1)).all(|yy| {
                    (x.saturating_sub(1)..=(x + 1).min(w - 1)).all(|xx| {
                        let j = yy * w + xx;
                        (0..3).all(|c| rgb[j][c].abs_diff(rgb[i][c]) <= 12)
                    })
                });
        }
    }
    // An uncertain interface band is not itself ink. Keep source color cores
    // that the learned model does not mark as lines and no fitted stroke owns.
    // This prevents the padding from erasing a genuine narrow cel shadow.
    if let Some(hint) = hint {
        for i in 0..w * h {
            if coherent[i] && mask[i] == 0 && hint[i] < 0.12 {
                unknown[i] = 0;
            }
        }
    }
    if unknown.contains(&255) {
        let inverse: Vec<u8> = unknown.iter().map(|m| 255 - m).collect();
        let inverse_data = core::Mat::from_slice(&inverse)?;
        let inverse_mat = inverse_data.reshape(1, h as i32)?;
        let mut reach = core::Mat::default();
        imgproc::distance_transform_def(&inverse_mat, &mut reach, imgproc::DIST_L2, 5)?;
        let reach = reach.data_typed::<f32>()?;
        let mut queue: std::collections::VecDeque<_> = unknown
            .iter()
            .enumerate()
            .filter(|(_, m)| **m != 0)
            .map(|(i, _)| i)
            .collect();
        while let Some(i) = queue.pop_front() {
            let (x, y) = (i % w, i / w);
            for j in [
                if x > 0 { Some(i - 1) } else { None },
                if x + 1 < w { Some(i + 1) } else { None },
                if y > 0 { Some(i - w) } else { None },
                if y + 1 < h { Some(i + w) } else { None },
            ]
            .into_iter()
            .flatten()
            {
                if unknown[j] == 0
                    && !coherent[j]
                    && reach[j] as f64 <= options.stroke_max_width * 0.5
                {
                    unknown[j] = 255;
                    queue.push_back(j);
                }
            }
        }
    }
    let count = unknown.iter().filter(|p| **p != 0).count();
    let mut report = Report {
        restored_pixels: count,
        clean_donor_pixels: w * h - count,
        donor_core_fallback: false,
        method: "full-profile-and-clean-interface-donors".into(),
        line_model: crate::lineart::identity(options.line_model.as_deref())?,
        harmonic_iterations: 0,
        harmonic_residual: 0.,
        harmonic_converged: true,
        source_guided_pixels: 0,
        preserved_observations: 0,
    };
    if count == 0 {
        return Ok(report);
    }
    ensure!(
        count < w * h,
        "No uncontaminated fill donors remain; reduce fill-edge-padding or review line model"
    );
    // Donors must lie in a coherent color core, not just outside the ink mask.
    // An isolated mixed-color fringe must never grow into a false shadow patch.
    let mut donor_mask = vec![255u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if unknown[i] != 0 {
                continue;
            }
            let stable = (y.saturating_sub(1)..=(y + 1).min(h - 1)).all(|yy| {
                (x.saturating_sub(1)..=(x + 1).min(w - 1)).all(|xx| {
                    let j = yy * w + xx;
                    labels[j] == labels[i] && (0..3).all(|c| rgb[j][c].abs_diff(rgb[i][c]) <= 12)
                })
            });
            if stable {
                donor_mask[i] = 0;
            }
        }
    }
    if !donor_mask.contains(&0) {
        // Extremely fine textures may have no region interiors. Preserve their
        // source donors, expose the fallback, and never invent a uniform color.
        donor_mask.copy_from_slice(&unknown);
        report.donor_core_fallback = true;
    }
    report.clean_donor_pixels = donor_mask.iter().filter(|m| **m == 0).count();
    let donors: Vec<_> = rgb
        .iter()
        .zip(labels.iter())
        .zip(&donor_mask)
        .filter(|(_, m)| **m == 0)
        .map(|((c, l), _)| (*c, *l))
        .collect();
    let (mut distances, mut nearest) = (core::Mat::default(), core::Mat::default());
    let donor_data = core::Mat::from_slice(&donor_mask)?;
    let donor_mat = donor_data.reshape(1, h as i32)?;
    imgproc::distance_transform_with_labels(
        &donor_mat,
        &mut distances,
        &mut nearest,
        imgproc::DIST_L2,
        5,
        imgproc::DIST_LABEL_PIXEL,
    )?;
    let mut recovered = rgb.to_vec();
    let mut recovered_labels = labels.to_vec();
    for (i, &near) in nearest.data_typed::<i32>()?.iter().enumerate() {
        if donor_mask[i] != 0 {
            ensure!(
                near > 0 && near as usize <= donors.len(),
                "Invalid clean donor"
            );
            let (color, label) = donors[near as usize - 1];
            recovered[i] = color;
            recovered_labels[i] = label;
            if unknown[i] != 0 {
                labels[i] = label;
            }
        }
    }
    let mut solve_domain = donor_mask.clone();
    let (guided, preserved) = source_guided_donors(
        rgb,
        &mut recovered,
        &mut recovered_labels,
        &mut solve_domain,
        size,
        strokes,
        options.stroke_max_width * 2. + options.fill_edge_padding,
    )?;
    report.source_guided_pixels = guided;
    report.preserved_observations = preserved;
    for i in 0..w * h {
        if unknown[i] != 0 {
            labels[i] = recovered_labels[i];
        }
    }
    let (iterations, residual, converged) =
        harmonic(&mut recovered, size, &solve_domain, &recovered_labels);
    report.harmonic_iterations = iterations;
    report.harmonic_residual = residual;
    report.harmonic_converged = converged;
    report.method = "source-and-stroke-side-guided-harmonic".into();
    for (i, color) in recovered.into_iter().enumerate() {
        if unknown[i] != 0 {
            rgb[i] = color;
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn asymmetric_clean_cores_do_not_move_a_curved_hidden_boundary() -> Result<()> {
        let (w, h) = (80usize, 64usize);
        let boundary = |y: usize| 34. + 5. * (y as f64 / 11.).sin();
        let left = [70, 130, 220];
        let right = [225, 170, 115];
        let mut expected = vec![right; w * h];
        let mut rgb = expected.clone();
        let mut labels = vec![1; w * h];
        let mut mask = vec![0; w * h];
        let mut samples = Vec::new();
        for y in 0..h {
            let bx = boundary(y);
            let slope = 5. / 11. * (y as f64 / 11.).cos();
            let n = crate::curves::unit([1., -slope]);
            samples.push(crate::strokes::Sample {
                center: [bx, y as f64 + 0.5],
                normal: n,
                left: 1.6,
                right: 1.6,
                support_left: 2.,
                support_right: 2.,
                color: [20; 3],
                confidence: 1.,
                inferred: false,
                derivative: 0.,
                curvature: 1.,
                ridge_scale: 1.,
            });
            for x in 0..w {
                let i = y * w + x;
                let side = x as f64 + 0.5 - bx;
                expected[i] = if side < 0. { left } else { right };
                rgb[i] = if side.abs() < 1.6 {
                    [20; 3]
                } else {
                    expected[i]
                };
                labels[i] = if side.abs() < 1.6 {
                    2
                } else {
                    usize::from(side >= 0.)
                };
                if (-7. ..12.).contains(&side) {
                    mask[i] = 255;
                }
            }
        }
        let stroke = crate::strokes::Stroke {
            id: 0,
            start_node: 0,
            end_node: h - 1,
            closed: false,
            samples,
            center_curve: crate::curves::Fit {
                cubics: vec![],
                max_sample_error: 0.,
            },
            outlines: vec![],
            fill: "#141414".into(),
            color: [20; 3],
            occlusion_clip: None,
            clip_curves: None,
            nodes: vec![],
            channels: Default::default(),
            surface: None,
        };
        let options = crate::strokes::Options {
            fill_edge_padding: 0.,
            ..Default::default()
        };
        let report = restore(
            &mut rgb,
            &mut labels,
            (w, h),
            &mask,
            &[stroke],
            None,
            &options,
        )?;
        assert!(report.harmonic_converged, "{report:?}");
        assert!(report.source_guided_pixels > 0 && report.preserved_observations > 0);
        let mut wrong = 0;
        for y in 3..h - 3 {
            for x in 0..w {
                let i = y * w + x;
                if rgb[i] != expected[i] {
                    wrong += 1;
                    assert!(
                        (x as f64 + 0.5 - boundary(y)).abs() < 0.8,
                        "boundary moved at {x},{y}: {:?}, expected {:?}",
                        rgb[i],
                        expected[i]
                    );
                }
            }
        }
        assert!(
            wrong < (h - 6),
            "more than one ambiguous boundary pixel per row"
        );
        Ok(())
    }

    #[test]
    fn masked_color_ramp_is_recovered_without_donor_steps_or_shadow_leakage() -> Result<()> {
        let (w, h) = (64, 24);
        let expected: Vec<_> = (0..w * h)
            .map(|i| {
                [crate::mesh_fill::srgb(0.1 + 0.6 * (i % w) as f64 / (w - 1) as f64).round() as u8;
                    3]
            })
            .collect();
        let mut rgb = expected.clone();
        let mut labels = vec![0; w * h];
        let mut mask = vec![0; w * h];
        for y in 0..h {
            for x in 23..41 {
                mask[y * w + x] = 255;
            }
        }
        let options = crate::strokes::Options {
            fill_edge_padding: 0.,
            ..Default::default()
        };
        let report = restore(&mut rgb, &mut labels, (w, h), &mask, &[], None, &options)?;
        assert!(report.harmonic_converged, "{report:?}");
        assert!(
            rgb.iter()
                .zip(&expected)
                .all(|(a, b)| a[0].abs_diff(b[0]) <= 1),
            "gradient became donor plateaus"
        );
        for y in 0..h {
            for x in 0..w {
                rgb[y * w + x] = if x < 32 { [220; 3] } else { [70; 3] };
                labels[y * w + x] = usize::from(x >= 32);
            }
        }
        let before = rgb.clone();
        let report = restore(&mut rgb, &mut labels, (w, h), &mask, &[], None, &options)?;
        assert!(report.harmonic_converged);
        assert_eq!(rgb, before, "real cel-shadow boundary leaked");
        Ok(())
    }
    #[test]
    fn ai_color_boundary_evidence_does_not_remove_a_clean_gradient() -> Result<()> {
        let (w, h) = (64, 32);
        let mut rgb: Vec<_> = (0..w * h).map(|i| [(80 + i % w * 2) as u8; 3]).collect();
        let before = rgb.clone();
        let mut labels: Vec<_> = (0..w * h).map(|i| (i % w) / 2).collect();
        let mut hint = vec![0.; w * h];
        for y in 0..h {
            hint[y * w + 32] = 1.;
        }
        let result = restore(
            &mut rgb,
            &mut labels,
            (w, h),
            &vec![0; w * h],
            &[],
            Some(&hint),
            &Default::default(),
        )?;
        assert_eq!(result.restored_pixels, 0);
        assert_eq!(rgb, before);
        Ok(())
    }
    #[test]
    fn learned_interface_padding_preserves_a_narrow_real_shadow_core() -> Result<()> {
        let (w, h) = (32, 24);
        let mut rgb = vec![[220; 3]; w * h];
        let mut labels = vec![0; w * h];
        let mut hint = vec![0.; w * h];
        for y in 0..h {
            for x in 12..17 {
                rgb[y * w + x] = [70; 3];
                labels[y * w + x] = 1;
            }
            hint[y * w + 12] = 1.;
            hint[y * w + 16] = 1.;
        }
        let report = restore(
            &mut rgb,
            &mut labels,
            (w, h),
            &vec![0; w * h],
            &[],
            Some(&hint),
            &Default::default(),
        )?;
        assert!(!report.donor_core_fallback);
        for y in 3..21 {
            assert_eq!(rgb[y * w + 14], [70; 3]);
        }
        assert_eq!(rgb[12 * w + 6], [220; 3]);
        Ok(())
    }
    #[test]
    fn isolated_fringe_is_not_copied_into_a_shadow_patch() -> Result<()> {
        let (w, h) = (32, 24);
        let mut rgb = vec![[220u8; 3]; w * h];
        let mut labels = vec![0; w * h];
        let mut mask = vec![0; w * h];
        for y in 7..12 {
            for x in 0..w {
                mask[y * w + x] = 255;
            }
        }
        rgb[12 * w + 16] = [120; 3];
        labels[12 * w + 16] = 1;
        for y in 13..17 {
            rgb[y * w + 16] = [150; 3];
            labels[y * w + 16] = 1;
        }
        let options = crate::strokes::Options {
            fill_edge_padding: 0.,
            ..Default::default()
        };
        let report = restore(&mut rgb, &mut labels, (w, h), &mask, &[], None, &options)?;
        assert_eq!(rgb[11 * w + 16], [220; 3]);
        assert_eq!(labels[11 * w + 16], 0);
        for y in 12..17 {
            assert_eq!(rgb[y * w + 16], [220; 3], "connected residual at row {y}");
        }
        assert!(!report.donor_core_fallback);
        Ok(())
    }
    #[test]
    fn clean_gradients_stay_identical_and_exhausted_donors_are_rejected() -> Result<()> {
        let mut rgb: Vec<_> = (0..64).map(|i| [i as u8 * 4; 3]).collect();
        let before = rgb.clone();
        let mut labels = vec![0; 64];
        let settings = crate::strokes::Options::default();
        let result = restore(
            &mut rgb,
            &mut labels,
            (8, 8),
            &[0; 64],
            &[],
            None,
            &settings,
        )?;
        assert_eq!(result.restored_pixels, 0);
        assert_eq!(rgb, before);
        assert!(
            restore(
                &mut rgb,
                &mut labels,
                (8, 8),
                &[255; 64],
                &[],
                None,
                &settings
            )
            .is_err()
        );
        Ok(())
    }
}
