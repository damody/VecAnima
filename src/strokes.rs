//! Local-contrast candidates, topology-preserving thinning and independent profiles.
use crate::curves::{self, Point, add, mul, norm, sub, unit};
use anyhow::{Result, ensure};
use clap::{Args, ValueEnum};
use opencv::{core, imgproc, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
pub enum Model {
    Baseline,
    Profile,
}

#[derive(Clone, Debug, Serialize, Deserialize, Args)]
pub struct Options {
    /// Optional local AniLines basic RGB ONNX model; never downloads or invokes Python.
    #[arg(long)]
    pub line_model: Option<std::path::PathBuf>,
    /// Extra working-pixel safety band for clean fill reconstruction.
    #[arg(long, default_value_t = 2.)]
    pub fill_edge_padding: f64,
    #[arg(long,value_enum,default_value_t=Model::Profile)]
    pub stroke_model: Model,
    #[arg(long, default_value_t = 12.)]
    pub stroke_contrast: f64,
    #[arg(long, default_value_t = 16.)]
    pub stroke_max_width: f64,
    #[arg(long, default_value_t = 0.35)]
    pub curve_error: f64,
    #[arg(long, default_value_t = 24.)]
    pub stroke_color_error: f64,
    #[arg(long, default_value_t = 3.)]
    pub stroke_gap_max: f64,
    /// Minimum (centerline length + width) / width, including caps; width is its 90th percentile.
    #[arg(long, default_value_t = 2.)]
    pub stroke_min_aspect: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            line_model: None,
            fill_edge_padding: 2.,
            stroke_model: Model::Profile,
            stroke_contrast: 12.,
            stroke_max_width: 16.,
            curve_error: 0.35,
            stroke_color_error: 24.,
            stroke_gap_max: 3.,
            stroke_min_aspect: 2.,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.fill_edge_padding.is_finite() && (0. ..=8.).contains(&self.fill_edge_padding),
            "fill-edge-padding must be in 0..=8"
        );
        if let Some(path) = &self.line_model {
            ensure!(
                path.is_file(),
                "Line model does not exist: {}",
                path.display()
            );
        }
        ensure!(
            self.stroke_contrast.is_finite()
                && self.stroke_contrast > 0.
                && self.stroke_contrast < 255.,
            "stroke-contrast must be in (0,255)"
        );
        ensure!(
            self.stroke_max_width.is_finite() && (2. ..=128.).contains(&self.stroke_max_width),
            "stroke-max-width must be in 2..128"
        );
        ensure!(
            self.curve_error.is_finite() && self.curve_error > 0.,
            "curve-error must be positive and finite"
        );
        ensure!(
            self.stroke_color_error.is_finite() && self.stroke_color_error > 0.,
            "stroke-color-error must be positive and finite"
        );
        ensure!(
            self.stroke_gap_max.is_finite() && (0. ..=16.).contains(&self.stroke_gap_max),
            "stroke-gap-max must be in 0..16"
        );
        ensure!(
            self.stroke_min_aspect.is_finite() && self.stroke_min_aspect >= 1.,
            "stroke-min-aspect must be finite and at least 1"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    pub center: Point,
    pub normal: Point,
    pub left: f64,
    pub right: f64,
    #[serde(default)]
    pub support_left: f64,
    #[serde(default)]
    pub support_right: f64,
    pub color: [u8; 3],
    pub confidence: f64,
    pub inferred: bool,
    pub derivative: f64,
    pub curvature: f64,
    pub ridge_scale: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stroke {
    pub id: usize,
    pub start_node: usize,
    pub end_node: usize,
    pub closed: bool,
    pub samples: Vec<Sample>,
    pub center_curve: curves::Fit,
    pub outlines: Vec<curves::Fit>,
    pub fill: String,
    pub color: [u8; 3],
    pub occlusion_clip: Option<Vec<Vec<[i32; 2]>>>,
    pub clip_curves: Option<Vec<curves::Fit>>,
    pub nodes: Vec<Node>,
    pub channels: Channels,
    pub surface: Option<Surface>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub arc: f64,
    pub center: Point,
    pub left: f64,
    pub right: f64,
    pub color_linear: [f64; 3],
    pub confidence: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Channels {
    pub left: crate::spline::Scalar,
    pub right: crate::spline::Scalar,
    pub color: [crate::spline::Scalar; 3],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Surface {
    pub geometry: crate::geometry::MeshData,
    pub colors_linear: Vec<[[f64; 3]; 3]>,
}
pub struct Detection {
    pub strokes: Vec<Stroke>,
    pub mask: Vec<u8>,
    pub line_hint: Option<Vec<f64>>,
}

fn bilinear(values: &[f64], w: usize, h: usize, p: Point) -> f64 {
    let x = p[0].clamp(0., (w - 1) as f64);
    let y = p[1].clamp(0., (h - 1) as f64);
    let (ix, iy) = (x.floor() as usize, y.floor() as usize);
    let (fx, fy) = (x - ix as f64, y - iy as f64);
    let (jx, jy) = ((ix + 1).min(w - 1), (iy + 1).min(h - 1));
    values[iy * w + ix] * (1. - fx) * (1. - fy)
        + values[iy * w + jx] * fx * (1. - fy)
        + values[jy * w + ix] * (1. - fx) * fy
        + values[jy * w + jx] * fx * fy
}

/// Strong learned support seeds connected weak support; a weak component with
/// no seed is never promoted. Source profiles still validate every path.
fn learned_support(hint: &[f64], w: usize, h: usize) -> Vec<u8> {
    let mut mask = vec![0; hint.len()];
    let mut queue = Vec::new();
    for (i, value) in hint.iter().enumerate() {
        if *value >= 0.2 {
            mask[i] = 1;
            queue.push(i);
        }
    }
    while let Some(i) = queue.pop() {
        let (x, y) = (i % w, i / w);
        for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                let j = yy * w + xx;
                if mask[j] == 0 && hint[j] >= 0.05 {
                    mask[j] = 1;
                    queue.push(j);
                }
            }
        }
    }
    mask
}

fn rgb_at(rgb: &[[u8; 3]], size: (usize, usize), p: Point) -> [u8; 3] {
    let (w, h) = size;
    let x = p[0].clamp(0., (w - 1) as f64);
    let y = p[1].clamp(0., (h - 1) as f64);
    let (ix, iy) = (x.floor() as usize, y.floor() as usize);
    let (dx, dy) = (x - ix as f64, y - iy as f64);
    let corners = [
        (ix, iy, (1. - dx) * (1. - dy)),
        ((ix + 1).min(w - 1), iy, dx * (1. - dy)),
        (ix, (iy + 1).min(h - 1), (1. - dx) * dy),
        ((ix + 1).min(w - 1), (iy + 1).min(h - 1), dx * dy),
    ];
    std::array::from_fn(|k| {
        crate::mesh_fill::srgb(
            corners
                .iter()
                .map(|&(x, y, weight)| weight * crate::mesh_fill::linear(rgb[y * w + x][k] as f64))
                .sum(),
        )
        .round()
        .clamp(0., 255.) as u8
    })
}

fn ink_peak(ink: &[f64], size: (usize, usize), p: Point, normal: Point, radius: f64) -> Point {
    let (w, h) = size;
    let mut best = p;
    let mut score = bilinear(ink, w, h, p);
    for step in -6..=6 {
        let offset = step as f64 * radius / 6.;
        let q = add(p, mul(normal, offset));
        // A locality penalty resolves broad/ambiguous ridges without jumping
        // to a separate nearby contour.
        let value = bilinear(ink, w, h, q) - offset.abs();
        if value > score {
            best = q;
            score = value;
        }
    }
    best
}

/// Two parallel deletion passes. Borders are handled with explicit zero padding.
fn thin(mask: &[u8], w: usize, h: usize) -> Vec<u8> {
    let stride = w + 2;
    let mut pixels = vec![0u8; stride * (h + 2)];
    for y in 0..h {
        for x in 0..w {
            pixels[(y + 1) * stride + x + 1] = u8::from(mask[y * w + x] != 0);
        }
    }
    loop {
        let mut changed = false;
        for pass in 0..2 {
            let mut erase = Vec::new();
            for y in 1..=h {
                for x in 1..=w {
                    let i = y * stride + x;
                    if pixels[i] == 0 {
                        continue;
                    }
                    let p = [
                        pixels[i - stride],
                        pixels[i - stride + 1],
                        pixels[i + 1],
                        pixels[i + stride + 1],
                        pixels[i + stride],
                        pixels[i + stride - 1],
                        pixels[i - 1],
                        pixels[i - stride - 1],
                    ];
                    let neighbors = p.iter().sum::<u8>();
                    if !(2..=6).contains(&neighbors) {
                        continue;
                    }
                    let changes = (0..8)
                        .filter(|j| p[*j] == 0 && p[(*j + 1) % 8] != 0)
                        .count();
                    let keep = if pass == 0 {
                        p[0] * p[2] * p[4] != 0 || p[2] * p[4] * p[6] != 0
                    } else {
                        p[0] * p[2] * p[6] != 0 || p[0] * p[4] * p[6] != 0
                    };
                    if changes == 1 && !keep {
                        erase.push(i);
                    }
                }
            }
            changed |= !erase.is_empty();
            for i in erase {
                pixels[i] = 0;
            }
        }
        if !changed {
            break;
        }
    }
    (0..h)
        .flat_map(|y| {
            let p = &pixels;
            (0..w).map(move |x| p[(y + 1) * stride + x + 1])
        })
        .collect()
}

fn graph(mask: &[u8], w: usize, h: usize) -> Vec<Vec<usize>> {
    let mut edges = vec![Vec::new(); mask.len()];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if mask[i] == 0 {
                continue;
            }
            for (dx, dy) in [(1isize, 0isize), (1, 1), (0, 1), (-1, 1)] {
                let (xx, yy) = (x as isize + dx, y as isize + dy);
                if xx < 0 || xx >= w as isize || yy >= h as isize {
                    continue;
                }
                let j = yy as usize * w + xx as usize;
                if mask[j] == 0 {
                    continue;
                }
                // A diagonal is only needed if neither orthogonal path exists.
                if dx != 0
                    && dy != 0
                    && (mask[y * w + xx as usize] != 0 || mask[yy as usize * w + x] != 0)
                {
                    continue;
                }
                edges[i].push(j);
                edges[j].push(i);
            }
        }
    }
    for e in &mut edges {
        e.sort_unstable();
    }
    edges
}

fn paths(edges: &[Vec<usize>], width: usize) -> Vec<Vec<usize>> {
    use std::collections::BTreeMap;
    let point = |i: usize| [(i % width) as f64, (i / width) as f64];
    let mut pairings = vec![BTreeMap::new(); edges.len()];
    for (v, neighbors) in edges.iter().enumerate() {
        let direction = |next: usize| {
            let (mut previous, mut at) = (v, next);
            for _ in 0..3 {
                if edges[at].len() != 2 {
                    break;
                }
                let next = *edges[at].iter().find(|n| **n != previous).unwrap();
                previous = at;
                at = next;
            }
            unit(sub(point(at), point(v)))
        };
        let mut choices = Vec::new();
        for (i, &a) in neighbors.iter().enumerate() {
            for &b in &neighbors[i + 1..] {
                let score = curves::dot(direction(a), direction(b));
                if neighbors.len() == 2 || score < -0.4 {
                    choices.push((score, a, b));
                }
            }
        }
        choices.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        for (_, a, b) in choices {
            if !pairings[v].contains_key(&a) && !pairings[v].contains_key(&b) {
                pairings[v].insert(a, b);
                pairings[v].insert(b, a);
            }
        }
    }
    let halfedges: Vec<_> = edges
        .iter()
        .enumerate()
        .flat_map(|(a, n)| n.iter().map(move |b| (a, *b)))
        .collect();
    let starts: Vec<_> = halfedges
        .iter()
        .copied()
        .filter(|(a, b)| !pairings[*a].contains_key(b))
        .chain(halfedges.iter().copied())
        .collect();
    let edge = |a: usize, b: usize| (a.min(b), a.max(b));
    let mut used = BTreeSet::new();
    let mut paths = Vec::new();
    for (start, next) in starts {
        if !used.insert(edge(start, next)) {
            continue;
        }
        let mut path = vec![start];
        let (mut previous, mut current) = (start, next);
        loop {
            path.push(current);
            if current == start {
                break;
            }
            let Some(&next) = pairings[current].get(&previous) else {
                break;
            };
            if !used.insert(edge(current, next)) {
                break;
            }
            previous = current;
            current = next;
        }
        paths.push(path);
    }
    paths
}

fn profile(
    gray: &[f64],
    rgb: &[[u8; 3]],
    size: (usize, usize),
    center: Point,
    normal: Point,
    ridge_scale: f64,
    options: &Options,
) -> Option<Sample> {
    let (w, h) = size;
    let at = |t: f64| bilinear(gray, w, h, add(center, mul(normal, t)));
    // Quadratic derivative root, restricted to the local skeleton cell.
    let (a, b, c) = (at(-1.), at(0.), at(1.));
    let curvature = a - 2. * b + c;
    let offset = if curvature > 1e-3 {
        ((a - c) / (2. * curvature)).clamp(-0.65, 0.65)
    } else {
        0.
    };
    let center = add(center, mul(normal, offset));
    let at = |t: f64| bilinear(gray, w, h, add(center, mul(normal, t)));
    let valley = at(0.);
    let step = 0.25;
    let limit = options.stroke_max_width;
    let mut widths = [0.; 2];
    let mut supports = [0.; 2];
    let mut contrasts = [0.; 2];
    for (side, sign) in [-1., 1.].into_iter().enumerate() {
        let mut values = vec![valley];
        let mut peak = valley;
        let mut strongest_gradient = 0f64;
        let mut quiet = 0;
        for k in 1..=(limit / step) as usize {
            let p = add(center, mul(normal, sign * k as f64 * step));
            if p[0] < 0. || p[1] < 0. || p[0] > (w - 1) as f64 || p[1] > (h - 1) as f64 {
                break;
            }
            let value = at(sign * k as f64 * step);
            if value < valley - 2. {
                // After recovery this is a neighboring feature, not a reason
                // to discard a valid first crossing (e.g. eyelid beside iris).
                if peak - valley >= options.stroke_contrast {
                    break;
                }
                return None;
            }
            let gradient = (value - *values.last().unwrap()).max(0.);
            strongest_gradient = strongest_gradient.max(gradient);
            values.push(value);
            peak = peak.max(value);
            if peak - valley >= options.stroke_contrast && gradient < strongest_gradient * 0.15 {
                quiet += 1;
            } else {
                quiet = 0;
            }
            if quiet >= 4 {
                break;
            }
            // Stop at a falling neighboring feature after sufficient recovery.
            if peak - valley >= options.stroke_contrast
                && peak - value > options.stroke_contrast * 0.5
            {
                break;
            }
        }
        let contrast = peak - valley;
        if contrast < options.stroke_contrast {
            return None;
        }
        contrasts[side] = contrast;
        let level = valley + contrast * 0.5;
        let edge = values
            .windows(2)
            .enumerate()
            .find(|(_, v)| v[0] <= level && v[1] >= level)?;
        let (i, v) = edge;
        let crossing = (i as f64 + (level - v[0]) / (v[1] - v[0]).max(1e-9)) * step;
        if crossing < 0.2 || crossing >= limit - 0.5 {
            return None;
        }
        widths[side] = crossing;
        // FWHM is geometric width, not the entire source contamination footprint.
        let clean_level = peak - (contrast * 0.02).max(1.);
        supports[side] = values
            .windows(2)
            .enumerate()
            .find(|(_, v)| v[0] <= clean_level && v[1] >= clean_level)
            .map(|(i, v)| (i as f64 + (clean_level - v[0]) / (v[1] - v[0]).max(1e-9)) * step)
            .unwrap_or(crossing)
            .max(crossing);
    }
    if widths[0] + widths[1] > options.stroke_max_width {
        return None;
    }
    let x = center[0].floor().clamp(0., (w - 1) as f64) as usize;
    let y = center[1].floor().clamp(0., (h - 1) as f64) as usize;
    let (dx, dy) = (center[0] - x as f64, center[1] - y as f64);
    let corners = [
        (x, y, (1. - dx) * (1. - dy)),
        ((x + 1).min(w - 1), y, dx * (1. - dy)),
        (x, (y + 1).min(h - 1), (1. - dx) * dy),
        ((x + 1).min(w - 1), (y + 1).min(h - 1), dx * dy),
    ];
    let color = std::array::from_fn(|k| {
        crate::mesh_fill::srgb(
            corners
                .iter()
                .map(|(x, y, weight)| weight * crate::mesh_fill::linear(rgb[y * w + x][k] as f64))
                .sum(),
        )
        .round()
        .clamp(0., 255.) as u8
    });
    Some(Sample {
        center: [center[0] + 0.5, center[1] + 0.5],
        normal,
        left: widths[0],
        right: widths[1],
        support_left: supports[0],
        support_right: supports[1],
        color,
        confidence: (contrasts[0].min(contrasts[1]) / 64.).min(1.),
        inferred: false,
        derivative: (c - a) * 0.5,
        curvature,
        ridge_scale,
    })
}

pub fn rebuild(stroke: &mut Stroke, error: f64) {
    let points: Vec<_> = stroke.samples.iter().map(|s| s.center).collect();
    let centers = curves::smooth_points(&points, stroke.closed, 0.65);
    let n = centers.len();
    // Reject isolated cross-section spikes before interpolating width. Color
    // edges and junctions can contaminate one profile even on a correct path.
    let robust_widths: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            std::array::from_fn(|side| {
                let mut neighborhood: Vec<_> = (-2isize..=2)
                    .map(|offset| {
                        let j = if stroke.closed {
                            (i as isize + offset).rem_euclid((n - 1).max(1) as isize) as usize
                        } else {
                            (i as isize + offset).clamp(0, n as isize - 1) as usize
                        };
                        if side == 0 {
                            stroke.samples[j].left
                        } else {
                            stroke.samples[j].right
                        }
                    })
                    .collect();
                neighborhood.sort_by(f64::total_cmp);
                neighborhood[neighborhood.len() / 2]
            })
        })
        .collect();
    let mut arc = 0.;
    stroke.nodes.clear();
    for (i, &center) in centers.iter().enumerate() {
        if i > 0 {
            arc += norm(sub(center, centers[i - 1]));
        }
        if stroke
            .nodes
            .last()
            .is_some_and(|node| arc - node.arc < 1e-8)
        {
            continue;
        }
        let mut sum = [0.; 5];
        let mut weight = 0.;
        for offset in -2isize..=2 {
            let j = if stroke.closed {
                (i as isize + offset).rem_euclid((n - 1).max(1) as isize) as usize
            } else {
                (i as isize + offset).clamp(0, n as isize - 1) as usize
            };
            let sample = &stroke.samples[j];
            let distance = offset.unsigned_abs() as f64;
            let w = (-distance * distance / 2.).exp() * sample.confidence.max(0.1);
            sum[0] += w * robust_widths[j][0];
            sum[1] += w * robust_widths[j][1];
            for k in 0..3 {
                sum[k + 2] += w * crate::mesh_fill::linear(sample.color[k] as f64);
            }
            weight += w;
        }
        let sample = &stroke.samples[i];
        let left = (sum[0] / weight).max(0.1);
        let right = (sum[1] / weight).max(0.1);
        stroke.nodes.push(Node {
            arc,
            center,
            left,
            right,
            color_linear: std::array::from_fn(|j| sum[j + 2] / weight),
            confidence: sample.confidence,
        });
    }
    if stroke.nodes.len() < 2 {
        return;
    }
    if stroke.closed {
        let first = stroke.nodes[0].clone();
        let last = stroke.nodes.last_mut().unwrap();
        last.center = first.center;
        last.left = first.left;
        last.right = first.right;
        last.color_linear = first.color_linear;
    }
    let knots: Vec<_> = stroke.nodes.iter().map(|n| n.arc).collect();
    let channel =
        |values: Vec<f64>| crate::spline::Scalar::new(knots.clone(), values, stroke.closed);
    stroke.channels = Channels {
        left: channel(stroke.nodes.iter().map(|n| n.left).collect()),
        right: channel(stroke.nodes.iter().map(|n| n.right).collect()),
        color: std::array::from_fn(|j| {
            channel(stroke.nodes.iter().map(|n| n.color_linear[j]).collect())
        }),
    };
    stroke.center_curve = curves::fit_smooth(
        &stroke.nodes.iter().map(|n| n.center).collect::<Vec<_>>(),
        stroke.closed,
        error,
    );
    let line = curves::flatten(&stroke.center_curve, 0.04);
    if line.len() < 2 {
        return;
    }
    let mut lengths = vec![0.];
    for pair in line.windows(2) {
        lengths.push(lengths.last().unwrap() + norm(sub(pair[1], pair[0])));
    }
    let total = *lengths.last().unwrap();
    if total < 1e-8 {
        return;
    }
    let source_total = *knots.last().unwrap();
    let mut positions: Vec<_> = (0..=(total / 0.75).ceil() as usize)
        .map(|i| (i as f64 * 0.75).min(total))
        .collect();
    positions.extend(knots.iter().map(|v| v / source_total * total));
    positions.push(total);
    positions.sort_by(f64::total_cmp);
    positions.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let on_line = |s: f64| {
        let i = lengths
            .partition_point(|v| *v <= s)
            .saturating_sub(1)
            .min(line.len() - 2);
        let t = (s - lengths[i]) / (lengths[i + 1] - lengths[i]).max(1e-12);
        add(line[i], mul(sub(line[i + 1], line[i]), t.clamp(0., 1.)))
    };
    let mut points = Vec::new();
    let mut vertex_colors = Vec::new();
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &s in &positions {
        let center = on_line(s);
        let tangent = unit(sub(
            on_line((s + 0.4).min(total)),
            on_line((s - 0.4).max(0.)),
        ));
        let normal = [-tangent[1], tangent[0]];
        let u = s / total * source_total;
        let l = add(center, mul(normal, -stroke.channels.left.at(u)));
        let r = add(center, mul(normal, stroke.channels.right.at(u)));
        left.push(l);
        right.push(r);
        points.extend([l, r]);
        let color = std::array::from_fn(|j| stroke.channels.color[j].at(u).clamp(0., 1.));
        vertex_colors.extend([color, color]);
    }
    let mut triangles = Vec::new();
    for i in 0..positions.len() - 1 {
        triangles.extend([[i * 2, i * 2 + 1, i * 2 + 3], [i * 2, i * 2 + 3, i * 2 + 2]]);
    }
    let mut ring = left.clone();
    if stroke.closed {
        let mut reversed = right.clone();
        reversed.reverse();
        stroke.outlines = vec![
            curves::fit_smooth(&left, true, error),
            curves::fit_smooth(&reversed, true, error),
        ];
    } else {
        for (i, outward) in [
            (
                positions.len() - 1,
                unit(sub(on_line(total), on_line((total - 0.5).max(0.)))),
            ),
            (0, unit(sub(on_line(0.), on_line(0.5f64.min(total))))),
        ] {
            let (a, b) = if i == 0 {
                (right[i], left[i])
            } else {
                (left[i], right[i])
            };
            let middle = mul(add(a, b), 0.5);
            let axis = mul(sub(a, b), 0.5);
            let radius = norm(axis);
            let center_index = points.len();
            points.push(middle);
            vertex_colors.push(vertex_colors[i * 2]);
            let mut previous = if i == 0 { i * 2 + 1 } else { i * 2 };
            let mut cap = Vec::new();
            for k in 1..=12 {
                let t = k as f64 * std::f64::consts::PI / 12.;
                let p = add(
                    middle,
                    add(mul(axis, t.cos()), mul(outward, radius * t.sin())),
                );
                let index = if k == 12 {
                    if i == 0 { i * 2 } else { i * 2 + 1 }
                } else {
                    let id = points.len();
                    points.push(p);
                    vertex_colors.push(vertex_colors[i * 2]);
                    id
                };
                triangles.push([center_index, previous, index]);
                previous = index;
                cap.push(p);
            }
            if i > 0 {
                ring.extend(cap);
                ring.extend(right.iter().rev().copied());
            } else {
                ring.extend(cap);
            }
        }
        ring.push(ring[0]);
        stroke.outlines = vec![curves::fit_smooth(&ring, true, error)];
    }
    triangles.retain_mut(|t| {
        let direction = crate::geometry::orient(points[t[0]], points[t[1]], points[t[2]]);
        if direction < 0 {
            t.swap(1, 2);
        }
        direction != 0
    });
    let colors_linear = triangles
        .iter()
        .map(|t| t.map(|i| vertex_colors[i]))
        .collect();
    stroke.surface = Some(Surface {
        geometry: crate::geometry::MeshData {
            points,
            triangles,
            constraints: Vec::new(),
        },
        colors_linear,
    });
}

fn repair_gaps(
    skeleton: &mut [u8],
    gray: &[f64],
    background: &[u8],
    rgb: &[[u8; 3]],
    size: (usize, usize),
    options: &Options,
) -> Vec<bool> {
    let (w, h) = size;
    let edges = graph(skeleton, w, h);
    let endpoints: Vec<_> = edges
        .iter()
        .enumerate()
        .filter(|(_, ns)| ns.len() == 1)
        .map(|(i, _)| i)
        .collect();
    let point = |i: usize| [(i % w) as f64, (i / w) as f64];
    let direction = |i: usize| {
        let mut previous = i;
        let mut at = edges[i][0];
        for _ in 0..2 {
            if edges[at].len() != 2 {
                break;
            }
            let next = *edges[at].iter().find(|j| **j != previous).unwrap();
            previous = at;
            at = next;
        }
        unit(sub(point(i), point(at)))
    };
    let mut candidates = Vec::new();
    for (index, &a) in endpoints.iter().enumerate() {
        for &b in endpoints.iter().skip(index + 1) {
            let delta = sub(point(b), point(a));
            let distance = norm(delta);
            if distance <= 1.5 || distance > options.stroke_gap_max + 1. {
                continue;
            }
            let axis = unit(delta);
            if curves::dot(direction(a), axis) < 0.9
                || curves::dot(direction(b), mul(axis, -1.)) < 0.9
                || norm_color(rgb[a], rgb[b]) > options.stroke_color_error
            {
                continue;
            }
            let count = delta[0].abs().max(delta[1].abs()).ceil() as usize;
            let mut pixels = Vec::new();
            let mut valid = true;
            for k in 1..count {
                let p = add(point(a), mul(delta, k as f64 / count as f64));
                let (x, y) = (p[0].round() as usize, p[1].round() as usize);
                let i = y * w + x;
                if skeleton[i] != 0
                    || background[i] as f64 - gray[i] < options.stroke_contrast * 0.4
                    || norm_color(rgb[i], rgb[a]) > options.stroke_color_error
                {
                    valid = false;
                    break;
                }
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (xx, yy) = (x as isize + dx, y as isize + dy);
                        if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                            continue;
                        }
                        let j = yy as usize * w + xx as usize;
                        if skeleton[j] != 0
                            && norm(sub(point(j), point(a))) > 1.5
                            && norm(sub(point(j), point(b))) > 1.5
                        {
                            valid = false;
                        }
                    }
                }
                pixels.push(i);
            }
            if valid && !pixels.is_empty() {
                candidates.push((distance, a, b, pixels));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut used = BTreeSet::new();
    let mut repaired = vec![false; w * h];
    for (_, a, b, pixels) in candidates {
        if used.contains(&a) || used.contains(&b) || pixels.iter().any(|i| repaired[*i]) {
            continue;
        }
        used.insert(a);
        used.insert(b);
        for i in pixels {
            skeleton[i] = 1;
            repaired[i] = true;
        }
    }
    repaired
}
pub fn detect(image: &core::Mat, options: &Options) -> Result<Detection> {
    options.validate()?;
    let line_hint = options
        .line_model
        .as_deref()
        .map(|p| crate::lineart::predict(image, p))
        .transpose()?;
    detect_with_hint(image, options, line_hint)
}

fn detect_with_hint(
    image: &core::Mat,
    options: &Options,
    line_hint: Option<Vec<f64>>,
) -> Result<Detection> {
    let (w, h) = (image.cols() as usize, image.rows() as usize);
    if let Some(hint) = &line_hint {
        ensure!(hint.len() == w * h, "Invalid line-hint dimensions");
    }
    let guide_gray: Option<Vec<f64>> = line_hint
        .as_ref()
        .map(|hint| hint.iter().map(|v| (1. - v) * 255.).collect());
    let mut gray_mat = core::Mat::default();
    imgproc::cvt_color_def(image, &mut gray_mat, imgproc::COLOR_BGR2GRAY)?;
    // Morphological closing estimates the surrounding surface without the
    // dark-side response of Gaussian blur at a single color boundary.
    let radius = (options.stroke_max_width * 0.5).ceil() as i32;
    let kernel = imgproc::get_structuring_element_def(
        imgproc::MORPH_ELLIPSE,
        core::Size::new(radius * 2 + 1, radius * 2 + 1),
    )?;
    let mut background = core::Mat::default();
    imgproc::morphology_ex_def(&gray_mat, &mut background, imgproc::MORPH_CLOSE, &kernel)?;
    let source_gray = gray_mat.data_typed::<u8>()?;
    let closed_gray = background.data_typed::<u8>()?;
    let ink: Vec<f64> = source_gray
        .iter()
        .zip(closed_gray)
        .map(|(g, b)| b.saturating_sub(*g) as f64)
        .collect();
    // Profile the ink residual rather than unequal colors on the two sides.
    // The source RGB remains untouched for per-node color observations.
    let gray: Vec<f64> = ink.iter().map(|v| 255. - v).collect();
    let rgb: Vec<_> = image
        .data_typed::<core::Vec3b>()?
        .iter()
        .map(|p| [p[2], p[1], p[0]])
        .collect();
    let candidate: Vec<_> = line_hint
        .as_ref()
        .map(|hint| learned_support(hint, w, h))
        .unwrap_or_else(|| {
            ink.iter()
                .map(|v| u8::from(*v >= options.stroke_contrast))
                .collect()
        });
    let learned_radius = if line_hint.is_some() {
        let mask = core::Mat::from_slice(&candidate)?;
        let mut distances = core::Mat::default();
        imgproc::distance_transform_def(
            &mask.reshape(1, h as i32)?,
            &mut distances,
            imgproc::DIST_L2,
            imgproc::DIST_MASK_PRECISE,
        )?;
        Some(distances.data_typed::<f32>()?.to_vec())
    } else {
        None
    };
    let mut gray_float = core::Mat::default();
    let residual_mat = core::Mat::from_slice(&gray)?;
    residual_mat
        .reshape(1, h as i32)?
        .convert_to(&mut gray_float, core::CV_64F, 1., 0.)?;
    let mut scales = Vec::new();
    for sigma in [
        0.7,
        (options.stroke_max_width / 8.).max(1.2),
        (options.stroke_max_width / 4.).max(2.4),
    ] {
        let mut blurred = core::Mat::default();
        imgproc::gaussian_blur_def(&gray_float, &mut blurred, core::Size::new(0, 0), sigma)?;
        scales.push((sigma, blurred.data_typed::<f64>()?.to_vec()));
    }
    // Thin the connected ink support; ridge derivatives refine centers and
    // normals after connectivity is established, not before it.
    let mut skeleton = thin(&candidate, w, h);
    let repaired = repair_gaps(
        &mut skeleton,
        &gray,
        &vec![255; w * h],
        &rgb,
        (w, h),
        options,
    );
    let edges = graph(&skeleton, w, h);
    let paths = paths(&edges, w);
    let mut strokes = Vec::new();
    for path in paths {
        if path.len() < 3 {
            continue;
        }
        let closed = path.first() == path.last();
        let points: Vec<_> = path
            .iter()
            .map(|i| [(*i % w) as f64, (*i / w) as f64])
            .collect();
        let measured: Vec<_> = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let a = if closed {
                    points[(i + points.len() - 3) % (points.len() - 1)]
                } else {
                    points[i.saturating_sub(2)]
                };
                let b = if closed {
                    points[(i + 2) % (points.len() - 1)]
                } else {
                    points[(i + 2).min(points.len() - 1)]
                };
                let tangent = unit(sub(b, a));
                let reference = [-tangent[1], tangent[0]];
                let guide_sample = guide_gray
                    .as_ref()
                    .and_then(|guide| profile(guide, &rgb, (w, h), *p, reference, 1., options));
                // A source shadow beside a learned contour must not rotate its
                // normal or inflate its width into the neighboring region.
                let (normal, scale) = if guide_gray.is_some() {
                    (reference, 1.)
                } else {
                    ridge_normal(&scales, w, h, *p, reference)
                };
                profile(&gray, &rgb, (w, h), *p, normal, scale, options)
                    .or_else(|| {
                        guide_gray.as_ref().and_then(|guide| {
                            // A learned contour remains connected through weak source
                            // contrast. Its width is explicitly inferred, never silently
                            // presented as a measured source profile. Require source RGB
                            // evidence so a hallucinated line in flat color stays absent.
                            let (x, y) = (p[0] as usize, p[1] as usize);
                            let (mut lo, mut hi) = ([255u8; 3], [0u8; 3]);
                            for yy in y.saturating_sub(2)..=(y + 2).min(h - 1) {
                                for xx in x.saturating_sub(2)..=(x + 2).min(w - 1) {
                                    for c in 0..3 {
                                        lo[c] = lo[c].min(rgb[yy * w + xx][c]);
                                        hi[c] = hi[c].max(rgb[yy * w + xx][c]);
                                    }
                                }
                            }
                            let curved = (y.saturating_sub(1).max(1)
                                ..=(y + 1).min(h.saturating_sub(2)))
                                .any(|yy| {
                                    (x.saturating_sub(1).max(1)..=(x + 1).min(w.saturating_sub(2)))
                                        .any(|xx| {
                                            [(1isize, 0isize), (0, 1)].iter().any(|&(dx, dy)| {
                                                let a = rgb[(yy as isize - dy) as usize * w
                                                    + (xx as isize - dx) as usize];
                                                let b = rgb[(yy as isize + dy) as usize * w
                                                    + (xx as isize + dx) as usize];
                                                (0..3)
                                                    .map(|c| {
                                                        (a[c] as i32 + b[c] as i32
                                                            - 2 * rgb[yy * w + xx][c] as i32)
                                                            .abs()
                                                    })
                                                    .sum::<i32>()
                                                    >= 6
                                            })
                                        })
                                });
                            if ((0..3).map(|c| (hi[c] - lo[c]) as usize).sum::<usize>() < 12
                                || !curved)
                                && ink[path[i]] < 1.
                            {
                                return None;
                            }
                            profile(
                                guide,
                                &rgb,
                                (w, h),
                                *p,
                                [-tangent[1], tangent[0]],
                                1.,
                                options,
                            )
                            .map(|mut sample| {
                                sample.inferred = true;
                                sample.confidence *= 0.35;
                                sample
                            })
                        })
                    })
                    .map(|mut sample| {
                        if let Some(radius) = &learned_radius {
                            let cap = (radius[path[i]] as f64).max(0.5);
                            let (left, right) = guide_sample
                                .as_ref()
                                .map(|g| (g.left.min(cap), g.right.min(cap)))
                                .unwrap_or((cap, cap));
                            if sample.left > left + 0.25 || sample.right > right + 0.25 {
                                sample.inferred = true;
                                sample.confidence *= 0.5;
                            }
                            sample.left = sample.left.min(left + 0.25);
                            sample.right = sample.right.min(right + 0.25);
                            // Avoid alternating source/guide offsets along the
                            // same learned contour; source profiles still supply
                            // color and the complete removal footprint.
                            sample.center = add(*p, [0.5, 0.5]);
                            sample.color = rgb_at(
                                &rgb,
                                (w, h),
                                ink_peak(&ink, (w, h), *p, reference, (left + right).min(1.5)),
                            );
                        }
                        if repaired[path[i]] {
                            sample.inferred = true;
                            sample.confidence *= 0.5;
                        }
                        sample
                    })
            })
            .collect();
        // Only bridge an unmeasured junction/end when adjacent profile support exists.
        let valid: Vec<_> = measured
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.as_ref().map(|_| i))
            .collect();
        if valid.len() < 2 {
            continue;
        }
        let supported: Vec<Option<Sample>> = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if let Some(sample) = &measured[i] {
                    return Some(sample.clone());
                }
                let nearest = *valid.iter().min_by_key(|j| j.abs_diff(i)).unwrap();
                let distance = norm(sub(*p, points[nearest]));
                // Only uncertain pixels immediately around a supported junction can
                // inherit widths. A rejected long span is a break, never a shortcut.
                let anchors = valid.partition_point(|j| *j < i);
                let bracket = (anchors > 0 && anchors < valid.len())
                    .then(|| (valid[anchors - 1], valid[anchors]));
                let model_bridge = line_hint.is_some()
                    && ink[path[i]] >= 1.
                    && bracket.is_some_and(|(a, b)| {
                        b - a <= (options.stroke_max_width * 2.).ceil() as usize
                    });
                if !model_bridge {
                    if distance > options.stroke_max_width.min(3.) {
                        return None;
                    }
                    if norm_color(rgb[path[i]], measured[nearest].as_ref().unwrap().color)
                        > options.stroke_color_error
                    {
                        return None;
                    }
                }
                let mut sample = measured[nearest].clone().unwrap();
                sample.center = add(*p, [0.5, 0.5]);
                if model_bridge {
                    let (a, b) = bracket.unwrap();
                    let t = (i - a) as f64 / (b - a) as f64;
                    let (first, last) =
                        (measured[a].as_ref().unwrap(), measured[b].as_ref().unwrap());
                    sample.left = first.left * (1. - t) + last.left * t;
                    sample.right = first.right * (1. - t) + last.right * t;
                    sample.support_left = first.support_left * (1. - t) + last.support_left * t;
                    sample.support_right = first.support_right * (1. - t) + last.support_right * t;
                    let tangent = unit(sub(
                        points[(i + 2).min(points.len() - 1)],
                        points[i.saturating_sub(2)],
                    ));
                    sample.normal = [-tangent[1], tangent[0]];
                    sample.color =
                        rgb_at(&rgb, (w, h), ink_peak(&ink, (w, h), *p, sample.normal, 1.5));
                }
                sample.confidence *= 0.25;
                sample.inferred = true;
                Some(sample)
            })
            .collect();
        for run in supported_runs(&supported) {
            if run.len() < 3 {
                continue;
            }
            let samples: Vec<_> = run.iter().map(|i| supported[*i].clone().unwrap()).collect();
            let closed_segment = closed && run.len() == path.len();
            let centers: Vec<_> = samples.iter().map(|s| s.center).collect();
            let mut color = [0u8; 3];
            for (c, value) in color.iter_mut().enumerate() {
                *value = (samples.iter().map(|s| s.color[c] as usize).sum::<usize>()
                    / samples.len()) as u8;
            }
            let fill = format!("#{:02x}{:02x}{:02x}", color[0], color[1], color[2]);
            let mut stroke = Stroke {
                id: strokes.len(),
                start_node: path[run[0]],
                end_node: path[*run.last().unwrap()],
                closed: closed_segment,
                center_curve: curves::fit(&centers, options.curve_error),
                outlines: Vec::new(),
                samples,
                fill,
                color,
                occlusion_clip: None,
                nodes: Vec::new(),
                channels: Channels::default(),
                surface: None,
                clip_curves: None,
            };
            let length: f64 = stroke
                .samples
                .windows(2)
                .map(|p| norm(sub(p[1].center, p[0].center)))
                .sum();
            let mut widths: Vec<_> = stroke.samples.iter().map(|p| p.left + p.right).collect();
            widths.sort_by(f64::total_cmp);
            // A single junction profile must not reclassify an entire line.
            let characteristic_width = widths[((widths.len() - 1) * 9) / 10];
            if length + characteristic_width >= characteristic_width * options.stroke_min_aspect {
                rebuild(&mut stroke, options.curve_error);
                strokes.push(stroke);
            }
        }
    }
    let mut coverage = Vec::new();
    let mut owners = vec![usize::MAX; w * h];
    let mut costs = vec![f64::INFINITY; w * h];
    for stroke in &strokes {
        let mut mask = core::Mat::new_rows_cols_with_default(
            h as i32,
            w as i32,
            core::CV_8UC1,
            core::Scalar::all(0.),
        )?;
        let mut polygons = core::Vector::<core::Vector<core::Point>>::new();
        let (mut minx, mut miny, mut maxx, mut maxy) = (w as i32, h as i32, 0i32, 0i32);
        for outline in &stroke.outlines {
            let mut polygon = core::Vector::new();
            for c in &outline.cubics {
                for i in 0..8 {
                    let p = curves::at(c, i as f64 / 8.);
                    minx = minx.min(p[0].floor() as i32);
                    maxx = maxx.max(p[0].ceil() as i32);
                    miny = miny.min(p[1].floor() as i32);
                    maxy = maxy.max(p[1].ceil() as i32);
                    polygon.push(core::Point::new(
                        (p[0] * 16.).round() as i32,
                        (p[1] * 16.).round() as i32,
                    ));
                }
            }
            polygons.push(polygon);
        }
        imgproc::fill_poly(
            &mut mask,
            &polygons,
            core::Scalar::all(255.),
            imgproc::LINE_8,
            4,
            core::Point::new(0, 0),
        )?;
        let pixels = mask.data_typed::<u8>()?;
        let mut covered = Vec::new();
        for y in miny.max(0)..=maxy.min(h as i32 - 1) {
            for x in minx.max(0)..=maxx.min(w as i32 - 1) {
                let i = y as usize * w + x as usize;
                if pixels[i] == 0 {
                    continue;
                }
                let local = color_at(stroke, [x as f64 + 0.5, y as f64 + 0.5]);
                let cost = (0..3)
                    .map(|c| (rgb[i][c] as f64 - local[c]).powi(2))
                    .sum::<f64>();
                covered.push((i, cost));
                if cost < costs[i] {
                    costs[i] = cost;
                    owners[i] = stroke.id;
                }
            }
        }
        coverage.push(covered);
    }
    // A source-derived visibility partition resolves overlapping colored ribbons.
    // This supports spatially varying occlusion; a global dark-first order cannot.
    for i in 0..strokes.len() {
        let occluded = coverage[i].iter().any(|(p, cost)| {
            owners[*p] == usize::MAX || cost.sqrt() > costs[*p].sqrt() + options.stroke_color_error
        });
        if !occluded {
            continue;
        }
        let mut visible = vec![0u8; w * h];
        for (p, cost) in &coverage[i] {
            if owners[*p] != usize::MAX
                && cost.sqrt() <= costs[*p].sqrt() + options.stroke_color_error
            {
                visible[*p] = 1;
            }
        }
        let rings = crate::vectorize::mask_rings(&visible, w, h, 0.)?;
        strokes[i].clip_curves = Some(
            rings
                .iter()
                .map(|ring| {
                    let mut points: Vec<_> =
                        ring.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
                    points.push(points[0]);
                    curves::fit_smooth(&curves::smooth_points(&points, true, 0.45), true, 0.2)
                })
                .collect(),
        );
        strokes[i].occlusion_clip = Some(rings);
    }
    let mut separation: Vec<u8> = owners
        .iter()
        .map(|i| if *i == usize::MAX { 0 } else { 255 })
        .collect();
    let support = separation.clone();
    for y in 0..h {
        for x in 0..w {
            if support[y * w + x] != 0 || ink[y * w + x] < (options.stroke_contrast * 0.15).max(1.)
            {
                continue;
            }
            if (-2isize..=2).any(|dy| {
                (-2isize..=2).any(|dx| {
                    let (xx, yy) = (x as isize + dx, y as isize + dy);
                    xx >= 0
                        && yy >= 0
                        && xx < w as isize
                        && yy < h as isize
                        && support[yy as usize * w + xx as usize] != 0
                })
            }) {
                separation[y * w + x] = 255;
            }
        }
    }
    Ok(Detection {
        strokes,
        mask: separation,
        line_hint,
    })
}
fn supported_runs(samples: &[Option<Sample>]) -> Vec<Vec<usize>> {
    let mut runs = Vec::new();
    let mut run = Vec::new();
    for (i, sample) in samples.iter().enumerate() {
        if sample.is_some() {
            run.push(i);
        } else if !run.is_empty() {
            runs.push(std::mem::take(&mut run));
        }
    }
    if !run.is_empty() {
        runs.push(run);
    }
    runs
}

fn ridge_normal(
    scales: &[(f64, Vec<f64>)],
    w: usize,
    h: usize,
    p: Point,
    reference: Point,
) -> (Point, f64) {
    let mut chosen = (reference, 0.);
    let mut best = 0.;
    for (sigma, gray) in scales {
        let at = |dx, dy| bilinear(gray, w, h, add(p, [dx, dy]));
        let center = at(0., 0.);
        let xx = at(1., 0.) - 2. * center + at(-1., 0.);
        let yy = at(0., 1.) - 2. * center + at(0., -1.);
        let xy = (at(1., 1.) - at(1., -1.) - at(-1., 1.) + at(-1., -1.)) * 0.25;
        let eigen = (xx + yy + ((xx - yy).powi(2) + 4. * xy * xy).sqrt()) * 0.5;
        let mut normal = if xy.abs() > 1e-8 {
            unit([xy, eigen - xx])
        } else if xx > yy {
            [1., 0.]
        } else {
            [0., 1.]
        };
        let alignment = curves::dot(normal, reference);
        let score = eigen * sigma * sigma;
        if alignment.abs() >= 0.8 && score > best {
            if alignment < 0. {
                normal = mul(normal, -1.);
            }
            chosen = (normal, *sigma);
            best = score;
        }
    }
    chosen
}

fn norm_color(a: [u8; 3], b: [u8; 3]) -> f64 {
    (0..3)
        .map(|i| (a[i] as f64 - b[i] as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}
fn color_at(stroke: &Stroke, p: Point) -> [f64; 3] {
    let mut best = (f64::INFINITY, 0.);
    for nodes in stroke.nodes.windows(2) {
        let edge = sub(nodes[1].center, nodes[0].center);
        let v = curves::dot(edge, edge);
        let t = (curves::dot(sub(p, nodes[0].center), edge) / v.max(1e-12)).clamp(0., 1.);
        let d = norm(sub(p, add(nodes[0].center, mul(edge, t))));
        if d < best.0 {
            best = (d, nodes[0].arc + t * (nodes[1].arc - nodes[0].arc));
        }
    }
    std::array::from_fn(|j| crate::mesh_fill::srgb(stroke.channels.color[j].at(best.1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learned_contour_does_not_inherit_adjacent_shadow_width() -> Result<()> {
        let (w, h) = (100, 64);
        let mut image =
            core::Mat::new_rows_cols_with_default(h, w, core::CV_8UC3, core::Scalar::all(220.))?;
        let mut hint = vec![0.; w as usize * h as usize];
        for x in 8..92 {
            for y in 30..40 {
                let value = if y <= 32 { 20 } else { 100 };
                *image.at_2d_mut::<core::Vec3b>(y, x)? = core::Vec3b::from([value; 3]);
                if y <= 32 {
                    hint[y as usize * w as usize + x as usize] = 1.;
                }
            }
        }
        let mut detection = detect_with_hint(&image, &Default::default(), Some(hint))?;
        let stroke = detection
            .strokes
            .iter_mut()
            .max_by_key(|s| s.samples.len())
            .expect("missing contour");
        assert!(
            stroke
                .nodes
                .iter()
                .filter(|n| n.center[0] > 20. && n.center[0] < 80.)
                .all(|n| n.left + n.right < 4.),
            "adjacent shadow inflated the stroke"
        );
        // A single contaminated profile must not create a visible width blob.
        let middle = stroke.samples.len() / 2;
        stroke.samples[middle].left = 12.;
        stroke.samples[middle].right = 12.;
        rebuild(stroke, 0.35);
        assert!(stroke.nodes.iter().all(|n| n.left + n.right < 4.));
        Ok(())
    }
    #[test]
    fn learned_topology_connects_weak_source_ink_but_rejects_flat_hallucinations() -> Result<()> {
        let (w, h) = (128, 64);
        let mut image =
            core::Mat::new_rows_cols_with_default(h, w, core::CV_8UC3, core::Scalar::all(220.))?;
        imgproc::line(
            &mut image,
            core::Point::new(8, 32),
            core::Point::new(120, 32),
            core::Scalar::all(20.),
            2,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::rectangle(
            &mut image,
            core::Rect::new(48, 31, 32, 3),
            core::Scalar::all(218.),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        let mut hint = vec![0.; w as usize * h as usize];
        for y in 31..=33 {
            for x in 8..=120 {
                hint[y * w as usize + x] = if (48..80).contains(&x) { 0.1 } else { 1. };
            }
        }
        let detection = detect_with_hint(&image, &Default::default(), Some(hint.clone()))?;
        assert!(
            detection.strokes.iter().any(|s| {
                let lo = s
                    .nodes
                    .iter()
                    .map(|n| n.center[0])
                    .fold(f64::INFINITY, f64::min);
                let hi = s
                    .nodes
                    .iter()
                    .map(|n| n.center[0])
                    .fold(f64::NEG_INFINITY, f64::max);
                lo < 20.
                    && hi > 105.
                    && s.samples
                        .iter()
                        .any(|p| p.inferred && p.center[0] > 55. && p.center[0] < 72.)
            }),
            "weak section split the learned contour"
        );
        let flat =
            core::Mat::new_rows_cols_with_default(h, w, core::CV_8UC3, core::Scalar::all(220.))?;
        assert!(
            detect_with_hint(&flat, &Default::default(), Some(hint.clone()))?
                .strokes
                .is_empty()
        );
        let mut gradient = flat.clone();
        for y in 0..h as usize {
            for x in 0..w as usize {
                gradient.data_typed_mut::<core::Vec3b>()?[y * w as usize + x] =
                    core::Vec3b::from([(50 + x) as u8; 3]);
            }
        }
        assert!(
            detect_with_hint(&gradient, &Default::default(), Some(hint))?
                .strokes
                .is_empty(),
            "linear shading became an inferred stroke"
        );
        Ok(())
    }
    #[test]
    fn variable_width_and_color_nodes_drive_the_rendered_ribbon() -> Result<()> {
        let (w, h) = (128, 72);
        let mut image =
            core::Mat::new_rows_cols_with_default(h, w, core::CV_8UC3, core::Scalar::all(240.))?;
        for x in 12..116 {
            let center = 36. + (x as f64 / 18.).sin() * 7.;
            let width = 2. + (x - 12) as f64 / 30.;
            for y in 0..h {
                if (y as f64 + 0.5 - center).abs() < width {
                    *image.at_2d_mut::<core::Vec3b>(y, x)? =
                        core::Vec3b::from([10, (15 + x / 3) as u8, (20 + x / 2) as u8]);
                }
            }
        }
        let detection = detect(&image, &Options::default())?;
        let longest = detection
            .strokes
            .iter()
            .max_by_key(|s| s.nodes.len())
            .unwrap();
        assert!(longest.nodes.len() > 40);
        let widths: Vec<_> = longest.nodes.iter().map(|n| n.left + n.right).collect();
        assert!(
            widths.iter().copied().fold(0., f64::max)
                - widths.iter().copied().fold(f64::INFINITY, f64::min)
                > 2.
        );
        assert!(
            longest.channels.color[0]
                .values
                .iter()
                .copied()
                .fold(0., f64::max)
                - longest.channels.color[0]
                    .values
                    .iter()
                    .copied()
                    .fold(1., f64::min)
                > 0.02
        );
        let frame = crate::vectorize::Frame {
            version: 3,
            model: "continuous-stroke-test".into(),
            fill_recovery: None,
            width: w as usize,
            height: h as usize,
            background: "#f0f0f0".into(),
            settings: Default::default(),
            layers: Vec::new(),
            strokes: detection.strokes,
            mesh: None,
            regions: Vec::new(),
            boundaries: None,
        };
        let text = crate::vectorize::svg(&frame);
        assert!(text.contains("mesh-g-stroke-"));
        let render = crate::vectorize::rasterize(&text)?;
        for x in [35usize, 90] {
            let y = (36. + (x as f64 / 18.).sin() * 7.).floor() as usize;
            let pixel = &render.data()[(y * w as usize + x) * 4..][..3];
            assert!(
                (pixel[0] as i32 - (20 + x as i32 / 2)).abs() < 15,
                "x={x} rgb={pixel:?}"
            );
        }
        Ok(())
    }
    #[test]
    fn repairs_only_aligned_gaps_with_source_evidence() {
        let (w, h) = (24, 12);
        let mut skeleton = vec![0; w * h];
        for x in 3..21 {
            if x != 11 {
                skeleton[6 * w + x] = 1;
            }
        }
        let mut gray = vec![20.; w * h];
        let background = vec![60; w * h];
        let rgb = vec![[20; 3]; w * h];
        let repaired = repair_gaps(
            &mut skeleton,
            &gray,
            &background,
            &rgb,
            (w, h),
            &Options::default(),
        );
        assert!(repaired[6 * w + 11]);
        skeleton[6 * w + 11] = 0;
        gray[6 * w + 11] = 60.;
        assert!(
            !repair_gaps(
                &mut skeleton,
                &gray,
                &background,
                &rgb,
                (w, h),
                &Options::default()
            )[6 * w + 11]
        );
    }
    #[test]
    fn graph_retains_ring_and_junction_without_diagonal_shortcuts() {
        let m = [0, 1, 0, 1, 1, 1, 0, 1, 0];
        let e = graph(&m, 3, 3);
        assert_eq!(e[4].len(), 4);
        assert_eq!(paths(&e, 3).len(), 2);
        let m = [0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1];
        let e = graph(&thin(&m, 4, 4), 4, 4);
        assert!(!paths(&e, 4).is_empty());
    }
    #[test]
    fn rejects_large_dark_fills_and_measures_finite_thin_lines() -> Result<()> {
        let mut image =
            core::Mat::new_rows_cols_with_default(64, 96, core::CV_8UC3, core::Scalar::all(240.))?;
        for y in 20..25 {
            for x in 10..86 {
                *image.at_2d_mut::<core::Vec3b>(y, x)? = core::Vec3b::from([20, 20, 20]);
            }
        }
        let detection = detect(&image, &Options::default())?;
        assert!(!detection.strokes.is_empty());
        let middle: Vec<_> = detection
            .strokes
            .iter()
            .flat_map(|s| &s.samples)
            .filter(|p| p.center[0] > 30. && p.center[0] < 70. && !p.inferred)
            .collect();
        assert!(!middle.is_empty());
        for p in middle {
            assert!((p.left + p.right - 5.).abs() < 0.6);
        }
        let black =
            core::Mat::new_rows_cols_with_default(64, 96, core::CV_8UC3, core::Scalar::all(0.))?;
        assert!(detect(&black, &Options::default())?.strokes.is_empty());
        Ok(())
    }
    #[test]
    fn antialiased_diagonal_survives_visibility_and_has_continuous_width() -> Result<()> {
        let (w, h) = (96, 64);
        let mut image =
            core::Mat::new_rows_cols_with_default(h, w, core::CV_8UC3, core::Scalar::all(240.))?;
        imgproc::line(
            &mut image,
            core::Point::new(8, 12),
            core::Point::new(86, 50),
            core::Scalar::all(30.),
            2,
            imgproc::LINE_AA,
            0,
        )?;
        let detected = detect(&image, &Options::default())?;
        let supported = (20..75)
            .filter(|x| {
                let y = 12. + (*x as f64 - 8.) * 38. / 78.;
                detected
                    .strokes
                    .iter()
                    .flat_map(|s| &s.nodes)
                    .any(|n| norm(sub(n.center, [*x as f64 + 0.5, y + 0.5])) < 2.)
            })
            .count();
        assert!(
            supported > 48,
            "only {supported} supported diagonal positions"
        );
        assert!(
            detected
                .strokes
                .iter()
                .any(|s| s.nodes.len() > 30 && s.clip_curves.is_none())
        );
        Ok(())
    }
    #[test]
    fn recovered_profile_stops_before_a_darker_neighbor() {
        let (w, h) = (24, 12);
        let mut gray = vec![240.; w * h];
        for y in 0..h {
            gray[y * w + 10] = 30.;
            gray[y * w + 12] = 0.;
        }
        let rgb: Vec<_> = gray.iter().map(|v| [*v as u8; 3]).collect();
        let sample = profile(
            &gray,
            &rgb,
            (w, h),
            [10., 6.],
            [1., 0.],
            0.7,
            &Options::default(),
        )
        .unwrap();
        assert!((sample.left - 0.5).abs() < 0.1 && (sample.right - 0.5).abs() < 0.1);
    }
    #[test]
    fn short_thin_marks_are_strokes_while_compact_dark_disks_remain_fills() -> Result<()> {
        let mut image =
            core::Mat::new_rows_cols_with_default(64, 64, core::CV_8UC3, core::Scalar::all(220.))?;
        imgproc::line(
            &mut image,
            core::Point::new(8, 12),
            core::Point::new(13, 12),
            core::Scalar::all(20.),
            1,
            imgproc::LINE_AA,
            0,
        )?;
        imgproc::circle(
            &mut image,
            core::Point::new(44, 44),
            5,
            core::Scalar::all(20.),
            -1,
            imgproc::LINE_AA,
            0,
        )?;
        let d = detect(&image, &Options::default())?;
        assert!(d.strokes.iter().any(|s| {
            s.nodes
                .iter()
                .any(|n| norm(sub(n.center, [10.5, 12.5])) < 2.)
        }));
        assert_ne!(d.mask[12 * 64 + 10], 0);
        assert_eq!(d.mask[44 * 64 + 44], 0, "compact disk must remain a fill");
        Ok(())
    }
    #[test]
    fn one_sided_color_boundary_is_not_a_stroke() -> Result<()> {
        let mut image =
            core::Mat::new_rows_cols_with_default(64, 96, core::CV_8UC3, core::Scalar::all(240.))?;
        for y in 0..64 {
            for x in 0..48 {
                *image.at_2d_mut::<core::Vec3b>(y, x)? = core::Vec3b::from([20, 20, 20]);
            }
        }
        assert!(detect(&image, &Options::default())?.strokes.is_empty());
        Ok(())
    }
}
