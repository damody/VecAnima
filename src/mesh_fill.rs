//! Shared pixel-cell boundaries and a constrained, linear-light color mesh.
use crate::geometry::{Mesh, MeshData, Point};
use anyhow::{Context, Result, ensure};
use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
pub enum Model {
    Flat,
    Mesh,
}
#[derive(Clone, Debug, Serialize, Deserialize, Args)]
pub struct Options {
    #[arg(long, value_enum, default_value_t=Model::Flat)]
    pub fill_model: Model,
    #[arg(long, default_value_t = 8.)]
    pub mesh_error: f64,
    #[arg(long, default_value_t = 12000)]
    pub mesh_vertices: usize,
    #[arg(long, default_value_t = 16)]
    pub mesh_spacing: usize,
    #[arg(long, default_value_t = 2.)]
    pub svg_color_error: f64,
    #[arg(long, default_value_t = 18.)]
    pub mesh_boundary_contrast: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            fill_model: Model::Flat,
            mesh_error: 8.,
            mesh_vertices: 12000,
            mesh_spacing: 16,
            svg_color_error: 2.,
            mesh_boundary_contrast: 18.,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.mesh_error.is_finite() && self.mesh_error > 0.,
            "mesh-error must be positive and finite"
        );
        ensure!(
            self.svg_color_error.is_finite() && self.svg_color_error >= 1.,
            "svg-color-error must be at least 1 for 8-bit SVG colors"
        );
        ensure!(
            self.mesh_boundary_contrast.is_finite() && self.mesh_boundary_contrast >= 0.,
            "Invalid mesh boundary contrast"
        );
        ensure!(
            self.mesh_vertices >= 4 && self.mesh_spacing > 0,
            "Invalid mesh budget or spacing"
        );
        Ok(())
    }
}
#[derive(Serialize, Deserialize)]
pub struct Fill {
    pub geometry: MeshData,
    /// Per-corner colors permit a discontinuity across a shared constraint.
    pub colors_linear: Vec<[[f64; 3]; 3]>,
    pub region_labels: Vec<usize>,
    pub boundaries: Vec<Boundary>,
    pub max_channel_error: f64,
    pub unmet_pixels: usize,
    pub vertex_budget_reached: bool,
    pub svg_error_bound: f64,
    pub edge_band_pixels: usize,
    pub interior_max_channel_error: f64,
    #[serde(default)]
    pub suppressed_palette_boundaries: usize,
}
#[derive(Serialize, Deserialize)]
pub struct Boundary {
    pub vertices: Vec<usize>,
    pub regions: [usize; 2],
}
fn key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}
pub fn linear(v: f64) -> f64 {
    let v = v / 255.;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
pub fn srgb(v: f64) -> f64 {
    255. * if v <= 0.0031308 {
        12.92 * v.max(0.)
    } else {
        1.055 * v.powf(1. / 2.4) - 0.055
    }
}

/// Each interface is visited once. Only exactly collinear degree-two vertices
/// are removed; junctions and changes in the incident region pair survive.
fn boundaries(
    labels: &[usize],
    w: usize,
    h: usize,
    palette: &[[f32; 3]],
    contrast: f64,
) -> (Vec<Point>, Vec<Boundary>) {
    let stride = w + 1;
    let mut edges = BTreeMap::new();
    let outside = usize::MAX;
    let mut add = |a: usize, b: usize, l: usize, r: usize| {
        if l != r
            && (l == outside
                || r == outside
                || palette.is_empty()
                || (0..3)
                    .map(|j| (palette[l][j] - palette[r][j]).powi(2))
                    .sum::<f32>()
                    .sqrt()
                    >= contrast as f32)
        {
            edges.insert(key(a, b), key(l, r));
        }
    };
    for y in 0..=h {
        for x in 0..w {
            add(
                y * stride + x,
                y * stride + x + 1,
                if y == 0 {
                    outside
                } else {
                    labels[(y - 1) * w + x]
                },
                if y == h { outside } else { labels[y * w + x] },
            );
        }
    }
    for y in 0..h {
        for x in 0..=w {
            add(
                y * stride + x,
                (y + 1) * stride + x,
                if x == 0 {
                    outside
                } else {
                    labels[y * w + x - 1]
                },
                if x == w { outside } else { labels[y * w + x] },
            );
        }
    }
    let mut neighbors: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(a, b) in edges.keys() {
        neighbors.entry(a).or_default().push(b);
        neighbors.entry(b).or_default().push(a);
    }
    let keep = |v: usize| {
        let ns = &neighbors[&v];
        ns.len() != 2
            || edges[&key(v, ns[0])] != edges[&key(v, ns[1])]
            || (ns[0] % stride != ns[1] % stride && ns[0] / stride != ns[1] / stride)
    };
    let mut points = Vec::new();
    let mut ids = BTreeMap::new();
    for &v in neighbors.keys().filter(|v| keep(**v)) {
        ids.insert(v, points.len());
        points.push([(v % stride) as f64, (v / stride) as f64]);
    }
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for (&a, &id) in &ids {
        for &b in &neighbors[&a] {
            if !seen.insert(key(a, b)) {
                continue;
            }
            let pair = edges[&key(a, b)];
            let (mut previous, mut end) = (a, b);
            while !keep(end) {
                let next = neighbors[&end]
                    .iter()
                    .copied()
                    .find(|v| *v != previous)
                    .unwrap();
                seen.insert(key(end, next));
                previous = end;
                end = next;
            }
            result.push(Boundary {
                vertices: vec![id, ids[&end]],
                regions: [pair.0, pair.1],
            });
        }
    }
    (points, result)
}

struct Source<'a> {
    rgb: &'a [[u8; 3]],
    labels: &'a [usize],
    w: usize,
    h: usize,
}

/// Palette clustering can place different labels on a continuous color ramp.
/// A hard mesh interface needs a localized source jump, rather than a large
/// distance between remote palette centers.
fn source_discontinuity(
    rgb: &[[u8; 3]],
    w: usize,
    h: usize,
    line: &[Point],
    threshold: f64,
) -> bool {
    let at = |p: Point| -> [f64; 3] {
        let x = (p[0] - 0.5).clamp(0., (w - 1) as f64);
        let y = (p[1] - 0.5).clamp(0., (h - 1) as f64);
        let (ix, iy) = (x.floor() as usize, y.floor() as usize);
        let (dx, dy) = (x - ix as f64, y - iy as f64);
        std::array::from_fn(|c| {
            rgb[iy * w + ix][c] as f64 * (1. - dx) * (1. - dy)
                + rgb[iy * w + (ix + 1).min(w - 1)][c] as f64 * dx * (1. - dy)
                + rgb[(iy + 1).min(h - 1) * w + ix][c] as f64 * (1. - dx) * dy
                + rgb[(iy + 1).min(h - 1) * w + (ix + 1).min(w - 1)][c] as f64 * dx * dy
        })
    };
    let distance =
        |a: [f64; 3], b: [f64; 3]| (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt();
    let (mut total, mut supported) = (0., 0.);
    for edge in line.windows(2) {
        let delta = crate::curves::sub(edge[1], edge[0]);
        let length = crate::curves::norm(delta);
        if length < 1e-8 {
            continue;
        }
        let normal = [-delta[1] / length, delta[0] / length];
        let center = crate::curves::mul(crate::curves::add(edge[0], edge[1]), 0.5);
        let sample = |offset| {
            at(crate::curves::add(
                center,
                crate::curves::mul(normal, offset),
            ))
        };
        let (a, b, c, d) = (sample(-2.25), sample(-0.75), sample(0.75), sample(2.25));
        let inner = distance(b, c);
        let outer = distance(a, b).max(distance(c, d));
        total += length;
        if inner >= threshold && inner > outer * 1.5 + 1. {
            supported += length;
        }
    }
    total > 0. && supported >= total * 0.25
}
impl Source<'_> {
    fn index(&self, p: Point) -> usize {
        p[1].floor().clamp(0., (self.h - 1) as f64) as usize * self.w
            + p[0].floor().clamp(0., (self.w - 1) as f64) as usize
    }
    fn corner(&self, p: Point, sector: &[[Point; 3]], fallback: usize) -> [f64; 3] {
        let mut best = None;
        let (x, y) = (p[0].floor() as isize, p[1].floor() as isize);
        for dy in -2..=2 {
            for dx in -2..=2 {
                let (xx, yy) = (x + dx, y + dy);
                if xx < 0 || yy < 0 || xx >= self.w as isize || yy >= self.h as isize {
                    continue;
                }
                let i = yy as usize * self.w + xx as usize;
                if !sector.iter().any(|t| {
                    weights(*t, [xx as f64 + 0.5, yy as f64 + 0.5])
                        .iter()
                        .all(|v| *v >= -1e-12)
                }) {
                    continue;
                }
                let d = (xx as f64 + 0.5 - p[0]).powi(2) + (yy as f64 + 0.5 - p[1]).powi(2);
                if best.is_none_or(|(old, _)| d < old) {
                    best = Some((d, i));
                }
            }
        }
        self.rgb[best.map_or(fallback, |(_, i)| i)].map(|v| linear(v as f64))
    }
    fn colors(&self, data: &MeshData) -> (Vec<[[f64; 3]; 3]>, Vec<usize>) {
        // Union face corners only through unconstrained incident edges. Every
        // vertex has one color per topological sector, shared by all its faces.
        fn root(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        let mut parent: Vec<_> = (0..data.triangles.len() * 3).collect();
        let protected: BTreeSet<_> = data.constraints.iter().map(|e| key(e[0], e[1])).collect();
        let mut owners = BTreeMap::new();
        for (f, t) in data.triangles.iter().enumerate() {
            for k in 0..3 {
                let edge = key(t[k], t[(k + 1) % 3]);
                if protected.contains(&edge) {
                    continue;
                }
                if let Some((g, old)) = owners.insert(edge, (f, *t)) {
                    for v in [t[k], t[(k + 1) % 3]] {
                        let a = root(&mut parent, f * 3 + t.iter().position(|i| *i == v).unwrap());
                        let b = root(
                            &mut parent,
                            g * 3 + old.iter().position(|i| *i == v).unwrap(),
                        );
                        parent[a.max(b)] = a.min(b);
                    }
                }
            }
        }
        let mut sectors: BTreeMap<usize, Vec<[Point; 3]>> = BTreeMap::new();
        for (f, t) in data.triangles.iter().enumerate() {
            for k in 0..3 {
                let id = root(&mut parent, f * 3 + k);
                sectors
                    .entry(id)
                    .or_default()
                    .push(t.map(|i| data.points[i]));
            }
        }
        let mut sector_colors = BTreeMap::new();
        for (&id, sector) in &sectors {
            let vertex = data.triangles[id / 3][id % 3];
            let p = sector[0];
            let center = [
                (p[0][0] + p[1][0] + p[2][0]) / 3.,
                (p[0][1] + p[1][1] + p[2][1]) / 3.,
            ];
            sector_colors.insert(
                id,
                self.corner(data.points[vertex], sector, self.index(center)),
            );
        }
        let mut colors = Vec::new();
        let mut regions = Vec::new();
        for (f, t) in data.triangles.iter().enumerate() {
            let p = t.map(|i| data.points[i]);
            let center = [
                (p[0][0] + p[1][0] + p[2][0]) / 3.,
                (p[0][1] + p[1][1] + p[2][1]) / 3.,
            ];
            let i = self.index(center);
            let region = self.labels[i];
            regions.push(region);
            colors.push(std::array::from_fn(|k| {
                sector_colors[&root(&mut parent, f * 3 + k)]
            }));
        }
        (colors, regions)
    }
}

fn weights(p: [Point; 3], q: Point) -> [f64; 3] {
    let cross = |a: Point, b: Point, c: Point| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let area = cross(p[0], p[1], p[2]);
    [
        cross(q, p[1], p[2]) / area,
        cross(p[0], q, p[2]) / area,
        cross(p[0], p[1], q) / area,
    ]
}
fn visit(data: &MeshData, w: usize, h: usize, mut f: impl FnMut(usize, usize, [f64; 3])) {
    for (face, t) in data.triangles.iter().enumerate() {
        let p = t.map(|i| data.points[i]);
        let lo = std::array::from_fn::<_, 2, _>(|j| {
            p.iter()
                .map(|v| v[j])
                .fold(f64::INFINITY, f64::min)
                .floor()
                .max(0.) as usize
        });
        let hi = std::array::from_fn::<_, 2, _>(|j| {
            p.iter().map(|v| v[j]).fold(0., f64::max).ceil().max(0.) as usize
        });
        for y in lo[1]..hi[1].min(h) {
            for x in lo[0]..hi[0].min(w) {
                let b = weights(p, [x as f64 + 0.5, y as f64 + 0.5]);
                if b.iter().all(|v| *v >= -1e-12) {
                    f(face, y * w + x, b);
                }
            }
        }
    }
}
fn interpolate(c: [[f64; 3]; 3], b: [f64; 3]) -> [u8; 3] {
    std::array::from_fn(|j| {
        srgb((0..3).map(|i| c[i][j] * b[i]).sum())
            .round()
            .clamp(0., 255.) as u8
    })
}

pub fn build(
    rgb: &[[u8; 3]],
    labels: &[usize],
    w: usize,
    h: usize,
    options: &Options,
    palette: &[[f32; 3]],
    graph: Option<&crate::shared::Graph>,
) -> Result<Fill> {
    options.validate()?;
    ensure!(
        w > 0 && h > 0 && rgb.len() == w * h && labels.len() == w * h,
        "Invalid mesh image"
    );
    ensure!(
        palette.is_empty() || labels.iter().all(|l| *l < palette.len()),
        "Invalid mesh palette label"
    );
    let mut suppressed_palette_boundaries = 0;
    let (mut points, boundaries) = if let Some(graph) = graph {
        let mut points = Vec::new();
        let mut ids = BTreeMap::new();
        let mut result = Vec::new();
        for curve in &graph.curves {
            let [a, b] = curve.regions;
            if a != crate::shared::OUTSIDE
                && b != crate::shared::OUTSIDE
                && !palette.is_empty()
                && (0..3)
                    .map(|j| (palette[a][j] - palette[b][j]).powi(2))
                    .sum::<f32>()
                    .sqrt()
                    < options.mesh_boundary_contrast as f32
            {
                continue;
            }
            if a != crate::shared::OUTSIDE
                && b != crate::shared::OUTSIDE
                && !palette.is_empty()
                && !source_discontinuity(rgb, w, h, &curve.polyline, options.mesh_boundary_contrast)
            {
                suppressed_palette_boundaries += 1;
                continue;
            }
            let mut vertices = Vec::new();
            for p in &curve.polyline {
                let key = p.map(|v| if v == 0. { 0 } else { v.to_bits() });
                let id = *ids.entry(key).or_insert_with(|| {
                    let id = points.len();
                    points.push(*p);
                    id
                });
                if vertices.last() != Some(&id) {
                    vertices.push(id);
                }
            }
            result.push(Boundary {
                vertices,
                regions: curve.regions,
            });
        }
        (points, result)
    } else {
        boundaries(labels, w, h, palette, options.mesh_boundary_contrast)
    };
    ensure!(
        points.len() <= options.mesh_vertices,
        "Mandatory boundary vertices ({}) exceed mesh budget ({}); increase --mesh-vertices",
        points.len(),
        options.mesh_vertices
    );
    let mut unique: BTreeSet<_> = points
        .iter()
        .map(|p| p.map(|v| if v == 0. { 0 } else { v.to_bits() }))
        .collect();
    for y in (options.mesh_spacing..h).step_by(options.mesh_spacing) {
        for x in (options.mesh_spacing..w).step_by(options.mesh_spacing) {
            if points.len() < options.mesh_vertices
                && unique.insert([(x as f64).to_bits(), (y as f64).to_bits()])
            {
                points.push([x as f64, y as f64]);
            }
        }
    }
    let mut mesh = Mesh::new(points).context("Initial fill Delaunay mesh")?;
    mesh.constrain_all(
        &boundaries
            .iter()
            .flat_map(|b| b.vertices.windows(2).map(|v| [v[0], v[1]]))
            .collect::<Vec<_>>(),
    )
    .context("Shared fill constraints")?;
    let source = Source { rgb, labels, w, h };
    let mut edge_band = vec![false; w * h];
    if graph.is_some() {
        for boundary in &boundaries {
            if boundary.regions.contains(&usize::MAX) {
                continue;
            }
            for pair in boundary.vertices.windows(2) {
                let (a, b) = (mesh.points[pair[0]], mesh.points[pair[1]]);
                let delta = [b[0] - a[0], b[1] - a[1]];
                let length = delta[0] * delta[0] + delta[1] * delta[1];
                let lo = [
                    (a[0].min(b[0]) - 1.).floor().max(0.) as usize,
                    (a[1].min(b[1]) - 1.).floor().max(0.) as usize,
                ];
                let hi = [
                    (a[0].max(b[0]) + 1.).ceil().min(w as f64) as usize,
                    (a[1].max(b[1]) + 1.).ceil().min(h as f64) as usize,
                ];
                for y in lo[1]..hi[1] {
                    for x in lo[0]..hi[0] {
                        let p = [x as f64 + 0.5, y as f64 + 0.5];
                        let t = (((p[0] - a[0]) * delta[0] + (p[1] - a[1]) * delta[1])
                            / length.max(1e-20))
                        .clamp(0., 1.);
                        if (p[0] - a[0] - t * delta[0]).hypot(p[1] - a[1] - t * delta[1]) < 0.85 {
                            let normal = [-delta[1], delta[0]];
                            let norm = length.sqrt().max(1e-10);
                            let sample = |sign: f64| {
                                let xx = (p[0] + sign * normal[0] / norm)
                                    .floor()
                                    .clamp(0., (w - 1) as f64)
                                    as usize;
                                let yy = (p[1] + sign * normal[1] / norm)
                                    .floor()
                                    .clamp(0., (h - 1) as f64)
                                    as usize;
                                rgb[yy * w + xx]
                            };
                            let (a, b) = (sample(-1.25), sample(1.25));
                            let (inner_a, inner_b) = (sample(-0.25), sample(0.25));
                            if (0..3).any(|j| {
                                a[j].abs_diff(b[j]) as f64 > options.mesh_error * 2.
                                    && inner_a[j].abs_diff(inner_b[j]) as f64
                                        > 0.75 * a[j].abs_diff(b[j]) as f64
                            }) {
                                edge_band[y * w + x] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    loop {
        let data = mesh.data();
        let (colors, regions) = source.colors(&data);
        let mut worst = vec![(0., 0usize); data.triangles.len()];
        let mut maximum = 0f64;
        let mut interior_maximum = 0f64;
        let mut unmet = 0usize;
        visit(&data, w, h, |face, i, b| {
            let value = interpolate(colors[face], b);
            let error = (0..3)
                .map(|j| (value[j] as f64 - rgb[i][j] as f64).abs())
                .fold(0., f64::max);
            maximum = maximum.max(error);
            if edge_band[i] {
                return;
            }
            interior_maximum = interior_maximum.max(error);
            if error > options.mesh_error {
                unmet += 1;
            }
            if error > worst[face].0 {
                worst[face] = (error, i);
            }
        });
        if unmet == 0 || mesh.points.len() >= options.mesh_vertices {
            mesh.validate()?;
            return Ok(Fill {
                geometry: data,
                colors_linear: colors,
                region_labels: regions,
                boundaries,
                max_channel_error: maximum,
                unmet_pixels: unmet,
                vertex_budget_reached: unmet > 0,
                svg_error_bound: options.svg_color_error,
                edge_band_pixels: edge_band.iter().filter(|v| **v).count(),
                interior_max_channel_error: interior_maximum,
                suppressed_palette_boundaries,
            });
        }
        worst.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let before = mesh.points.len();
        for (error, i) in worst
            .into_iter()
            .filter(|p| p.0 > options.mesh_error)
            .take((options.mesh_vertices - before).min(256))
        {
            let _ = error;
            mesh.insert([(i % w) as f64 + 0.5, (i / w) as f64 + 0.5])
                .context("Error-directed fill refinement")?;
        }
        ensure!(
            mesh.points.len() > before,
            "Mesh refinement cannot progress; error={}",
            maximum
        );
    }
}
pub fn raster(fill: &Fill, w: usize, h: usize) -> Result<resvg::tiny_skia::Pixmap> {
    // Integrate four subpixel samples in linear light, including shared edges.
    let scale = 2usize;
    let data = MeshData {
        points: fill
            .geometry
            .points
            .iter()
            .map(|p| [p[0] * scale as f64, p[1] * scale as f64])
            .collect(),
        triangles: fill.geometry.triangles.clone(),
        constraints: Vec::new(),
    };
    let mut sum = vec![[0.; 3]; w * h];
    let mut covered = vec![false; w * h * scale * scale];
    visit(&data, w * scale, h * scale, |face, i, b| {
        if covered[i] {
            return;
        }
        covered[i] = true;
        let pixel = (i / (w * scale) / scale) * w + (i % (w * scale) / scale);
        for (j, value) in sum[pixel].iter_mut().enumerate() {
            *value += (0..3)
                .map(|k| fill.colors_linear[face][k][j] * b[k])
                .sum::<f64>();
        }
    });
    ensure!(
        covered.iter().all(|v| *v),
        "Mesh leaves uncovered subpixels"
    );
    let mut image =
        resvg::tiny_skia::Pixmap::new(w as u32, h as u32).context("Cannot allocate mesh raster")?;
    for (i, c) in sum.iter().enumerate() {
        let rgb = c.map(|v| srgb(v / 4.).round().clamp(0., 255.) as u8);
        image.data_mut()[i * 4..i * 4 + 4].copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
    Ok(image)
}

/// Little-endian float32 GPU vertices: clip-space x/y and linear RGB.
/// The JSON geometry remains the authoritative editable representation.
pub fn gpu_buffer(fill: &Fill, w: usize, h: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(fill.geometry.triangles.len() * 3 * 20);
    for (t, colors) in fill.geometry.triangles.iter().zip(&fill.colors_linear) {
        for (vertex, color) in t.iter().zip(colors) {
            let p = fill.geometry.points[*vertex];
            for value in [
                p[0] / w as f64 * 2. - 1.,
                1. - p[1] / h as f64 * 2.,
                color[0],
                color[1],
                color[2],
            ] {
                bytes.extend_from_slice(&(value as f32).to_le_bytes());
            }
        }
    }
    bytes
}

/// Piecewise constant vector export, with a conservative per-channel error
/// bound relative to the native linear-light mesh. No embedded raster image.
pub fn svg(fill: &Fill, text: &mut String) {
    use std::fmt::Write;
    // Paint connected mesh sectors independently, with one antialiased outer
    // clip. Internal paint tiles stay opaque and cannot produce alpha cracks.
    let data = &fill.geometry;
    let protected: BTreeSet<_> = data.constraints.iter().map(|e| key(e[0], e[1])).collect();
    let mut owners = BTreeMap::new();
    let mut adjacent = vec![Vec::new(); data.triangles.len()];
    for (f, t) in data.triangles.iter().enumerate() {
        for k in 0..3 {
            let e = key(t[k], t[(k + 1) % 3]);
            if let Some(g) = owners.insert(e, f)
                && !protected.contains(&e)
            {
                adjacent[f].push(g);
                adjacent[g].push(f);
            }
        }
    }
    let mut seen = vec![false; data.triangles.len()];
    for start in 0..seen.len() {
        if seen[start] {
            continue;
        }
        let mut faces = vec![start];
        seen[start] = true;
        let mut cursor = 0;
        while cursor < faces.len() {
            for g in &adjacent[faces[cursor]] {
                if !seen[*g] {
                    seen[*g] = true;
                    faces.push(*g);
                }
            }
            cursor += 1;
        }
        let mut border = BTreeMap::new();
        for f in &faces {
            let t = data.triangles[*f];
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                if border.remove(&(b, a)).is_none() {
                    border.insert((a, b), (*f, k));
                }
            }
        }
        write!(text, "<defs><clipPath id=\"fill-sector-{start}\"><path shape-rendering=\"geometricPrecision\" clip-rule=\"evenodd\" d=\"").unwrap();
        let mut edges: BTreeSet<_> = border.keys().copied().collect();
        while let Some(&(a, b)) = edges.first() {
            edges.remove(&(a, b));
            let p = data.points[a];
            write!(text, "M{:.6},{:.6}", p[0], p[1]).unwrap();
            let mut v = b;
            loop {
                let p = data.points[v];
                write!(text, "L{:.6},{:.6}", p[0], p[1]).unwrap();
                if v == a {
                    break;
                }
                let next = edges.range((v, 0)..=(v, usize::MAX)).next().copied();
                let Some(e) = next else {
                    break;
                };
                edges.remove(&e);
                v = e.1;
            }
            text.push('Z');
        }
        text.push_str("\"/></clipPath></defs>");
        write!(text, "<g clip-path=\"url(#fill-sector-{start})\">").unwrap();
        let mut sector = MeshData {
            points: data.points.clone(),
            triangles: Vec::new(),
            constraints: Vec::new(),
        };
        let mut colors = Vec::new();
        // Extrude edge colors outside the clip so partial edge pixels have
        // coverage even when the internal tile painter snaps to pixel centers.
        for (&(a, b), &(f, k)) in &border {
            let (p, q) = (data.points[a], data.points[b]);
            let d = [q[0] - p[0], q[1] - p[1]];
            let length = d[0].hypot(d[1]).max(1e-12);
            let n = [d[1] / length, -d[0] / length];
            let i = sector.points.len();
            sector
                .points
                .extend([[p[0] + n[0], p[1] + n[1]], [q[0] + n[0], q[1] + n[1]]]);
            let (ca, cb) = (fill.colors_linear[f][k], fill.colors_linear[f][(k + 1) % 3]);
            sector.triangles.extend([[a, i, i + 1], [a, i + 1, b]]);
            colors.extend([[ca, ca, cb], [ca, cb, cb]]);
        }
        for f in faces {
            sector.triangles.push(data.triangles[f]);
            colors.push(fill.colors_linear[f]);
        }
        svg_surface(
            &sector,
            &colors,
            fill.svg_error_bound,
            &format!("fill-{start}"),
            text,
        );
        text.push_str("</g>");
    }
}
pub fn svg_surface(
    data: &MeshData,
    colors: &[[[f64; 3]; 3]],
    limit: f64,
    prefix: &str,
    text: &mut String,
) {
    use std::fmt::Write;
    fn gradient(
        p: [Point; 3],
        c: [[f64; 3]; 3],
        limit: f64,
        text: &mut String,
        id: &mut usize,
        prefix: &str,
    ) -> bool {
        let (a, b) = [(0, 1), (1, 2), (2, 0)]
            .into_iter()
            .max_by(|(a, b), (d, e)| {
                let distance =
                    |a: usize, b: usize| (0..3).map(|j| (c[a][j] - c[b][j]).powi(2)).sum::<f64>();
                distance(*a, *b).total_cmp(&distance(*d, *e))
            })
            .unwrap();
        let axis = std::array::from_fn::<_, 3, _>(|j| c[b][j] - c[a][j]);
        let length = axis.iter().map(|v| v * v).sum::<f64>();
        if length < 1e-20 {
            return false;
        }
        let t = c.map(|v| (0..3).map(|j| (v[j] - c[a][j]) * axis[j]).sum::<f64>() / length);
        let mut residual = 0f64;
        for j in 0..3 {
            let e = (0..3)
                .map(|i| (c[i][j] - (c[a][j] + t[i] * axis[j])).abs())
                .fold(0., f64::max);
            let minimum = c
                .iter()
                .map(|v| v[j])
                .fold(f64::INFINITY, f64::min)
                .min(c[a][j])
                .min(c[b][j])
                .max(0.);
            residual = residual.max(srgb(minimum + e) - srgb(minimum));
        }
        if residual > limit - 0.75 {
            return false;
        }
        fn stops(
            a: [f64; 3],
            b: [f64; 3],
            lo: f64,
            hi: f64,
            error: f64,
            result: &mut Vec<(f64, [u8; 3])>,
        ) {
            let first = std::array::from_fn::<_, 3, _>(|j| a[j] + lo * (b[j] - a[j]));
            let last = std::array::from_fn::<_, 3, _>(|j| a[j] + hi * (b[j] - a[j]));
            let worst = (0..3)
                .map(|j| {
                    let x = first[j].min(last[j]);
                    let y = first[j].max(last[j]);
                    if y - x < 1e-15 {
                        return 0.;
                    }
                    let slope = (srgb(y) - srgb(x)) / (y - x);
                    let peak = (255. * 1.055 / 2.4 / slope)
                        .powf(2.4 / 1.4)
                        .clamp(x.max(0.0031308).min(y), y);
                    (srgb(peak) - srgb(x) - slope * (peak - x)).max(0.)
                })
                .fold(0., f64::max);
            if worst > error {
                let mid = (lo + hi) * 0.5;
                stops(a, b, lo, mid, error, result);
                stops(a, b, mid, hi, error, result);
            } else {
                result.push((hi, last.map(|v| srgb(v).round().clamp(0., 255.) as u8)));
            }
        }
        let mut stops_vec = vec![(0., c[a].map(|v| srgb(v).round().clamp(0., 255.) as u8))];
        stops(c[a], c[b], 0., 1., limit - residual - 0.5, &mut stops_vec);
        let det =
            (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
        let g = [
            ((t[1] - t[0]) * (p[2][1] - p[0][1]) - (t[2] - t[0]) * (p[1][1] - p[0][1])) / det,
            ((p[1][0] - p[0][0]) * (t[2] - t[0]) - (p[2][0] - p[0][0]) * (t[1] - t[0])) / det,
        ];
        let length = g[0] * g[0] + g[1] * g[1];
        let start = [
            p[0][0] - t[0] * g[0] / length,
            p[0][1] - t[0] * g[1] / length,
        ];
        let end = [start[0] + g[0] / length, start[1] + g[1] / length];
        *id += 1;
        write!(text,"<defs><linearGradient id=\"mesh-g-{prefix}-{id}\" gradientUnits=\"userSpaceOnUse\" x1=\"{:.9}\" y1=\"{:.9}\" x2=\"{:.9}\" y2=\"{:.9}\">",start[0],start[1],end[0],end[1]).unwrap();
        for (offset, rgb) in stops_vec {
            write!(
                text,
                "<stop offset=\"{offset:.9}\" stop-color=\"#{:02x}{:02x}{:02x}\"/>",
                rgb[0], rgb[1], rgb[2]
            )
            .unwrap();
        }
        writeln!(text,"</linearGradient></defs><path fill=\"url(#mesh-g-{prefix}-{id})\" d=\"M{:.6},{:.6}L{:.6},{:.6}L{:.6},{:.6}Z\"/>",p[0][0],p[0][1],p[1][0],p[1][1],p[2][0],p[2][1]).unwrap();
        true
    }
    fn triangle(
        p: [Point; 3],
        c: [[f64; 3]; 3],
        limit: f64,
        text: &mut String,
        id: &mut usize,
        prefix: &str,
    ) {
        let bounds = std::array::from_fn::<_, 3, _>(|j| {
            let low = c.iter().map(|v| v[j]).fold(f64::INFINITY, f64::min);
            let high = c.iter().map(|v| v[j]).fold(0., f64::max);
            srgb(high) - srgb(low)
        });
        if bounds.iter().any(|v| *v > 1e-5) && gradient(p, c, limit, text, id, prefix) {
            return;
        }
        if bounds.iter().any(|v| *v > limit) {
            if bounds.iter().any(|v| *v > limit * 2.5) && gradient(p, c, limit, text, id, prefix) {
                return;
            }
            let m = std::array::from_fn::<_, 3, _>(|i| {
                [
                    (p[i][0] + p[(i + 1) % 3][0]) * 0.5,
                    (p[i][1] + p[(i + 1) % 3][1]) * 0.5,
                ]
            });
            let v = std::array::from_fn::<_, 3, _>(|i| {
                std::array::from_fn(|j| (c[i][j] + c[(i + 1) % 3][j]) * 0.5)
            });
            for (a, b) in [
                ([p[0], m[0], m[2]], [c[0], v[0], v[2]]),
                ([m[0], p[1], m[1]], [v[0], c[1], v[1]]),
                ([m[2], m[1], p[2]], [v[2], v[1], c[2]]),
                ([m[0], m[1], m[2]], [v[0], v[1], v[2]]),
            ] {
                triangle(a, b, limit, text, id, prefix);
            }
            return;
        }
        let value = interpolate(c, [1. / 3.; 3]);
        writeln!(
            text,
            "<path fill=\"#{:02x}{:02x}{:02x}\" d=\"M{:.6},{:.6}L{:.6},{:.6}L{:.6},{:.6}Z\"/>",
            value[0], value[1], value[2], p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1]
        )
        .unwrap();
    }
    writeln!(
        text,
        "<g id=\"linear-light-mesh-{prefix}\" shape-rendering=\"crispEdges\">"
    )
    .unwrap();
    let mut id = 0;
    for (t, c) in data.triangles.iter().zip(colors) {
        triangle(t.map(|i| data.points[i]), *c, limit, text, &mut id, prefix);
    }
    text.push_str("</g>\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn palette_partition_does_not_turn_a_continuous_ramp_into_a_hard_interface() -> Result<()> {
        let (w, h) = (64, 32);
        let labels: Vec<_> = (0..w * h).map(|i| usize::from(i % w >= 32)).collect();
        let rgb: Vec<_> = (0..w * h).map(|i| [(30 + i % w * 3) as u8; 3]).collect();
        let palette = [[30.; 3], [220.; 3]];
        let graph = crate::shared::build(&labels, w, h, 0.65)?;
        let fill = build(
            &rgb,
            &labels,
            w,
            h,
            &Options::default(),
            &palette,
            Some(&graph),
        )?;
        assert!(fill.suppressed_palette_boundaries > 0);
        assert!(
            fill.boundaries
                .iter()
                .all(|b| b.regions.contains(&usize::MAX))
        );
        let image = raster(&fill, w, h)?;
        let at = |x: usize| image.data()[(16 * w + x) * 4] as i32;
        assert!((at(32) - at(31)).abs() <= 5, "palette seam became visible");
        let step: Vec<_> = labels
            .iter()
            .map(|l| if *l == 0 { [30; 3] } else { [220; 3] })
            .collect();
        let fill = build(
            &step,
            &labels,
            w,
            h,
            &Options::default(),
            &palette,
            Some(&graph),
        )?;
        assert_eq!(fill.suppressed_palette_boundaries, 0);
        assert!(
            fill.boundaries
                .iter()
                .any(|b| !b.regions.contains(&usize::MAX))
        );
        Ok(())
    }
    #[test]
    fn svg_gradient_export_respects_color_bound_against_independent_renderer() -> Result<()> {
        let fill = Fill {
            geometry: MeshData {
                points: vec![[0., 0.], [64., 0.], [64., 64.], [0., 64.]],
                triangles: vec![[0, 1, 2], [0, 2, 3]],
                constraints: Vec::new(),
            },
            colors_linear: vec![
                [[0., 0., 0.], [1., 1., 1.], [1., 1., 1.]],
                [[0., 0., 0.], [1., 1., 1.], [0., 0., 0.]],
            ],
            region_labels: vec![0, 0],
            boundaries: Vec::new(),
            max_channel_error: 0.,
            unmet_pixels: 0,
            vertex_budget_reached: false,
            svg_error_bound: 12.,
            edge_band_pixels: 0,
            interior_max_channel_error: 0.,
            suppressed_palette_boundaries: 0,
        };
        let native = raster(&fill, 64, 64)?;
        let mut text =
            String::from("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"64\" height=\"64\">");
        svg(&fill, &mut text);
        text.push_str("</svg>");
        assert!(text.contains("linearGradient"));
        let exported = crate::vectorize::rasterize(&text)?;
        let maximum = native
            .data()
            .iter()
            .zip(exported.data())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(maximum <= 13, "SVG color error {maximum}");
        Ok(())
    }
    #[test]
    fn slanted_discontinuous_edge_has_partial_coverage_without_alpha_cracks() -> Result<()> {
        let fill = Fill {
            geometry: MeshData {
                points: vec![
                    [0., 0.],
                    [16., 0.],
                    [16., 16.],
                    [0., 16.],
                    [5., 0.],
                    [11., 16.],
                ],
                triangles: vec![[0, 4, 3], [4, 5, 3], [4, 1, 2], [4, 2, 5]],
                constraints: vec![[4, 5]],
            },
            colors_linear: vec![[[0.; 3]; 3], [[0.; 3]; 3], [[1.; 3]; 3], [[1.; 3]; 3]],
            region_labels: vec![0, 0, 1, 1],
            boundaries: Vec::new(),
            max_channel_error: 0.,
            unmet_pixels: 0,
            vertex_budget_reached: false,
            svg_error_bound: 12.,
            edge_band_pixels: 0,
            interior_max_channel_error: 0.,
            suppressed_palette_boundaries: 0,
        };
        let native = raster(&fill, 16, 16)?;
        assert!(
            native
                .data()
                .chunks_exact(4)
                .any(|p| p[0] > 0 && p[0] < 255)
        );
        let mut text = String::from(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\"><rect width=\"16\" height=\"16\" fill=\"white\"/>",
        );
        svg(&fill, &mut text);
        text.push_str("</svg>");
        let exported = crate::vectorize::rasterize(&text)?;
        assert!(
            exported
                .data()
                .chunks_exact(4)
                .any(|p| p[0] > 0 && p[0] < 255)
        );
        Ok(())
    }
    #[test]
    fn shared_interface_and_discontinuous_colors_are_exact() -> Result<()> {
        let (w, h) = (12, 8);
        let labels: Vec<_> = (0..w * h).map(|i| usize::from(i % w >= 6)).collect();
        let rgb: Vec<_> = labels
            .iter()
            .map(|l| if *l == 0 { [255, 0, 0] } else { [0, 0, 255] })
            .collect();
        let fill = build(&rgb, &labels, w, h, &Options::default(), &[], None)?;
        assert_eq!(
            fill.boundaries
                .iter()
                .filter(|b| b.regions == [0, 1])
                .count(),
            1
        );
        let image = raster(&fill, w, h)?;
        for (a, b) in image.data().chunks_exact(4).zip(rgb) {
            assert_eq!(&a[..3], &b);
        }
        Ok(())
    }
    #[test]
    fn linear_gradient_and_budget_are_reported() -> Result<()> {
        let (w, h) = (24, 12);
        let rgb: Vec<_> = (0..w * h)
            .map(|i| {
                let v = srgb((i % w) as f64 / (w - 1) as f64).round() as u8;
                [v, v, v]
            })
            .collect();
        let fill = build(
            &rgb,
            &vec![0; w * h],
            w,
            h,
            &Options {
                mesh_vertices: 150,
                ..Options::default()
            },
            &[],
            None,
        )?;
        assert!(fill.max_channel_error <= 8. || fill.vertex_budget_reached);
        assert_eq!(fill.geometry.constraints.len(), 4);
        raster(&fill, w, h)?;
        Ok(())
    }
}
