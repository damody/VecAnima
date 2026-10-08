//! Pyramidal Lucas–Kanade, robust affine camera motion and stroke identities.
use crate::{
    curves::{Point, norm, sub},
    strokes::Stroke,
    vectorize::Frame,
};
use anyhow::{Result, ensure};
use clap::Args;
use opencv::{core, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize, Args)]
pub struct Options {
    #[arg(long)]
    pub temporal: bool,
    #[arg(long, default_value_t = 0.25)]
    pub temporal_strength: f64,
    #[arg(long, default_value_t = 0.45)]
    pub shot_threshold: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            temporal: false,
            temporal_strength: 0.25,
            shot_threshold: 0.45,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temporal_strength.is_finite() && (0. ..=1.).contains(&self.temporal_strength),
            "temporal-strength must be in 0..1"
        );
        ensure!(
            self.shot_threshold.is_finite() && (0. ..=1.).contains(&self.shot_threshold),
            "shot-threshold must be in 0..1"
        );
        Ok(())
    }
}
#[derive(Clone)]
struct Image {
    w: usize,
    h: usize,
    v: Vec<f64>,
}
impl Image {
    fn at(&self, p: Point) -> f64 {
        let (x, y) = (
            p[0].clamp(0., (self.w - 1) as f64),
            p[1].clamp(0., (self.h - 1) as f64),
        );
        let (a, b) = (x.floor() as usize, y.floor() as usize);
        let (c, d) = ((a + 1).min(self.w - 1), (b + 1).min(self.h - 1));
        let (u, v) = (x - a as f64, y - b as f64);
        self.v[b * self.w + a] * (1. - u) * (1. - v)
            + self.v[b * self.w + c] * u * (1. - v)
            + self.v[d * self.w + a] * (1. - u) * v
            + self.v[d * self.w + c] * u * v
    }
    fn inside(&self, p: Point, r: f64) -> bool {
        p[0] >= r && p[1] >= r && p[0] < (self.w - 1) as f64 - r && p[1] < (self.h - 1) as f64 - r
    }
}
fn pyramid(mat: &core::Mat) -> Result<Vec<Image>> {
    let mut result = vec![Image {
        w: mat.cols() as usize,
        h: mat.rows() as usize,
        v: mat
            .data_typed::<core::Vec3b>()?
            .iter()
            .map(|p| 0.114 * p[0] as f64 + 0.587 * p[1] as f64 + 0.299 * p[2] as f64)
            .collect(),
    }];
    while result.len() < 4 && result.last().unwrap().w.min(result.last().unwrap().h) >= 32 {
        let prev = result.last().unwrap();
        let (w, h) = (prev.w / 2, prev.h / 2);
        let mut v = vec![0.; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let weight = if dx == 0 { 2. } else { 1. } * if dy == 0 { 2. } else { 1. };
                        sum += weight
                            * prev.at([2. * x as f64 + dx as f64, 2. * y as f64 + dy as f64]);
                    }
                }
                v[y * w + x] = sum / 16.;
            }
        }
        result.push(Image { w, h, v });
    }
    Ok(result)
}
fn flow(a: &[Image], b: &[Image], p: Point, initial: Point) -> Option<(Point, f64)> {
    let mut delta = initial;
    let mut error = 0.;
    for level in (0..a.len().min(b.len())).rev() {
        let scale = (1usize << level) as f64;
        let q = [p[0] / scale, p[1] / scale];
        let x = &a[level];
        let y = &b[level];
        if !x.inside(q, 4.) {
            continue;
        }
        let mut d = [delta[0] / scale, delta[1] / scale];
        for _ in 0..15 {
            if !y.inside([q[0] + d[0], q[1] + d[1]], 4.) {
                return None;
            }
            let (mut xx, mut xy, mut yy, mut ex, mut ey, mut total) = (0., 0., 0., 0., 0., 0.);
            for j in -3..=3 {
                for i in -3..=3 {
                    let old = [q[0] + i as f64, q[1] + j as f64];
                    let at = [old[0] + d[0], old[1] + d[1]];
                    let gx = (y.at([at[0] + 1., at[1]]) - y.at([at[0] - 1., at[1]])) * 0.5;
                    let gy = (y.at([at[0], at[1] + 1.]) - y.at([at[0], at[1] - 1.])) * 0.5;
                    let e = x.at(old) - y.at(at);
                    xx += gx * gx;
                    xy += gx * gy;
                    yy += gy * gy;
                    ex += gx * e;
                    ey += gy * e;
                    total += e.abs();
                }
            }
            let determinant = xx * yy - xy * xy;
            if determinant <= 1e-6 || determinant / (xx + yy).max(1.) < 2. {
                return None;
            }
            let step = [
                (yy * ex - xy * ey) / determinant,
                (xx * ey - xy * ex) / determinant,
            ];
            if norm(step) > 4. {
                return None;
            }
            d[0] += step[0];
            d[1] += step[1];
            error = total / 49.;
            if norm(step) < 0.01 {
                break;
            }
        }
        delta = [d[0] * scale, d[1] * scale];
    }
    if error > 20. {
        None
    } else {
        Some(([p[0] + delta[0], p[1] + delta[1]], error))
    }
}
fn consistent(a: &[Image], b: &[Image], p: Point, initial: Point) -> Option<(Point, f64)> {
    let (q, e) = flow(a, b, p, initial)?;
    let (back, _) = flow(b, a, q, [p[0] - q[0], p[1] - q[1]])?;
    let fb = norm(sub(back, p));
    if fb > 1.5 {
        None
    } else {
        Some((q, (1. - fb / 1.5) * (1. - e / 20.)))
    }
}
fn apply(m: [f64; 6], p: Point) -> Point {
    [
        m[0] * p[0] + m[1] * p[1] + m[2],
        m[3] * p[0] + m[4] * p[1] + m[5],
    ]
}
fn solve(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for k in 0..3 {
        let pivot = (k..3).max_by(|i, j| a[*i][k].abs().total_cmp(&a[*j][k].abs()))?;
        a.swap(k, pivot);
        b.swap(k, pivot);
        if a[k][k].abs() < 1e-10 {
            return None;
        }
        let scale = a[k][k];
        for value in a[k].iter_mut().skip(k) {
            *value /= scale;
        }
        b[k] /= scale;
        for i in 0..3 {
            if i != k {
                let v = a[i][k];
                let pivot_row = a[k];
                for (value, pivot) in a[i].iter_mut().zip(pivot_row).skip(k) {
                    *value -= v * pivot;
                }
                b[i] -= v * b[k];
            }
        }
    }
    Some(b)
}
fn camera(a: &[Image], b: &[Image]) -> ([f64; 6], usize, f64) {
    let image = &a[0];
    let mut candidates = Vec::new();
    for y in (6..image.h.saturating_sub(6)).step_by(8) {
        for x in (6..image.w.saturating_sub(6)).step_by(8) {
            let mut xx = 0.;
            let mut xy = 0.;
            let mut yy = 0.;
            for j in -2..=2 {
                for i in -2..=2 {
                    let p = [x as f64 + i as f64, y as f64 + j as f64];
                    let gx = image.at([p[0] + 1., p[1]]) - image.at([p[0] - 1., p[1]]);
                    let gy = image.at([p[0], p[1] + 1.]) - image.at([p[0], p[1] - 1.]);
                    xx += gx * gx;
                    xy += gx * gy;
                    yy += gy * gy;
                }
            }
            let score = (xx + yy - ((xx - yy).powi(2) + 4. * xy * xy).sqrt()) * 0.5;
            if score > 100. {
                candidates.push((score, [x as f64, y as f64]));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let pairs: Vec<_> = candidates
        .into_iter()
        .take(128)
        .filter_map(|(_, p)| consistent(a, b, p, [0., 0.]).map(|(q, c)| (p, q, c)))
        .collect();
    let mut m = [1., 0., 0., 0., 1., 0.];
    let mut weights: Vec<_> = pairs.iter().map(|p| p.2).collect();
    if pairs.len() < 6 {
        return (m, pairs.len(), 0.);
    }
    for _ in 0..5 {
        let mut lhs = [[0.; 3]; 3];
        let mut rhs = [[0.; 3]; 2];
        for ((p, q, _), weight) in pairs.iter().zip(&weights) {
            let v = [p[0], p[1], 1.];
            for i in 0..3 {
                for j in 0..3 {
                    lhs[i][j] += weight * v[i] * v[j];
                }
                for k in 0..2 {
                    rhs[k][i] += weight * v[i] * q[k];
                }
            }
        }
        if let (Some(x), Some(y)) = (solve(lhs, rhs[0]), solve(lhs, rhs[1])) {
            m = [x[0], x[1], x[2], y[0], y[1], y[2]];
        } else {
            break;
        }
        for ((p, q, c), weight) in pairs.iter().zip(&mut weights) {
            let residual = norm(sub(apply(m, *p), *q));
            *weight = c * (1.5 / residual.max(1.5));
        }
    }
    let inliers = pairs
        .iter()
        .filter(|(p, q, _)| norm(sub(apply(m, *p), *q)) <= 2.)
        .count();
    let confidence = inliers as f64 / pairs.len() as f64;
    (m, inliers, confidence)
}
fn histogram(image: &Image) -> [f64; 32] {
    let mut h = [0.; 32];
    for v in &image.v {
        h[(*v as usize / 8).min(31)] += 1. / image.v.len() as f64;
    }
    h
}
fn center(s: &Stroke) -> Point {
    let n = s.samples.len() as f64;
    s.samples.iter().fold([0., 0.], |a, p| {
        [a[0] + p.center[0] / n, a[1] + p.center[1] / n]
    })
}
#[derive(Clone)]
struct Track {
    id: usize,
    stroke: Stroke,
}
#[derive(Serialize)]
pub struct Association {
    pub stroke: usize,
    pub track: usize,
    pub confidence: f64,
    pub filtered: bool,
}
#[derive(Serialize)]
pub struct RegionAssociation {
    pub region: usize,
    pub track: usize,
    pub confidence: f64,
    pub filtered: bool,
}
#[derive(Serialize)]
pub struct Event {
    pub kind: &'static str,
    pub tracks: Vec<usize>,
}
#[derive(Serialize)]
pub struct Record {
    pub shot: usize,
    pub cut: bool,
    pub histogram_distance: f64,
    pub camera_affine: [f64; 6],
    pub camera_inliers: usize,
    pub camera_confidence: f64,
    pub associations: Vec<Association>,
    pub events: Vec<Event>,
    pub regions: Vec<RegionAssociation>,
}
#[derive(Default)]
pub struct Tracker {
    previous: Option<Vec<Image>>,
    tracks: Vec<Track>,
    next: usize,
    shot: usize,
    regions: Vec<(usize, crate::vectorize::Region)>,
}
impl Tracker {
    pub fn process(
        &mut self,
        image: &core::Mat,
        frame: &mut Frame,
        options: &Options,
    ) -> Result<Record> {
        let current = pyramid(image)?;
        let distance = self.previous.as_ref().map_or(0., |p| {
            histogram(&p[0])
                .iter()
                .zip(histogram(&current[0]))
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                * 0.5
        });
        let (motion, inliers, confidence) = self
            .previous
            .as_ref()
            .map_or(([1., 0., 0., 0., 1., 0.], 0, 0.), |p| camera(p, &current));
        let cut = self.previous.is_some() && distance > options.shot_threshold && confidence < 0.5;
        let mut record = Record {
            shot: self.shot,
            cut,
            histogram_distance: distance,
            camera_affine: motion,
            camera_inliers: inliers,
            camera_confidence: confidence,
            associations: Vec::new(),
            events: Vec::new(),
            regions: Vec::new(),
        };
        if cut {
            self.shot += 1;
            record.shot = self.shot;
            record.events.push(Event {
                kind: "cut",
                tracks: self.tracks.iter().map(|t| t.id).collect(),
            });
            self.tracks.clear();
            self.regions.clear();
        }
        let mut candidates = Vec::new();
        let mut links = vec![Vec::new(); frame.strokes.len()];
        for (old, t) in self.tracks.iter().enumerate() {
            let p = center(&t.stroke);
            let prediction = apply(motion, p);
            let initial = sub(prediction, p);
            let tracked = self
                .previous
                .as_ref()
                .and_then(|prev| consistent(prev, &current, p, initial));
            let (predicted, flow_conf) = tracked.unwrap_or((prediction, confidence * 0.5));
            for (new, s) in frame.strokes.iter().enumerate() {
                if s.closed != t.stroke.closed {
                    continue;
                }
                let color = (0..3)
                    .map(|j| (s.color[j] as f64 - t.stroke.color[j] as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let d = norm(sub(center(s), predicted));
                let length = (s.samples.len() as f64 / t.stroke.samples.len() as f64)
                    .ln()
                    .abs();
                if d < 6. && color < 24. && length < 0.7 && flow_conf > 0.05 {
                    let score = d + color / 12. + length * 3.;
                    candidates.push((score, old, new, flow_conf, predicted));
                    links[new].push(old);
                }
            }
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let (mut used_old, mut used_new) = (BTreeSet::new(), BTreeSet::new());
        let mut next = Vec::new();
        for (score, old, new, flow_conf, predicted) in candidates {
            if used_old.contains(&old) || used_new.contains(&new) {
                continue;
            }
            used_old.insert(old);
            used_new.insert(new);
            let t = &self.tracks[old];
            let s = &mut frame.strokes[new];
            let residual = norm(sub(center(s), predicted));
            let c = flow_conf * (-score / 6.).exp();
            let mut filtered = false;
            if options.temporal
                && residual < 0.75
                && s.samples.len().abs_diff(t.stroke.samples.len()) <= 2
            {
                let strength = options.temporal_strength * c * (-residual * residual / 0.25).exp();
                if strength > 0.01 {
                    let sample_count = s.samples.len();
                    for (i, sample) in s.samples.iter_mut().enumerate() {
                        let j = i * (t.stroke.samples.len() - 1) / (sample_count.max(2) - 1);
                        let before = &t.stroke.samples[j];
                        // Width filtering is confidence gated. Centers remain source
                        // measured so fast facial changes cannot be spatially smeared.
                        if (sample.left - before.left).abs() < 0.5
                            && (sample.right - before.right).abs() < 0.5
                        {
                            sample.left = sample.left * (1. - strength) + before.left * strength;
                            sample.right = sample.right * (1. - strength) + before.right * strength;
                            filtered = true;
                        }
                    }
                    if filtered {
                        crate::strokes::rebuild(s, frame.settings.strokes.curve_error);
                    }
                }
            }
            record.associations.push(Association {
                stroke: s.id,
                track: t.id,
                confidence: c,
                filtered,
            });
            next.push(Track {
                id: t.id,
                stroke: s.clone(),
            });
        }
        for (i, s) in frame.strokes.iter().enumerate() {
            if !used_new.contains(&i) {
                let id = self.next;
                self.next += 1;
                record.events.push(Event {
                    kind: "birth",
                    tracks: vec![id],
                });
                record.associations.push(Association {
                    stroke: s.id,
                    track: id,
                    confidence: 1.,
                    filtered: false,
                });
                next.push(Track {
                    id,
                    stroke: s.clone(),
                });
            }
        }
        for (i, t) in self.tracks.iter().enumerate() {
            if !used_old.contains(&i) {
                record.events.push(Event {
                    kind: "death-or-occlusion",
                    tracks: vec![t.id],
                });
            }
        }
        for old in 0..self.tracks.len() {
            let children: Vec<_> = links
                .iter()
                .enumerate()
                .filter(|(_, v)| v.contains(&old))
                .filter_map(|(i, _)| {
                    record
                        .associations
                        .iter()
                        .find(|a| a.stroke == frame.strokes[i].id)
                        .map(|a| a.track)
                })
                .collect();
            if children.len() > 1 {
                record.events.push(Event {
                    kind: "split-candidate",
                    tracks: children,
                });
            }
        }
        for parents in links.iter().filter(|v| v.len() > 1) {
            record.events.push(Event {
                kind: "merge-candidate",
                tracks: parents.iter().map(|i| self.tracks[*i].id).collect(),
            });
        }
        record.associations.sort_by_key(|a| a.stroke);
        // Region identities use spatially indexed camera-compensated centers,
        // area and source color; they are connected color regions, not objects.
        let mut bins: std::collections::BTreeMap<(i32, i32), Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, r) in frame.regions.iter().enumerate() {
            bins.entry((
                (r.center[0] / 16.).floor() as i32,
                (r.center[1] / 16.).floor() as i32,
            ))
            .or_default()
            .push(i);
        }
        let mut region_candidates = Vec::new();
        for (old, (_, r)) in self.regions.iter().enumerate() {
            let p = apply(motion, r.center);
            let bin = ((p[0] / 16.).floor() as i32, (p[1] / 16.).floor() as i32);
            for y in -1..=1 {
                for x in -1..=1 {
                    if let Some(items) = bins.get(&(bin.0 + x, bin.1 + y)) {
                        for &new in items {
                            let s = &frame.regions[new];
                            let distance = norm(sub(p, s.center));
                            let area = (s.area as f64 / r.area as f64).ln().abs();
                            let color = (0..3)
                                .map(|j| (r.color[j] as f64 - s.color[j] as f64).powi(2))
                                .sum::<f64>()
                                .sqrt();
                            if distance < 12. && area < 0.75 && color < 24. {
                                region_candidates.push((
                                    distance + area * 4. + color / 8.,
                                    old,
                                    new,
                                ));
                            }
                        }
                    }
                }
            }
        }
        region_candidates
            .sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let (mut old_used, mut new_used, mut region_next) =
            (BTreeSet::new(), BTreeSet::new(), Vec::new());
        for (score, old, new) in region_candidates {
            if old_used.contains(&old) || new_used.contains(&new) {
                continue;
            }
            old_used.insert(old);
            new_used.insert(new);
            let id = self.regions[old].0;
            record.regions.push(RegionAssociation {
                region: frame.regions[new].id,
                track: id,
                confidence: (-score / 8.).exp() * confidence,
                filtered: false,
            });
            region_next.push((id, frame.regions[new].clone()));
        }
        for (i, r) in frame.regions.iter().enumerate() {
            if !new_used.contains(&i) {
                let id = self.next;
                self.next += 1;
                record.events.push(Event {
                    kind: "region-birth",
                    tracks: vec![id],
                });
                record.regions.push(RegionAssociation {
                    region: r.id,
                    track: id,
                    confidence: 1.,
                    filtered: false,
                });
                region_next.push((id, r.clone()));
            }
        }
        for (i, (id, _)) in self.regions.iter().enumerate() {
            if !old_used.contains(&i) {
                record.events.push(Event {
                    kind: "region-death-or-occlusion",
                    tracks: vec![*id],
                });
            }
        }
        record.regions.sort_by_key(|a| a.region);
        self.regions = region_next;
        self.tracks = next;
        self.previous = Some(current);
        Ok(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pyramidal_flow_and_affine_fit_follow_translation() -> Result<()> {
        let mut a =
            core::Mat::new_rows_cols_with_default(96, 128, core::CV_8UC3, core::Scalar::all(100.))?;
        let mut b = a.try_clone()?;
        for y in 10..80 {
            for x in 10..110 {
                let value = (((x * 17 + y * 31) % 97) * 2 + 30) as u8;
                *a.at_2d_mut::<core::Vec3b>(y, x)? = core::Vec3b::from([value; 3]);
                *b.at_2d_mut::<core::Vec3b>(y + 2, x + 3)? = core::Vec3b::from([value; 3]);
            }
        }
        let (p, q) = (pyramid(&a)?, pyramid(&b)?);
        let (m, n, c) = camera(&p, &q);
        assert!(n >= 6 && c > 0.6, "{m:?} {n} {c}");
        assert!(norm(sub(apply(m, [60., 40.]), [63., 42.])) < 0.3);
        Ok(())
    }
}
