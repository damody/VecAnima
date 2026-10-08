//! A planar shared boundary graph: one smooth curve, two incident region sides.
use crate::{
    curves::{self, Fit, Point},
    geometry,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
pub const OUTSIDE: usize = usize::MAX;
#[derive(Serialize, Deserialize)]
pub struct Graph {
    pub curves: Vec<Boundary>,
    pub loops: BTreeMap<usize, Vec<Vec<Directed>>>,
    pub smoothing_retries: usize,
}
#[derive(Serialize, Deserialize)]
pub struct Boundary {
    pub start: usize,
    pub end: usize,
    pub regions: [usize; 2],
    pub closed: bool,
    pub curve: Fit,
    pub polyline: Vec<Point>,
    pub smoothing_scale: f64,
    pub start_direction: usize,
    pub end_direction: usize,
}
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Directed {
    pub curve: usize,
    pub reverse: bool,
}
fn key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}
fn direction(a: usize, b: usize, stride: usize) -> usize {
    if b == a + 1 {
        0
    } else if b == a + stride {
        1
    } else if a == b + 1 {
        2
    } else {
        3
    }
}
fn dense(curve: &Fit) -> Vec<Point> {
    let line = curves::flatten(curve, 0.04);
    let mut out = Vec::new();
    if let Some(p) = line.first() {
        out.push(*p);
    }
    for pair in line.windows(2) {
        let d = curves::sub(pair[1], pair[0]);
        let n = (curves::norm(d) / 8.).ceil().max(1.) as usize;
        for i in 1..=n {
            out.push(if i == n {
                pair[1]
            } else {
                curves::add(pair[0], curves::mul(d, i as f64 / n as f64))
            });
        }
    }
    out
}
fn intersection(curves: &[Boundary]) -> Option<(usize, usize)> {
    struct Segment {
        curve: usize,
        index: usize,
        a: Point,
        b: Point,
    }
    let mut segments: Vec<Segment> = Vec::new();
    let mut bins: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (c, curve) in curves.iter().enumerate() {
        for (index, pair) in curve.polyline.windows(2).enumerate() {
            let (a, b) = (pair[0], pair[1]);
            let lo = [
                (a[0].min(b[0]) / 16.).floor() as i32,
                (a[1].min(b[1]) / 16.).floor() as i32,
            ];
            let hi = [
                (a[0].max(b[0]) / 16.).floor() as i32,
                (a[1].max(b[1]) / 16.).floor() as i32,
            ];
            let mut tested = BTreeSet::new();
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    if let Some(items) = bins.get(&(x, y)) {
                        for &id in items {
                            if !tested.insert(id) {
                                continue;
                            }
                            let old = &segments[id];
                            if old.curve == c
                                && (old.index.abs_diff(index) <= 1
                                    || (curve.closed
                                        && old.index == 0
                                        && index + 2 == curve.polyline.len()))
                            {
                                continue;
                            }
                            // Shared graph nodes are permitted, crossings away from them are not.
                            if geometry::crosses(a, b, old.a, old.b) {
                                return Some((c, old.curve));
                            }
                        }
                    }
                }
            }
            let id = segments.len();
            segments.push(Segment {
                curve: c,
                index,
                a,
                b,
            });
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    bins.entry((x, y)).or_default().push(id);
                }
            }
        }
    }
    None
}
pub fn build(labels: &[usize], w: usize, h: usize, smoothing: f64) -> Result<Graph> {
    ensure!(
        w > 0 && h > 0 && labels.len() == w * h,
        "Invalid boundary dimensions"
    );
    let stride = w + 1;
    let mut edges = BTreeMap::new();
    let mut add = |a, b, positive, negative| {
        if positive != negative {
            edges.insert(key(a, b), [positive, negative]);
        }
    };
    for y in 0..=h {
        for x in 0..w {
            add(
                y * stride + x,
                y * stride + x + 1,
                if y == h { OUTSIDE } else { labels[y * w + x] },
                if y == 0 {
                    OUTSIDE
                } else {
                    labels[(y - 1) * w + x]
                },
            );
        }
    }
    for y in 0..h {
        for x in 0..=w {
            add(
                y * stride + x,
                (y + 1) * stride + x,
                if x == 0 {
                    OUTSIDE
                } else {
                    labels[y * w + x - 1]
                },
                if x == w { OUTSIDE } else { labels[y * w + x] },
            );
        }
    }
    let mut neighbors: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(a, b) in edges.keys() {
        neighbors.entry(a).or_default().push(b);
        neighbors.entry(b).or_default().push(a);
    }
    let anchor = |v: usize| {
        let n = &neighbors[&v];
        n.len() != 2
            || key(edges[&key(v, n[0])][0], edges[&key(v, n[0])][1])
                != key(edges[&key(v, n[1])][0], edges[&key(v, n[1])][1])
            || ((v.is_multiple_of(stride) || v % stride == w)
                && (v / stride == 0 || v / stride == h))
    };
    // Shared junctions also need subpixel positions. Keeping every three-way
    // palette junction fixed on grid corners defeats curve smoothing entirely.
    // Relax the planar graph with a positional anchor and a strict < half-cell
    // displacement bound; all incident interfaces use the same coordinates.
    let original: BTreeMap<_, _> = neighbors
        .keys()
        .map(|v| (*v, [(v % stride) as f64, (v / stride) as f64]))
        .collect();
    let mut positions = original.clone();
    for _ in 0..8 {
        let mut next = positions.clone();
        for (&v, adjacent) in &neighbors {
            if v % stride == 0 || v % stride == w || v / stride == 0 || v / stride == h {
                continue;
            }
            let mut mean = [0.; 2];
            for n in adjacent {
                mean = curves::add(mean, positions[n]);
            }
            mean = curves::mul(mean, 1. / adjacent.len() as f64);
            let target = curves::add(curves::mul(original[&v], 0.25), curves::mul(mean, 0.75));
            let delta = curves::sub(target, original[&v]);
            next.insert(
                v,
                curves::add(
                    original[&v],
                    curves::mul(
                        delta,
                        (smoothing.min(0.45) / curves::norm(delta).max(1e-20)).min(1.),
                    ),
                ),
            );
        }
        positions = next;
    }
    let mut seen = BTreeSet::new();
    let mut raw = Vec::new();
    let mut boundaries = Vec::new();
    let starts: Vec<_> = neighbors
        .keys()
        .copied()
        .filter(|v| anchor(*v))
        .chain(neighbors.keys().copied().filter(|v| !anchor(*v)))
        .collect();
    for start in starts {
        for &next in &neighbors[&start] {
            if !seen.insert(key(start, next)) {
                continue;
            }
            let mut chain = vec![start, next];
            let (mut previous, mut at) = (start, next);
            while at != start && !anchor(at) {
                let n = *neighbors[&at]
                    .iter()
                    .find(|i| **i != previous)
                    .context("Broken shared chain")?;
                ensure!(
                    seen.insert(key(at, n)),
                    "Shared boundary walk repeated an edge"
                );
                chain.push(n);
                previous = at;
                at = n;
            }
            let regions = if start < next {
                edges[&key(start, next)]
            } else {
                let mut pair = edges[&key(start, next)];
                pair.swap(0, 1);
                pair
            };
            let closed = at == start;
            let points: Vec<_> = chain.iter().map(|v| positions[v]).collect();
            let smooth = curves::smooth_points(&points, closed, smoothing);
            let curve = curves::fit_smooth(&smooth, closed, 0.2);
            let polyline = dense(&curve);
            boundaries.push(Boundary {
                start,
                end: at,
                regions,
                closed,
                curve,
                polyline,
                smoothing_scale: smoothing,
                start_direction: direction(start, next, stride),
                end_direction: direction(chain[chain.len() - 2], at, stride),
            });
            raw.push(points);
        }
    }
    let mut retries = 0;
    while let Some((a, b)) = intersection(&boundaries) {
        retries += 1;
        ensure!(
            retries <= boundaries.len() * 16,
            "Cannot construct noncrossing smooth boundaries"
        );
        for id in BTreeSet::from([a, b]) {
            let boundary = &mut boundaries[id];
            boundary.smoothing_scale *= 0.5;
            ensure!(
                boundary.smoothing_scale >= 1e-6,
                "Smooth boundary collapsed topology at curve {id}"
            );
            // Reduce both vertex displacement and handle reach. Merely reducing
            // sample-fit error does not bound Bezier overshoot between samples.
            let p = curves::smooth_points(&raw[id], boundary.closed, boundary.smoothing_scale);
            let n = p.len();
            let tangent = |i: usize| {
                if boundary.closed && (i == 0 || i + 1 == n) {
                    curves::unit(curves::sub(p[1], p[n - 2]))
                } else if i == 0 {
                    curves::unit(curves::sub(p[1], p[0]))
                } else if i + 1 == n {
                    curves::unit(curves::sub(p[i], p[i - 1]))
                } else {
                    curves::unit(curves::sub(p[i + 1], p[i - 1]))
                }
            };
            let handles: Vec<_> = (0..n)
                .map(|i| {
                    let before = if i > 0 {
                        curves::norm(curves::sub(p[i], p[i - 1]))
                    } else if boundary.closed {
                        curves::norm(curves::sub(p[0], p[n - 2]))
                    } else {
                        curves::norm(curves::sub(p[1], p[0]))
                    };
                    let after = if i + 1 < n {
                        curves::norm(curves::sub(p[i + 1], p[i]))
                    } else if boundary.closed {
                        curves::norm(curves::sub(p[1], p[0]))
                    } else {
                        before
                    };
                    before.min(after) * boundary.smoothing_scale.min(0.3)
                })
                .collect();
            boundary.curve = Fit {
                cubics: (0..n - 1)
                    .map(|i| {
                        [
                            p[i],
                            curves::add(p[i], curves::mul(tangent(i), handles[i])),
                            curves::sub(p[i + 1], curves::mul(tangent(i + 1), handles[i + 1])),
                            p[i + 1],
                        ]
                    })
                    .collect(),
                max_sample_error: 0.,
            };
            boundary.polyline = dense(&boundary.curve);
        }
    }
    let mut outgoing: BTreeMap<(usize, usize), Vec<Directed>> = BTreeMap::new();
    for (i, b) in boundaries.iter().enumerate() {
        for (side, region) in b.regions.iter().enumerate() {
            if *region != OUTSIDE {
                outgoing
                    .entry((*region, if side == 0 { b.start } else { b.end }))
                    .or_default()
                    .push(Directed {
                        curve: i,
                        reverse: side == 1,
                    });
            }
        }
    }
    let mut loops: BTreeMap<usize, Vec<Vec<Directed>>> = BTreeMap::new();
    let mut visited = BTreeSet::new();
    for (&(region, start), items) in &outgoing {
        for &first in items {
            if visited.contains(&(first.curve, first.reverse)) {
                continue;
            }
            let mut current = first;
            let mut ring = Vec::new();
            loop {
                ensure!(
                    visited.insert((current.curve, current.reverse)),
                    "Shared region loop repeated a curve"
                );
                ring.push(current);
                let boundary = &boundaries[current.curve];
                let end = if current.reverse {
                    boundary.start
                } else {
                    boundary.end
                };
                if end == start {
                    break;
                }
                let incoming = if current.reverse {
                    (boundary.start_direction + 2) % 4
                } else {
                    boundary.end_direction
                };
                current = *outgoing
                    .get(&(region, end))
                    .context("Unclosed shared region")?
                    .iter()
                    .filter(|v| !visited.contains(&(v.curve, v.reverse)))
                    .min_by_key(|v| {
                        let b = &boundaries[v.curve];
                        let direction = if v.reverse {
                            (b.end_direction + 2) % 4
                        } else {
                            b.start_direction
                        };
                        let turn = (direction + 4 - incoming) % 4;
                        [1, 0, 3, 2].iter().position(|v| *v == turn).unwrap()
                    })
                    .context("No shared region successor")?;
            }
            loops.entry(region).or_default().push(ring);
        }
    }
    Ok(Graph {
        curves: boundaries,
        loops,
        smoothing_retries: retries,
    })
}
impl Graph {
    pub fn ring(&self, ring: &[Directed]) -> Fit {
        let mut fit = Fit {
            cubics: Vec::new(),
            max_sample_error: 0.,
        };
        for part in ring {
            let curve = &self.curves[part.curve].curve;
            fit.max_sample_error = fit.max_sample_error.max(curve.max_sample_error);
            if part.reverse {
                fit.cubics
                    .extend(curve.cubics.iter().rev().map(|c| [c[3], c[2], c[1], c[0]]));
            } else {
                fit.cubics.extend_from_slice(&curve.cubics);
            }
        }
        fit
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_multicolor_junctions_preserve_closed_planar_regions() -> Result<()> {
        let (w, h) = (24, 16);
        let mut seed = 19u64;
        let labels: Vec<_> = (0..w * h)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 32) as usize % 4
            })
            .collect();
        let graph = build(&labels, w, h, 0.65)?;
        assert!(intersection(&graph.curves).is_none());
        for rings in graph.loops.values() {
            for ring in rings {
                let curve = graph.ring(ring);
                assert_eq!(
                    curve.cubics.first().unwrap()[0],
                    curve.cubics.last().unwrap()[3]
                );
            }
        }
        Ok(())
    }
    #[test]
    fn shared_smooth_curves_close_regions_and_holes_without_crossings() -> Result<()> {
        let (w, h) = (32, 24);
        let mut labels = vec![0; w * h];
        for y in 3..21 {
            for x in 3..29 {
                if ((x as f64 - 16.).powi(2) + (y as f64 - 12.).powi(2)) < 80. {
                    labels[y * w + x] = 1;
                }
            }
        }
        for y in 10..14 {
            for x in 14..18 {
                labels[y * w + x] = 0;
            }
        }
        let graph = build(&labels, w, h, 0.65)?;
        assert!(intersection(&graph.curves).is_none());
        assert_eq!(graph.loops[&1].len(), 2);
        for loops in graph.loops.values() {
            for ring in loops {
                let fit = graph.ring(ring);
                assert_eq!(
                    fit.cubics.first().unwrap()[0],
                    fit.cubics.last().unwrap()[3]
                );
            }
        }
        assert!(
            graph
                .curves
                .iter()
                .any(|b| b.curve.cubics.len() * 3 < b.polyline.len())
        );
        Ok(())
    }
}
