//! Independently implemented least-squares cubic fitting with bounded sample error.
use serde::{Deserialize, Serialize};
pub type Point = [f64; 2];
pub type Cubic = [Point; 4];
pub fn add(a: Point, b: Point) -> Point {
    [a[0] + b[0], a[1] + b[1]]
}
pub fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}
pub fn mul(a: Point, s: f64) -> Point {
    [a[0] * s, a[1] * s]
}
pub fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
pub fn norm(a: Point) -> f64 {
    dot(a, a).sqrt()
}
pub fn unit(a: Point) -> Point {
    let n = norm(a);
    if n > 1e-12 { mul(a, 1. / n) } else { [1., 0.] }
}
pub fn at(c: &Cubic, t: f64) -> Point {
    let u = 1. - t;
    add(
        add(mul(c[0], u * u * u), mul(c[1], 3. * u * u * t)),
        add(mul(c[2], 3. * u * t * t), mul(c[3], t * t * t)),
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fit {
    pub cubics: Vec<Cubic>,
    pub max_sample_error: f64,
}

pub fn fit(points: &[Point], tolerance: f64) -> Fit {
    let mut output = Fit {
        cubics: Vec::new(),
        max_sample_error: 0.,
    };
    if points.len() < 2 {
        return output;
    }
    // Split true corners before fitting; do not smooth a sharp turn away.
    let mut begin = 0;
    for i in 1..points.len() - 1 {
        let a = unit(sub(points[i], points[i - 1]));
        let b = unit(sub(points[i + 1], points[i]));
        if dot(a, b) < 0.6 {
            fit_piece(&points[begin..=i], tolerance, &mut output);
            begin = i;
        }
    }
    fit_piece(&points[begin..], tolerance, &mut output);
    output
}

fn fit_piece(points: &[Point], tolerance: f64, out: &mut Fit) {
    if points.len() < 2 {
        return;
    }
    let a = points[0];
    let b = *points.last().unwrap();
    let left = unit(sub(points[1], a));
    let right = unit(sub(points[points.len() - 2], b));
    fit_tangents(points, tolerance, out, left, right);
}
fn fit_tangents(points: &[Point], tolerance: f64, out: &mut Fit, left: Point, right: Point) {
    let a = points[0];
    let b = *points.last().unwrap();
    let mut parameters = vec![0.];
    for pair in points.windows(2) {
        parameters.push(parameters.last().unwrap() + norm(sub(pair[1], pair[0])));
    }
    let length = *parameters.last().unwrap();
    if length <= 1e-12 {
        return;
    }
    for t in &mut parameters {
        *t /= length;
    }
    let (mut aa, mut ab, mut bb, mut ar, mut br) = (0., 0., 0., 0., 0.);
    for (p, t) in points.iter().zip(&parameters) {
        let u = 1. - t;
        let b1 = 3. * u * u * t;
        let b2 = 3. * u * t * t;
        let x = mul(left, b1);
        let y = mul(right, b2);
        let base = add(mul(a, u * u * u + b1), mul(b, b2 + t * t * t));
        let r = sub(*p, base);
        aa += dot(x, x);
        ab += dot(x, y);
        bb += dot(y, y);
        ar += dot(x, r);
        br += dot(y, r);
    }
    let determinant = aa * bb - ab * ab;
    let mut lengths = [norm(sub(b, a)) / 3.; 2];
    if determinant.abs() > 1e-12 {
        let x = (ar * bb - br * ab) / determinant;
        let y = (br * aa - ar * ab) / determinant;
        if x > 0. && y > 0. && x <= length * 2. && y <= length * 2. {
            lengths = [x, y];
        }
    }
    let cubic = [
        a,
        add(a, mul(left, lengths[0])),
        add(b, mul(right, lengths[1])),
        b,
    ];
    let mut worst = 0.;
    let mut split = points.len() / 2;
    for i in 1..points.len() - 1 {
        let distance = norm(sub(points[i], at(&cubic, parameters[i])));
        if distance > worst {
            worst = distance;
            split = i;
        }
    }
    if worst <= tolerance || points.len() == 2 {
        out.max_sample_error = out.max_sample_error.max(worst);
        out.cubics.push(cubic);
    } else {
        // Iteratively smaller slices guarantee termination without a recursion cutoff.
        let tangent = unit(sub(points[split + 1], points[split - 1]));
        fit_tangents(&points[..=split], tolerance, out, left, mul(tangent, -1.));
        fit_tangents(&points[split..], tolerance, out, tangent, right);
    }
}

/// Denoise digitization steps while retaining corners supported over several
/// pixels. Closed curves use periodic neighbors and a common seam tangent.
pub fn smooth_points(points: &[Point], closed: bool, maximum: f64) -> Vec<Point> {
    if points.len() < 4 || maximum <= 0. {
        return points.to_vec();
    }
    let n = points.len() - usize::from(closed && points.first() == points.last());
    let at = |i: isize| {
        points[if closed {
            i.rem_euclid(n as isize) as usize
        } else {
            i.clamp(0, n as isize - 1) as usize
        }]
    };
    let mut scores = vec![1.; n];
    for (i, score) in scores.iter_mut().enumerate() {
        let mut ends = [at(i as isize); 2];
        for (side, direction) in [-1, 1].into_iter().enumerate() {
            let mut traveled = 0.;
            let mut previous = at(i as isize);
            for k in 1..n.min(16) {
                let p = at(i as isize + direction * k as isize);
                traveled += norm(sub(p, previous));
                ends[side] = p;
                previous = p;
                if traveled >= 3. {
                    break;
                }
            }
        }
        *score = dot(unit(sub(points[i], ends[0])), unit(sub(ends[1], points[i])));
    }
    let corner = |i: usize| {
        scores[i] < 0.65
            && [-2isize, -1, 1, 2].iter().all(|d| {
                let j = if closed {
                    (i as isize + d).rem_euclid(n as isize) as usize
                } else {
                    (i as isize + d).clamp(0, n as isize - 1) as usize
                };
                scores[i] <= scores[j]
            })
    };
    let mut output = Vec::with_capacity(points.len());
    for i in 0..n {
        if (!closed && (i == 0 || i + 1 == n)) || corner(i) {
            output.push(points[i]);
            continue;
        }
        let mut sum = points[i];
        let mut total = 1.;
        for direction in [-1, 1] {
            let mut traveled = 0.;
            let mut previous = points[i];
            for k in 1..n.min(12) {
                let j = if closed {
                    (i as isize + direction * k as isize).rem_euclid(n as isize) as usize
                } else {
                    (i as isize + direction * k as isize).clamp(0, n as isize - 1) as usize
                };
                let p = points[j];
                traveled += norm(sub(p, previous));
                previous = p;
                if traveled > 3. || corner(j) {
                    break;
                }
                let weight = (-traveled * traveled / 2.).exp();
                sum = add(sum, mul(p, weight));
                total += weight;
            }
        }
        let shift = sub(mul(sum, 1. / total), points[i]);
        output.push(add(
            points[i],
            mul(shift, (maximum / norm(shift).max(maximum)).min(1.)),
        ));
    }
    if closed {
        output.push(output[0]);
    }
    output
}
pub fn fit_smooth(points: &[Point], closed: bool, tolerance: f64) -> Fit {
    let mut out = Fit {
        cubics: Vec::new(),
        max_sample_error: 0.,
    };
    if points.len() < 2 {
        return out;
    }
    let n = points.len();
    let tangent = |i: usize| {
        if closed && (i == 0 || i == n - 1) {
            unit(sub(points[2.min(n - 2)], points[n - 3]))
        } else {
            unit(sub(points[(i + 2).min(n - 1)], points[i.saturating_sub(2)]))
        }
    };
    // Tangents are shared at all recursive joins. Detect corners at a larger
    // spatial scale than adjacent skeleton pixels, never at each grid step.
    let mut begin = 0;
    for i in 1..n - 1 {
        let left = unit(sub(points[i], points[i.saturating_sub(3)]));
        let right = unit(sub(points[(i + 3).min(n - 1)], points[i]));
        if dot(left, right) < 0.5 {
            fit_tangents(
                &points[begin..=i],
                tolerance,
                &mut out,
                if begin == 0 {
                    tangent(begin)
                } else {
                    unit(sub(points[(begin + 2).min(i)], points[begin]))
                },
                mul(left, -1.),
            );
            begin = i;
        }
    }
    if begin + 1 < n {
        fit_tangents(
            &points[begin..],
            tolerance,
            &mut out,
            if begin == 0 {
                tangent(0)
            } else {
                unit(sub(points[(begin + 2).min(n - 1)], points[begin]))
            },
            mul(tangent(n - 1), -1.),
        );
    }
    out
}

pub fn flatten(fit: &Fit, error: f64) -> Vec<Point> {
    fn piece(c: Cubic, error: f64, out: &mut Vec<Point>) {
        let chord = sub(c[3], c[0]);
        let length = norm(chord);
        let distance = |p: Point| {
            if length < 1e-12 {
                norm(sub(p, c[0]))
            } else {
                let v = sub(p, c[0]);
                (v[0] * chord[1] - v[1] * chord[0]).abs() / length
            }
        };
        let polygon = norm(sub(c[1], c[0])) + norm(sub(c[2], c[1])) + norm(sub(c[3], c[2]));
        if distance(c[1]).max(distance(c[2])) <= error && polygon - length <= error {
            out.push(c[3]);
            return;
        }
        let a = mul(add(c[0], c[1]), 0.5);
        let b = mul(add(c[1], c[2]), 0.5);
        let d = mul(add(c[2], c[3]), 0.5);
        let e = mul(add(a, b), 0.5);
        let f = mul(add(b, d), 0.5);
        let m = mul(add(e, f), 0.5);
        piece([c[0], a, e, m], error, out);
        piece([m, f, d, c[3]], error, out);
    }
    let mut out = Vec::new();
    if let Some(c) = fit.cubics.first() {
        out.push(c[0]);
    }
    for c in &fit.cubics {
        piece(*c, error, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn digitized_diagonal_becomes_tangent_continuous_without_grid_corners() {
        let points: Vec<_> = (0..80).map(|i| [i as f64 * 0.5, (i / 2) as f64]).collect();
        let smoothed = smooth_points(&points, false, 0.65);
        let fit = fit_smooth(&smoothed, false, 0.25);
        assert!(fit.cubics.len() < points.len() / 4);
        for pair in fit.cubics.windows(2) {
            let a = unit(sub(pair[0][3], pair[0][2]));
            let b = unit(sub(pair[1][1], pair[1][0]));
            assert!(dot(a, b) > 0.9999);
        }
    }
    #[test]
    fn flatten_retains_collinear_backtracking_extrema() {
        let fit = Fit {
            cubics: vec![[[0., 0.], [4., 0.], [-3., 0.], [1., 0.]]],
            max_sample_error: 0.,
        };
        let line = flatten(&fit, 0.01);
        assert!(line.len() > 4);
        assert!(line.iter().any(|p| p[0] > 1.1));
        assert!(line.iter().any(|p| p[0] < -0.1));
    }
    #[test]
    fn fitting_preserves_corners_endpoints_and_error() {
        let points: Vec<_> = (0..100)
            .map(|i| {
                let t = i as f64 * 0.03;
                [t, t.sin()]
            })
            .collect();
        let f = fit(&points, 0.01);
        assert!(f.max_sample_error <= 0.01);
        assert_eq!(f.cubics[0][0], points[0]);
        assert_eq!(f.cubics.last().unwrap()[3], *points.last().unwrap());
        for pair in f.cubics.windows(2) {
            assert_eq!(pair[0][3], pair[1][0]);
        }
        let corner = fit(&[[0., 0.], [1., 0.], [1., 1.]], 0.01);
        assert_eq!(corner.cubics.len(), 2);
    }
}
