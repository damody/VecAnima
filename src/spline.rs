//! Shape-preserving cubic Hermite interpolation for width and color channels.
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Scalar {
    pub knots: Vec<f64>,
    pub values: Vec<f64>,
    pub slopes: Vec<f64>,
}
impl Scalar {
    pub fn new(knots: Vec<f64>, values: Vec<f64>, closed: bool) -> Self {
        assert_eq!(knots.len(), values.len());
        let n = knots.len();
        let mut slopes = vec![0.; n];
        if n < 2 {
            return Self {
                knots,
                values,
                slopes,
            };
        }
        let h: Vec<_> = knots.windows(2).map(|p| p[1] - p[0]).collect();
        assert!(h.iter().all(|v| *v > 0.));
        let d: Vec<_> = values
            .windows(2)
            .zip(&h)
            .map(|(v, h)| (v[1] - v[0]) / h)
            .collect();
        let mean = |a: f64, b: f64, ha: f64, hb: f64| {
            if a * b <= 0. {
                0.
            } else {
                let w1 = 2. * hb + ha;
                let w2 = hb + 2. * ha;
                (w1 + w2) / (w1 / a + w2 / b)
            }
        };
        for i in 1..n - 1 {
            slopes[i] = mean(d[i - 1], d[i], h[i - 1], h[i]);
        }
        slopes[0] = d[0];
        slopes[n - 1] = d[n - 2];
        if closed && n > 2 {
            let slope = mean(d[n - 2], d[0], h[n - 2], h[0]);
            slopes[0] = slope;
            slopes[n - 1] = slope;
        }
        Self {
            knots,
            values,
            slopes,
        }
    }
    pub fn at(&self, x: f64) -> f64 {
        if self.knots.len() < 2 {
            return self.values.first().copied().unwrap_or(0.);
        }
        let i = self
            .knots
            .partition_point(|k| *k <= x)
            .saturating_sub(1)
            .min(self.knots.len() - 2);
        let h = self.knots[i + 1] - self.knots[i];
        let t = ((x - self.knots[i]) / h).clamp(0., 1.);
        let u = 1. - t;
        (1. + 2. * t) * u * u * self.values[i]
            + t * u * u * h * self.slopes[i]
            + t * t * (3. - 2. * t) * self.values[i + 1]
            - t * t * u * h * self.slopes[i + 1]
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positive_width_and_monotone_color_never_overshoot() {
        let s = Scalar::new(vec![0., 1., 2., 3.], vec![0.2, 4., 1., 0.4], false);
        for i in 0..300 {
            assert!((0.2..=4.).contains(&s.at(i as f64 / 100.)));
        }
        let c = Scalar::new(vec![0., 1., 3.], vec![0., 0.2, 1.], false);
        for i in 0..300 {
            assert!(c.at(i as f64 / 100.) <= c.at((i + 1) as f64 / 100.));
        }
    }
}
