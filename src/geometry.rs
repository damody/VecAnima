//! Independent robust predicates, incremental Delaunay and constrained cavities.
//! BigInt is only used for exact dyadic arithmetic, never for triangulation.
use anyhow::{Context, Result, bail, ensure};
use num_bigint::{BigInt, Sign};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
pub type Point = [f64; 2];
type Edge = (usize, usize);
fn edge(a: usize, b: usize) -> Edge {
    if a < b { (a, b) } else { (b, a) }
}
fn sign(x: f64) -> i8 {
    if x > 0. {
        1
    } else if x < 0. {
        -1
    } else {
        0
    }
}
fn integer_sign(x: BigInt) -> i8 {
    match x.sign() {
        Sign::Minus => -1,
        Sign::NoSign => 0,
        Sign::Plus => 1,
    }
}

fn dyadic(value: f64) -> (i64, i32) {
    debug_assert!(value.is_finite());
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 2047) as i32;
    let mantissa = if exponent == 0 {
        bits & ((1u64 << 52) - 1)
    } else {
        (bits & ((1u64 << 52) - 1)) | (1u64 << 52)
    };
    (
        (if bits >> 63 == 0 { 1 } else { -1 }) * mantissa as i64,
        if exponent == 0 {
            -1074
        } else {
            exponent - 1075
        },
    )
}
fn exact(points: &[Point]) -> Vec<[BigInt; 2]> {
    let parts: Vec<_> = points
        .iter()
        .map(|p| [dyadic(p[0]), dyadic(p[1])])
        .collect();
    let minimum = parts
        .iter()
        .flatten()
        .filter(|p| p.0 != 0)
        .map(|p| p.1)
        .min()
        .unwrap_or(0);
    parts
        .iter()
        .map(|p| {
            std::array::from_fn(|i| {
                if p[i].0 == 0 {
                    BigInt::from(0)
                } else {
                    BigInt::from(p[i].0) << ((p[i].1 - minimum) as usize)
                }
            })
        })
        .collect()
}
pub fn orient(a: Point, b: Point, c: Point) -> i8 {
    let (ax, ay, bx, by) = (a[0] - c[0], a[1] - c[1], b[0] - c[0], b[1] - c[1]);
    let (p, q) = (ax * by, ay * bx);
    let determinant = p - q;
    let bound = (p.abs() + q.abs()) * 4e-16;
    if determinant.is_finite() && determinant.abs() > bound && bound > f64::MIN_POSITIVE {
        return sign(determinant);
    }
    let p = exact(&[a, b, c]);
    integer_sign(
        (&p[0][0] - &p[2][0]) * (&p[1][1] - &p[2][1])
            - (&p[0][1] - &p[2][1]) * (&p[1][0] - &p[2][0]),
    )
}
/// Positive means d is inside abc's circumcircle, assuming abc is CCW.
pub fn incircle(a: Point, b: Point, c: Point, d: Point) -> i8 {
    let (ax, ay, bx, by, cx, cy) = (
        a[0] - d[0],
        a[1] - d[1],
        b[0] - d[0],
        b[1] - d[1],
        c[0] - d[0],
        c[1] - d[1],
    );
    let (al, bl, cl) = (ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy);
    let value = al * (bx * cy - by * cx) + bl * (cx * ay - cy * ax) + cl * (ax * by - ay * bx);
    let permanent = al * (bx.abs() * cy.abs() + by.abs() * cx.abs())
        + bl * (cx.abs() * ay.abs() + cy.abs() * ax.abs())
        + cl * (ax.abs() * by.abs() + ay.abs() * bx.abs());
    let bound = permanent * 2e-15;
    if value.is_finite() && value.abs() > bound && bound > f64::MIN_POSITIVE {
        return sign(value);
    }
    let p = exact(&[a, b, c, d]);
    let delta: Vec<_> = (0..3)
        .map(|i| [&p[i][0] - &p[3][0], &p[i][1] - &p[3][1]])
        .collect();
    let lift: Vec<_> = delta
        .iter()
        .map(|p| &p[0] * &p[0] + &p[1] * &p[1])
        .collect();
    integer_sign(
        &lift[0] * (&delta[1][0] * &delta[2][1] - &delta[1][1] * &delta[2][0])
            + &lift[1] * (&delta[2][0] * &delta[0][1] - &delta[2][1] * &delta[0][0])
            + &lift[2] * (&delta[0][0] * &delta[1][1] - &delta[0][1] * &delta[1][0]),
    )
}
pub fn crosses(a: Point, b: Point, c: Point, d: Point) -> bool {
    orient(a, b, c) * orient(a, b, d) < 0 && orient(c, d, a) * orient(c, d, b) < 0
}
fn on_segment(a: Point, b: Point, p: Point) -> bool {
    orient(a, b, p) == 0
        && p[0] >= a[0].min(b[0])
        && p[0] <= a[0].max(b[0])
        && p[1] >= a[1].min(b[1])
        && p[1] <= a[1].max(b[1])
}

#[derive(Clone, Debug)]
struct Face {
    v: [usize; 3],
    adj: [Option<usize>; 3],
    alive: bool,
}
#[derive(Debug)]
pub struct Mesh {
    pub points: Vec<Point>,
    faces: Vec<Face>,
    edges: HashMap<Edge, Vec<(usize, usize)>>,
    fans: Vec<HashSet<usize>>,
    pub constraints: BTreeSet<Edge>,
    hint: usize,
}
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MeshData {
    pub points: Vec<Point>,
    pub triangles: Vec<[usize; 3]>,
    pub constraints: Vec<[usize; 2]>,
}
#[derive(Deserialize)]
pub struct Input {
    pub points: Vec<Point>,
    #[serde(default)]
    pub constraints: Vec<[usize; 2]>,
    #[serde(default)]
    pub rings: Vec<Vec<usize>>,
}
pub fn triangulate(input: Input) -> Result<MeshData> {
    let mut mesh = Mesh::new(input.points)?;
    let mut segments: BTreeSet<_> = input.constraints.iter().map(|e| edge(e[0], e[1])).collect();
    let mut rings = Vec::new();
    for ring in input.rings {
        ensure!(
            ring.len() >= 3 && ring.iter().all(|i| *i < mesh.points.len()),
            "Invalid domain ring"
        );
        for (a, b) in ring
            .iter()
            .zip(ring.iter().cycle().skip(1))
            .take(ring.len())
        {
            segments.insert(edge(*a, *b));
        }
        rings.push(ring.iter().map(|i| mesh.points[*i]).collect::<Vec<_>>());
    }
    mesh.constrain_all(&segments.iter().map(|e| [e.0, e.1]).collect::<Vec<_>>())?;
    mesh.validate()?;
    let mut data = mesh.data();
    if !rings.is_empty() {
        data.triangles.retain(|t| {
            let p = [
                (data.points[t[0]][0] + data.points[t[1]][0] + data.points[t[2]][0]) / 3.,
                (data.points[t[0]][1] + data.points[t[1]][1] + data.points[t[2]][1]) / 3.,
            ];
            contains(&rings, p)
        });
    }
    Ok(data)
}

impl Mesh {
    pub fn new(points: Vec<Point>) -> Result<Self> {
        ensure!(points.len() >= 3, "Need at least three vertices");
        let mut unique = HashSet::new();
        for p in &points {
            ensure!(
                p.iter().all(|v| v.is_finite()),
                "Coordinates must be finite"
            );
            let key = p.map(|v| if v == 0. { 0 } else { v.to_bits() });
            ensure!(
                unique.insert(key),
                "Duplicate vertices must be merged before triangulation"
            );
        }
        let mut low = [f64::INFINITY; 2];
        let mut high = [f64::NEG_INFINITY; 2];
        for p in &points {
            for c in 0..2 {
                low[c] = low[c].min(p[c]);
                high[c] = high[c].max(p[c]);
            }
        }
        let size = (high[0] - low[0]).max(high[1] - low[1]);
        ensure!(
            size.is_finite() && size > 0.,
            "Invalid coordinate span; rescale input"
        );
        let middle = [
            low[0] + (high[0] - low[0]) * 0.5,
            low[1] + (high[1] - low[1]) * 0.5,
        ];
        let n = points.len();
        let mut all = points;
        all.extend([
            [middle[0] - size * 32., middle[1] - size * 16.],
            [middle[0] + size * 32., middle[1] - size * 16.],
            [middle[0], middle[1] + size * 32.],
        ]);
        ensure!(
            all.iter().flatten().all(|v| v.is_finite()),
            "Super-triangle exceeds finite coordinate range; rescale input"
        );
        let mut mesh = Self {
            fans: vec![HashSet::new(); all.len()],
            points: all,
            faces: Vec::new(),
            edges: HashMap::new(),
            constraints: BTreeSet::new(),
            hint: 0,
        };
        mesh.add_face([n, n + 1, n + 2])?;
        for i in 0..n {
            mesh.insert_index(i)?;
        }
        let remove: Vec<_> = mesh
            .faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.alive && f.v.iter().any(|v| *v >= n))
            .map(|(i, _)| i)
            .collect();
        for face in remove {
            mesh.remove_face(face);
        }
        mesh.points.truncate(n);
        mesh.fans.truncate(n);
        mesh.hint = mesh
            .faces
            .iter()
            .position(|f| f.alive)
            .context("All input vertices are collinear")?;
        Ok(mesh)
    }
    fn add_face(&mut self, mut vertices: [usize; 3]) -> Result<usize> {
        match orient(
            self.points[vertices[0]],
            self.points[vertices[1]],
            self.points[vertices[2]],
        ) {
            0 => bail!("Degenerate triangle"),
            -1 => vertices.swap(1, 2),
            _ => {}
        }
        let id = self.faces.len();
        self.faces.push(Face {
            v: vertices,
            adj: [None; 3],
            alive: true,
        });
        for v in vertices {
            self.fans[v].insert(id);
        }
        for s in 0..3 {
            let key = edge(vertices[s], vertices[(s + 1) % 3]);
            let neighbors = self.edges.entry(key).or_default();
            ensure!(neighbors.len() < 2, "Non-manifold triangulation edge");
            if let Some((other, os)) = neighbors.first().copied() {
                self.faces[id].adj[s] = Some(other);
                self.faces[other].adj[os] = Some(id);
            }
            neighbors.push((id, s));
        }
        Ok(id)
    }
    fn remove_face(&mut self, id: usize) {
        if !self.faces[id].alive {
            return;
        }
        let vertices = self.faces[id].v;
        for v in vertices {
            self.fans[v].remove(&id);
        }
        for s in 0..3 {
            let key = edge(vertices[s], vertices[(s + 1) % 3]);
            let neighbors = self.edges.get_mut(&key).unwrap();
            neighbors.retain(|(f, _)| *f != id);
            if let Some((f, s)) = neighbors.first().copied() {
                self.faces[f].adj[s] = None;
            }
            if neighbors.is_empty() {
                self.edges.remove(&key);
            }
        }
        self.faces[id].alive = false;
    }
    fn locate(&self, p: Point) -> Result<usize> {
        let mut current = if self.faces[self.hint].alive {
            self.hint
        } else {
            self.faces
                .iter()
                .position(|f| f.alive)
                .context("Empty mesh")?
        };
        let mut visited = HashSet::new();
        loop {
            ensure!(visited.insert(current), "Triangle location walk cycled");
            let f = &self.faces[current];
            let exit = (0..3)
                .find(|s| orient(self.points[f.v[*s]], self.points[f.v[(*s + 1) % 3]], p) < 0);
            if let Some(s) = exit {
                current = f.adj[s].context("Vertex outside triangulated domain")?;
            } else {
                return Ok(current);
            }
        }
    }
    fn insert_index(&mut self, index: usize) -> Result<()> {
        let p = self.points[index];
        let start = self.locate(p)?;
        let split = (0..3)
            .map(|s| edge(self.faces[start].v[s], self.faces[start].v[(s + 1) % 3]))
            .find(|e| {
                self.constraints.contains(e) && on_segment(self.points[e.0], self.points[e.1], p)
            });
        let mut bad = HashSet::new();
        let mut pending = vec![start];
        while let Some(id) = pending.pop() {
            if !bad.insert(id) {
                continue;
            }
            let face = &self.faces[id];
            for s in 0..3 {
                let key = edge(face.v[s], face.v[(s + 1) % 3]);
                if self.constraints.contains(&key) && Some(key) != split {
                    continue;
                }
                if let Some(other) = face.adj[s] {
                    let v = self.faces[other].v;
                    if !bad.contains(&other)
                        && incircle(self.points[v[0]], self.points[v[1]], self.points[v[2]], p) >= 0
                    {
                        pending.push(other);
                    }
                }
            }
        }
        let mut boundary = Vec::new();
        for id in &bad {
            let f = &self.faces[*id];
            for s in 0..3 {
                if f.adj[s].is_none_or(|n| !bad.contains(&n)) {
                    boundary.push((f.v[s], f.v[(s + 1) % 3]));
                }
            }
        }
        boundary.sort_unstable();
        for id in bad {
            self.remove_face(id);
        }
        for (a, b) in boundary {
            if orient(self.points[a], self.points[b], p) != 0 {
                self.hint = self.add_face([a, b, index])?;
            }
        }
        if let Some(key) = split {
            self.constraints.remove(&key);
            self.constraints.insert(edge(key.0, index));
            self.constraints.insert(edge(index, key.1));
        }
        Ok(())
    }
    pub fn insert(&mut self, p: Point) -> Result<usize> {
        ensure!(p.iter().all(|v| v.is_finite()), "Nonfinite inserted vertex");
        // Exact duplicate insertion is a no-op, including either sign of zero.
        if let Some(i) = self.points.iter().position(|v| *v == p) {
            return Ok(i);
        }
        self.locate(p)?; // Reject outside insertion before changing the vertex arrays.
        let index = self.points.len();
        self.points.push(p);
        self.fans.push(HashSet::new());
        self.insert_index(index)?;
        Ok(index)
    }
    #[cfg(test)]
    pub fn constrain(&mut self, a: usize, b: usize) -> Result<()> {
        self.constrain_raw(a, b)?;
        self.legalize()
    }
    pub fn constrain_all(&mut self, segments: &[[usize; 2]]) -> Result<()> {
        for [a, b] in segments {
            self.constrain_raw(*a, *b)?;
        }
        self.legalize()
    }
    fn constrain_raw(&mut self, a: usize, b: usize) -> Result<()> {
        ensure!(
            a < self.points.len() && b < self.points.len() && a != b,
            "Invalid constraint endpoints"
        );
        for &(c, d) in &self.constraints {
            ensure!(
                !crosses(
                    self.points[a],
                    self.points[b],
                    self.points[c],
                    self.points[d]
                ),
                "Crossing constraints require an explicit intersection vertex"
            );
        }
        let mut chain: Vec<_> = (0..self.points.len())
            .filter(|i| on_segment(self.points[a], self.points[b], self.points[*i]))
            .collect();
        let axis = usize::from(
            (self.points[a][1] - self.points[b][1]).abs()
                > (self.points[a][0] - self.points[b][0]).abs(),
        );
        chain.sort_by(|i, j| self.points[*i][axis].total_cmp(&self.points[*j][axis]));
        for pair in chain.windows(2) {
            self.recover(pair[0], pair[1])?;
            self.constraints.insert(edge(pair[0], pair[1]));
        }
        Ok(())
    }
    fn recover(&mut self, a: usize, b: usize) -> Result<()> {
        if self.edges.contains_key(&edge(a, b)) {
            return Ok(());
        }
        let mut first: Vec<_> = self.fans[a].iter().copied().collect();
        first.sort_unstable();
        let mut current = first
            .into_iter()
            .find(|id| {
                let v = self.faces[*id].v;
                let k = v.iter().position(|i| *i == a).unwrap();
                orient(self.points[a], self.points[v[(k + 1) % 3]], self.points[b]) >= 0
                    && orient(self.points[v[(k + 2) % 3]], self.points[a], self.points[b]) >= 0
            })
            .context("No triangle contains constraint ray")?;
        let mut cavity = BTreeSet::new();
        loop {
            ensure!(cavity.insert(current), "Constraint walk cycled");
            let f = &self.faces[current];
            if f.v.contains(&b) {
                break;
            }
            let exit = (0..3)
                .find(|s| {
                    f.adj[*s].is_none_or(|id| !cavity.contains(&id))
                        && crosses(
                            self.points[a],
                            self.points[b],
                            self.points[f.v[*s]],
                            self.points[f.v[(*s + 1) % 3]],
                        )
                })
                .context("Constraint walk failed to find crossed edge")?;
            current = f.adj[exit].context("Constraint exits mesh domain")?;
        }
        let mut boundary = HashMap::new();
        let mut protected = Vec::new();
        for id in &cavity {
            let f = &self.faces[*id];
            for s in 0..3 {
                let key = edge(f.v[s], f.v[(s + 1) % 3]);
                if f.adj[s].is_none_or(|n| !cavity.contains(&n)) {
                    ensure!(
                        boundary.insert(f.v[s], f.v[(s + 1) % 3]).is_none(),
                        "Non-simple constraint cavity"
                    );
                } else if self.constraints.contains(&key) {
                    protected.push(key);
                }
            }
        }
        let mut ring = vec![a];
        let mut v = a;
        loop {
            v = *boundary.get(&v).context("Broken cavity boundary")?;
            if v == a {
                break;
            }
            ensure!(ring.len() <= boundary.len(), "Cavity boundary cycled");
            ring.push(v);
        }
        let split = ring
            .iter()
            .position(|v| *v == b)
            .context("Constraint endpoint missing from cavity")?;
        let one = ear_clip(&self.points, &ring[..=split])?;
        let mut other = ring[split..].to_vec();
        other.push(a);
        let two = ear_clip(&self.points, &other)?;
        for id in cavity {
            self.remove_face(id);
        }
        for t in one.into_iter().chain(two) {
            self.hint = self.add_face(t)?;
        }
        self.constraints.insert(edge(a, b));
        protected.sort_unstable();
        protected.dedup();
        for (c, d) in protected {
            if !self.edges.contains_key(&edge(c, d)) {
                self.recover(c, d)?;
            }
        }
        ensure!(
            self.edges.contains_key(&edge(a, b)),
            "Constraint recovery lost target"
        );
        Ok(())
    }
    fn legalize(&mut self) -> Result<()> {
        let mut pending: BTreeSet<_> = self
            .edges
            .keys()
            .copied()
            .filter(|e| !self.constraints.contains(e))
            .collect();
        let mut flips = 0;
        while let Some(key) = pending.pop_first() {
            if self.constraints.contains(&key) {
                continue;
            }
            let Some(adj) = self.edges.get(&key) else {
                continue;
            };
            if adj.len() != 2 {
                continue;
            }
            let (one, two) = (adj[0].0, adj[1].0);
            let f = self.faces[one].v;
            let g = self.faces[two].v;
            let c = *f.iter().find(|v| **v != key.0 && **v != key.1).unwrap();
            let d = *g.iter().find(|v| **v != key.0 && **v != key.1).unwrap();
            if !crosses(
                self.points[key.0],
                self.points[key.1],
                self.points[c],
                self.points[d],
            ) || incircle(
                self.points[f[0]],
                self.points[f[1]],
                self.points[f[2]],
                self.points[d],
            ) <= 0
            {
                continue;
            }
            self.remove_face(one);
            self.remove_face(two);
            for t in [[c, d, key.0], [d, c, key.1]] {
                let id = self.add_face(t)?;
                for s in 0..3 {
                    pending.insert(edge(self.faces[id].v[s], self.faces[id].v[(s + 1) % 3]));
                }
            }
            flips += 1;
            ensure!(
                flips <= self.points.len() * self.points.len().max(100),
                "Delaunay legalization did not converge"
            );
        }
        Ok(())
    }
    pub fn data(&self) -> MeshData {
        MeshData {
            points: self.points.clone(),
            triangles: self.faces.iter().filter(|f| f.alive).map(|f| f.v).collect(),
            constraints: self.constraints.iter().map(|e| [e.0, e.1]).collect(),
        }
    }
    pub fn validate(&self) -> Result<()> {
        for key in &self.constraints {
            ensure!(self.edges.contains_key(key), "Missing protected edge");
        }
        for (i, f) in self.faces.iter().enumerate().filter(|(_, f)| f.alive) {
            ensure!(
                orient(
                    self.points[f.v[0]],
                    self.points[f.v[1]],
                    self.points[f.v[2]]
                ) > 0,
                "Inverted triangle"
            );
            for s in 0..3 {
                let key = edge(f.v[s], f.v[(s + 1) % 3]);
                let adj = &self.edges[&key];
                ensure!(adj.contains(&(i, s)), "Invalid edge owner");
                if let Some(n) = f.adj[s] {
                    ensure!(
                        self.faces[n].alive && self.faces[n].adj.contains(&Some(i)),
                        "Invalid neighboring face"
                    );
                }
            }
        }
        Ok(())
    }
}

pub fn ear_clip(points: &[Point], polygon: &[usize]) -> Result<Vec<[usize; 3]>> {
    if polygon.len() < 3 {
        return Ok(Vec::new());
    }
    let mut ring = polygon.to_vec();
    let extreme = *ring
        .iter()
        .min_by(|a, b| {
            points[**a][0]
                .total_cmp(&points[**b][0])
                .then(points[**a][1].total_cmp(&points[**b][1]))
        })
        .unwrap();
    let i = ring.iter().position(|v| *v == extreme).unwrap();
    let orientation = orient(
        points[ring[(i + ring.len() - 1) % ring.len()]],
        points[ring[i]],
        points[ring[(i + 1) % ring.len()]],
    );
    if orientation < 0 {
        ring.reverse();
    }
    let mut output = Vec::new();
    while ring.len() > 3 {
        let n = ring.len();
        let ear = (0..n)
            .find(|i| {
                let (a, b, c) = (ring[(i + n - 1) % n], ring[*i], ring[(i + 1) % n]);
                orient(points[a], points[b], points[c]) > 0
                    && !ring
                        .iter()
                        .filter(|p| **p != a && **p != b && **p != c)
                        .any(|p| {
                            orient(points[a], points[b], points[*p]) >= 0
                                && orient(points[b], points[c], points[*p]) >= 0
                                && orient(points[c], points[a], points[*p]) >= 0
                        })
            })
            .context("Polygon is self-intersecting or degenerate")?;
        output.push([ring[(ear + n - 1) % n], ring[ear], ring[(ear + 1) % n]]);
        ring.remove(ear);
    }
    ensure!(
        orient(points[ring[0]], points[ring[1]], points[ring[2]]) > 0,
        "Degenerate polygon tail"
    );
    output.push([ring[0], ring[1], ring[2]]);
    Ok(output)
}

pub fn contains(rings: &[Vec<Point>], p: Point) -> bool {
    let mut inside = false;
    for ring in rings {
        for (a, b) in ring
            .iter()
            .zip(ring.iter().cycle().skip(1))
            .take(ring.len())
        {
            if (a[1] > p[1]) != (b[1] > p[1])
                && orient(*a, *b, p) == if b[1] > a[1] { 1 } else { -1 }
            {
                inside = !inside;
            }
        }
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_predicates_cover_subnormal_overflow_and_cocircular() {
        assert_eq!(orient([0., 0.], [1., 1.], [2., 2.]), 0);
        assert_eq!(orient([0., 0.], [1e-300, 0.], [0., 1e-300]), 1);
        assert_eq!(orient([0., 0.], [1e300, 0.], [0., 1e300]), 1);
        assert_eq!(incircle([0., 0.], [1., 0.], [1., 1.], [0., 1.]), 0);
        assert_eq!(incircle([0., 0.], [1., 0.], [0., 1.], [0.1, 0.1]), 1);
        assert_eq!(
            incircle([0., 0.], [1e-300, 0.], [0., 1e-300], [1e-301, 1e-301]),
            1
        );
    }
    #[test]
    fn delaunay_insertion_constraints_and_collinear_chains() -> Result<()> {
        let p = vec![
            [0., 0.],
            [8., 0.],
            [8., 8.],
            [0., 8.],
            [1., 3.],
            [3., 2.],
            [4., 4.],
            [6., 5.],
            [4., 0.],
        ];
        let mut mesh = Mesh::new(p)?;
        mesh.validate()?;
        mesh.constrain(0, 2)?;
        mesh.constrain(0, 1)?;
        mesh.validate()?;
        assert!(mesh.constraints.contains(&edge(0, 6)) && mesh.constraints.contains(&edge(6, 2)));
        mesh.insert([2., 3.])?;
        mesh.validate()?;
        let data = mesh.data();
        let area: f64 = data
            .triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| data.points[i]);
                ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5
            })
            .sum();
        assert!((area - 64.).abs() < 1e-10);
        Ok(())
    }
    #[test]
    fn hole_evenodd_and_invalid_inputs() -> Result<()> {
        let rings = vec![
            vec![[0., 0.], [10., 0.], [10., 10.], [0., 10.]],
            vec![[3., 3.], [7., 3.], [7., 7.], [3., 7.]],
        ];
        assert!(contains(&rings, [1., 1.]));
        assert!(!contains(&rings, [5., 5.]));
        assert!(Mesh::new(vec![[0., 0.], [1., 0.], [2., 0.]]).is_err());
        assert!(Mesh::new(vec![[0., 0.], [1., 0.], [f64::NAN, 1.]]).is_err());
        let mut m = Mesh::new(vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]])?;
        m.constrain(0, 2)?;
        assert!(m.constrain(1, 3).is_err());
        Ok(())
    }
    #[test]
    fn constrained_hole_has_exact_area_and_grid_is_complete() -> Result<()> {
        let points = vec![
            [0., 0.],
            [10., 0.],
            [10., 10.],
            [0., 10.],
            [3., 3.],
            [7., 3.],
            [7., 7.],
            [3., 7.],
        ];
        let data = triangulate(Input {
            points,
            constraints: Vec::new(),
            rings: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
        })?;
        let area: f64 = data
            .triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| data.points[i]);
                ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5
            })
            .sum();
        assert!((area - 84.).abs() < 1e-12);
        let points: Vec<_> = (0..9)
            .flat_map(|y| (0..9).map(move |x| [x as f64, y as f64]))
            .collect();
        let mut mesh = Mesh::new(points)?;
        mesh.constrain_all(&[[0, 80], [8, 72]])?;
        mesh.validate()?;
        assert_eq!(mesh.data().triangles.len(), 128);
        Ok(())
    }
    #[test]
    fn arbitrary_corridors_and_incremental_refinement_preserve_constraints() -> Result<()> {
        for seed in 0..16u64 {
            let mut state = seed + 1;
            let mut points = vec![[0., 0.], [100., 0.], [100., 100.], [0., 100.]];
            for _ in 0..60 {
                let mut next = || {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    1. + ((state >> 32) % 98000) as f64 / 1000.
                };
                points.push([next(), next()]);
            }
            let mut mesh = Mesh::new(points)?;
            mesh.constrain(4, 40)?;
            mesh.constrain(4, 55)?;
            mesh.insert([50., 50.])?;
            mesh.validate()?;
            let before = mesh.points.len();
            assert!(mesh.insert([-10., 50.]).is_err());
            assert_eq!(before, mesh.points.len());
        }
        Ok(())
    }
}
