"""Independent simple-polygon triangulation baseline, without CGAL.

Algorithm reference: David Eberly, Triangulation by Ear Clipping (Geometric Tools).
Numerical motivation: Jonathan Shewchuk, Robust Predicates (1996/1997).
This implementation uses Fraction fallback, not Shewchuk expansion arithmetic.
Holes and constrained Delaunay refinement are intentionally not implemented yet.
"""
from fractions import Fraction
import math
import sys


def orient(a, b, c):
    """Return -1, 0 or 1; uncertain floating determinants use exact binary rationals."""
    ax, ay = a[0] - c[0], a[1] - c[1]
    bx, by = b[0] - c[0], b[1] - c[1]
    left, right = ax * by, ay * bx
    det = left - right
    bound = 8 * sys.float_info.epsilon * (abs(left) + abs(right))
    if math.isfinite(det) and abs(det) > bound:
        return 1 if det > 0 else -1
    aa, bb, cc = [tuple(Fraction(x) for x in p) for p in (a, b, c)]
    exact = (aa[0]-cc[0])*(bb[1]-cc[1]) - (aa[1]-cc[1])*(bb[0]-cc[0])
    return (exact > 0) - (exact < 0)


def on_segment(a, b, p):
    return orient(a, b, p) == 0 and all(min(a[k], b[k]) <= p[k] <= max(a[k], b[k]) for k in (0, 1))


def intersects(a, b, c, d):
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0 and o3 * o4 < 0) or any((
        o1 == 0 and on_segment(a, b, c), o2 == 0 and on_segment(a, b, d),
        o3 == 0 and on_segment(c, d, a), o4 == 0 and on_segment(c, d, b)))


def triangulate_simple(points):
    """Return CCW triangles as original vertex indices, rejecting self-intersections.

    Adjacent duplicates, a closing duplicate and collinear intermediate vertices
    are removed. Input must be a finite simple ring; holes require a future API.
    """
    p = [tuple(map(float, v)) for v in points]
    if any(len(v) != 2 or not all(math.isfinite(x) for x in v) for v in p):
        raise ValueError("Expected finite 2D vertices")
    active = []
    for i in range(len(p)):
        if not active or p[i] != p[active[-1]]:
            active.append(i)
    if len(active) > 1 and p[active[0]] == p[active[-1]]:
        active.pop()
    if len(set(p[i] for i in active)) != len(active):
        raise ValueError("Non-adjacent repeated vertex")
    changed = True
    while changed and len(active) > 3:
        changed = False
        for j, i in enumerate(active):
            if on_segment(p[active[j-1]], p[active[(j+1) % len(active)]], p[i]):
                active.pop(j)
                changed = True
                break
    if len(active) < 3:
        raise ValueError("Polygon has fewer than three distinct vertices")
    n = len(active)
    for j in range(n):
        for k in range(j+1, n):
            if k == j+1 or (j == 0 and k == n-1):
                continue
            if intersects(p[active[j]], p[active[(j+1) % n]], p[active[k]], p[active[(k+1) % n]]):
                raise ValueError("Polygon is not simple")
    # Exact area avoids losing winding when coordinates have a large offset.
    area = sum(Fraction(p[active[j]][0])*Fraction(p[active[(j+1) % n]][1])
               - Fraction(p[active[j]][1])*Fraction(p[active[(j+1) % n]][0]) for j in range(n))
    if area == 0:
        raise ValueError("Polygon has zero area")
    if area < 0:
        active.reverse()
    triangles = []
    while len(active) > 3:
        for j, b in enumerate(active):
            a, c = active[j-1], active[(j+1) % len(active)]
            if orient(p[a], p[b], p[c]) <= 0:
                continue
            blocked = any(all(orient(p[u], p[v], p[k]) >= 0 for u, v in ((a,b), (b,c), (c,a)))
                          for k in active if k not in (a,b,c))
            if not blocked:
                triangles.append((a,b,c))
                active.pop(j)
                break
        else:
            raise ValueError("Cannot find a valid ear")
    triangles.append(tuple(active))
    return triangles
