from fractions import Fraction

import pytest

from vecanima.geometry import orient, triangulate_simple


def area2(p):
    return sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(p, p[1:]+p[:1]))


@pytest.mark.parametrize("points", [
    [(0,0),(5,0),(5,5),(0,5)],
    [(0,0),(5,0),(5,5),(3,2),(0,5)],
    [(0,0),(2,0),(5,0),(5,5),(0,5),(0,0)],
    [(0,0),(0,5),(3,2),(5,5),(5,0)],
    [(0,0),(0,0),(10,0),(10,.000001),(0,.000001)],
])
def test_triangulation_preserves_area_and_orientation(points):
    triangles = triangulate_simple(points)
    assert all(orient(*(points[i] for i in t)) == 1 for t in triangles)
    total = sum(area2([points[i] for i in t]) for t in triangles)
    assert total == pytest.approx(abs(area2(points)))


@pytest.mark.parametrize("points", [
    [(0,0),(5,5),(0,5),(5,0)],
    [(0,0),(1,1),(2,2)],
    [(0,0),(1,0)],
    [(0,0),(1,float("inf")),(2,2)],
    [(0,0),(5,0),(5,5),(0,0),(0,5)],
])
def test_invalid_geometry_is_rejected(points):
    with pytest.raises(ValueError):
        triangulate_simple(points)


def test_near_collinear_predicate_matches_exact_arithmetic():
    a, b, c = (0.,0.), (1.,1e-20), (2.,2e-20+1e-35)
    expected = Fraction(b[0])*Fraction(c[1])-Fraction(b[1])*Fraction(c[0])
    assert orient(a,b,c) == (expected > 0)-(expected < 0)
