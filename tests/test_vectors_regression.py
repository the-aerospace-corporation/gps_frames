# Copyright (c) 2022 The Aerospace Corporation
"""Regression tests for Vector, UnitVector, and SerializeableVector in gps_frames.vectors.

Verifies vector space axioms, inner/cross product identities, frame cross-operations,
UnitVector normalizations, and type validation.
"""

import numpy as np
import pytest

from gps_frames.vectors import Vector, UnitVector, SerializeableVector
from gps_time import GPSTime


class TestVectorSpaceAxioms:
    """Verifies that Vector arithmetic strictly satisfies the axioms of a linear vector space."""

    @pytest.fixture
    def vectors_abc(self):
        t = GPSTime(2100, 0.0)
        u = Vector(np.array([12.3, -45.6, 78.9]), t, "ECEF")
        v = Vector(np.array([-98.7, 65.4, -32.1]), t, "ECEF")
        w = Vector(np.array([10.0, 20.0, -30.0]), t, "ECEF")
        return u, v, w

    def test_additive_commutativity(self, vectors_abc):
        u, v, _ = vectors_abc
        assert np.allclose((u + v).coordinates, (v + u).coordinates, atol=1e-12)

    def test_additive_associativity(self, vectors_abc):
        u, v, w = vectors_abc
        left = (u + v) + w
        right = u + (v + w)
        assert np.allclose(left.coordinates, right.coordinates, atol=1e-12)

    def test_additive_identity_and_inverse(self, vectors_abc):
        u, _, _ = vectors_abc
        zero = Vector(np.array([0.0, 0.0, 0.0]), u.frame_time, u.frame)

        # u + 0 == u
        assert np.allclose((u + zero).coordinates, u.coordinates, atol=1e-12)

        # u + (-u) == 0
        neg_u = -u
        assert np.allclose((u + neg_u).coordinates, zero.coordinates, atol=1e-12)

        # u - u == 0
        assert np.allclose((u - u).coordinates, zero.coordinates, atol=1e-12)

    def test_scalar_multiplication_distributivity(self, vectors_abc):
        u, v, _ = vectors_abc
        alpha = 3.5

        # alpha * (u + v) == alpha * u + alpha * v
        left = (u + v) * alpha
        right = (u * alpha) + (v * alpha)
        assert np.allclose(left.coordinates, right.coordinates, atol=1e-12)


class TestDotAndCrossProducts:
    """Verifies algebraic and geometric properties of dot and cross products."""

    @pytest.fixture
    def vector_pair(self):
        t = GPSTime(2100, 0.0)
        u = Vector(np.array([1.0, 3.0, -5.0]), t, "ECI")
        v = Vector(np.array([4.0, -2.0, -1.0]), t, "ECI")
        return u, v

    def test_dot_product_properties(self, vector_pair):
        u, v = vector_pair
        # Symmetry
        assert np.isclose(u.dot_product(v), v.dot_product(u), atol=1e-12)
        # With numpy array
        assert np.isclose(u.dot_product(v.coordinates), u.dot_product(v), atol=1e-12)
        # Self dot product == magnitude^2
        assert np.isclose(u.dot_product(u), u.magnitude ** 2, atol=1e-12)

    def test_cross_product_properties(self, vector_pair):
        u, v = vector_pair
        cross_uv = u.cross_product(v)
        cross_vu = v.cross_product(u)

        # Anticommutativity: u x v == -(v x u)
        assert np.allclose(cross_uv.coordinates, (-cross_vu).coordinates, atol=1e-12)

        # Orthogonality: (u x v) . u == 0 and (u x v) . v == 0
        assert np.isclose(cross_uv.dot_product(u), 0.0, atol=1e-12)
        assert np.isclose(cross_uv.dot_product(v), 0.0, atol=1e-12)

        # Lagrange's identity: ||u x v||^2 == ||u||^2 * ||v||^2 - (u . v)^2
        lhs = cross_uv.magnitude ** 2
        rhs = (u.magnitude ** 2) * (v.magnitude ** 2) - (u.dot_product(v) ** 2)
        assert np.isclose(lhs, rhs, atol=1e-10)

    def test_cross_product_parallel_vectors_is_zero(self):
        t = GPSTime(0, 0)
        u = Vector(np.array([2.0, 4.0, 6.0]), t, "ECEF")
        v = Vector(np.array([1.0, 2.0, 3.0]), t, "ECEF")
        cross = u.cross_product(v)
        assert np.allclose(cross.coordinates, [0.0, 0.0, 0.0], atol=1e-12)


class TestUnitVectorInvariants:
    """Verifies UnitVector normalization and properties."""

    @pytest.mark.parametrize(
        "coords",
        [
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-50.0, 120.0, -30.0],
            [1e-4, 2e-4, -3e-4],
        ],
    )
    def test_always_unit_magnitude(self, coords):
        t = GPSTime(0, 0)
        u = UnitVector(np.array(coords), t, "ECI")
        assert np.isclose(u.magnitude, 1.0, atol=1e-14)
        assert np.isclose(np.linalg.norm(u.coordinates), 1.0, atol=1e-14)

    def test_switch_frame_maintains_unit_magnitude(self):
        t = GPSTime(2100, 3600.0)
        u = UnitVector(np.array([1.0, 2.0, 3.0]), t, "ECI")
        u.switch_frame("ECEF")
        assert np.isclose(u.magnitude, 1.0, atol=1e-14)

    def test_scalar_multiplication_yields_vector(self):
        t = GPSTime(0, 0)
        u = UnitVector(np.array([0.0, 1.0, 0.0]), t, "ECEF")
        scaled = u * 42.5
        assert isinstance(scaled, Vector)
        assert not isinstance(scaled, UnitVector)
        assert np.isclose(scaled.magnitude, 42.5, atol=1e-12)


class TestVectorCrossFrameOperations:
    """Verifies that vector addition and dot products handle cross-frame operands transparently."""

    def test_cross_frame_addition(self):
        t = GPSTime(2100, 0.0)  # aligned at t=0
        v_ecef = Vector(np.array([10.0, 0.0, 0.0]), t, "ECEF")
        v_eci = Vector(np.array([0.0, 20.0, 0.0]), t, "ECI")

        # Result is in v_ecef's frame (ECEF)
        res = v_ecef + v_eci
        assert res.frame == "ECEF"
        assert np.allclose(res.coordinates, [10.0, 20.0, 0.0], atol=1e-10)

    def test_cross_time_addition(self):
        t1 = GPSTime(2100, 0.0)
        t2 = GPSTime(2100, 3600.0)
        v1 = Vector(np.array([100.0, 0.0, 0.0]), t1, "ECEF")
        v2 = Vector(np.array([0.0, 100.0, 0.0]), t2, "ECEF")

        # v2 should be rotated to t1's ECEF frame before adding
        res = v1 + v2
        assert res.frame_time == t1

    def test_invalid_type_raises(self):
        t = GPSTime(0, 0)
        v = Vector(np.array([1, 2, 3]), t, "ECI")
        with pytest.raises(TypeError, match="must be a float"):
            _ = v * "invalid"
        with pytest.raises(TypeError, match="other must be a Vector"):
            v.dot_product("invalid")
