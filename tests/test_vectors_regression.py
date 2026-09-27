# Copyright (c) 2022 The Aerospace Corporation
"""Regression tests for Vector, UnitVector, and SerializeableVector in gps_frames.vectors.

Verifies:
- Linear vector space axioms (commutativity, associativity, additive identity/inverse, scalar distributivity)
- Dot and cross product algebraic identities (symmetry, self-inner product, anticommutativity, Lagrange's identity)
- UnitVector automatic normalization, scaling degradation to Vector, and frame-switch norm preservation
- Automatic frame conversion and frame_time synchronization in cross-frame operations
"""

import numpy as np
import pytest

from gps_frames.vectors import SerializeableVector, UnitVector, Vector
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
        """Verify vector addition commutativity: u + v == v + u.

        Testing:
            Vector.__add__() order independence.

        Expected Result:
            (u + v).coordinates == (v + u).coordinates within atol=1e-12.
        """
        u, v, _ = vectors_abc
        assert np.allclose((u + v).coordinates, (v + u).coordinates, atol=1e-12)

    def test_additive_associativity(self, vectors_abc):
        """Verify vector addition associativity: (u + v) + w == u + (v + w).

        Testing:
            Vector.__add__() grouping independence.

        Expected Result:
            ((u + v) + w).coordinates == (u + (v + w)).coordinates within atol=1e-12.
        """
        u, v, w = vectors_abc
        left = (u + v) + w
        right = u + (v + w)
        assert np.allclose(left.coordinates, right.coordinates, atol=1e-12)

    def test_additive_identity_and_inverse(self, vectors_abc):
        """Verify additive identity u + 0 == u and additive inverse u + (-u) == 0.

        Testing:
            Vector identity and inverse operations.

        Expected Result:
            - u + 0 == u within atol=1e-12
            - u + (-u) == 0 within atol=1e-12
            - u - u == 0 within atol=1e-12
        """
        u, _, _ = vectors_abc
        zero = Vector(np.array([0.0, 0.0, 0.0]), u.frame_time, u.frame)

        assert np.allclose((u + zero).coordinates, u.coordinates, atol=1e-12)

        neg_u = -u
        assert np.allclose((u + neg_u).coordinates, zero.coordinates, atol=1e-12)
        assert np.allclose((u - u).coordinates, zero.coordinates, atol=1e-12)

    def test_scalar_multiplication_distributivity(self, vectors_abc):
        """Verify scalar multiplication distributivity: alpha * (u + v) == alpha * u + alpha * v.

        Testing:
            Vector scalar scaling and vector addition distribution.

        Expected Result:
            ((u + v) * 3.5).coordinates matches ((u * 3.5) + (v * 3.5)).coordinates within atol=1e-12.
        """
        u, v, _ = vectors_abc
        alpha = 3.5

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
        """Verify dot product symmetry, array compatibility, and self inner product.

        Testing:
            u.dot_product(v):
            1. Symmetry: u . v == v . u = 1(4) + 3(-2) + (-5)(-1) = 4 - 6 + 5 = 3.0
            2. Equivalence with array argument u.dot_product(v.coordinates)
            3. Self inner product: u . u == ||u||^2 = 1^2 + 3^2 + (-5)^2 = 1 + 9 + 25 = 35.0

        Expected Result:
            u . v == 3.0 and u . u == 35.0.
        """
        u, v = vector_pair
        assert np.isclose(u.dot_product(v), 3.0, atol=1e-12)
        assert np.isclose(u.dot_product(v), v.dot_product(u), atol=1e-12)
        assert np.isclose(u.dot_product(v.coordinates), u.dot_product(v), atol=1e-12)
        assert np.isclose(u.dot_product(u), u.magnitude**2, atol=1e-12)

    def test_cross_product_properties(self, vector_pair):
        """Verify cross product anticommutativity, orthogonality, and Lagrange's identity.

        Testing:
            u.cross_product(v):
            1. Anticommutativity: u x v == -(v x u)
            2. Orthogonality: (u x v) . u == 0 and (u x v) . v == 0
            3. Lagrange's identity: ||u x v||^2 == ||u||^2 * ||v||^2 - (u . v)^2

        Expected Result:
            For u = [1, 3, -5] and v = [4, -2, -1]:
                u x v = [3(-1) - (-5)(-2), (-5)(4) - 1(-1), 1(-2) - 3(4)]
                      = [-3 - 10, -20 + 1, -2 - 12] = [-13, -19, -14]
            All identities verified within atol=1e-10.
        """
        u, v = vector_pair
        cross_uv = u.cross_product(v)
        cross_vu = v.cross_product(u)

        assert np.allclose(
            cross_uv.coordinates, np.array([-13.0, -19.0, -14.0]), atol=1e-12
        )
        assert np.allclose(cross_uv.coordinates, (-cross_vu).coordinates, atol=1e-12)
        assert np.isclose(cross_uv.dot_product(u), 0.0, atol=1e-12)
        assert np.isclose(cross_uv.dot_product(v), 0.0, atol=1e-12)

        lhs = cross_uv.magnitude**2
        rhs = (u.magnitude**2) * (v.magnitude**2) - (u.dot_product(v) ** 2)
        assert np.isclose(lhs, rhs, atol=1e-10)

    def test_cross_product_parallel_vectors_is_zero(self):
        """Verify that the cross product of parallel vectors is zero: u x (2*u) == 0.

        Testing:
            u = [2, 4, 6] and v = [1, 2, 3] = 0.5 * u.

        Expected Result:
            cross.coordinates == [0.0, 0.0, 0.0].
        """
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
        """Verify that UnitVector constructor always normalizes input coordinates to norm 1.0.

        Testing:
            UnitVector(coords, t, frame) normalizes coordinates by ||coords||.

        Expected Result:
            u.magnitude == 1.0 and np.linalg.norm(u.coordinates) == 1.0 within atol=1e-14.
        """
        t = GPSTime(0, 0)
        u = UnitVector(np.array(coords), t, "ECI")
        assert np.isclose(u.magnitude, 1.0, atol=1e-14)
        assert np.isclose(np.linalg.norm(u.coordinates), 1.0, atol=1e-14)

    def test_switch_frame_maintains_unit_magnitude(self):
        """Verify that UnitVector.switch_frame preserves unit magnitude across rotations.

        Testing:
            Rigid body rotation via switch_frame("ECEF") preserves unit norm.

        Expected Result:
            u.magnitude == 1.0 within atol=1e-14 after frame transformation.
        """
        t = GPSTime(2100, 3600.0)
        u = UnitVector(np.array([1.0, 2.0, 3.0]), t, "ECI")
        u.switch_frame("ECEF")
        assert np.isclose(u.magnitude, 1.0, atol=1e-14)

    def test_scalar_multiplication_yields_vector(self):
        """Verify that scaling a UnitVector degrades the object type to a regular Vector.

        Testing:
            UnitVector.__mul__(scalar): since the magnitude becomes != 1.0, the result
            must be an instance of Vector, not UnitVector.

        Expected Result:
            isinstance(scaled, Vector) is True, isinstance(scaled, UnitVector) is False,
            and scaled.magnitude == 42.5.
        """
        t = GPSTime(0, 0)
        u = UnitVector(np.array([0.0, 1.0, 0.0]), t, "ECEF")
        scaled = u * 42.5
        assert isinstance(scaled, Vector)
        assert not isinstance(scaled, UnitVector)
        assert np.isclose(scaled.magnitude, 42.5, atol=1e-12)


class TestVectorCrossFrameOperations:
    """Verifies that vector addition and dot products handle cross-frame operands transparently."""

    def test_cross_frame_addition(self):
        """Verify adding an ECI vector to an ECEF vector automatically converts the second operand.

        Testing:
            Vector.__add__() frame alignment at t=0.

        Expected Result:
            v_ecef = [10, 0, 0] + v_eci = [0, 20, 0] yields [10.0, 20.0, 0.0] in ECEF.
        """
        t = GPSTime(2100, 0.0)
        v_ecef = Vector(np.array([10.0, 0.0, 0.0]), t, "ECEF")
        v_eci = Vector(np.array([0.0, 20.0, 0.0]), t, "ECI")

        res = v_ecef + v_eci
        assert res.frame == "ECEF"
        assert np.allclose(res.coordinates, [10.0, 20.0, 0.0], atol=1e-10)

    def test_cross_time_addition(self):
        """Verify adding vectors with different frame_times synchronizes epochs to the first vector.

        Testing:
            Vector.__add__() epoch synchronization.

        Expected Result:
            res.frame_time == t1.
        """
        t1 = GPSTime(2100, 0.0)
        t2 = GPSTime(2100, 3600.0)
        v1 = Vector(np.array([100.0, 0.0, 0.0]), t1, "ECEF")
        v2 = Vector(np.array([0.0, 100.0, 0.0]), t2, "ECEF")

        res = v1 + v2
        assert res.frame_time == t1

    def test_invalid_type_raises(self):
        """Verify that multiplying or taking dot products with invalid types raises TypeError.

        Testing:
            Type checking in Vector.__mul__() and Vector.dot_product().

        Expected Result:
            TypeError raised for string operands.
        """
        t = GPSTime(0, 0)
        v = Vector(np.array([1, 2, 3]), t, "ECI")
        with pytest.raises(TypeError, match="must be a float"):
            _ = v * "invalid"
        with pytest.raises(TypeError, match="other must be a Vector"):
            v.dot_product("invalid")
