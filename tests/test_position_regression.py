# Copyright (c) 2022 The Aerospace Corporation
"""Regression tests for Position class and distance calculations in gps_frames.position.

Verifies:
- Metric space properties of distance(): identity of indiscernibles, non-negativity, symmetry, and triangle inequality
- Invariance of physical separation across coordinate frames {ECEF, ECI, LLA}
- WGS84 ellipsoidal geometry: equatorial radius vs polar radius
- Exact algebraic relations between HAE, MSL, and spherical altitude
- Vector addition/subtraction affine point arithmetic
- State management: deep copy in get_position() vs in-place mutation in switch_frame()
- YAML serialization round-trip
"""

import copy
from io import StringIO
import numpy as np
import pytest
import ruamel.yaml

from gps_frames.parameters import EarthParam, GeoidData
from gps_frames.position import Position, distance
from gps_frames.vectors import Vector
from gps_time import GPSTime


class TestDistanceMetricProperties:
    """Verifies that distance() satisfies metric space axioms across different frames and times."""

    @pytest.fixture
    def sample_positions(self):
        t1 = GPSTime(2100, 1000.0)
        t2 = GPSTime(2100, 5000.0)
        pA = Position(np.array([EarthParam.r_e, 0.0, 0.0]), t1, "ECEF")
        pB = Position(np.array([0.0, EarthParam.r_e + 500e3, 0.0]), t1, "ECI")
        pC = Position(np.array([np.pi / 4, -np.pi / 3, 10000.0]), t2, "LLA")
        return pA, pB, pC

    def test_identity_and_non_negativity(self, sample_positions):
        """Verify metric space axioms: d(A, A) == 0 and d(A, B) > 0 for distinct points.

        Testing:
            Non-negativity and identity of indiscernibles for distance().

        Expected Result:
            - distance(pA, pA) == 0.0 and distance(pB, pB) == 0.0
            - distance(pA, pB) > 0.0
        """
        pA, pB, _ = sample_positions
        assert np.isclose(distance(pA, pA), 0.0, atol=1e-12)
        assert np.isclose(distance(pB, pB), 0.0, atol=1e-12)
        assert distance(pA, pB) > 0.0

    def test_symmetry(self, sample_positions):
        """Verify metric space symmetry: d(A, B) == d(B, A).

        Testing:
            Commutativity of distance() across mixed frames (ECEF, ECI, LLA) and epochs.

        Expected Result:
            distance(pA, pB) == distance(pB, pA) within atol=1e-10.
        """
        pA, pB, pC = sample_positions
        assert np.isclose(distance(pA, pB), distance(pB, pA), atol=1e-10)
        assert np.isclose(distance(pA, pC), distance(pC, pA), atol=1e-10)
        assert np.isclose(distance(pB, pC), distance(pC, pB), atol=1e-10)

    def test_triangle_inequality(self, sample_positions):
        """Verify metric space triangle inequality: d(A, C) <= d(A, B) + d(B, C).

        Testing:
            Geometric triangle inequality in 3D Euclidean space.

        Expected Result:
            distance(pA, pC) <= distance(pA, pB) + distance(pB, pC).
        """
        pA, pB, pC = sample_positions
        d_AC = distance(pA, pC)
        d_AB = distance(pA, pB)
        d_BC = distance(pB, pC)

        assert d_AC <= d_AB + d_BC + 1e-9

    def test_distance_frame_invariance(self):
        """Verify that physical separation is invariant under change of coordinate frames.

        Testing:
            distance(p1, p2) computed with mixed frames (ECEF vs ECI), both in ECEF,
            both in ECI, and both in LLA.

        Expected Result:
            All 4 evaluations yield identical distance within atol=1e-6 m (1 um).
        """
        t = GPSTime(2150, 43200.0)
        p1 = Position(np.array([EarthParam.r_e, 10000.0, 5000.0]), t, "ECEF")
        p2 = Position(np.array([0.0, EarthParam.r_e + 20200e3, 0.0]), t, "ECI")

        d_mixed = distance(p1, p2)
        d_both_ecef = distance(p1.get_position("ECEF"), p2.get_position("ECEF"))
        d_both_eci = distance(p1.get_position("ECI"), p2.get_position("ECI"))
        d_both_lla = distance(p1.get_position("LLA"), p2.get_position("LLA"))

        assert np.isclose(d_mixed, d_both_ecef, atol=1e-6)
        assert np.isclose(d_mixed, d_both_eci, atol=1e-6)
        assert np.isclose(d_mixed, d_both_lla, atol=1e-6)


class TestPositionAltitudesAndRadii:
    """Verifies relationships between HAE, MSL, spherical altitude, and Earth radius."""

    def test_equator_vs_pole_radius(self):
        """Verify oblate spheroidal Earth geometry: equatorial radius > polar radius.

        Testing:
            Geocentric radius at equator (lat = 0) vs North pole (lat = pi/2) at HAE = 0:
                R_equator = EarthParam.wgs84a = 6378137.0 m
                R_pole = EarthParam.wgs84b ~= 6356752.3142 m

        Expected Result:
            r_eq == 6378137.0 m, r_pole == 6356752.3142 m, and r_eq - r_pole ~= 21384.6858 m.
        """
        t = GPSTime(0, 0)
        pos_equator = Position(np.array([0.0, 0.0, 0.0]), t, "LLA")
        pos_pole = Position(np.array([np.pi / 2, 0.0, 0.0]), t, "LLA")

        r_eq = pos_equator.get_radius()
        r_pole = pos_pole.get_radius()

        assert np.isclose(r_eq, EarthParam.wgs84a, atol=1e-3)
        assert np.isclose(r_pole, EarthParam.wgs84b, atol=1e-3)
        assert r_eq > r_pole

    def test_altitude_relationships(self):
        """Verify mathematical relationships between HAE, MSL, and spherical altitude.

        Testing:
            For a point at lat = 34 deg N, lon = -118 deg W with HAE = 500 m:
            1. get_altitude_hae() == 500.0 m
            2. get_altitude_msl() == 500.0 - GeoidData.get_geoid_height(lat, lon)
            3. get_altitude_spherical() == pos.get_radius() - EarthParam.r_e

        Expected Result:
            All altitude values match their exact theoretical formulations.
        """
        t = GPSTime(2000, 3600.0)
        lat = np.deg2rad(34.0)
        lon = np.deg2rad(-118.0)
        hae = 500.0
        pos = Position(np.array([lat, lon, hae]), t, "LLA")

        # 1. HAE check
        assert np.isclose(pos.get_altitude_hae(), hae, atol=1e-6)

        # 2. MSL check: HAE - geoid_height
        geoid_h = GeoidData.get_geoid_height(lat, lon, units="rad")
        expected_msl = hae - geoid_h
        assert np.isclose(pos.get_altitude_msl(), expected_msl, atol=1e-6)

        # 3. Spherical altitude check: radius - r_e
        expected_spherical = pos.get_radius() - EarthParam.r_e
        assert np.isclose(pos.get_altitude_spherical(), expected_spherical, atol=1e-6)


class TestPositionVectorArithmetic:
    """Verifies adding and subtracting vectors to/from Positions."""

    def test_add_subtract_round_trip(self):
        """Verify affine arithmetic round-trip: (Pos + Vec) - Vec == Pos.

        Testing:
            Position translation by displacement Vector and inverse translation.

        Expected Result:
            pos_restored == pos with coordinates matching within atol=1e-12.
        """
        t = GPSTime(2100, 0.0)
        pos = Position(np.array([6378137.0, 500.0, -1000.0]), t, "ECEF")
        vec = Vector(np.array([100.0, -250.0, 400.0]), t, "ECEF")

        pos_shifted = pos + vec
        pos_restored = pos_shifted - vec

        assert isinstance(pos_shifted, Position)
        assert isinstance(pos_restored, Position)
        assert np.allclose(pos_restored.coordinates, pos.coordinates, atol=1e-12)
        assert pos_restored == pos

    def test_add_zero_vector(self):
        """Verify identity property: Pos + 0 == Pos.

        Testing:
            Adding a null vector leaves position coordinates unmodified.

        Expected Result:
            pos_new.coordinates == [1.0, 2.0, 3.0].
        """
        t = GPSTime(2100, 0.0)
        pos = Position(np.array([1.0, 2.0, 3.0]), t, "ECI")
        zero_vec = Vector(np.array([0.0, 0.0, 0.0]), t, "ECI")

        pos_new = pos + zero_vec
        assert np.allclose(pos_new.coordinates, pos.coordinates, atol=1e-14)

    def test_add_vector_with_frame_conversion(self):
        """Verify automatic coordinate frame conversion when adding an ECI vector to an ECEF position.

        Testing:
            Vector is transformed to match Position.frame prior to coordinate addition.

        Expected Result:
            At t=0 (ECI and ECEF aligned):
                pos = [10.0, 20.0, 30.0], vec = [1.0, 2.0, 3.0]
                res = [10+1, 20+2, 30+3] = [11.0, 22.0, 33.0] in ECEF.
        """
        t = GPSTime(2100, 0.0)
        pos = Position(np.array([10.0, 20.0, 30.0]), t, "ECEF")
        vec = Vector(np.array([1.0, 2.0, 3.0]), t, "ECI")

        res = pos + vec
        assert res.frame == "ECEF"
        assert np.allclose(res.coordinates, [11.0, 22.0, 33.0], atol=1e-10)

    def test_invalid_arithmetic_raises(self):
        """Verify that invalid arithmetic operations (Pos + Pos, Pos - Pos, Pos + scalar) raise TypeError.

        Testing:
            Type checking in Position.__add__() and Position.__sub__().

        Expected Result:
            TypeError("must be a vector") raised for all invalid operands.
        """
        t = GPSTime(0, 0)
        pos = Position(np.array([1, 2, 3]), t, "ECI")
        with pytest.raises(TypeError, match="must be a vector"):
            _ = pos + pos
        with pytest.raises(TypeError, match="must be a vector"):
            _ = pos - pos
        with pytest.raises(TypeError, match="must be a vector"):
            _ = pos + 42


class TestPositionStateManagement:
    """Verifies immutability of get_position vs mutability of switch_frame."""

    def test_get_position_creates_copy(self):
        """Verify that get_position() returns a new Position instance leaving the original intact.

        Testing:
            Non-destructive frame conversion via get_position().

        Expected Result:
            pos_ecef.frame == "ECEF", pos_eci.frame == "ECI", and pos_ecef is not pos_eci.
        """
        t = GPSTime(0, 0)
        pos_ecef = Position(np.array([EarthParam.r_e, 0.0, 0.0]), t, "ECEF")
        pos_eci = pos_ecef.get_position("ECI")

        assert pos_ecef.frame == "ECEF"
        assert pos_eci.frame == "ECI"
        assert pos_ecef is not pos_eci

    def test_switch_frame_in_place(self):
        """Verify that switch_frame() modifies the Position instance in place.

        Testing:
            In-place frame conversion via switch_frame().

        Expected Result:
            pos.frame changes to "LLA" and coordinates convert to [0.0 rad, 0.0 rad, 0.0 m].
        """
        t = GPSTime(0, 0)
        pos = Position(np.array([EarthParam.r_e, 0.0, 0.0]), t, "ECEF")
        pos.switch_frame("LLA")
        assert pos.frame == "LLA"
        assert np.isclose(pos.coordinates[0], 0.0, atol=1e-8)
        assert np.isclose(pos.coordinates[1], 0.0, atol=1e-8)

    def test_update_frame_time_multi_epoch(self):
        """Verify multi-week time updates on ECI positions preserve radial magnitude.

        Testing:
            update_frame_time() across 5 weeks in ECI frame.

        Expected Result:
            pos_eci.frame_time == t2 and radial distance norm ||r|| is invariant.
        """
        t1 = GPSTime(100, 0.0)
        t2 = GPSTime(105, 0.0)

        pos_eci = Position(np.array([1000.0, 2000.0, 3000.0]), t1, "ECI")
        pos_eci.update_frame_time(t2)

        assert pos_eci.frame_time == t2
        assert np.isclose(
            np.linalg.norm(pos_eci.coordinates), np.linalg.norm([1000.0, 2000.0, 3000.0])
        )


class TestPositionSerialization:
    """Verifies YAML serialization and deserialization of Position."""

    def test_yaml_round_trip(self):
        """Verify that Position serializes to YAML and deserializes with identical values.

        Testing:
            YAML round-trip via ruamel.yaml.

        Expected Result:
            Loaded Position matches original in frame, frame_time, and coordinates.
        """
        yaml = ruamel.yaml.YAML()
        yaml.register_class(Position)

        t = GPSTime(2150, 12345.0)
        pos = Position(np.array([1234567.89, -987654.32, 456789.01]), t, "ECEF")

        stream = StringIO()
        yaml.dump(pos, stream)
        yaml_str = stream.getvalue()

        assert "!SerializeableVector.Position" in yaml_str
        assert "frame: ECEF" in yaml_str

        loaded = yaml.load(yaml_str)
        assert isinstance(loaded, Position)
        assert loaded.frame == pos.frame
        assert loaded.frame_time == pos.frame_time
        assert np.allclose(loaded.coordinates, pos.coordinates, atol=1e-6)
        assert loaded == pos
