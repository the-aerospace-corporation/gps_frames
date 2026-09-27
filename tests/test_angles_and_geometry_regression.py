# Copyright (c) 2022 The Aerospace Corporation
"""Regression tests for relative angles, range/azimuth/elevation, and obscuration in gps_frames.

Verifies get_relative_angles across all axis permutations and quadrants,
get_range_azimuth_elevation for cardinal directions and zenith/nadir,
and check_earth_obscuration geometry.
"""

import numpy as np
import pytest

from gps_frames import (
    get_relative_angles,
    get_range_azimuth_elevation,
    get_azimuth_elevation,
    check_earth_obscuration,
)
from gps_frames.basis import Basis
from gps_frames.position import Position, distance
from gps_frames.vectors import UnitVector
from gps_frames.parameters import EarthParam
from gps_time import GPSTime


class TestRelativeAnglesAllPermutations:
    """Verifies get_relative_angles for all combinations of look and reference axes."""

    @pytest.fixture
    def canonical_basis(self):
        t = GPSTime(0, 0)
        origin = Position(np.array([0.0, 0.0, 0.0]), t, "ECEF")
        u1 = UnitVector(np.array([1.0, 0.0, 0.0]), t, "ECEF")
        u2 = UnitVector(np.array([0.0, 1.0, 0.0]), t, "ECEF")
        u3 = UnitVector(np.array([0.0, 0.0, 1.0]), t, "ECEF")
        return Basis(origin, u1, u2, u3)

    @pytest.mark.parametrize(
        "look_axis, ref_axis, cyclic",
        [
            (1, 2, True),   # 1 -> 2 -> 3
            (2, 3, True),   # 2 -> 3 -> 1
            (3, 1, True),   # 3 -> 1 -> 2
            (1, 3, False),  # 1 -> 3 (non-cyclic)
            (2, 1, False),  # 2 -> 1 (non-cyclic)
            (3, 2, False),  # 3 -> 2 (non-cyclic)
        ],
    )
    def test_target_on_look_axis(self, canonical_basis, look_axis, ref_axis, cyclic):
        """Target exactly along look axis must have 0 off-boresight angle."""
        coords = np.zeros(3)
        coords[look_axis - 1] = 100.0  # 100 meters along look axis
        target = Position(coords, GPSTime(0, 0), "ECEF")

        off_bore, angle_ref = get_relative_angles(canonical_basis, target, look_axis, ref_axis)
        assert np.isclose(off_bore, 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "look_axis, ref_axis",
        [(1, 2), (2, 3), (3, 1), (1, 3), (2, 1), (3, 2)],
    )
    def test_target_on_reference_axis(self, canonical_basis, look_axis, ref_axis):
        """Target along reference axis must have off-boresight = pi/2 and angle from ref = 0."""
        coords = np.zeros(3)
        coords[ref_axis - 1] = 50.0
        target = Position(coords, GPSTime(0, 0), "ECEF")

        off_bore, angle_ref = get_relative_angles(canonical_basis, target, look_axis, ref_axis)
        assert np.isclose(off_bore, np.pi / 2, atol=1e-12)
        assert np.isclose(angle_ref, 0.0, atol=1e-12)

    def test_invalid_axis_specifications(self, canonical_basis):
        target = Position(np.array([1, 0, 0]), GPSTime(0, 0), "ECEF")
        # Same axis
        with pytest.raises(ValueError, match="must be different"):
            get_relative_angles(canonical_basis, target, 1, 1)
        # Invalid look axis
        with pytest.raises(ValueError, match="look_axis must be 1, 2, or 3"):
            get_relative_angles(canonical_basis, target, 0, 2)
        with pytest.raises(ValueError, match="look_axis must be 1, 2, or 3"):
            get_relative_angles(canonical_basis, target, 4, 2)
        # Invalid reference axis
        with pytest.raises(ValueError, match="reference_axis must be 1, 2, or 3"):
            get_relative_angles(canonical_basis, target, 1, 0)
        with pytest.raises(ValueError, match="reference_axis must be 1, 2, or 3"):
            get_relative_angles(canonical_basis, target, 1, 5)


class TestRangeAzimuthElevation:
    """Verifies range, azimuth, and elevation against cardinal geometry."""

    @pytest.fixture
    def enu_ground_station(self):
        """Constructs an ENU basis where Axis 0 is East, Axis 1 is North, Axis 2 is Up."""
        t = GPSTime(2100, 0.0)
        origin = Position(np.array([1000.0, 2000.0, 3000.0]), t, "ECEF")
        east = UnitVector(np.array([1.0, 0.0, 0.0]), t, "ECEF")
        north = UnitVector(np.array([0.0, 1.0, 0.0]), t, "ECEF")
        up = UnitVector(np.array([0.0, 0.0, 1.0]), t, "ECEF")
        return Basis(origin, east, north, up)

    def test_cardinal_directions(self, enu_ground_station):
        t = GPSTime(2100, 0.0)
        orig_coords = enu_ground_station.origin.coordinates
        dist = 1000.0

        # Due North: East = 0, North = +dist, Up = 0 -> Az = 0, El = 0
        p_north = Position(orig_coords + np.array([0.0, dist, 0.0]), t, "ECEF")
        rng, az, el = get_range_azimuth_elevation(enu_ground_station, p_north)
        assert np.isclose(rng, dist, atol=1e-10)
        assert np.isclose(az, 0.0, atol=1e-10)
        assert np.isclose(el, 0.0, atol=1e-10)

        # Due East: East = +dist, North = 0, Up = 0 -> Az = pi/2, El = 0
        p_east = Position(orig_coords + np.array([dist, 0.0, 0.0]), t, "ECEF")
        rng, az, el = get_range_azimuth_elevation(enu_ground_station, p_east)
        assert np.isclose(rng, dist, atol=1e-10)
        assert np.isclose(az, np.pi / 2, atol=1e-10)
        assert np.isclose(el, 0.0, atol=1e-10)

        # Due South: East = 0, North = -dist, Up = 0 -> Az = pi (or -pi), El = 0
        p_south = Position(orig_coords + np.array([0.0, -dist, 0.0]), t, "ECEF")
        rng, az, el = get_range_azimuth_elevation(enu_ground_station, p_south)
        assert np.isclose(rng, dist, atol=1e-10)
        assert np.isclose(np.abs(az), np.pi, atol=1e-10)
        assert np.isclose(el, 0.0, atol=1e-10)

        # Due West: East = -dist, North = 0, Up = 0 -> Az = -pi/2, El = 0
        p_west = Position(orig_coords + np.array([-dist, 0.0, 0.0]), t, "ECEF")
        rng, az, el = get_range_azimuth_elevation(enu_ground_station, p_west)
        assert np.isclose(rng, dist, atol=1e-10)
        assert np.isclose(az, -np.pi / 2, atol=1e-10)
        assert np.isclose(el, 0.0, atol=1e-10)

    def test_zenith_and_nadir(self, enu_ground_station):
        t = GPSTime(2100, 0.0)
        orig_coords = enu_ground_station.origin.coordinates
        dist = 5000.0

        # Zenith: Up = +dist -> Elevation = +pi/2 (90 deg)
        p_zenith = Position(orig_coords + np.array([0.0, 0.0, dist]), t, "ECEF")
        rng, az, el = get_range_azimuth_elevation(enu_ground_station, p_zenith)
        assert np.isclose(rng, dist, atol=1e-10)
        assert np.isclose(el, np.pi / 2, atol=1e-10)

        # Nadir: Up = -dist -> Elevation = -pi/2 (-90 deg)
        p_nadir = Position(orig_coords + np.array([0.0, 0.0, -dist]), t, "ECEF")
        rng, az, el = get_range_azimuth_elevation(enu_ground_station, p_nadir)
        assert np.isclose(rng, dist, atol=1e-10)
        assert np.isclose(el, -np.pi / 2, atol=1e-10)

    def test_wrapper_consistency(self, enu_ground_station):
        """get_azimuth_elevation output matches get_range_azimuth_elevation."""
        t = GPSTime(2100, 0.0)
        target = Position(enu_ground_station.origin.coordinates + [100, 200, 300], t, "ECEF")
        r, az1, el1 = get_range_azimuth_elevation(enu_ground_station, target)
        az2, el2 = get_azimuth_elevation(enu_ground_station, target)

        assert az1 == az2
        assert el1 == el2
        assert np.isclose(r, distance(enu_ground_station.origin, target), atol=1e-6)


class TestEarthObscurationRegression:
    """Verifies geometric line of sight and Earth obscuration."""

    def test_clear_zenith_line_of_sight(self):
        t = GPSTime(0, 0)
        # Station on surface at equator (lon=0)
        gs = Position(np.array([EarthParam.r_e, 0.0, 0.0]), t, "ECEF")
        # Satellite directly overhead at 20,200 km altitude
        sat = Position(np.array([EarthParam.r_e + 20200e3, 0.0, 0.0]), t, "ECEF")

        assert check_earth_obscuration(gs, sat)
        assert check_earth_obscuration(sat, gs)

    def test_antipodal_obscuration(self):
        t = GPSTime(0, 0)
        # Station on surface
        gs = Position(np.array([EarthParam.r_e, 0.0, 0.0]), t, "ECEF")
        # Satellite on exact opposite side of Earth
        sat_opp = Position(np.array([-(EarthParam.r_e + 20200e3), 0.0, 0.0]), t, "ECEF")

        assert not check_earth_obscuration(gs, sat_opp)
        assert not check_earth_obscuration(sat_opp, gs)
