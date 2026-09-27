# Copyright (c) 2022 The Aerospace Corporation
"""Regression tests for relative angles, range/azimuth/elevation, and obscuration in gps_frames.

Verifies:
- get_relative_angles across all 6 permutations of look and reference axes
- Cardinal directions (North, East, South, West) and zenith/nadir tracking geometry
- Line-of-sight Earth obscuration checks for clear overhead and antipodal satellite positions
"""

import numpy as np
import pytest

from gps_frames import (
    check_earth_obscuration,
    get_azimuth_elevation,
    get_range_azimuth_elevation,
    get_relative_angles,
)
from gps_frames.basis import Basis
from gps_frames.parameters import EarthParam
from gps_frames.position import Position, distance
from gps_frames.vectors import UnitVector
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
            (1, 2, True),  # 1 -> 2 -> 3
            (2, 3, True),  # 2 -> 3 -> 1
            (3, 1, True),  # 3 -> 1 -> 2
            (1, 3, False),  # 1 -> 3 (non-cyclic)
            (2, 1, False),  # 2 -> 1 (non-cyclic)
            (3, 2, False),  # 3 -> 2 (non-cyclic)
        ],
    )
    def test_target_on_look_axis(self, canonical_basis, look_axis, ref_axis, cyclic):
        """Verify that a target positioned directly along the look axis has zero off-boresight angle.

        Testing:
            get_relative_angles(canonical_basis, target, look_axis, ref_axis).
            When target lies on look_axis, the line-of-sight vector is collinear with look_axis:
                cos(theta) = (LOS . u_look) / ||LOS|| = 1.0 -> theta = 0.0 rad.

        Expected Result:
            off_bore == 0.0 rad within atol=1e-12.
        """
        coords = np.zeros(3)
        coords[look_axis - 1] = 100.0
        target = Position(coords, GPSTime(0, 0), "ECEF")

        off_bore, angle_ref = get_relative_angles(
            canonical_basis, target, look_axis, ref_axis
        )
        assert np.isclose(off_bore, 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "look_axis, ref_axis",
        [(1, 2), (2, 3), (3, 1), (1, 3), (2, 1), (3, 2)],
    )
    def test_target_on_reference_axis(self, canonical_basis, look_axis, ref_axis):
        """Verify that a target along the reference axis has 90 deg off-bore and 0 clock angle.

        Testing:
            When target lies along ref_axis (orthogonal to look_axis):
            1. Off-boresight angle: cos(theta) = 0.0 -> theta = pi/2 rad (90.0 deg)
            2. Clock angle from reference: projected vector is collinear with ref_axis -> phi = 0.0 rad.

        Expected Result:
            off_bore == pi/2 rad (within 1e-12) and angle_ref == 0.0 rad (within 1e-12).
        """
        coords = np.zeros(3)
        coords[ref_axis - 1] = 50.0
        target = Position(coords, GPSTime(0, 0), "ECEF")

        off_bore, angle_ref = get_relative_angles(
            canonical_basis, target, look_axis, ref_axis
        )
        assert np.isclose(off_bore, np.pi / 2, atol=1e-12)
        assert np.isclose(angle_ref, 0.0, atol=1e-12)

    def test_invalid_axis_specifications(self, canonical_basis):
        """Verify exception handling when providing identical or out-of-range axis indices.

        Testing:
            get_relative_angles input validation.

        Expected Result:
            - Identical look and ref axis raises ValueError("must be different").
            - Out-of-bounds indices (< 1 or > 3) raise ValueError.
        """
        target = Position(np.array([1, 0, 0]), GPSTime(0, 0), "ECEF")
        with pytest.raises(ValueError, match="must be different"):
            get_relative_angles(canonical_basis, target, 1, 1)
        with pytest.raises(ValueError, match="look_axis must be 1, 2, or 3"):
            get_relative_angles(canonical_basis, target, 0, 2)
        with pytest.raises(ValueError, match="look_axis must be 1, 2, or 3"):
            get_relative_angles(canonical_basis, target, 4, 2)
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
        """Verify tracking angles for targets along the 4 cardinal directions (N, E, S, W).

        Testing:
            Range, Azimuth, Elevation on ENU horizon plane:
            - North: [East=0, North=+d, Up=0] -> Az = arctan2(0, d) = 0.0 rad,   El = 0.0 rad
            - East:  [East=+d, North=0, Up=0] -> Az = arctan2(d, 0) = pi/2 rad,  El = 0.0 rad
            - South: [East=0, North=-d, Up=0] -> Az = arctan2(0, -d) = +/-pi rad, El = 0.0 rad
            - West:  [East=-d, North=0, Up=0] -> Az = arctan2(-d, 0) = -pi/2 rad, El = 0.0 rad

        Expected Result:
            All cardinal tracking angles match exact values within atol=1e-10 rad.
        """
        t = GPSTime(2100, 0.0)
        orig_coords = enu_ground_station.origin.coordinates
        dist = 1000.0

        # Due North: Az = 0, El = 0
        p_north = Position(orig_coords + np.array([0.0, dist, 0.0]), t, "ECEF")
        rng, az, el = get_range_azimuth_elevation(enu_ground_station, p_north)
        assert np.isclose(rng, dist, atol=1e-10)
        assert np.isclose(az, 0.0, atol=1e-10)
        assert np.isclose(el, 0.0, atol=1e-10)

        # Due East: Az = pi/2 (90 deg), El = 0
        p_east = Position(orig_coords + np.array([dist, 0.0, 0.0]), t, "ECEF")
        rng, az, el = get_range_azimuth_elevation(enu_ground_station, p_east)
        assert np.isclose(rng, dist, atol=1e-10)
        assert np.isclose(az, np.pi / 2, atol=1e-10)
        assert np.isclose(el, 0.0, atol=1e-10)

        # Due South: Az = pi (180 deg), El = 0
        p_south = Position(orig_coords + np.array([0.0, -dist, 0.0]), t, "ECEF")
        rng, az, el = get_range_azimuth_elevation(enu_ground_station, p_south)
        assert np.isclose(rng, dist, atol=1e-10)
        assert np.isclose(np.abs(az), np.pi, atol=1e-10)
        assert np.isclose(el, 0.0, atol=1e-10)

        # Due West: Az = -pi/2 (-90 deg), El = 0
        p_west = Position(orig_coords + np.array([-dist, 0.0, 0.0]), t, "ECEF")
        rng, az, el = get_range_azimuth_elevation(enu_ground_station, p_west)
        assert np.isclose(rng, dist, atol=1e-10)
        assert np.isclose(az, -np.pi / 2, atol=1e-10)
        assert np.isclose(el, 0.0, atol=1e-10)

    def test_zenith_and_nadir(self, enu_ground_station):
        """Verify tracking angles for Zenith (+Up) and Nadir (-Up).

        Testing:
            - Zenith: [0, 0, +d] -> Elevation = arcsin(+d/d) = +pi/2 rad (+90 deg)
            - Nadir:  [0, 0, -d] -> Elevation = arcsin(-d/d) = -pi/2 rad (-90 deg)

        Expected Result:
            Elevation matches +pi/2 and -pi/2 within atol=1e-10 rad.
        """
        t = GPSTime(2100, 0.0)
        orig_coords = enu_ground_station.origin.coordinates
        dist = 5000.0

        p_zenith = Position(orig_coords + np.array([0.0, 0.0, dist]), t, "ECEF")
        rng, az, el = get_range_azimuth_elevation(enu_ground_station, p_zenith)
        assert np.isclose(rng, dist, atol=1e-10)
        assert np.isclose(el, np.pi / 2, atol=1e-10)

        p_nadir = Position(orig_coords + np.array([0.0, 0.0, -dist]), t, "ECEF")
        rng, az, el = get_range_azimuth_elevation(enu_ground_station, p_nadir)
        assert np.isclose(rng, dist, atol=1e-10)
        assert np.isclose(el, -np.pi / 2, atol=1e-10)

    def test_wrapper_consistency(self, enu_ground_station):
        """Verify consistency between get_azimuth_elevation and get_range_azimuth_elevation.

        Testing:
            get_azimuth_elevation() convenience function.

        Expected Result:
            az1 == az2, el1 == el2, and computed range matches Euclidean distance().
        """
        t = GPSTime(2100, 0.0)
        target = Position(
            enu_ground_station.origin.coordinates + [100, 200, 300], t, "ECEF"
        )
        r, az1, el1 = get_range_azimuth_elevation(enu_ground_station, target)
        az2, el2 = get_azimuth_elevation(enu_ground_station, target)

        assert az1 == az2
        assert el1 == el2
        assert np.isclose(r, distance(enu_ground_station.origin, target), atol=1e-6)


class TestEarthObscurationRegression:
    """Verifies geometric line of sight and Earth obscuration."""

    def test_clear_zenith_line_of_sight(self):
        """Verify that line-of-sight to a satellite directly overhead is unobstructed.

        Testing:
            check_earth_obscuration(gs, sat) when sat is directly above gs.

        Expected Result:
            Returns True (visible / unobstructed) symmetrically in both directions.
        """
        t = GPSTime(0, 0)
        gs = Position(np.array([EarthParam.r_e, 0.0, 0.0]), t, "ECEF")
        sat = Position(np.array([EarthParam.r_e + 20200e3, 0.0, 0.0]), t, "ECEF")

        assert check_earth_obscuration(gs, sat)
        assert check_earth_obscuration(sat, gs)

    def test_antipodal_obscuration(self):
        """Verify that line-of-sight through the center of the Earth is obstructed.

        Testing:
            check_earth_obscuration(gs, sat_opp) when gs and sat are on opposite sides of Earth.

        Expected Result:
            Returns False (obscured by Earth) in both directions.
        """
        t = GPSTime(0, 0)
        gs = Position(np.array([EarthParam.r_e, 0.0, 0.0]), t, "ECEF")
        sat_opp = Position(
            np.array([-(EarthParam.r_e + 20200e3), 0.0, 0.0]), t, "ECEF"
        )

        assert not check_earth_obscuration(gs, sat_opp)
        assert not check_earth_obscuration(sat_opp, gs)
