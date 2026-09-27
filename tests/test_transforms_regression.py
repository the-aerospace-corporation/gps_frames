# Copyright (c) 2022 The Aerospace Corporation
"""Regression tests for coordinate and velocity transformations in gps_frames.transforms.

Verifies LLA <-> ECEF <-> ECI conversions across a global grid, polar/equatorial boundaries,
extreme altitudes, time evolution, and velocity transformations.
"""

import numpy as np
import pytest
from gps_frames import transforms as trans
from gps_frames.parameters import EarthParam
from gps_time import GPSTime


class TestLlaEcefRoundTrips:
    """Verifies that converting between LLA and ECEF is invertible across global coordinates."""

    @pytest.mark.parametrize(
        "lat_deg",
        [-89.999, -75.0, -45.0, -30.0, 0.0, 30.0, 45.0, 75.0, 89.999],
    )
    @pytest.mark.parametrize(
        "lon_deg",
        [-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0],
    )
    @pytest.mark.parametrize(
        "alt_m",
        [-400.0, 0.0, 1000.0, 10000.0, 400000.0, 20200000.0, 35786000.0],
    )
    def test_lla_to_ecef_to_lla_grid(self, lat_deg, lon_deg, alt_m):
        """Global grid of coordinates must round-trip with high precision."""
        lat_rad = np.deg2rad(lat_deg)
        lon_rad = np.deg2rad(lon_deg)
        orig_lla = np.array([lat_rad, lon_rad, alt_m], dtype=float)

        ecef = trans.lla2ecef(orig_lla)
        rec_lla = trans.ecef2lla(ecef)

        # Latitude within 1e-11 rad (~sub-millimeter surface error)
        assert np.isclose(rec_lla[0], orig_lla[0], atol=1e-10)

        # Longitude: handle wraparound at +/- pi
        diff_lon = np.arctan2(np.sin(rec_lla[1] - orig_lla[1]), np.cos(rec_lla[1] - orig_lla[1]))
        assert np.isclose(diff_lon, 0.0, atol=1e-10)

        # Altitude within 1 mm (1e-3 m)
        assert np.isclose(rec_lla[2], orig_lla[2], atol=1e-3)

    def test_exact_poles(self):
        """At North and South poles, X and Y must be 0 and Z must match ellipsoid semiminor axis."""
        b = EarthParam.wgs84b

        # North Pole
        north_lla = np.array([np.pi / 2, 0.0, 0.0])
        north_ecef = trans.lla2ecef(north_lla)
        assert np.isclose(north_ecef[0], 0.0, atol=1e-6)
        assert np.isclose(north_ecef[1], 0.0, atol=1e-6)
        assert np.isclose(north_ecef[2], b, atol=1e-6)

        rec_north = trans.ecef2lla(north_ecef)
        assert np.isclose(rec_north[0], np.pi / 2, atol=1e-10)
        assert np.isclose(rec_north[2], 0.0, atol=1e-3)

        # South Pole
        south_lla = np.array([-np.pi / 2, 0.0, 0.0])
        south_ecef = trans.lla2ecef(south_lla)
        assert np.isclose(south_ecef[0], 0.0, atol=1e-6)
        assert np.isclose(south_ecef[1], 0.0, atol=1e-6)
        assert np.isclose(south_ecef[2], -b, atol=1e-6)

        rec_south = trans.ecef2lla(south_ecef)
        assert np.isclose(rec_south[0], -np.pi / 2, atol=1e-10)
        assert np.isclose(rec_south[2], 0.0, atol=1e-3)

    def test_exact_equator(self):
        """At equator (lat=0, alt=0), radius is exactly semimajor axis a."""
        a = EarthParam.wgs84a

        # Lon = 0
        eq0_ecef = trans.lla2ecef(np.array([0.0, 0.0, 0.0]))
        assert np.allclose(eq0_ecef, [a, 0.0, 0.0], atol=1e-6)

        # Lon = 90 deg (pi/2)
        eq90_ecef = trans.lla2ecef(np.array([0.0, np.pi / 2, 0.0]))
        assert np.allclose(eq90_ecef, [0.0, a, 0.0], atol=1e-6)

        # Lon = 180 deg (pi)
        eq180_ecef = trans.lla2ecef(np.array([0.0, np.pi, 0.0]))
        assert np.allclose(eq180_ecef, [-a, 0.0, 0.0], atol=1e-6)


class TestEcefEciTransformations:
    """Verifies time-dependent Earth rotation transformations between ECEF and ECI."""

    def test_ecef_eci_round_trip(self):
        """ECEF <-> ECI conversions must be perfectly reversible at any time of week."""
        ecef_orig = np.array([4500000.0, -1200000.0, 4300000.0])
        tows = [0.0, 100.0, 3600.0, 43200.0, 86400.0, 604799.0]

        for tow in tows:
            eci = trans.ecef2eci(ecef_orig, tow)
            rec_ecef = trans.eci2ecef(eci, tow)
            assert np.allclose(rec_ecef, ecef_orig, atol=1e-12)

    def test_sidereal_day_rotation(self):
        """After one full sidereal day (2*pi / w_e seconds), ECEF and ECI orientation repeats."""
        sidereal_period = 2 * np.pi / EarthParam.w_e
        pos = np.array([EarthParam.r_e, 1000.0, 5000.0])

        eci_0 = trans.ecef2eci(pos, 0.0)
        eci_1period = trans.ecef2eci(pos, sidereal_period)
        assert np.allclose(eci_1period, eci_0, atol=1e-6)

    def test_quarter_rotation(self):
        """After a quarter turn (angle = pi/2), X axis rotates to -Y in passive rotation."""
        quarter_period = (np.pi / 2) / EarthParam.w_e
        pos_ecef = np.array([1.0, 0.0, 0.0])

        # At t = quarter_period, angle = pi/2. standard_rotation(3, -pi/2, [1,0,0])
        # R_3(-pi/2) rotates [1,0,0] to [0, 1, 0]
        pos_eci = trans.ecef2eci(pos_ecef, quarter_period)
        assert np.allclose(pos_eci, [0.0, 1.0, 0.0], atol=1e-14)

    def test_rotate_ecef_time_shifts(self):
        """rotate_ecef shifts coordinates by omega_e * delta_t."""
        t1 = GPSTime(2100, 1000.0)
        t2 = GPSTime(2100, 2000.0)  # delta = 1000 s
        coords = np.array([6378137.0, 0.0, 0.0])

        rotated = trans.rotate_ecef(t1, t2, coords)
        # Angle = omega_e * 1000
        expected_angle = EarthParam.w_e * 1000.0
        expected = trans.standard_rotation(3, expected_angle, coords)
        assert np.allclose(rotated, expected, atol=1e-14)

        # Reverse shift restores coordinates
        rev_rotated = trans.rotate_ecef(t2, t1, rotated)
        assert np.allclose(rev_rotated, coords, atol=1e-8)

    def test_add_weeks_eci(self):
        """add_weeks_eci advances ECI frame by integer weeks."""
        coords = np.array([1000.0, 2000.0, 3000.0])

        # 0 weeks is identity
        assert np.allclose(trans.add_weeks_eci(0, coords), coords)

        # 2 weeks forward then 2 weeks backward
        fwd = trans.add_weeks_eci(2, coords)
        back = trans.add_weeks_eci(-2, fwd)
        assert np.allclose(back, coords, atol=1e-12)


class TestVelocityTransformations:
    """Verifies kinematic velocity transformations between frames."""

    def test_velocity_round_trip(self):
        """Transforming ECEF -> ECI -> ECEF must restore original velocity and position."""
        pos_ecef = np.array([3000000.0, 4000000.0, 2000000.0])
        vel_ecef = np.array([100.0, -250.0, 50.0])
        t = GPSTime(2200, 12345.67)

        # Transform to ECI
        vel_eci = trans.velocity_transform("ECEF", "ECI", pos_ecef, vel_ecef, t)
        pos_eci = trans.position_transform("ECEF", "ECI", pos_ecef, t)

        # Transform back to ECEF
        vel_rec = trans.velocity_transform("ECI", "ECEF", pos_eci, vel_eci, t)
        assert np.allclose(vel_rec, vel_ecef, atol=1e-10)

    def test_equatorial_surface_velocity_in_eci(self):
        """A stationary point on the equator in ECEF moves eastward in ECI at omega_e * a."""
        a = EarthParam.wgs84a
        w_e = EarthParam.w_e
        t0 = GPSTime(2000, 0.0)

        pos_ecef = np.array([a, 0.0, 0.0])
        vel_ecef = np.array([0.0, 0.0, 0.0])

        vel_eci = trans.velocity_transform("ECEF", "ECI", pos_ecef, vel_ecef, t0)
        # At t0=0, ECI and ECEF are aligned. The velocity is pure +Y: [0, w_e * a, 0]
        expected_v_y = w_e * a  # approx 465.101 m/s
        assert np.isclose(vel_eci[0], 0.0, atol=1e-10)
        assert np.isclose(vel_eci[1], expected_v_y, atol=1e-8)
        assert np.isclose(vel_eci[2], 0.0, atol=1e-10)

    def test_polar_surface_velocity_in_eci(self):
        """A stationary point on the pole (on rotation axis) has zero tangential velocity in ECI."""
        b = EarthParam.wgs84b
        t0 = GPSTime(2000, 0.0)

        pos_north = np.array([0.0, 0.0, b])
        vel_north = np.array([0.0, 0.0, 0.0])

        vel_eci = trans.velocity_transform("ECEF", "ECI", pos_north, vel_north, t0)
        assert np.allclose(vel_eci, [0.0, 0.0, 0.0], atol=1e-10)

    def test_velocity_transform_invalid_frames(self):
        """LLA velocity transformation is explicitly unsupported and raises ValueError."""
        pos = np.array([1.0, 0.0, 0.0])
        vel = np.array([0.0, 0.0, 0.0])
        t = GPSTime(0, 0)

        with pytest.raises(ValueError, match="cannot be LLA"):
            trans.velocity_transform("LLA", "ECEF", pos, vel, t)
        with pytest.raises(ValueError, match="cannot be LLA"):
            trans.velocity_transform("ECEF", "LLA", pos, vel, t)
        with pytest.raises(ValueError, match="not valid"):
            trans.velocity_transform("INVALID", "ECEF", pos, vel, t)
        with pytest.raises(ValueError, match="not valid"):
            trans.velocity_transform("ECEF", "INVALID", pos, vel, t)
