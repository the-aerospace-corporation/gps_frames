# Copyright (c) 2022 The Aerospace Corporation
"""Regression tests for coordinate and velocity transformations in gps_frames.transforms.

Verifies:
- LLA <-> ECEF round-trip invertibility across a global 567-point grid spanning latitudes, longitudes, and altitudes
- Exact polar and equatorial boundary values against WGS84 semimajor (a) and semiminor (b) axes
- Sidereal day rotation periodicity and quarter-turn geometry
- Forward and backward multi-week Greenwich sidereal frame evolution
- Kinematic velocity transformations, Coriolis velocity at equator (~465.1 m/s), and polar zero-velocity
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
        """Verify round-trip LLA -> ECEF -> LLA across global grid (latitudes, longitudes, altitudes).

        Testing:
            Invertibility of Bowring's geodetic transformation:
                ecef = lla2ecef(orig_lla)
                rec_lla = ecef2lla(ecef)
            across 567 combinations of:
            - Latitudes from South Pole (-89.999 deg) to North Pole (+89.999 deg)
            - Longitudes across all 4 quadrants (-180 to +180 deg)
            - Altitudes from Dead Sea (-400 m) to GEO (35,786 km).

        Expected Result:
            - Recovered latitude matches within atol=1e-10 rad (sub-millimeter surface precision).
            - Recovered longitude matches within atol=1e-10 rad (handling +/- pi branch cut).
            - Recovered altitude matches within atol=1e-3 m (1 mm).
        """
        lat_rad = np.deg2rad(lat_deg)
        lon_rad = np.deg2rad(lon_deg)
        orig_lla = np.array([lat_rad, lon_rad, alt_m], dtype=float)

        ecef = trans.lla2ecef(orig_lla)
        rec_lla = trans.ecef2lla(ecef)

        assert np.isclose(rec_lla[0], orig_lla[0], atol=1e-10)
        diff_lon = np.arctan2(
            np.sin(rec_lla[1] - orig_lla[1]), np.cos(rec_lla[1] - orig_lla[1])
        )
        assert np.isclose(diff_lon, 0.0, atol=1e-10)
        assert np.isclose(rec_lla[2], orig_lla[2], atol=1e-3)

    def test_exact_poles(self):
        """Verify exact Cartesian coordinates at North and South poles on reference ellipsoid.

        Testing:
            At lat = +/- pi/2 and alt = 0:
                X = 0, Y = 0, Z = +/- b
            where b = EarthParam.wgs84b ~= 6356752.3142 m (semiminor axis).

        Expected Result:
            - North Pole: [0.0, 0.0, +b] and round-trips to lat = pi/2, alt = 0.
            - South Pole: [0.0, 0.0, -b] and round-trips to lat = -pi/2, alt = 0.
        """
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
        """Verify exact Cartesian coordinates on the Equator (lat = 0, alt = 0).

        Testing:
            At lat = 0, alt = 0:
                X = a * cos(lon), Y = a * sin(lon), Z = 0
            where a = EarthParam.wgs84a = 6378137.0 m.

        Expected Result:
            - lon = 0 deg:   [+a, 0, 0]
            - lon = 90 deg:  [0, +a, 0]
            - lon = 180 deg: [-a, 0, 0]
        """
        a = EarthParam.wgs84a

        eq0_ecef = trans.lla2ecef(np.array([0.0, 0.0, 0.0]))
        assert np.allclose(eq0_ecef, [a, 0.0, 0.0], atol=1e-6)

        eq90_ecef = trans.lla2ecef(np.array([0.0, np.pi / 2, 0.0]))
        assert np.allclose(eq90_ecef, [0.0, a, 0.0], atol=1e-6)

        eq180_ecef = trans.lla2ecef(np.array([0.0, np.pi, 0.0]))
        assert np.allclose(eq180_ecef, [-a, 0.0, 0.0], atol=1e-6)


class TestEcefEciTransformations:
    """Verifies time-dependent Earth rotation transformations between ECEF and ECI."""

    def test_ecef_eci_round_trip(self):
        """Verify that ECEF <-> ECI transformations are perfectly invertible across time-of-week.

        Testing:
            rec_ecef = eci2ecef(ecef2eci(ecef, tow), tow) for tow in [0, 100, 3600, 43200, 86400, 604799].

        Expected Result:
            rec_ecef == ecef_orig within atol=1e-12 m.
        """
        ecef_orig = np.array([4500000.0, -1200000.0, 4300000.0])
        tows = [0.0, 100.0, 3600.0, 43200.0, 86400.0, 604799.0]

        for tow in tows:
            eci = trans.ecef2eci(ecef_orig, tow)
            rec_ecef = trans.eci2ecef(eci, tow)
            assert np.allclose(rec_ecef, ecef_orig, atol=1e-12)

    def test_sidereal_day_rotation(self):
        """Verify that advancing by one full sidereal day (2*pi / w_e seconds) reproduces initial orientation.

        Testing:
            T_sidereal = 2 * pi / EarthParam.w_e ~= 86164.0905 s (~23h 56m 4.09s).
            After one period, Earth rotation angle theta = w_e * T_sidereal = 2*pi == 0 (mod 2*pi).

        Expected Result:
            trans.ecef2eci(pos, T_sidereal) matches trans.ecef2eci(pos, 0.0) within atol=1e-6 m.
        """
        sidereal_period = 2 * np.pi / EarthParam.w_e
        pos = np.array([EarthParam.r_e, 1000.0, 5000.0])

        eci_0 = trans.ecef2eci(pos, 0.0)
        eci_1period = trans.ecef2eci(pos, sidereal_period)
        assert np.allclose(eci_1period, eci_0, atol=1e-6)

    def test_quarter_rotation(self):
        """Verify coordinate transformation after a 90-degree Earth rotation (t = (pi/2) / w_e).

        Testing:
            Passive coordinate frame rotation from ECEF to ECI across angle theta = pi/2:
                r_ECI = R_z(-pi/2) * r_ECEF
                [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]^T @ [1, 0, 0] = [0, 1, 0].

        Expected Result:
            Unit X vector in ECEF [1.0, 0.0, 0.0] rotates to unit Y vector in ECI [0.0, 1.0, 0.0].
        """
        quarter_period = (np.pi / 2) / EarthParam.w_e
        pos_ecef = np.array([1.0, 0.0, 0.0])

        pos_eci = trans.ecef2eci(pos_ecef, quarter_period)
        assert np.allclose(pos_eci, [0.0, 1.0, 0.0], atol=1e-14)

    def test_rotate_ecef_time_shifts(self):
        """Verify time-evolution of ECEF coordinates via rotate_ecef.

        Testing:
            rotate_ecef(t1, t2, coords) applies Z-axis rotation angle:
                theta = omega_e * delta_t = 7.292115e-5 * 1000.0 = 0.07292115 rad.

        Expected Result:
            Forward rotated coordinates match trans.standard_rotation(3, theta, coords)
            and reverse shift restores original coordinates within atol=1e-8.
        """
        t1 = GPSTime(2100, 1000.0)
        t2 = GPSTime(2100, 2000.0)
        coords = np.array([6378137.0, 0.0, 0.0])

        rotated = trans.rotate_ecef(t1, t2, coords)
        expected_angle = EarthParam.w_e * 1000.0
        expected = trans.standard_rotation(3, expected_angle, coords)
        assert np.allclose(rotated, expected, atol=1e-14)

        rev_rotated = trans.rotate_ecef(t2, t1, rotated)
        assert np.allclose(rev_rotated, coords, atol=1e-8)

    def test_add_weeks_eci(self):
        """Verify shifting ECI coordinates across integer weeks.

        Testing:
            trans.add_weeks_eci(num_weeks, coords) with 0 weeks (identity) and +2 then -2 weeks.

        Expected Result:
            - 0 weeks returns identical coordinates.
            - Advancing +2 weeks then -2 weeks restores original coordinates within atol=1e-12.
        """
        coords = np.array([1000.0, 2000.0, 3000.0])

        assert np.allclose(trans.add_weeks_eci(0, coords), coords)

        fwd = trans.add_weeks_eci(2, coords)
        back = trans.add_weeks_eci(-2, fwd)
        assert np.allclose(back, coords, atol=1e-12)


class TestVelocityTransformations:
    """Verifies kinematic velocity transformations between frames."""

    def test_velocity_round_trip(self):
        """Verify velocity transformation round-trip: ECEF -> ECI -> ECEF.

        Testing:
            Full kinematic velocity round-trip accounting for position-dependent rotational terms.

        Expected Result:
            Recovered velocity matches original ECEF velocity within atol=1e-10 m/s.
        """
        pos_ecef = np.array([3000000.0, 4000000.0, 2000000.0])
        vel_ecef = np.array([100.0, -250.0, 50.0])
        t = GPSTime(2200, 12345.67)

        vel_eci = trans.velocity_transform("ECEF", "ECI", pos_ecef, vel_ecef, t)
        pos_eci = trans.position_transform("ECEF", "ECI", pos_ecef, t)

        vel_rec = trans.velocity_transform("ECI", "ECEF", pos_eci, vel_eci, t)
        assert np.allclose(vel_rec, vel_ecef, atol=1e-10)

    def test_equatorial_surface_velocity_in_eci(self):
        """Verify that a stationary equatorial surface point has eastward inertial velocity of omega_e * a.

        Testing:
            Kinematic rotational velocity at equator (r = a = 6378137.0 m):
                v_y = omega_e * a = 7.2921151467e-5 * 6378137.0 ~= 465.10113 m/s
                v_x = 0.0, v_z = 0.0.

        Expected Result:
            vel_eci matches [0.0, 465.10113, 0.0] m/s.
        """
        a = EarthParam.wgs84a
        w_e = EarthParam.w_e
        t0 = GPSTime(2000, 0.0)

        pos_ecef = np.array([a, 0.0, 0.0])
        vel_ecef = np.array([0.0, 0.0, 0.0])

        vel_eci = trans.velocity_transform("ECEF", "ECI", pos_ecef, vel_ecef, t0)
        expected_v_y = w_e * a
        assert np.isclose(vel_eci[0], 0.0, atol=1e-10)
        assert np.isclose(vel_eci[1], expected_v_y, atol=1e-8)
        assert np.isclose(vel_eci[2], 0.0, atol=1e-10)

    def test_polar_surface_velocity_in_eci(self):
        """Verify that a stationary point at the Earth's pole has zero tangential velocity in ECI.

        Testing:
            At the North Pole, position vector is collinear with Earth's spin axis:
                r_pole = [0, 0, b], omega_e = [0, 0, w_e]
                omega_e x r_pole = [0, 0, 0].

        Expected Result:
            Inertial velocity is exactly [0.0, 0.0, 0.0] m/s.
        """
        b = EarthParam.wgs84b
        t0 = GPSTime(2000, 0.0)

        pos_north = np.array([0.0, 0.0, b])
        vel_north = np.array([0.0, 0.0, 0.0])

        vel_eci = trans.velocity_transform("ECEF", "ECI", pos_north, vel_north, t0)
        assert np.allclose(vel_eci, [0.0, 0.0, 0.0], atol=1e-10)

    def test_velocity_transform_invalid_frames(self):
        """Verify error handling when attempting velocity transforms with LLA or unsupported frame names.

        Testing:
            trans.velocity_transform raises ValueError for LLA or unknown frame strings.

        Expected Result:
            Raises ValueError with descriptive message.
        """
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
