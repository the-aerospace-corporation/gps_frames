# Copyright (c) 2022 The Aerospace Corporation
"""Golden characterization and multi-system integration tests.

Pins down end-to-end values across orbit kinematics, frame transformations,
station look angles, and velocities to guarantee zero regression during optimizations.
"""

import numpy as np
import pytest

from gps_time import GPSTime
from gps_frames.position import Position, distance
from gps_frames.vectors import Vector, UnitVector
from gps_frames.velocity import Velocity
from gps_frames.parameters import EarthParam
from gps_frames import (
    get_east_north_up_basis,
    get_range_azimuth_elevation,
    get_azimuth_elevation,
    get_relative_angles,
    check_earth_obscuration,
)
from gps_frames.basis import Basis


class TestGoldenOrbitCharacterization:
    """Golden baseline test verifying integrated orbit state and station geometry."""

    def test_gps_orbit_snapshot_characterization(self):
        """Validates that a GPS MEO orbit state transforms to exact baseline values."""
        t = GPSTime(2150, 3600.0)
        a = 26560000.0
        inc = np.deg2rad(55.0)
        u = np.deg2rad(45.0)

        # 1. ECI Position
        x_eci = a * np.cos(u)
        y_eci = a * np.sin(u) * np.cos(inc)
        z_eci = a * np.sin(u) * np.sin(inc)
        pos_eci = Position(np.array([x_eci, y_eci, z_eci]), t, "ECI")

        # Pinned ECI coordinates
        expected_eci = np.array([18780756.1083147, 10772199.16058529, 15384294.75941896])
        assert np.allclose(pos_eci.coordinates, expected_eci, atol=1e-6)

        # 2. ECEF Position
        pos_ecef = pos_eci.get_position("ECEF")
        expected_ecef = np.array([20932836.22488863, 5529325.66750274, 15384294.75941896])
        assert np.allclose(pos_ecef.coordinates, expected_ecef, atol=1e-5)

        # 3. LLA Position
        pos_lla = pos_eci.get_position("LLA")
        expected_lla = np.array([0.618541592, 0.258247631, 20189037.3])
        assert np.isclose(pos_lla.coordinates[0], expected_lla[0], atol=1e-7)  # lat rad (~35.44 deg)
        assert np.isclose(pos_lla.coordinates[1], expected_lla[1], atol=1e-7)  # lon rad (~14.79 deg)
        assert np.isclose(pos_lla.coordinates[2], expected_lla[2], atol=1.0)   # alt m (~20,189 km)

        # 4. Velocities
        v_mag = np.sqrt(EarthParam.mu / a)
        vx_eci = -v_mag * np.sin(u)
        vy_eci = v_mag * np.cos(u) * np.cos(inc)
        vz_eci = v_mag * np.cos(u) * np.sin(inc)
        vel_eci = Velocity(pos_eci, Vector(np.array([vx_eci, vy_eci, vz_eci]), t, "ECI"))

        expected_v_eci = np.array([-2739.30182216, 1571.19897724, 2243.90468755])
        assert np.allclose(vel_eci.velocity.coordinates, expected_v_eci, atol=1e-6)

        vel_ecef = vel_eci.get_velocity("ECEF")
        expected_v_ecef = np.array([-1834.50482291, 701.80309858, 2243.90468755])
        assert np.allclose(vel_ecef.velocity.coordinates, expected_v_ecef, atol=1e-5)

        # 5. Ground Station Look Angles (Los Angeles station: 34 deg N, 118 deg W, 100m HAE)
        gs_lla = Position(np.array([np.deg2rad(34.0), np.deg2rad(-118.0), 100.0]), t, "LLA")
        gs_enu = get_east_north_up_basis(gs_lla)

        rng, az, el = get_range_azimuth_elevation(gs_enu, pos_ecef)
        expected_rng = 28153763.786
        expected_az = 0.64766886
        expected_el = -0.36173240

        assert np.isclose(rng, expected_rng, atol=1.0)
        assert np.isclose(az, expected_az, atol=1e-5)
        assert np.isclose(el, expected_el, atol=1e-5)

        # 6. Obscuration (elevation is negative ~ -20.7 deg, so obscured)
        obs = check_earth_obscuration(gs_lla, pos_ecef)
        assert not obs
        assert obs == False

    def test_satellite_nadir_pointing_boresight(self):
        """Satellite pointed nadir (towards Earth center) looking at ground station."""
        t = GPSTime(2150, 0.0)
        # Satellite on Z axis at 26,560 km
        sat_pos = Position(np.array([0.0, 0.0, 26560000.0]), t, "ECEF")

        # Nadir is along -Z. So Look Axis is -Z (or define basis with Z-axis along -Z)
        # Let look_axis = 3, pointing along -Z: axis3 = [0, 0, -1]
        u1 = UnitVector(np.array([1.0, 0.0, 0.0]), t, "ECEF")
        u2 = UnitVector(np.array([0.0, -1.0, 0.0]), t, "ECEF")
        u3 = UnitVector(np.array([0.0, 0.0, -1.0]), t, "ECEF")
        sat_basis = Basis(sat_pos, u1, u2, u3)

        # Ground station directly below satellite on surface
        sub_sat_gs = Position(np.array([0.0, 0.0, EarthParam.r_e]), t, "ECEF")
        off_bore, angle_ref = get_relative_angles(sat_basis, sub_sat_gs, look_axis=3, reference_axis=1)
        # Directly along nadir boresight
        assert np.isclose(off_bore, 0.0, atol=1e-12)

        # Ground station at edge of Earth disk (limb)
        # Limb angle asin(Re / Rsat) = asin(6378137 / 26560000) approx 13.9 deg
        expected_limb_angle = np.arcsin(EarthParam.r_e / 26560000.0)
        limb_gs = Position(np.array([EarthParam.r_e, 0.0, 0.0]), t, "ECEF")
        off_bore_limb, _ = get_relative_angles(sat_basis, limb_gs, look_axis=3, reference_axis=1)
        # Line from [0, 0, 26560e3] to [Re, 0, 0]: angle off nadir
        # tan(theta) = Re / 26560e3
        expected_off_bore = np.arctan2(EarthParam.r_e, 26560000.0)
        assert np.isclose(off_bore_limb, expected_off_bore, atol=1e-6)
