# Copyright (c) 2022 The Aerospace Corporation
"""Regression tests for physical, Earth, and GPS constants, and the EGM-96 geoid model.

Verifies exact specification values, derived relationships, immutability, and GeoidData interpolation.
"""

import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from gps_frames.parameters import PhysicsParam, EarthParam, GPSparam, GeoidData


class TestPhysicalAndEarthConstants:
    """Verifies that physical and Earth parameters match IS-GPS-200 and WGS-84 specifications."""

    def test_physics_param_values_and_immutability(self):
        assert np.isclose(PhysicsParam.c, 2.99792458e8)
        assert np.isclose(PhysicsParam.k_b, 1.38064852e-23)

        p = PhysicsParam()
        with pytest.raises(FrozenInstanceError):
            p.c = 3e8

    def test_earth_param_values_and_relationships(self):
        # Semi-major axis
        assert np.isclose(EarthParam.wgs84a, 6378137.0)
        assert np.isclose(EarthParam.r_e, 6378137.0)

        # Flattening
        expected_f = 1.0 / 298.257223563
        assert np.isclose(EarthParam.wgs84f, expected_f)

        # Derived semi-minor axis b = a * (1 - f)
        expected_b = EarthParam.wgs84a * (1.0 - EarthParam.wgs84f)
        assert np.isclose(EarthParam.wgs84b, expected_b, atol=1e-12)

        # Eccentricity squared: e^2 = 1 - (1 - f)^2
        expected_e2 = 1.0 - (1.0 - EarthParam.wgs84f) ** 2
        assert np.isclose(EarthParam.wgs84ecc_squared, expected_e2, atol=1e-15)

        # Eccentricity: e = sqrt(e^2)
        assert np.isclose(EarthParam.wgs84ecc, np.sqrt(expected_e2), atol=1e-15)

        # Gravitational parameter mu
        assert np.isclose(EarthParam.mu, 3.986005e14)

        # Earth angular velocity w_e
        assert np.isclose(EarthParam.w_e, 7.2921151467e-5)

        # Relativistic parameter F
        assert np.isclose(EarthParam.F, -4.442807633e-10)

        # Immutability of instance
        e = EarthParam()
        with pytest.raises(FrozenInstanceError):
            e.mu = 4e14

    def test_gps_param_values(self):
        assert np.isclose(GPSparam.L1_CENTER_FREQ_Hz, 1575.42e6)
        assert np.isclose(GPSparam.L2_CENTER_FREQ_Hz, 1227.6e6)
        assert np.isclose(GPSparam.L5_CENTER_FREQ_Hz, 1176.45e6)

        # Gamma ratio: (f_L1 / f_L2)^2
        expected_gamma = (1575.42 / 1227.6) ** 2
        assert np.isclose(GPSparam.gamma, expected_gamma, atol=1e-12)

        # Pi specification in IS-GPS-200
        assert np.isclose(GPSparam.pi, 3.1415926535898, atol=1e-13)

        # Epochs
        assert GPSparam.lnav_t_oc_epoch == 16
        assert GPSparam.lnav_t_oe_epoch == 16
        assert GPSparam.lnav_t_oa_epoch == 4096


class TestGeoidModelData:
    """Verifies EGM-96 geoid interpolation across grid nodes and coordinates."""

    def test_geoid_grid_structure(self):
        # Latitudes span -90 to +90 deg
        assert GeoidData.latitudes[0] == -90.0
        assert GeoidData.latitudes[-1] == 90.0
        # Longitudes span -180 to +180 deg
        assert GeoidData.longitudes[0] == -180.0
        assert GeoidData.longitudes[-1] == 180.0
        # Grid shape matches
        assert GeoidData.geoid_heights.shape == (len(GeoidData.latitudes), len(GeoidData.longitudes))

    @pytest.mark.parametrize(
        "lat_deg, lon_deg",
        [
            (0.0, 0.0),
            (34.0, -118.0),
            (51.5, 0.0),
            (-33.8, 151.2),
            (90.0, 0.0),
            (-90.0, 0.0),
        ],
    )
    def test_geoid_unit_consistency(self, lat_deg, lon_deg):
        """get_geoid_height must return identical results whether inputs are in rad or deg."""
        h_deg = GeoidData.get_geoid_height(lat_deg, lon_deg, units="deg")
        h_rad = GeoidData.get_geoid_height(np.deg2rad(lat_deg), np.deg2rad(lon_deg), units="rad")
        assert np.isclose(h_deg, h_rad, atol=1e-10)

    def test_invalid_geoid_units_raises(self):
        with pytest.raises(NotImplementedError, match="Only rad and deg"):
            GeoidData.get_geoid_height(0.0, 0.0, units="arcsec")
