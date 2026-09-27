# Copyright (c) 2022 The Aerospace Corporation
"""Regression tests for physical, Earth, and GPS constants, and the EGM-96 geoid model.

Verifies:
- Exact SI physical constants and instance immutability (PhysicsParam)
- WGS84 Earth reference ellipsoid parameters and derived relations (EarthParam)
- IS-GPS-200 carrier frequencies, ionospheric gamma ratio, and navigation epochs (GPSparam)
- EGM-96 global geoid grid extent and unit consistency across radians and degrees (GeoidData)
"""

from dataclasses import FrozenInstanceError
import numpy as np
import pytest

from gps_frames.parameters import EarthParam, GeoidData, GPSparam, PhysicsParam


class TestPhysicalAndEarthConstants:
    """Verifies that physical and Earth parameters match IS-GPS-200 and WGS-84 specifications."""

    def test_physics_param_values_and_immutability(self):
        """Verify SI physical constants and dataclass immutability.

        Testing:
            PhysicsParam definition:
            - Speed of light: c = 2.99792458e8 m/s
            - Boltzmann constant: k_b = 1.38064852e-23 J/K
            Instance attribute reassignment raises FrozenInstanceError.

        Expected Result:
            Constants match exact values and instance modification is prohibited.
        """
        assert np.isclose(PhysicsParam.c, 2.99792458e8)
        assert np.isclose(PhysicsParam.k_b, 1.38064852e-23)

        p = PhysicsParam()
        with pytest.raises(FrozenInstanceError):
            p.c = 3e8

    def test_earth_param_values_and_relationships(self):
        """Verify WGS-84 ellipsoid constants and analytical relationships.

        Testing:
            EarthParam values:
            - Semimajor axis: a = 6378137.0 m
            - Reciprocal flattening: 1 / f = 298.257223563
            - Semiminor axis: b = a * (1 - f) ~= 6356752.3142 m
            - First eccentricity squared: e^2 = 1 - (1 - f)^2 ~= 0.00669437999014
            - First eccentricity: e = sqrt(e^2) ~= 0.0818191908426
            - Earth gravitational parameter: mu = 3.986005e14 m^3/s^2
            - Earth rotation rate: w_e = 7.2921151467e-5 rad/s
            - Relativistic clock parameter: F = -4.442807633e-10 s/sqrt(m)

        Expected Result:
            All derived parameters match exact formulas within 1e-12.
        """
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
        """Verify IS-GPS-200 carrier frequencies, gamma ratio, and broadcast epochs.

        Testing:
            GPSparam definitions:
            - L1: 1575.42 MHz, L2: 1227.60 MHz, L5: 1176.45 MHz
            - gamma = (1575.42 / 1227.6)^2 ~= 1.646944444
            - IS-GPS-200 pi constant = 3.1415926535898
            - LNAV epochs: t_oc = 16 s, t_oe = 16 s, t_oa = 4096 s

        Expected Result:
            All constants match specification values within 1e-12.
        """
        assert np.isclose(GPSparam.L1_CENTER_FREQ_Hz, 1575.42e6)
        assert np.isclose(GPSparam.L2_CENTER_FREQ_Hz, 1227.6e6)
        assert np.isclose(GPSparam.L5_CENTER_FREQ_Hz, 1176.45e6)

        expected_gamma = (1575.42 / 1227.6) ** 2
        assert np.isclose(GPSparam.gamma, expected_gamma, atol=1e-12)

        assert np.isclose(GPSparam.pi, 3.1415926535898, atol=1e-13)

        assert GPSparam.lnav_t_oc_epoch == 16
        assert GPSparam.lnav_t_oe_epoch == 16
        assert GPSparam.lnav_t_oa_epoch == 4096


class TestGeoidModelData:
    """Verifies EGM-96 geoid interpolation across grid nodes and coordinates."""

    def test_geoid_grid_structure(self):
        """Verify global extent of EGM-96 regular latitude and longitude grids.

        Testing:
            GeoidData grid boundaries:
            - Latitudes span [-90.0, +90.0] deg
            - Longitudes span [-180.0, +180.0] deg
            - 2D grid matrix shape matches (n_lat, n_lon).

        Expected Result:
            Grids span full planetary sphere.
        """
        assert GeoidData.latitudes[0] == -90.0
        assert GeoidData.latitudes[-1] == 90.0
        assert GeoidData.longitudes[0] == -180.0
        assert GeoidData.longitudes[-1] == 180.0
        assert GeoidData.geoid_heights.shape == (
            len(GeoidData.latitudes),
            len(GeoidData.longitudes),
        )

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
        """Verify that get_geoid_height yields identical undulations in degrees and radians.

        Testing:
            get_geoid_height(lat, lon, units='deg') == get_geoid_height(rad(lat), rad(lon), units='rad').

        Expected Result:
            Both unit modes agree within atol=1e-10 m.
        """
        h_deg = GeoidData.get_geoid_height(lat_deg, lon_deg, units="deg")
        h_rad = GeoidData.get_geoid_height(
            np.deg2rad(lat_deg), np.deg2rad(lon_deg), units="rad"
        )
        assert np.isclose(h_deg, h_rad, atol=1e-10)

    def test_invalid_geoid_units_raises(self):
        """Verify that unsupported angle unit strings raise NotImplementedError.

        Testing:
            get_geoid_height() unit validation.

        Expected Result:
            NotImplementedError("Only rad and deg") is raised for units="arcsec".
        """
        with pytest.raises(NotImplementedError, match="Only rad and deg"):
            GeoidData.get_geoid_height(0.0, 0.0, units="arcsec")
