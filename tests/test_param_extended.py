# Copyright (c) 2022 The Aerospace Corporation
"""Extended unit tests verifying physical, geodetic, and GPS constants in gps_frames.parameters.

Verifies:
- Standard physics parameters (speed of light c, Boltzmann constant k_b)
- WGS84 Earth ellipsoid derived geometric relations (eccentricity, semi-minor axis b)
- GPS signal parameters (carrier center frequencies L1, L2, L5, and dual-frequency ratio gamma)
"""

import pytest

from gps_frames.parameters import EarthParam, GPSparam, PhysicsParam


def test_physics_param_values():
    """Verify standard SI physics constants.

    Testing:
        PhysicsParam physical constants definitions.

    Expected Result:
        - Speed of light in vacuum: c == 2.99792458e8 m/s (exact SI definition)
        - Boltzmann constant: k_b == 1.38064852e-23 J/K
    """
    assert PhysicsParam.c == 2.99792458e8
    assert PhysicsParam.k_b == 1.38064852e-23


def test_earth_param_values():
    """Verify internal consistency of WGS84 reference ellipsoid geometric parameters.

    Testing:
        Derived ellipsoid parameters from equatorial radius a and flattening f:
            e^2 = 1 - (1 - f)^2 = 2f - f^2
            e = sqrt(e^2)
            b = a * (1 - f)

    Expected Result:
        - EarthParam.wgs84ecc_squared matches 1 - (1 - f)^2 within 1e-15 (~0.00669437999014)
        - EarthParam.wgs84ecc matches sqrt(e^2) within 1e-15 (~0.0818191908426)
        - EarthParam.wgs84b (semi-minor axis) matches a * (1 - f) within 1e-15 (~6356752.3142 m)
    """
    expected_ecc_sq = 1 - (1 - EarthParam.wgs84f) ** 2
    assert abs(EarthParam.wgs84ecc_squared - expected_ecc_sq) < 1e-15
    assert abs(EarthParam.wgs84ecc - expected_ecc_sq**0.5) < 1e-15

    expected_b = EarthParam.wgs84a * (1 - EarthParam.wgs84f)
    assert abs(EarthParam.wgs84b - expected_b) < 1e-15


def test_gps_param_values():
    """Verify GPS signal carrier frequencies and dual-frequency ionospheric ratio gamma.

    Testing:
        GPS signal definition parameters:
            L1 = 1575.42 MHz = 154 * 10.23 MHz
            L2 = 1227.60 MHz = 120 * 10.23 MHz
            L5 = 1176.45 MHz = 115 * 10.23 MHz
            gamma = (f_L1 / f_L2)^2 = (1575.42 / 1227.60)^2 = (77 / 60)^2 ~= 1.646944444

    Expected Result:
        - GPSparam.gamma matches (1575.42 / 1227.6)^2 within 1e-15
        - Carrier frequencies match nominal center values in Hz.
    """
    expected_gamma = (1575.42 / 1227.6) ** 2
    assert abs(GPSparam.gamma - expected_gamma) < 1e-15

    assert GPSparam.L1_CENTER_FREQ_Hz == 1575.42e6
    assert GPSparam.L2_CENTER_FREQ_Hz == 1227.6e6
    assert GPSparam.L5_CENTER_FREQ_Hz == 1176.45e6
