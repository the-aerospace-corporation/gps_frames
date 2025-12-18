
import pytest
from gps_frames.parameters import PhysicsParam, EarthParam, GPSparam

def test_physics_param_values():
    """Test standard physics parameters values."""
    # Check speed of light
    assert PhysicsParam.c == 2.99792458e8
    # Check Boltzmann constant
    assert PhysicsParam.k_b == 1.38064852e-23

def test_earth_param_values():
    """Test Earth parameters derived values."""
    # Check if wgs84ecc is consistent with wgs84f
    expected_ecc_sq = 1 - (1 - EarthParam.wgs84f)**2
    assert abs(EarthParam.wgs84ecc_squared - expected_ecc_sq) < 1e-15
    assert abs(EarthParam.wgs84ecc - expected_ecc_sq**0.5) < 1e-15
    
    # Check semi-minor axis
    expected_b = EarthParam.wgs84a * (1 - EarthParam.wgs84f)
    assert abs(EarthParam.wgs84b - expected_b) < 1e-15

def test_gps_param_values():
    """Test GPS parameters values."""
    # Check gamma calculation
    expected_gamma = (1575.42 / 1227.6) ** 2
    assert abs(GPSparam.gamma - expected_gamma) < 1e-15
    
    # Check frequencies
    assert GPSparam.L1_CENTER_FREQ_Hz == 1575.42e6
    assert GPSparam.L2_CENTER_FREQ_Hz == 1227.6e6
    assert GPSparam.L5_CENTER_FREQ_Hz == 1176.45e6
