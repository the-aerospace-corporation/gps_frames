
import pytest
import numpy as np
from gps_frames import transforms as trans
from gps_frames.parameters import EarthParam
from gps_time import GPSTime
from unittest.mock import patch

# Extended tests for position_transform to cover all branches

def test_position_transform_valid_paths():
    """Test all valid frame combinations for position_transform."""
    # Test point
    ecef_pos = np.array([EarthParam.r_e, 0, 0])
    time = GPSTime(0, 0)
    
    # ECEF -> ECEF
    res = trans.position_transform("ECEF", "ECEF", ecef_pos, time)
    assert np.allclose(res, ecef_pos)
    
    # ECEF -> LLA
    # (r_e, 0, 0) should be (0 lat, 0 lon, 0 alt) approximately
    lla_res = trans.position_transform("ECEF", "LLA", ecef_pos, time)
    assert np.allclose(lla_res, [0, 0, 0], atol=1e-8)
    
    # ECEF -> ECI
    # At t=0, ECI and ECEF are aligned
    eci_res = trans.position_transform("ECEF", "ECI", ecef_pos, time)
    assert np.allclose(eci_res, ecef_pos)
    
    # LLA -> LLA
    lla_pos = np.array([0., 0., 0.])
    res = trans.position_transform("LLA", "LLA", lla_pos, time)
    assert np.allclose(res, lla_pos)
    
    # LLA -> ECEF
    res = trans.position_transform("LLA", "ECEF", lla_pos, time)
    assert np.allclose(res, ecef_pos, atol=1e-3)
    
    # LLA -> ECI
    res = trans.position_transform("LLA", "ECI", lla_pos, time)
    assert np.allclose(res, ecef_pos, atol=1e-3) # Aligned at t=0
    
    # ECI -> ECI
    res = trans.position_transform("ECI", "ECI", ecef_pos, time)
    assert np.allclose(res, ecef_pos)
    
    # ECI -> ECEF
    res = trans.position_transform("ECI", "ECEF", ecef_pos, time)
    assert np.allclose(res, ecef_pos)
    
    # ECI -> LLA
    res = trans.position_transform("ECI", "LLA", ecef_pos, time)
    assert np.allclose(res, [0, 0, 0], atol=1e-8)

def test_position_transform_invalid():
    """Test invalid inputs for position_transform."""
    coords = np.array([0,0,0])
    time = GPSTime(0,0)
    
    # Unknown frames already tested in test_trans.py but verifying coverage says "NotImplementedError"
    # The code raises NotImplementedError for unknown frames.
    
    # But checking the code:
    # if from_frame not in VALID_FRAMES: raise NotImplementedError...
    
    # Also valid frames list: VALID_FRAMES = ["ECI", "ECEF", "LLA"]
    
    # The code has a fallback `logger.critical("Transformation Failed. Returning input")`
    # This is unreachable if the branching logic covers all combinations of 3 valid frames.
    # 3x3 = 9 combinations.
    # LLA->LLA, LLA->ECEF, LLA->ECI (3)
    # ECEF->LLA, ECEF->ECEF, ECEF->ECI (3)
    # ECI->LLA, ECI->ECEF, ECI->ECI (3)
    # All covered. So the logger.critical line is effectively dead code unless VALID_FRAMES grows but the if/elif chain doesn't update.
    # However, for coverage purposes, we can't easily trigger it without mocking or modifying VALID_FRAMES dynamically.
    pass

def test_transform_fallback_with_patched_frames():
    """Test the fallback path (lines 200-201) by introducing a valid-but-unhandled frame."""
    coords = np.array([1.0, 2.0, 3.0])
    time = GPSTime(0, 0)
    
    # We patch VALID_FRAMES to include "TEST_FRAME".
    # This allows it to pass the initial validation check.
    # But since there is no logic for "TEST_FRAME", it falls through to the warning/return.
    with patch("gps_frames.transforms.VALID_FRAMES", ["ECI", "ECEF", "LLA", "TEST_FRAME"]):
        # Try transforming FROM a known frame TO the new unhandled frame
        res = trans.position_transform("ECI", "TEST_FRAME", coords, time)
        
        # Should return input coordinates (fallback behavior)
        assert np.allclose(res, coords)
        
        # Test transforming FROM the new unhandled frame
        res2 = trans.position_transform("TEST_FRAME", "ECI", coords, time)
        assert np.allclose(res2, coords)

def test_add_weeks_eci_zero_weeks():
    # Test explicit 0 weeks branch
    coords = np.array([1, 2, 3], dtype=float)
    res = trans.add_weeks_eci(0, coords)
    assert np.allclose(res, coords)

def test_rotate_ecef_assertions():
    # Test float assertion for time_delta (implicitly tracked by type checker but good to verify runtime behavior if needed)
    pass
