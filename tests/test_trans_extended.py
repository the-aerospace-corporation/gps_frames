# Copyright (c) 2022 The Aerospace Corporation
"""Extended unit tests for gps_frames.transforms.

Verifies:
- All 9 valid pairwise coordinate frame combinations for position_transform:
  {ECEF, ECI, LLA} x {ECEF, ECI, LLA}
- Error handling for invalid/unsupported frames in position_transform
- Fallback handler for unhandled valid frames
- Zero-week boundary behavior for trans.add_weeks_eci
- Time delta assertions and return properties in rotate_ecef
"""

from unittest.mock import patch
import numpy as np
import pytest

from gps_frames import transforms as trans
from gps_frames.parameters import EarthParam
from gps_time import GPSTime


def test_position_transform_valid_paths():
    """Verify all 9 pairwise frame permutations in position_transform.

    Testing:
        trans.position_transform(from_frame, to_frame, coords, time) for all 3x3 combinations:
        1. ECEF -> ECEF (identity)
        2. ECEF -> LLA  (Bowring inverse geodetic)
        3. ECEF -> ECI  (Sidereal rotation R_z(-theta); identity at t=0)
        4. LLA  -> LLA  (identity)
        5. LLA  -> ECEF (Bowring forward geodetic)
        6. LLA  -> ECI  (Forward geodetic + sidereal rotation)
        7. ECI  -> ECI  (identity)
        8. ECI  -> ECEF (Sidereal rotation R_z(theta); identity at t=0)
        9. ECI  -> LLA  (Sidereal rotation + inverse geodetic)

    Expected Result:
        - At t=0 (theta = 0, ECI and ECEF coincide):
          ecef_pos = [EarthParam.r_e, 0, 0] = [6378137.0, 0.0, 0.0] m
          LLA coordinates = [0.0 rad, 0.0 rad, 0.0 m] (Equator at Prime Meridian, HAE = 0)
        - All 9 transformations return expected values matching theoretical coordinates.
    """
    ecef_pos = np.array([EarthParam.r_e, 0, 0])
    time = GPSTime(0, 0)

    # 1. ECEF -> ECEF
    res = trans.position_transform("ECEF", "ECEF", ecef_pos, time)
    assert np.allclose(res, ecef_pos)

    # 2. ECEF -> LLA: [r_e, 0, 0] corresponds to [0 lat, 0 lon, 0 alt]
    lla_res = trans.position_transform("ECEF", "LLA", ecef_pos, time)
    assert np.allclose(lla_res, [0, 0, 0], atol=1e-8)

    # 3. ECEF -> ECI: At t=0, sidereal rotation angle theta = 0 -> identity
    eci_res = trans.position_transform("ECEF", "ECI", ecef_pos, time)
    assert np.allclose(eci_res, ecef_pos)

    # 4. LLA -> LLA
    lla_pos = np.array([0.0, 0.0, 0.0])
    res = trans.position_transform("LLA", "LLA", lla_pos, time)
    assert np.allclose(res, lla_pos)

    # 5. LLA -> ECEF: [0, 0, 0] maps to equatorial radius [r_e, 0, 0]
    res = trans.position_transform("LLA", "ECEF", lla_pos, time)
    assert np.allclose(res, ecef_pos, atol=1e-3)

    # 6. LLA -> ECI
    res = trans.position_transform("LLA", "ECI", lla_pos, time)
    assert np.allclose(res, ecef_pos, atol=1e-3)

    # 7. ECI -> ECI
    res = trans.position_transform("ECI", "ECI", ecef_pos, time)
    assert np.allclose(res, ecef_pos)

    # 8. ECI -> ECEF
    res = trans.position_transform("ECI", "ECEF", ecef_pos, time)
    assert np.allclose(res, ecef_pos)

    # 9. ECI -> LLA
    res = trans.position_transform("ECI", "LLA", ecef_pos, time)
    assert np.allclose(res, [0, 0, 0], atol=1e-8)


def test_position_transform_invalid():
    """Verify error handling in position_transform for unsupported or invalid frame names.

    Testing:
        trans.position_transform(from_frame, to_frame, coords, time) raises NotImplementedError
        when either from_frame or to_frame is not recognized.

    Expected Result:
        NotImplementedError is raised when specifying an unrecognized frame like 'INVALID_FRAME'.
    """
    coords = np.array([0.0, 0.0, 0.0])
    time = GPSTime(0, 0)

    with pytest.raises(NotImplementedError):
        trans.position_transform("INVALID_FRAME", "ECEF", coords, time)

    with pytest.raises(NotImplementedError):
        trans.position_transform("ECEF", "INVALID_FRAME", coords, time)


def test_transform_fallback_with_patched_frames():
    """Verify fallback return path when a recognized frame has no explicit handler.

    Testing:
        When VALID_FRAMES includes a frame not handled by the if/elif dispatch chain,
        position_transform logs a critical warning and returns the input coordinates unmodified.

    Expected Result:
        res and res2 return exact input coordinates [1.0, 2.0, 3.0].
    """
    coords = np.array([1.0, 2.0, 3.0])
    time = GPSTime(0, 0)

    with patch(
        "gps_frames.transforms.VALID_FRAMES", ["ECI", "ECEF", "LLA", "TEST_FRAME"]
    ):
        res = trans.position_transform("ECI", "TEST_FRAME", coords, time)
        assert np.allclose(res, coords)

        res2 = trans.position_transform("TEST_FRAME", "ECI", coords, time)
        assert np.allclose(res2, coords)


def test_add_weeks_eci_zero_weeks():
    """Verify that shifting ECI by zero weeks leaves coordinates unchanged.

    Testing:
        trans.add_weeks_eci(0, coords) applies angle = w_e * 0 = 0 rad, yielding identity rotation.

    Expected Result:
        Output coordinates equal input coordinates [1.0, 2.0, 3.0].
    """
    coords = np.array([1.0, 2.0, 3.0], dtype=float)
    res = trans.add_weeks_eci(0, coords)
    assert np.allclose(res, coords)


def test_rotate_ecef_assertions():
    """Verify time difference assertion and execution in rotate_ecef.

    Testing:
        trans.rotate_ecef(old_time, new_time, coordinates) computes time_delta = new_time - old_time
        and asserts isinstance(time_delta, float).

    Expected Result:
        Executing rotate_ecef with GPSTime instances succeeds and returns rotated float coordinates.
    """
    t1 = GPSTime(2000, 100.0)
    t2 = GPSTime(2000, 200.0)
    coords = np.array([6378137.0, 0.0, 0.0])

    res = trans.rotate_ecef(t1, t2, coords)
    assert isinstance(res, np.ndarray)
    assert res.shape == (3,)
