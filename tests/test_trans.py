# Copyright (c) 2022 The Aerospace Corporation

import numpy as np
import pytest

from gps_frames import transforms as trans
from gps_frames.parameters import EarthParam
from gps_time import GPSTime


def test_velocity_transform_LLA():
    with pytest.raises(ValueError):
        trans.velocity_transform("LLA", "ECI", (0, 0, 0), (0, 0, 0), GPSTime(0, 0))


def test_velocity_transform_from_frame():
    with pytest.raises(ValueError):
        trans.velocity_transform("asdf", "ECI", (0, 0, 0), (0, 0, 0), GPSTime(0, 0))


def test_velocity_transform_to_frame():
    with pytest.raises(ValueError):
        trans.velocity_transform("ECI", "asdf", (0, 0, 0), (0, 0, 0), GPSTime(0, 0))


def test_velocity_transform_ecef_to_ecef():
    outVel = trans.velocity_transform(
        "ECEF", "ECEF", (EarthParam.r_e, 0, 0), (0, 12389, 1243), GPSTime(0, 0)
    )
    assert (outVel == np.array([0, 12389, 1243])).all()


@pytest.mark.parametrize(
    "pos, vel, expected",
    [
        (
            np.array((EarthParam.r_e, 0, 0), dtype=float),
            np.array((0, 0, 0), dtype=float),
            np.array((0, EarthParam.w_e * EarthParam.r_e, 0), dtype=float),
        ),
    ],
)
def test_velocity_transform_ecef_to_eci(pos, vel, expected):
    out = trans.velocity_transform("ECEF", "ECI", pos, vel, GPSTime(0, 0))
    assert np.allclose(out, expected)


def test_velocity_transform_eci_to_eci():
    outVel = trans.velocity_transform(
        "ECI", "ECI", (EarthParam.r_e, 98321, 2667541), (0, 12389, 1243), GPSTime(0, 0)
    )
    assert (outVel == np.array([0, 12389, 1243])).all()


@pytest.mark.parametrize("angle", [np.pi / 2, np.pi / 4, np.pi / 13])
def test_velocity_transform_eci_to_ecef(angle):
    sec = angle / EarthParam.w_e
    gTime = GPSTime(236, sec)
    outVel = trans.velocity_transform(
        "ECI",
        "ECEF",
        (0, EarthParam.r_e, 0),
        (-EarthParam.w_e * EarthParam.r_e, 0, 10),
        gTime,
    )
    assert np.allclose(outVel, np.array([0, 0, 10]))


def test_position_transform_from_frame():
    with pytest.raises(NotImplementedError):
        trans.position_transform("asdf", "ECI", (0, 0, 0), GPSTime(0, 0))


def test_position_transform_to_frame():
    with pytest.raises(NotImplementedError):
        trans.position_transform("ECI", "asdf", (0, 0, 0), GPSTime(0, 0))


@pytest.mark.parametrize(
    "LLA, ECEF",
    [
        (
            np.array([34.0, 67.0, 453.0]),
            np.array([2068387.54328875, 4872815.68729718, 3546699.87811555]),
        ),
        (
            np.array([68.0, 12.0, 123981232]),
            np.array([47773104.5218264, 10154486.837462, 120844489.588087]),
        ),
    ],
)
def test_lla2ecef(LLA, ECEF):

    LLA[0] *= np.pi / 180
    LLA[1] *= np.pi / 180
    out = trans.lla2ecef(LLA)
    assert np.allclose(out, ECEF)


@pytest.mark.parametrize(
    "ECEF, angle",
    [
        (np.array([EarthParam.r_e + 781, 63562.4, 44]), np.pi / 3),
        (np.array([EarthParam.r_e + 3432, 729.4, 6785]), np.pi / 3),
    ],
)
def test_ecef2eci(ECEF, angle):

    sec = angle / EarthParam.w_e
    c = np.cos(-angle)
    s = np.sin(-angle)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    expected = rot @ ECEF
    assert np.allclose(trans.ecef2eci(ECEF, sec), expected)


@pytest.mark.parametrize(
    "LLA, ECEF",
    [
        (
            np.array([34.0, 67.0, 453.0]),
            np.array([2068387.54328875, 4872815.68729718, 3546699.87811555]),
        ),
        (
            np.array([68.0, 12.0, 123981232]),
            np.array([47773104.5218264, 10154486.837462, 120844489.588087]),
        ),
    ],
)
def test_lla2eci(LLA, ECEF):

    LLA[0] *= np.pi / 180
    LLA[1] *= np.pi / 180
    sec = 716237

    ECEF = trans.lla2ecef(LLA)
    ECI = trans.ecef2eci(ECEF, sec)
    out = trans.lla2eci(LLA, sec)
    assert np.allclose(out, ECI)


@pytest.mark.parametrize(
    "LLA, ECEF",
    [
        (
            np.array([34.0, 67.0, 453.0]),
            np.array([2068387.54328875, 4872815.68729718, 3546699.87811555]),
        ),
        (
            np.array([68.0, 12.0, 123981232]),
            np.array([47773104.5218264, 10154486.837462, 120844489.588087]),
        ),
    ],
)
def test_ecef2lla(LLA, ECEF):

    LLA[0] *= np.pi / 180
    LLA[1] *= np.pi / 180
    out = trans.ecef2lla(ECEF)
    assert np.allclose(out, LLA)


def test_ecef2lla_no_convergence():

    ECEF = np.array((0.0, 0.0, 0.0))
    trans.ecef2lla(ECEF)


@pytest.mark.parametrize(
    "ECI, angle",
    [
        (np.array([EarthParam.r_e + 781, 63562.4, 44]), np.pi / 3),
        (np.array([EarthParam.r_e + 3432, 729.4, 6785]), np.pi / 3),
    ],
)
def test_eci2ecef(ECI, angle):

    sec = angle / EarthParam.w_e
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    expected = rot @ ECI
    assert np.allclose(trans.eci2ecef(ECI, sec), expected)


@pytest.mark.parametrize(
    "ECI, angle",
    [
        (np.array([EarthParam.r_e + 781, 63562.4, 44]), np.pi / 3),
        (np.array([EarthParam.r_e + 3432, 729.4, 6785]), np.pi / 3),
    ],
)
def test_eci2lla(ECI, angle):

    sec = angle / EarthParam.w_e
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    expected = trans.ecef2lla(rot @ ECI)
    out = trans.eci2lla(ECI, sec)
    assert np.allclose(out, expected)


@pytest.mark.parametrize(
    "ECI, num_weeks",
    [
        (np.array([EarthParam.r_e + 781, 63562.4, 44]), 14),
        (np.array([EarthParam.r_e + 781, 63562.4, 44]), 0),
        (np.array([EarthParam.r_e + 3432, 729.4, 6785]), 37),
    ],
)
def test_add_weeks_eci(ECI, num_weeks):

    angle = EarthParam.w_e * 604800 * num_weeks
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    expected = rot @ ECI
    out = trans.add_weeks_eci(num_weeks, ECI)

    assert np.allclose(out, expected)


@pytest.mark.parametrize(
    "ECEF, old_time, t_delt",
    [(np.array([12382, 6665, 1897273]), GPSTime(893, 6721), 8667632)],
)
def test_rotate_ecef(ECEF, old_time, t_delt):

    new_time = old_time + t_delt
    angle = EarthParam.w_e * t_delt
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    expected = rot @ ECEF
    out = trans.rotate_ecef(old_time, new_time, tuple(ECEF))
    assert np.allclose(out, expected)


def test_standard_rotation_matrix_axis1():
    expected = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    assert np.allclose(trans.standard_rotation_matrix(1, np.pi / 2), expected)


def test_standard_rotation_matrix_axis2():
    expected = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    assert np.allclose(trans.standard_rotation_matrix(2, np.pi / 2), expected)


def test_standard_rotation_matrix_axis3():
    expected = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    assert np.allclose(trans.standard_rotation_matrix(3, np.pi / 2), expected)


def test_standard_rotation_matrix_not_an_axis():
    with pytest.raises(AssertionError):
        trans.standard_rotation_matrix(-1, np.pi / 2)


def test_standard_rotation_matrix_rates_axis1():
    expected = [[0, 0, 0], [0, -1, 0], [0, 0, -1]]
    assert np.allclose(trans.standard_rotation_matrix_rates(1, np.pi / 2, 1), expected)


def test_standard_rotation_matrix_rates_axis2():
    expected = [[-1, 0, 0], [0, 0, 0], [0, 0, -1]]
    assert np.allclose(trans.standard_rotation_matrix_rates(2, np.pi / 2, 1), expected)


def test_standard_rotation_matrix_rates_axis3():
    expected = [[-1, 0, 0], [0, -1, 0], [0, 0, 0]]
    assert np.allclose(trans.standard_rotation_matrix_rates(3, np.pi / 2, 1), expected)


def test_standard_rotation_matrix_rates_not_an_axis():
    with pytest.raises(ValueError):
        trans.standard_rotation_matrix_rates(-1, np.pi / 2, 1)
