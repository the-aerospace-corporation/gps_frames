# Copyright (c) 2022 The Aerospace Corporation
"""Unit tests for coordinate and velocity transformations in gps_frames.transforms.

Verifies:
- Kinematic velocity transformations between ECI and ECEF (including omega x r Coriolis terms)
- Geodetic (LLA) <-> Cartesian (ECEF) ellipsoidal transformations (Bowring's algorithm)
- Inertial (ECI) <-> Earth-Fixed (ECEF) sidereal rotations across time and multi-week epochs
- Standard directional cosine matrices (axes 1, 2, 3) and their time derivative rate matrices
"""

import numpy as np
import pytest

from gps_frames import transforms as trans
from gps_frames.parameters import EarthParam
from gps_time import GPSTime


def test_velocity_transform_LLA():
    """Verify that attempting velocity transformations involving LLA raises ValueError.

    Testing:
        trans.velocity_transform() input validation. Velocity vectors are Euclidean rates (dx/dt, dy/dt, dz/dt)
        and cannot be defined in the curvilinear Latitude/Longitude/Altitude coordinate system.

    Expected Result:
        ValueError is raised when 'from' or 'to' frame is 'LLA'.
    """
    with pytest.raises(ValueError):
        trans.velocity_transform("LLA", "ECI", (0, 0, 0), (0, 0, 0), GPSTime(0, 0))


def test_velocity_transform_from_frame():
    """Verify that an invalid source frame string raises ValueError in velocity_transform.

    Testing:
        Frame name validation in trans.velocity_transform().

    Expected Result:
        ValueError is raised when 'from' frame is an unknown string like 'asdf'.
    """
    with pytest.raises(ValueError):
        trans.velocity_transform("asdf", "ECI", (0, 0, 0), (0, 0, 0), GPSTime(0, 0))


def test_velocity_transform_to_frame():
    """Verify that an invalid destination frame string raises ValueError in velocity_transform.

    Testing:
        Destination frame validation in trans.velocity_transform().

    Expected Result:
        ValueError is raised when 'to' frame is an unknown string like 'asdf'.
    """
    with pytest.raises(ValueError):
        trans.velocity_transform("ECI", "asdf", (0, 0, 0), (0, 0, 0), GPSTime(0, 0))


def test_velocity_transform_ecef_to_ecef():
    """Verify identity velocity transformation when source and destination are both ECEF.

    Testing:
        trans.velocity_transform('ECEF', 'ECEF', pos, vel, time) identity operation.

    Expected Result:
        Output velocity is identical to input velocity: [0, 12389, 1243] m/s.
    """
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
    """Verify ECEF to ECI kinematic velocity transformation for a stationary surface point.

    Testing:
        The kinematic velocity relationship between rotating (ECEF) and inertial (ECI) frames:
            v_ECI = R_z(-theta) * v_ECEF + omega_e x r_ECI
        At epoch t = 0 (theta = 0, R_z(0) = I):
            r_ECEF = [r_e, 0, 0] -> r_ECI = [r_e, 0, 0]
            v_ECEF = [0, 0, 0] (stationary on surface)
            omega_e x r_ECI = [0, 0, w_e] x [r_e, 0, 0] = [0, w_e * r_e, 0]

    Expected Result:
        Output velocity in ECI is:
            [0.0, w_e * r_e, 0.0] = [0.0, 7.2921151467e-5 * 6378137.0, 0.0]
                                  ~= [0.0, 465.10113, 0.0] m/s
    """
    out = trans.velocity_transform("ECEF", "ECI", pos, vel, GPSTime(0, 0))
    assert np.allclose(out, expected)


def test_velocity_transform_eci_to_eci():
    """Verify identity velocity transformation when source and destination are both ECI.

    Testing:
        trans.velocity_transform('ECI', 'ECI', pos, vel, time) identity operation.

    Expected Result:
        Output velocity is identical to input velocity: [0, 12389, 1243] m/s.
    """
    outVel = trans.velocity_transform(
        "ECI", "ECI", (EarthParam.r_e, 98321, 2667541), (0, 12389, 1243), GPSTime(0, 0)
    )
    assert (outVel == np.array([0, 12389, 1243])).all()


@pytest.mark.parametrize("angle", [np.pi / 2, np.pi / 4, np.pi / 13])
def test_velocity_transform_eci_to_ecef(angle):
    """Verify ECI to ECEF velocity transformation at various Earth rotation angles.

    Testing:
        Inertial-to-rotating kinematic velocity equation:
            v_ECEF = R_z(theta) * (v_ECI - omega_e x r_ECI)
        Given:
            r_ECI = [0, r_e, 0]
            v_ECI = [-w_e * r_e, 0, 10]
        The Earth rotation tangential velocity at this position is:
            omega_e x r_ECI = [0, 0, w_e] x [0, r_e, 0] = [-w_e * r_e, 0, 0]
        Subtracting rotation velocity yields:
            v_rel = v_ECI - omega_e x r_ECI = [0, 0, 10]
        Since rotation about the Z-axis leaves Z-components invariant:
            R_z(theta) * [0, 0, 10] = [0, 0, 10]

    Expected Result:
        outVel == [0.0, 0.0, 10.0] m/s for all rotation angles.
    """
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
    """Verify that an invalid source frame string raises NotImplementedError in position_transform.

    Testing:
        Error handling in trans.position_transform().

    Expected Result:
        NotImplementedError is raised when 'from' frame is unsupported ('asdf').
    """
    with pytest.raises(NotImplementedError):
        trans.position_transform("asdf", "ECI", (0, 0, 0), GPSTime(0, 0))


def test_position_transform_to_frame():
    """Verify that an invalid destination frame string raises NotImplementedError in position_transform.

    Testing:
        Error handling in trans.position_transform().

    Expected Result:
        NotImplementedError is raised when 'to' frame is unsupported ('asdf').
    """
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
    """Verify conversion from ellipsoidal LLA coordinates to Cartesian ECEF coordinates.

    Testing:
        Bowring's closed-form geodetic to Cartesian equations on WGS84 ellipsoid:
            N(phi) = a / sqrt(1 - e^2 * sin^2(phi))
            X = (N(phi) + h) * cos(phi) * cos(lambda)
            Y = (N(phi) + h) * cos(phi) * sin(lambda)
            Z = (N(phi) * (1 - e^2) + h) * sin(phi)
        where a = 6378137.0 m and e^2 ~= 0.00669437999014.

    Expected Result:
        - Param 1: [34 deg N, 67 deg E, 453 m]:
            phi = 0.593412 rad, lambda = 1.16937 rad, h = 453.0 m
            N(phi) ~= 6384883.21 m
            X ~= (6384883.21 + 453) * cos(34 deg) * cos(67 deg) ~= 2068387.54 m
            Y ~= (6384883.21 + 453) * cos(34 deg) * sin(67 deg) ~= 4872815.69 m
            Z ~= (6384883.21 * (1 - e^2) + 453) * sin(34 deg)  ~= 3546699.88 m
        - Param 2: High-altitude point [68 deg N, 12 deg E, 123981232 m]:
            X ~= 47773104.52 m, Y ~= 10154486.84 m, Z ~= 120844489.59 m.
    """
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
    """Verify conversion from Earth-fixed (ECEF) coordinates to inertial (ECI) coordinates.

    Testing:
        Passive coordinate frame rotation from ECEF to ECI across angle theta = w_e * t:
            r_ECI = R_z(-theta) * r_ECEF
        where R_z(-theta) has form:
            [[ cos(-theta),  sin(-theta), 0],
             [-sin(-theta),  cos(-theta), 0],
             [           0,            0, 1]]

    Expected Result:
        At angle = pi/3 (60 deg, t = (pi/3) / w_e ~= 14360.7 s):
            cos(-pi/3) = 0.5, sin(-pi/3) = -sqrt(3)/2 ~= -0.866025
            r_ECI[0] = 0.5 * X - 0.866025 * Y
            r_ECI[1] = 0.866025 * X + 0.5 * Y
            r_ECI[2] = Z
        out matches expected matrix multiplication rot @ ECEF.
    """
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
    """Verify direct conversion from LLA to ECI coordinates.

    Testing:
        trans.lla2eci(LLA, sec) executes the composite transformation:
            LLA -> lla2ecef(LLA) -> ecef2eci(ECEF, sec).

    Expected Result:
        out matches ECI coordinates obtained from two-step conversion trans.ecef2eci(trans.lla2ecef(LLA), sec).
    """
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
    """Verify iterative conversion from Cartesian ECEF to ellipsoidal LLA coordinates.

    Testing:
        trans.ecef2lla(ECEF) inverts Cartesian coordinates to geodetic latitude, longitude,
        and height above ellipsoid (HAE) using Bowring's iterative method.

    Expected Result:
        - Param 1: [2068387.54, 4872815.69, 3546699.88] -> [34 deg, 67 deg, 453 m] in radians/meters.
        - Param 2: [47773104.52, 10154486.84, 120844489.59] -> [68 deg, 12 deg, 123981232 m].
    """
    LLA[0] *= np.pi / 180
    LLA[1] *= np.pi / 180
    out = trans.ecef2lla(ECEF)
    assert np.allclose(out, LLA)


def test_ecef2lla_no_convergence():
    """Verify that ecef2lla handles the coordinate origin (0, 0, 0) without infinite looping.

    Testing:
        Singularity handling at r = (0, 0, 0) where longitude and latitude are indeterminate.

    Expected Result:
        The function terminates gracefully without raising unhandled exceptions or looping infinitely.
    """
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
    """Verify conversion from inertial (ECI) to Earth-fixed (ECEF) coordinates.

    Testing:
        Passive frame transformation from ECI to ECEF:
            r_ECEF = R_z(theta) * r_ECI
        where theta = w_e * t = angle.

    Expected Result:
        At angle = pi/3 (60 deg):
            rot = [[cos(60 deg), sin(60 deg), 0],
                   [-sin(60 deg), cos(60 deg), 0],
                   [0, 0, 1]]
        out matches expected rotation rot @ ECI.
    """
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
    """Verify composite conversion from inertial ECI coordinates to geodetic LLA coordinates.

    Testing:
        trans.eci2lla(ECI, sec) performs:
            r_ECEF = trans.eci2ecef(ECI, sec)
            LLA = trans.ecef2lla(r_ECEF)

    Expected Result:
        out matches trans.ecef2lla(rot @ ECI).
    """
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
    """Verify advancing the ECI frame origin across integer GPS weeks.

    Testing:
        trans.add_weeks_eci(num_weeks, ECI) applies Greenwich sidereal rotation:
            angle = w_e * 604800 s * num_weeks
            r_shifted = R_z(angle) * ECI

    Expected Result:
        - num_weeks = 0: angle = 0, rot = I, r_shifted == ECI.
        - num_weeks = 14: angle = 14 * 604800 * 7.292115e-5 ~= 617.44 rad.
          out matches rot @ ECI.
    """
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
    """Verify time-evolution rotation of ECEF coordinates.

    Testing:
        trans.rotate_ecef(t_old, t_new, coords) rotates coordinates about the Earth's spin axis (Z)
        by the elapsed sidereal angle:
            angle = w_e * (t_new - t_old) = w_e * t_delt
            r_new = R_z(angle) * r_old

    Expected Result:
        For ECEF = [12382, 6665, 1897273] and t_delt = 8667632 s:
            angle = 7.292115e-5 * 8667632 ~= 632.0538 rad
            rot = [[cos(angle), sin(angle), 0],
                   [-sin(angle), cos(angle), 0],
                   [0, 0, 1]]
        out matches rot @ ECEF.
    """
    new_time = old_time + t_delt
    angle = EarthParam.w_e * t_delt
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    expected = rot @ ECEF
    out = trans.rotate_ecef(old_time, new_time, tuple(ECEF))
    assert np.allclose(out, expected)


def test_standard_rotation_matrix_axis1():
    """Verify standard rotation matrix about axis 1 (X-axis) by 90 degrees (pi/2).

    Testing:
        trans.standard_rotation_matrix(1, pi/2) computes:
            R_1(theta) = [[1,          0,           0],
                          [0,  cos(theta),  sin(theta)],
                          [0, -sin(theta),  cos(theta)]]

    Expected Result:
        For theta = pi/2: cos(pi/2) = 0, sin(pi/2) = 1
            expected = [[1, 0,  0],
                        [0, 0,  1],
                        [0, -1, 0]]
    """
    expected = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    assert np.allclose(trans.standard_rotation_matrix(1, np.pi / 2), expected)


def test_standard_rotation_matrix_axis2():
    """Verify standard rotation matrix about axis 2 (Y-axis) by 90 degrees (pi/2).

    Testing:
        trans.standard_rotation_matrix(2, pi/2) computes:
            R_2(theta) = [[ cos(theta), 0, -sin(theta)],
                          [          0, 1,           0],
                          [ sin(theta), 0,  cos(theta)]]

    Expected Result:
        For theta = pi/2: cos(pi/2) = 0, sin(pi/2) = 1
            expected = [[0, 0, -1],
                        [0, 1,  0],
                        [1, 0,  0]]
    """
    expected = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    assert np.allclose(trans.standard_rotation_matrix(2, np.pi / 2), expected)


def test_standard_rotation_matrix_axis3():
    """Verify standard rotation matrix about axis 3 (Z-axis) by 90 degrees (pi/2).

    Testing:
        trans.standard_rotation_matrix(3, pi/2) computes:
            R_3(theta) = [[ cos(theta),  sin(theta), 0],
                          [-sin(theta),  cos(theta), 0],
                          [          0,           0, 1]]

    Expected Result:
        For theta = pi/2: cos(pi/2) = 0, sin(pi/2) = 1
            expected = [[ 0, 1, 0],
                        [-1, 0, 0],
                        [ 0, 0, 1]]
    """
    expected = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    assert np.allclose(trans.standard_rotation_matrix(3, np.pi / 2), expected)


def test_standard_rotation_matrix_not_an_axis():
    """Verify that an invalid axis number (< 1 or > 3) raises AssertionError.

    Testing:
        Input validation in trans.standard_rotation_matrix().

    Expected Result:
        trans.standard_rotation_matrix(-1, pi/2) raises AssertionError.
    """
    with pytest.raises(AssertionError):
        trans.standard_rotation_matrix(-1, np.pi / 2)


def test_standard_rotation_matrix_rates_axis1():
    """Verify standard rotation rate matrix (time derivative) about axis 1 (X-axis).

    Testing:
        trans.standard_rotation_matrix_rates(1, theta, theta_dot) computes d/dt R_1(theta(t)):
            d/dt R_1 = theta_dot * [[0,           0,            0],
                                    [0, -sin(theta),   cos(theta)],
                                    [0, -cos(theta),  -sin(theta)]]

    Expected Result:
        For theta = pi/2, theta_dot = 1:
            expected = [[0,  0,  0],
                        [0, -1,  0],
                        [0,  0, -1]]
    """
    expected = [[0, 0, 0], [0, -1, 0], [0, 0, -1]]
    assert np.allclose(trans.standard_rotation_matrix_rates(1, np.pi / 2, 1), expected)


def test_standard_rotation_matrix_rates_axis2():
    """Verify standard rotation rate matrix (time derivative) about axis 2 (Y-axis).

    Testing:
        trans.standard_rotation_matrix_rates(2, theta, theta_dot) computes d/dt R_2(theta(t)):
            d/dt R_2 = theta_dot * [[-sin(theta), 0, -cos(theta)],
                                    [          0, 0,           0],
                                    [ cos(theta), 0, -sin(theta)]]

    Expected Result:
        For theta = pi/2, theta_dot = 1:
            expected = [[-1, 0,  0],
                        [ 0, 0,  0],
                        [ 0, 0, -1]]
    """
    expected = [[-1, 0, 0], [0, 0, 0], [0, 0, -1]]
    assert np.allclose(trans.standard_rotation_matrix_rates(2, np.pi / 2, 1), expected)


def test_standard_rotation_matrix_rates_axis3():
    """Verify standard rotation rate matrix (time derivative) about axis 3 (Z-axis).

    Testing:
        trans.standard_rotation_matrix_rates(3, theta, theta_dot) computes d/dt R_3(theta(t)):
            d/dt R_3 = theta_dot * [[-sin(theta),  cos(theta), 0],
                                    [-cos(theta), -sin(theta), 0],
                                    [          0,           0, 0]]

    Expected Result:
        For theta = pi/2, theta_dot = 1:
            expected = [[-1,  0, 0],
                        [ 0, -1, 0],
                        [ 0,  0, 0]]
    """
    expected = [[-1, 0, 0], [0, -1, 0], [0, 0, 0]]
    assert np.allclose(trans.standard_rotation_matrix_rates(3, np.pi / 2, 1), expected)


def test_standard_rotation_matrix_rates_not_an_axis():
    """Verify that an invalid axis number raises ValueError in standard_rotation_matrix_rates.

    Testing:
        Input validation in trans.standard_rotation_matrix_rates().

    Expected Result:
        trans.standard_rotation_matrix_rates(-1, pi/2, 1) raises ValueError.
    """
    with pytest.raises(ValueError):
        trans.standard_rotation_matrix_rates(-1, np.pi / 2, 1)
