# Copyright (c) 2022 The Aerospace Corporation

import numpy as np
import pytest
import copy
from gps_frames.__init__ import check_earth_obscuration
from gps_frames.__init__ import get_azimuth_elevation
from gps_frames.__init__ import get_range_azimuth_elevation
from gps_frames.__init__ import _get_spherical_radius
from gps_frames.__init__ import get_relative_angles
from gps_frames.__init__ import get_east_north_up_basis
from gps_frames import position
from gps_frames import basis
from gps_frames import vectors
from gps_frames.parameters import EarthParam

from gps_time import GPSTime


@pytest.mark.parametrize(
    "pos, expected",
    [
        (
            position.Position(np.array([0, 0, 100], dtype=float), GPSTime(0, 0), "LLA"),
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        ),
        (
            position.Position(
                np.array([np.pi / 4, 0, 100], dtype=float), GPSTime(0, 0), "LLA"
            ),
            np.array(
                [
                    [0, 1, 0],
                    [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
                    [np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
                ],
                dtype=float,
            ),
        ),
    ],
)
def test_get_east_north_up_basis(pos, expected):

    ENUB = get_east_north_up_basis(pos)
    _basis = np.array(
        [ENUB.axes[0].coordinates, ENUB.axes[1].coordinates, ENUB.axes[2].coordinates]
    )
    assert np.allclose(_basis, expected)


def test_get_relative_angles_look_ref_axis_different():

    _basis = basis.get_ecef_basis(GPSTime(0, 0))
    target = position.Position(np.array([0, 0, 100], dtype=float), GPSTime(0, 0), "LLA")

    with pytest.raises(ValueError):
        get_relative_angles(_basis, target, 1, 1)


def test_get_relative_angles_look_axis_defined():

    _basis = basis.get_ecef_basis(GPSTime(0, 0))
    target = position.Position(np.array([0, 0, 100], dtype=float), GPSTime(0, 0), "LLA")

    with pytest.raises(ValueError):
        get_relative_angles(_basis, target, 4, 1)


def test_get_relative_angles_ref_axis_defined():

    _basis = basis.get_ecef_basis(GPSTime(0, 0))
    target = position.Position(np.array([0, 0, 100], dtype=float), GPSTime(0, 0), "LLA")

    with pytest.raises(ValueError):
        get_relative_angles(_basis, target, 1, 0)


@pytest.mark.parametrize(
    "target, look_axis, ref_axis, angle",
    [
        (
            position.Position(np.array([0, 0, 100], dtype=float), GPSTime(0, 0), "LLA"),
            1,
            3,
            0,
        ),
        (
            position.Position(np.array([0, 0, 100], dtype=float), GPSTime(0, 0), "LLA"),
            2,
            3,
            np.pi / 2,
        ),
        (
            position.Position(np.array([0, 0, 100], dtype=float), GPSTime(0, 0), "LLA"),
            2,
            1,
            np.pi / 2,
        ),
        (
            position.Position(
                np.array([5, 5, -np.sqrt(50)], dtype=float), GPSTime(0, 0), "ECEF"
            ),
            3,
            1,
            3 * np.pi / 4,
        ),
    ],
)
def test_get_relative_angles_angle_off_look_axis(target, look_axis, ref_axis, angle):

    _basis = basis.get_ecef_basis(GPSTime(0, 0))
    look_angle, ref_angle = get_relative_angles(_basis, target, look_axis, ref_axis)
    assert look_angle == angle


@pytest.mark.parametrize(
    "target, look_axis, ref_axis, angle",
    [
        (
            position.Position(np.array([0, 0, 100], dtype=float), GPSTime(0, 0), "LLA"),
            1,
            3,
            0,
        ),
        (
            position.Position(np.array([0, 0, 100], dtype=float), GPSTime(0, 0), "LLA"),
            2,
            3,
            np.pi / 2,
        ),
        (
            position.Position(np.array([0, 0, 100], dtype=float), GPSTime(0, 0), "LLA"),
            2,
            1,
            0,
        ),
        (
            position.Position(np.array([5, 5, -np.sqrt(50)]), GPSTime(0, 0), "ECEF"),
            3,
            1,
            np.pi / 4,
        ),
    ],
)
def test_get_relative_angles_angle_from_ref_axis(target, look_axis, ref_axis, angle):

    _basis = basis.get_ecef_basis(GPSTime(0, 0))
    look_angle, ref_angle = get_relative_angles(_basis, target, look_axis, ref_axis)
    assert ref_angle == angle


@pytest.mark.parametrize(
    "displacement, expected_az",
    [
        (np.array([0, EarthParam.r_e, 0], dtype=float), np.pi / 2.0),
        (np.array([0, EarthParam.r_e, EarthParam.r_e], dtype=float), np.pi / 4.0),
        (
            np.array(
                [EarthParam.r_e * np.sqrt(2), EarthParam.r_e, EarthParam.r_e],
                dtype=float,
            ),
            np.pi / 4.0,
        ),
    ],
)
def test_get_rae_azimuth(displacement, expected_az):

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "LLA")
    ENUB = get_east_north_up_basis(pos)

    target = pos.get_position("ECEF")
    target.coordinates += displacement

    rng, az, el = get_range_azimuth_elevation(ENUB, target)

    assert az == expected_az


@pytest.mark.parametrize(
    "displacement, expected_az",
    [
        (np.array([0, EarthParam.r_e, 0], dtype=float), np.pi / 2.0),
        (np.array([0, EarthParam.r_e, EarthParam.r_e], dtype=float), np.pi / 4.0),
        (
            np.array(
                [EarthParam.r_e * np.sqrt(2), EarthParam.r_e, EarthParam.r_e],
                dtype=float,
            ),
            np.pi / 4.0,
        ),
    ],
)
def test_get_ae_azimuth(displacement, expected_az):

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "LLA")
    ENUB = get_east_north_up_basis(pos)

    target = pos.get_position("ECEF")
    target.coordinates += displacement

    az, el = get_azimuth_elevation(ENUB, target)

    assert az == expected_az


@pytest.mark.parametrize(
    "displacement, expected_el",
    [
        (np.array([0, EarthParam.r_e, 0], dtype=float), 0),
        (np.array([0, EarthParam.r_e, EarthParam.r_e], dtype=float), 0),
        (
            np.array(
                [EarthParam.r_e * np.sqrt(2), EarthParam.r_e, EarthParam.r_e],
                dtype=float,
            ),
            np.pi / 4.0,
        ),
    ],
)
def test_get_rae_elevation(displacement, expected_el):

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "LLA")
    ENUB = get_east_north_up_basis(pos)

    target = pos.get_position("ECEF")
    target.coordinates += displacement

    rng, az, el = get_range_azimuth_elevation(ENUB, target)
    assert abs(el - expected_el) < 1e-9


@pytest.mark.parametrize(
    "displacement, expected_el",
    [
        (np.array([0, EarthParam.r_e, 0], dtype=float), 0),
        (np.array([0, EarthParam.r_e, EarthParam.r_e], dtype=float), 0),
        (
            np.array(
                [EarthParam.r_e * np.sqrt(2), EarthParam.r_e, EarthParam.r_e],
                dtype=float,
            ),
            np.pi / 4.0,
        ),
    ],
)
def test_get_ae_elevation(displacement, expected_el):

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "LLA")
    ENUB = get_east_north_up_basis(pos)

    target = pos.get_position("ECEF")
    target.coordinates += displacement

    az, el = get_azimuth_elevation(ENUB, target)
    assert abs(el - expected_el) < 1e-9


@pytest.mark.parametrize(
    "displacement, expected_rng",
    [
        (np.array([0, EarthParam.r_e, 0], dtype=float), EarthParam.r_e),
        (
            np.array([0, EarthParam.r_e, EarthParam.r_e], dtype=float),
            EarthParam.r_e * np.sqrt(2),
        ),
        (
            np.array(
                [EarthParam.r_e * np.sqrt(2), EarthParam.r_e, EarthParam.r_e],
                dtype=float,
            ),
            2 * EarthParam.r_e,
        ),
    ],
)
def test_get_rae_range(displacement, expected_rng):

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "LLA")
    ENUB = get_east_north_up_basis(pos)

    target = pos.get_position("ECEF")
    target.coordinates += displacement

    rng, az, el = get_range_azimuth_elevation(ENUB, target)
    assert np.isclose(rng, expected_rng)


# @pytest.mark.parametrize("displacement, expected_az", [(np.array([0, EarthParam.r_e, 0]), np.pi / 2.0),
#                                                        (np.array(
#                                                            [0, EarthParam.r_e, EarthParam.r_e]), np.pi / 4.0),
#                                                        (np.array([EarthParam.r_e * np.sqrt(2), EarthParam.r_e, EarthParam.r_e]), np.pi / 4.0)])
# def test_get_ra_azimuth(displacement, expected_az):

#     pos = position.Position(np.array([0, 0, 0]), GPSTime(0, 0), 'LLA')
#     ENUB = get_east_north_up_basis(pos)

#     target = pos.get_position('ECEF')
#     target.coordinates += displacement

#     az, el = get_azimuth_elevation(ENUB, target)

#     assert(az == expected_az)


# @pytest.mark.parametrize("displacement, expected_el", [(np.array([0, EarthParam.r_e, 0]), 0),
#                                                        (np.array(
#                                                            [0, EarthParam.r_e, EarthParam.r_e]), 0),
#                                                        (np.array([EarthParam.r_e * np.sqrt(2), EarthParam.r_e, EarthParam.r_e]), np.pi / 4.0)])
# def test_get_ra_elevation(displacement, expected_el):

#     pos = position.Position(np.array([0, 0, 0]), GPSTime(0, 0), 'LLA')
#     ENUB = get_east_north_up_basis(pos)

#     target = pos.get_position('ECEF')
#     target.coordinates += displacement

#     az, el = get_azimuth_elevation(ENUB, target)
#     assert(el == expected_el)


@pytest.mark.parametrize(
    "pos1, pos2, hae, adjust, expected",
    [
        (
            position.Position(
                np.array([0, 0, 100e3], dtype=float), GPSTime(0, 0), "LLA"
            ),
            position.Position(
                np.array([np.pi / 2, 0, 100e3], dtype=float), GPSTime(0, 0), "LLA"
            ),
            False,
            0,
            False,
        ),
        (
            position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "LLA"),
            position.Position(
                np.array([EarthParam.wgs84a + 200, 200, 0], dtype=float),
                GPSTime(0, 0),
                "ECEF",
            ),
            False,
            0,
            True,
        ),
        (
            position.Position(np.array([0, 0, 100], dtype=float), GPSTime(0, 0), "LLA"),
            position.Position(
                np.array([1.0 * np.pi / 180, 0, 100], dtype=float),
                GPSTime(0, 0),
                "LLA",
            ),
            False,
            29e3,
            False,
        ),
        (
            position.Position(np.array([0, 0, 100], dtype=float), GPSTime(0, 0), "LLA"),
            position.Position(
                np.array([1.0 * np.pi / 180, 0, 100], dtype=float),
                GPSTime(0, 0),
                "LLA",
            ),
            False,
            30e3,
            True,
        ),
        (
            position.Position(
                np.array([0, 0, 30e3], dtype=float), GPSTime(0, 0), "LLA"
            ),
            position.Position(
                np.array([np.pi / 180, 0, 0], dtype=float), GPSTime(0, 0), "LLA"
            ),
            False,
            0,
            True,
        ),
        (
            position.Position(
                np.array([0, 0, 30e3], dtype=float), GPSTime(0, 0), "LLA"
            ),
            position.Position(
                np.array([np.pi / 180, 0, 0], dtype=float), GPSTime(0, 0), "LLA"
            ),
            False,
            -100,
            True,
        ),
    ],
)
def test_earth_obscuration(pos1, pos2, hae, adjust, expected):
    assert (
        check_earth_obscuration(pos1, pos2, hae, adjust, elevation_mask_angle_rad=0.0)
        == expected
    )
    assert (
        check_earth_obscuration(
            pos1,
            pos2,
            hae,
            adjust,
            elevation_mask_angle_rad=0.0,
            transition_altitude_m=1.0,
        )
        == expected
    )


@pytest.mark.parametrize(
    "pos",
    [
        position.Position(np.array([36, -176, 10], dtype=float), GPSTime(0, 0), "LLA"),
        position.Position(
            np.array([89743763, 363551, -1287398], dtype=float),
            GPSTime(0, 0),
            "ECEF",
        ),
    ],
)
def test_get_spherical_radius(pos):

    expected = pos.get_radius()

    assert _get_spherical_radius(pos, False) == expected
