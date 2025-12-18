# Copyright (c) 2022 The Aerospace Corporation

import numpy as np
import pytest
import copy
from gps_frames import vectors
from gps_frames import velocity
from gps_frames import position
from gps_frames import transforms as trans
from gps_frames.parameters import GeoidData, EarthParam
from gps_time import GPSTime

############### position tests ###############


@pytest.mark.parametrize(
    "pos1, pos2",
    [
        (
            position.Position(
                np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECI"
            ),
            position.Position(
                np.array([300, 27, -36], dtype=float), GPSTime(2109, 259200), "ECI"
            ),
        ),
        (
            position.Position(
                np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECEF"
            ),
            position.Position(
                np.array([300, 27, -36], dtype=float), GPSTime(2109, 8712), "ECI"
            ),
        ),
    ],
)
def test_distance(pos1, pos2):
    """Test distance calculation between two positions."""
    pos2 = pos2.get_position(pos1.frame)
    pos2.update_frame_time(pos1.frame_time)
    distCalc = position.distance(pos1, pos2)
    distance_vec = pos1.coordinates - pos2.coordinates

    assert np.isclose(np.linalg.norm(distance_vec), distCalc)


def test_pos_post_reshape():
    """Test that position coordinates are reshaped to (3,)."""
    pos = position.Position(np.array([[1], [2], [3]]), GPSTime(0, 0), "ECI")
    assert np.shape(pos.coordinates) == (3,)


def test_pos_post_too_many_dimension():
    """Test that initializing Position with wrong dimensions raises ValueError."""
    with pytest.raises(ValueError):
        pos = position.Position(np.array([[[1]], [[2]], [[3]]]), 0, "ECI")


def test_posSwitch():
    """Test frame switching logic for Position."""
    pos = position.Position(
        np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECI"
    )
    posECEF = copy.copy(pos)
    posECEF.switch_frame("ECEF")

    assert np.allclose(
        posECEF.coordinates,
        trans.position_transform(pos.frame, "ECEF", pos.coordinates, pos.frame_time),
    )


def test_pos_update_frame_time_ECI():
    """Test updating frame time for ECI position."""
    pos = position.Position(
        np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECI"
    )
    pos.update_frame_time(GPSTime(2111, 259200))
    assert np.allclose(
        pos.coordinates, trans.add_weeks_eci(2, np.array((1, 2, 3), dtype=float))
    )


def test_pos_update_frame_time_ECEF():
    """Test updating frame time for ECEF position."""
    pos = position.Position(
        np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECEF"
    )
    pos.update_frame_time(GPSTime(2111, 259200))
    assert np.allclose(
        pos.coordinates,
        trans.rotate_ecef(
            GPSTime(2109, 259200),
            GPSTime(2111, 259200),
            np.array((1, 2, 3), dtype=float),
        ),
    )


def test_pos_update_frame_time_LLA():
    """Test updating frame time for LLA position."""
    pos = position.Position(
        np.array([0, 0, 100], dtype=float), GPSTime(2109, 259200), "LLA"
    )
    posLLA = copy.copy(pos)
    posLLA.update_frame_time(GPSTime(2112, 259200))

    pos.switch_frame("ECEF")
    pos.coordinates = trans.rotate_ecef(
        GPSTime(2109, 259200), GPSTime(2112, 259200), tuple(pos.coordinates)
    )
    pos.switch_frame("LLA")
    assert np.allclose(posLLA.coordinates, pos.coordinates)


def test_get_alt_msl():
    """Test getting altitude above Mean Sea Level."""
    posECI = position.Position(np.array([10e06, 0, 0]), GPSTime(200, 0), "ECI")
    posLLA = posECI.get_position("LLA")

    assert posLLA.get_altitude_msl() == posLLA.coordinates[
        2
    ] - GeoidData.get_geoid_height(posLLA.coordinates[0], posLLA.coordinates[1])


@pytest.mark.parametrize(
    "pos",
    [
        position.Position(np.array([10e06, 0, 0]), GPSTime(200, 0), "ECI"),
        position.Position(np.array([34, 12, 0]), GPSTime(200, 0), "LLA"),
        position.Position(np.array([10e06, 3213, 0]), GPSTime(200, 0), "ECEF"),
    ],
)
def test_get_alt_hae(pos):
    """Test getting altitude Height Above Ellipsoid."""
    posLLA = pos.get_position("LLA")
    assert posLLA.get_altitude_hae() == posLLA.coordinates[2]


@pytest.mark.parametrize(
    "pos",
    [
        position.Position(
            np.array([10e06, 3213, 0], dtype=float), GPSTime(200, 0), "ECEF"
        ),
        position.Position(
            np.array([124213, 8982e07, 9871623], dtype=float), GPSTime(200, 0), "ECI"
        ),
        position.Position(
            np.array([48, 32, 689687], dtype=float), GPSTime(200, 0), "LLA"
        ),
    ],
)
def test_get_radius(pos):
    """Test getting radius from center of Earth."""
    eci = pos.get_position("ECI")
    assert np.linalg.norm(eci.coordinates) == pos.get_radius()


@pytest.mark.parametrize(
    "pos",
    [
        position.Position(
            np.array([10e06, 3213, 0], dtype=float), GPSTime(200, 0), "ECEF"
        ),
        position.Position(
            np.array([124213, 8982e07, 9871623], dtype=float), GPSTime(200, 0), "ECI"
        ),
        position.Position(
            np.array([48, 32, 689687], dtype=float), GPSTime(200, 0), "LLA"
        ),
    ],
)
def test_get_altitude_spherical(pos):
    """Test getting spherical altitude (radius - Earth radius)."""
    eci = pos.get_position("ECI")
    assert pos.get_altitude_spherical() == pos.get_radius() - EarthParam.r_e


def test_from_vector_coordinates():
    """Test creating Position from Vector (coordinates check)."""
    vec = vectors.Vector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    pos = position.Position.from_vector(vec)
    assert (pos.coordinates == vec.coordinates).all()


def test_hash():
    """Test hash function (mostly ensuring no crash)."""
    pos = position.Position(np.array([10e06, 0, 0]), GPSTime(200, 0), "ECI")
    pos.__hash__()


def test_from_vector_frame():
    """Test creating Position from Vector (frame check)."""
    vec = vectors.Vector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    pos = position.Position.from_vector(vec)
    assert pos.frame == vec.frame


def test_from_vector_frame_time():
    """Test creating Position from Vector (time check)."""
    vec = vectors.Vector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    pos = position.Position.from_vector(vec)
    assert pos.frame_time == vec.frame_time


def test_to_vector_coordinates():
    """Test converting Position to Vector (coordinates check)."""
    pos = position.Position(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    vec = position.Position.to_vector(pos)
    assert (pos.coordinates == vec.coordinates).all()


def test_to_vector_frame():
    """Test converting Position to Vector (frame check)."""
    pos = position.Position(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    vec = position.Position.to_vector(pos)
    assert pos.frame == vec.frame


def test_to_vector_frame_time():
    """Test converting Position to Vector (time check)."""
    pos = position.Position(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    vec = position.Position.to_vector(pos)
    assert pos.frame_time == vec.frame_time


def test_pos_eq_diff_class():
    """Test that Position is not equal to a Vector."""
    pos = position.Position(np.array([0, 0, 0]), GPSTime(200, 0), "ECI")
    vec = position.Position.to_vector(pos)
    with pytest.raises(TypeError):
        pos.__eq__(vec)


@pytest.mark.parametrize(
    "pos1, pos2, expected",
    [
        (
            position.Position(np.array([0, 0, 0], dtype=float), GPSTime(200, 0), "ECI"),
            position.Position(
                np.array([1e-6, 0, 0], dtype=float), GPSTime(200, 0), "ECI"
            ),
            False,
        ),
        (
            position.Position(np.array([0, 0, 0], dtype=float), GPSTime(200, 0), "ECI"),
            position.Position(
                np.array([1e-7, 0, 0], dtype=float), GPSTime(200, 0), "ECI"
            ),
            True,
        ),
        (
            position.Position(np.array([1, 0, 0], dtype=float), GPSTime(200, 0), "ECI"),
            position.Position(np.array([0, 0, 0], dtype=float), GPSTime(201, 0), "ECI"),
            False,
        ),
    ],
)
def test_pos_eq(pos1, pos2, expected):
    """Test position equality (with tolerance and time check)."""
    assert pos1.__eq__(pos2) == expected


def test_pos_add_not_vec():
    """Test that adding two positions raises TypeError."""
    pos = position.Position(np.array([4, 7, 1]), GPSTime(0, 0), "ECI")
    with pytest.raises(TypeError):
        pos.__add__(pos)


def test_pos_add():
    """Test adding a Vector to a Position."""
    pos = position.Position(np.array([4, 7, 1]), GPSTime(0, 0), "ECI")
    vec = vectors.Vector(np.array([16, 39, -8]), GPSTime(0, 0), "ECI")
    assert (pos.__add__(vec).coordinates == pos.coordinates + vec.coordinates).any()


def test_pos_sub_not_vec():
    """Test that subtracting a Position from a Position raises TypeError (should use distance?)."""
    # Or maybe it's just not implemented to return a Vector?
    pos = position.Position(np.array([4, 7, 1]), GPSTime(0, 0), "ECI")
    with pytest.raises(TypeError):
        pos.__sub__(pos)


def test_pos_sub():
    """Test subtracting a Vector from a Position."""
    pos = position.Position(np.array([4, 7, 1]), GPSTime(0, 0), "ECI")
    vec = vectors.Vector(np.array([16, 39, -8]), GPSTime(0, 0), "ECI")
    assert (pos.__sub__(vec).coordinates == pos.coordinates - vec.coordinates).any()
