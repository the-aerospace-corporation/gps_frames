# Copyright (c) 2022 The Aerospace Corporation
"""Unit tests for the Position class in gps_frames.position.

Verifies distance calculations, frame transformations, time evolution,
altitude definitions (HAE, MSL, spherical), Vector interoperability,
and affine point arithmetic.
"""

import copy
import numpy as np
import pytest

from gps_frames import position, transforms as trans, vectors
from gps_frames.parameters import EarthParam, GeoidData
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
    """Verify 3D Euclidean distance calculation between two Position instances.

    Testing:
        position.distance(pos1, pos2) computes the Euclidean separation norm:
            d = ||r_1 - r_2|| = sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)
        Before computing separation, pos2 is transformed into pos1's frame and synchronized
        to pos1's frame epoch via pos2.get_position(pos1.frame) and pos2.update_frame_time(pos1.frame_time).

    Expected Result:
        - Param 1 (Same frame ECI, same epoch week 2109, sec 259200):
            pos1 = [1, 2, 3], pos2 = [300, 27, -36]
            delta_vec = [1 - 300, 2 - 27, 3 - (-36)] = [-299, -25, 39]
            ||delta_vec|| = sqrt((-299)^2 + (-25)^2 + 39^2)
                          = sqrt(89401 + 625 + 1521) = sqrt(91547) ~= 302.567347875 m
        - Param 2 (Different frames ECEF vs ECI, different epochs sec 259200 vs sec 8712):
            pos2 is converted from ECI to ECEF and rotated by Earth rotation rate omega_e across
            delta_t = 259200 - 8712 = 250488 s, yielding exact agreement with np.linalg.norm(distance_vec).
    """
    pos2 = pos2.get_position(pos1.frame)
    pos2.update_frame_time(pos1.frame_time)
    distCalc = position.distance(pos1, pos2)
    distance_vec = pos1.coordinates - pos2.coordinates

    assert np.isclose(np.linalg.norm(distance_vec), distCalc)


def test_pos_post_reshape():
    """Verify that 2D column arrays (3, 1) are reshaped into 1D (3,) coordinate arrays.

    Testing:
        Position.__post_init__() coordinate flattening logic for non-1D 3-element arrays.

    Expected Result:
        Input shape (3, 1) with values [[1], [2], [3]] is reshaped to shape (3,) with values [1, 2, 3].
    """
    pos = position.Position(np.array([[1], [2], [3]]), GPSTime(0, 0), "ECI")
    assert np.shape(pos.coordinates) == (3,)


def test_pos_post_too_many_dimension():
    """Verify that initializing Position with array dimensions greater than 2 raises ValueError.

    Testing:
        Input dimension check in Position.__post_init__().

    Expected Result:
        3D array with shape (3, 1, 1) triggers ValueError("coordinates must be 1- or 2-D").
    """
    with pytest.raises(ValueError):
        pos = position.Position(np.array([[[1]], [[2]], [[3]]]), 0, "ECI")


def test_posSwitch():
    """Verify in-place coordinate frame transformation via switch_frame().

    Testing:
        Position.switch_frame("ECEF") updates pos.coordinates in place by calling
        transforms.position_transform(pos.frame, "ECEF", pos.coordinates, pos.frame_time).

    Expected Result:
        An ECI position [1, 2, 3] at week 2109, sec 259200 is rotated by Earth's rotation angle:
            theta = omega_e * 259200 s ~= 7.292115e-5 * 259200 rad ~= 18.900 rad
        posECEF.coordinates matches trans.position_transform("ECI", "ECEF", [1, 2, 3], t).
    """
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
    """Verify time updating for an inertial (ECI) position across multi-week epochs.

    Testing:
        Position.update_frame_time() for ECI positions. ECI coordinates are invariant under
        intra-week time changes, but multi-week shifts apply Greenwich sidereal time adjustment
        via trans.add_weeks_eci(delta_weeks, coordinates).

    Expected Result:
        Updating from week 2109 to 2111 (delta_weeks = 2) applies a rotation of angle:
            theta = omega_e * (2 * 604800 s) ~= 88.2057 rad
        pos.coordinates matches trans.add_weeks_eci(2, [1, 2, 3]).
    """
    pos = position.Position(
        np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECI"
    )
    pos.update_frame_time(GPSTime(2111, 259200))
    assert np.allclose(
        pos.coordinates, trans.add_weeks_eci(2, np.array((1, 2, 3), dtype=float))
    )


def test_pos_update_frame_time_ECEF():
    """Verify time updating for an Earth-fixed (ECEF) position.

    Testing:
        Position.update_frame_time() for ECEF positions. ECEF coordinates rotate with Earth
        relative to the inertial frame over elapsed time delta_t via trans.rotate_ecef(t_old, t_new, coords).

    Expected Result:
        Advancing from week 2109, 259200 s to week 2111, 259200 s (delta_t = 1209600 s)
        rotates coordinates about the Z-axis by theta = omega_e * delta_t.
        pos.coordinates matches trans.rotate_ecef(t_old, t_new, [1, 2, 3]).
    """
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
    """Verify time updating for a geodetic (LLA) position.

    Testing:
        Position.update_frame_time() for LLA coordinates. Because LLA represents surface-fixed
        curvilinear coordinates, updating frame time transforms LLA -> ECEF, applies rotate_ecef,
        and transforms back ECEF -> LLA.

    Expected Result:
        posLLA.coordinates matches the result of converting to ECEF, rotating by Earth rotation,
        and converting back to LLA coordinates [lat, lon, alt].
    """
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
    """Verify calculation of Mean Sea Level (MSL) altitude from geodetic coordinates.

    Testing:
        Position.get_altitude_msl() calculates orthometric height relative to the geoid:
            h_MSL = h_HAE - N(lat, lon)
        where N is the geoid undulation from the EGM-96 spherical harmonic model (GeoidData).

    Expected Result:
        For ECI position [10e6, 0, 0] at week 200, converted to LLA (lat, lon, alt_hae),
        posLLA.get_altitude_msl() == alt_hae - GeoidData.get_geoid_height(lat, lon).
    """
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
    """Verify calculation of Height Above Ellipsoid (HAE) altitude.

    Testing:
        Position.get_altitude_hae() extracts the third coordinate (index 2) of the LLA representation,
        representing geometric distance normal to the WGS84 reference ellipsoid.

    Expected Result:
        posLLA.get_altitude_hae() equals posLLA.coordinates[2] (meters above WGS84 ellipsoid).
    """
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
    """Verify calculation of radial distance from Earth's center of mass.

    Testing:
        Position.get_radius() computes the Euclidean norm of the position in the inertial frame:
            R = ||r_ECI|| = sqrt(x_eci^2 + y_eci^2 + z_eci^2)

    Expected Result:
        pos.get_radius() equals np.linalg.norm(pos.get_position("ECI").coordinates).
    """
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
    """Verify calculation of spherical altitude above a spherical Earth of radius r_e.

    Testing:
        Position.get_altitude_spherical() computes:
            h_sph = R - EarthParam.r_e
        where R is the geocentric radius and EarthParam.r_e = 6378137.0 m (WGS84 equatorial radius).

    Expected Result:
        pos.get_altitude_spherical() equals pos.get_radius() - 6378137.0.
    """
    eci = pos.get_position("ECI")
    assert pos.get_altitude_spherical() == pos.get_radius() - EarthParam.r_e


def test_from_vector_coordinates():
    """Verify that Position.from_vector copies coordinates correctly from a Vector.

    Testing:
        Position.from_vector() constructor factory.

    Expected Result:
        Position coordinates equal Vector coordinates [1.0, 2.0, 3.0].
    """
    vec = vectors.Vector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    pos = position.Position.from_vector(vec)
    assert (pos.coordinates == vec.coordinates).all()


def test_hash():
    """Verify Position.__hash__() execution.

    Testing:
        Hashability of Position objects based on tuple(coordinates), frame, and frame_time.

    Expected Result:
        hash(pos) produces an integer without raising unhashable type errors.
    """
    pos = position.Position(np.array([10e06, 0, 0]), GPSTime(200, 0), "ECI")
    pos.__hash__()


def test_from_vector_frame():
    """Verify that Position.from_vector preserves the frame from the Vector.

    Testing:
        Position.from_vector() coordinate frame propagation.

    Expected Result:
        pos.frame == "ECI".
    """
    vec = vectors.Vector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    pos = position.Position.from_vector(vec)
    assert pos.frame == vec.frame


def test_from_vector_frame_time():
    """Verify that Position.from_vector preserves the frame_time from the Vector.

    Testing:
        Position.from_vector() frame time propagation.

    Expected Result:
        pos.frame_time == GPSTime(0, 0).
    """
    vec = vectors.Vector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    pos = position.Position.from_vector(vec)
    assert pos.frame_time == vec.frame_time


def test_to_vector_coordinates():
    """Verify that Position.to_vector converts Position coordinates to a Vector.

    Testing:
        Position.to_vector() method converting affine point to a Vector displacement from origin.

    Expected Result:
        vec.coordinates equals pos.coordinates [1.0, 2.0, 3.0].
    """
    pos = position.Position(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    vec = position.Position.to_vector(pos)
    assert (pos.coordinates == vec.coordinates).all()


def test_to_vector_frame():
    """Verify that Position.to_vector preserves the frame string.

    Testing:
        Position.to_vector() frame attribute propagation.

    Expected Result:
        vec.frame == "ECI".
    """
    pos = position.Position(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    vec = position.Position.to_vector(pos)
    assert pos.frame == vec.frame


def test_to_vector_frame_time():
    """Verify that Position.to_vector preserves the frame_time attribute.

    Testing:
        Position.to_vector() frame_time propagation.

    Expected Result:
        vec.frame_time == GPSTime(0, 0).
    """
    pos = position.Position(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    vec = position.Position.to_vector(pos)
    assert pos.frame_time == vec.frame_time


def test_pos_eq_diff_class():
    """Verify that comparing Position to an object of another class raises TypeError.

    Testing:
        Type safety in Position.__eq__().

    Expected Result:
        pos == vec raises TypeError("other value must be a Position").
    """
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
    """Verify Position.__eq__() coordinate tolerance (1 um) and frame time matching.

    Testing:
        Position equality requires matching frame_time, matching frame, and Euclidean distance < 1e-6 m.

    Expected Result:
        - pos1 = [0, 0, 0], pos2 = [1e-6, 0, 0] (separation = 1.0 um):
            Separation >= 1e-6 tolerance threshold -> False.
        - pos1 = [0, 0, 0], pos2 = [1e-7, 0, 0] (separation = 0.1 um):
            Separation < 1e-6 tolerance threshold -> True.
        - pos1 at week 200, pos2 at week 201:
            Mismatched frame times -> False.
    """
    assert pos1.__eq__(pos2) == expected


def test_pos_add_not_vec():
    """Verify that adding two Position objects raises TypeError.

    Testing:
        Affine space geometric rule: Points cannot be added to points (Position + Position is undefined;
        only Point + Displacement Vector = Point is defined).

    Expected Result:
        pos + pos raises TypeError.
    """
    pos = position.Position(np.array([4, 7, 1]), GPSTime(0, 0), "ECI")
    with pytest.raises(TypeError):
        pos.__add__(pos)


def test_pos_add():
    """Verify adding a displacement Vector to a Position.

    Testing:
        Position.__add__(Vector) computes:
            r_new = r_old + v
        displacing the point in 3D Euclidean space.

    Expected Result:
        pos = [4, 7, 1], vec = [16, 39, -8]
        r_new = [4 + 16, 7 + 39, 1 + (-8)] = [20, 46, -7]
    """
    pos = position.Position(np.array([4, 7, 1]), GPSTime(0, 0), "ECI")
    vec = vectors.Vector(np.array([16, 39, -8]), GPSTime(0, 0), "ECI")
    assert (pos.__add__(vec).coordinates == pos.coordinates + vec.coordinates).any()


def test_pos_sub_not_vec():
    """Verify that subtracting a Position from a Position raises TypeError.

    Testing:
        Type validation in Position.__sub__(). Position subtraction directly via '-' is disallowed;
        users must use position.distance(pos1, pos2) or convert to Vector.

    Expected Result:
        pos - pos raises TypeError.
    """
    pos = position.Position(np.array([4, 7, 1]), GPSTime(0, 0), "ECI")
    with pytest.raises(TypeError):
        pos.__sub__(pos)


def test_pos_sub():
    """Verify subtracting a displacement Vector from a Position.

    Testing:
        Position.__sub__(Vector) computes:
            r_new = r_old - v
        translating the point in reverse along the vector.

    Expected Result:
        pos = [4, 7, 1], vec = [16, 39, -8]
        r_new = [4 - 16, 7 - 39, 1 - (-8)] = [-12, -32, 9]
    """
    pos = position.Position(np.array([4, 7, 1]), GPSTime(0, 0), "ECI")
    vec = vectors.Vector(np.array([16, 39, -8]), GPSTime(0, 0), "ECI")
    assert (pos.__sub__(vec).coordinates == pos.coordinates - vec.coordinates).any()
