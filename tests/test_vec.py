# Copyright (c) 2022 The Aerospace Corporation

import numpy as np
import pytest
import copy

from gps_frames import vectors
from gps_frames import velocity
from gps_frames import position
from gps_frames import transforms as trans
from gps_time import GPSTime
from gps_frames.parameters import GeoidData

############### vector tests ###############

# no tests for serializable vector


def test_vec_post_init_convert_to_nparray():
    """Test that coordinates are converted to numpy array upon initialization."""
    vec = vectors.Vector([1, 2, 3], GPSTime(0, 0), "ECI")
    assert isinstance(vec.coordinates, np.ndarray)


def test_vec_post_init_no_LLA():
    """Test that initializing Vector with 'LLA' frame raises ValueError."""
    with pytest.raises(ValueError):
        vec = vectors.Vector(np.array([1, 2, 3]), 0, "LLA")


def test_vec_post_init_randframe():
    """Test that initializing Vector with invalid frame raises ValueError."""
    with pytest.raises(ValueError):
        vec = vectors.Vector(np.array([1, 2, 3]), 0, "foo")


def test_vec_post_init_reshape():
    """Test that coordinates are reshaped to (3,) flat array."""
    vec = vectors.Vector(np.array([[1], [2], [3]]), 0, "ECI")
    assert np.shape(vec.coordinates) == (3,)


def test_vec_post_init_too_many_dimensions():
    """Test that initializing with wrong dimensions raises ValueError."""
    with pytest.raises(ValueError):
        vec = vectors.Vector(np.array([[[1]], [[2]], [[3]]]), 0, "ECI")


@pytest.mark.parametrize(
    "vec", [vectors.Vector(np.array([93, 1232, 2]), GPSTime(1232, 122), "ECEF")]
)
def test_magnitude(vec):
    """Test standard magnitude calculation."""
    assert vec.magnitude == np.linalg.norm(vec.coordinates)


@pytest.mark.parametrize("array", [np.array([1, 1, 1]), np.array([9, 73, 83.11])])
def test_neg(array):
    """Test negation operator."""
    vec = vectors.Vector(array, GPSTime(0, 0), "ECI")
    negVec = vec.__neg__()
    assert (negVec.coordinates == -vec.coordinates).all()


@pytest.mark.parametrize(
    "vecOne, array, expected",
    [
        (
            vectors.Vector(np.array([-1, 2, 5]), GPSTime(0, 0), "ECI"),
            np.array([1, 1, 1]),
            6,
        )
    ],
)
def test_dot_basic(vecOne, array, expected):
    """Test dot product with numpy array."""
    assert vecOne.dot_product(array) == expected


@pytest.mark.parametrize(
    "vecOne, vecTwo, expected",
    [
        (
            vectors.Vector(np.array([-1, 2, 5]), GPSTime(0, 0), "ECI"),
            vectors.Vector(np.array([1, 1, 1]), GPSTime(0, 0), "ECI"),
            6,
        )
    ],
)
def test_dot_twovecs(vecOne, vecTwo, expected):
    """Test dot product with another Vector."""
    assert vecOne.dot_product(vecTwo) == expected


@pytest.mark.parametrize(
    "vecOne, vecTwo, expected",
    [
        (
            vectors.Vector(np.array([1, 0, 0]), GPSTime(0, 0), "ECI"),
            vectors.Vector(np.array([1, 0, 0]), GPSTime(0, 21541.024725943207), "ECEF"),
            0,
        )
    ],
)
def test_dot_with_diff_frames(vecOne, vecTwo, expected):
    """Test dot product behavior when vectors are in different frames."""
    # not exactly zero
    assert np.isclose(vecOne.dot_product(vecTwo), expected)


def test_dot_type():
    """Test that dot product raises TypeError for invalid input types."""
    vec = vectors.Vector(np.array([0, 0, 0]), GPSTime(0, 0), "ECI")
    with pytest.raises(TypeError):
        vec.dot_product("This is not a vector or an array")


@pytest.mark.parametrize(
    "vecOne, vecTwo, expected",
    [
        (
            vectors.Vector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI"),
            vectors.Vector(np.array([4, 5, 6]), GPSTime(0, 0), "ECI"),
            np.array([-3, 6, -3]),
        )
    ],
)
def test_cross(vecOne, vecTwo, expected):
    """Test cross product calculation."""
    assert (vecOne.cross_product(vecTwo).coordinates == expected).all()


def test_get_vec_coordinates():
    """Test get_vector coordinates tranformation result."""
    vecECI = vectors.Vector(
        np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECI"
    )
    assert np.allclose(
        trans.position_transform("ECI", "ECEF", vecECI.coordinates, vecECI.frame_time),
        vecECI.get_vector("ECEF").coordinates,
    )


def test_get_vec_frame():
    """Test get_vector returns vector in correct frame."""
    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    assert vecECI.get_vector("ECEF").frame == "ECEF"


def test_get_vec_LLA():
    """Test that get_vector raises ValueError for LLA frame."""
    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    with pytest.raises(ValueError):
        vecECI.get_vector("LLA")


def test_switch_frame_coordinates():
    """Test switch_frame updates coordinates correctly."""
    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    vecECEF = copy.copy(vecECI)
    vecECEF.switch_frame("ECEF")

    assert (
        vecECEF.coordinates
        == np.array(
            trans.position_transform(
                "ECI", "ECEF", tuple(vecECI.coordinates), vecECI.frame_time
            )
        )
    ).all()


def test_switch_frame_frame():
    """Test switch_frame updates the frame attribute."""
    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    vecECEF = copy.copy(vecECI)
    vecECEF.switch_frame("ECEF")
    assert vecECEF.frame == "ECEF"


def test_switch_frame_LLA():
    """Test that switch_frame raises ValueError for LLA."""
    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    with pytest.raises(ValueError):
        vecECI.switch_frame("LLA")


def test_update_frame_time_ECI():
    """Test update_frame_time for ECI frame (rotates frame)."""
    vecECI = vectors.Vector(
        np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECI"
    )
    newVecECI = copy.copy(vecECI)
    newVecECI.update_frame_time(GPSTime(2111, 259200))
    assert np.array(
        (trans.add_weeks_eci(2, vecECI.coordinates)) == newVecECI.coordinates
    ).all()


def test_update_frame_time_ECEF():
    """Test update_frame_time for ECEF frame (rotates coordinates)."""
    # testECEF
    vecECEF = vectors.Vector(
        np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECEF"
    )
    newVecECEF = copy.copy(vecECEF)
    newVecECEF.update_frame_time(GPSTime(2111, 259200))
    assert np.allclose(
        trans.rotate_ecef(
            GPSTime(2109, 259200), GPSTime(2111, 259200), vecECEF.coordinates
        ),
        newVecECEF.coordinates,
    )


@pytest.mark.parametrize(
    "vecOne, vecTwo",
    [
        (
            vectors.Vector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI"),
            vectors.Vector(np.array([4, 5, 6]), GPSTime(0, 0), "ECI"),
        ),
        (
            vectors.Vector(np.array([0, 0, 200]), GPSTime(0, 0), "ECI"),
            vectors.Vector(np.array([4, 5, 6]), GPSTime(0, 98748923), "ECEF"),
        ),
    ],
)
def test_add(vecOne, vecTwo):
    """Test vector addition (handles frame transformation implicitly)."""
    vecResult = vecOne.__add__(vecTwo)

    vecTwo.switch_frame(vecOne.frame)
    vecTwo.update_frame_time(vecOne.frame_time)
    assert (vecResult.coordinates == vecOne.coordinates + vecTwo.coordinates).all()


@pytest.mark.parametrize(
    "vecOne, vecTwo",
    [
        (
            vectors.Vector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI"),
            vectors.Vector(np.array([4, 5, 6]), GPSTime(0, 0), "ECI"),
        ),
        (
            vectors.Vector(np.array([1, 2, 3]), GPSTime(3431, 8834), "ECI"),
            vectors.Vector(np.array([4, 5, 6]), GPSTime(0, 0), "ECEF"),
        ),
    ],
)
def test_sub(vecOne, vecTwo):
    """Test vector subtraction (handles frame transformation implicitly)."""
    vecResult = vecOne.__sub__(vecTwo)

    vecTwo.switch_frame(vecOne.frame)
    vecTwo.update_frame_time(vecOne.frame_time)
    assert (vecResult.coordinates == vecOne.coordinates - vecTwo.coordinates).all()


@pytest.mark.parametrize(
    "vec, num",
    [
        (vectors.Vector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI"), 6),
        (
            vectors.Vector(np.array([9274, 83.122, -814.3]), GPSTime(42, 1232), "ECEF"),
            -6.676,
        ),
    ],
)
def test_mult(vec, num):
    """Test scalar multiplication."""
    vecResult = vec * num

    assert (vecResult.coordinates == num * vec.coordinates).all()


def test_mult_type_error():
    """Test that scalar multiplication checks types."""
    with pytest.raises(TypeError):
        vec = vectors.Vector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
        vec * ("not a number")


@pytest.mark.parametrize(
    "unVec",
    [
        vectors.UnitVector(np.array([7, 48, -17]), GPSTime(0, 0), "ECI"),
        vectors.UnitVector(
            np.array([-2093.34, 9993123, 17.177732]), GPSTime(2312, 232), "ECEF"
        ),
    ],
)
def test_unit_norm(unVec):
    """Test that UnitVector normalizes coordinates on initialization."""
    assert np.abs(np.linalg.norm(unVec.coordinates) - 1.0) < 1e-12


@pytest.mark.parametrize(
    "vec, expected",
    [
        (
            vectors.Vector(np.array([7, 48, -17]), GPSTime(0, 0), "ECI"),
            vectors.UnitVector(np.array([7, 48, -17]), GPSTime(0, 0), "ECI"),
        ),
        (
            vectors.Vector(
                np.array([-2093.34, 9993123, 17.177732]), GPSTime(2312, 232), "ECEF"
            ),
            vectors.UnitVector(
                np.array([-2093.34, 9993123, 17.177732]), GPSTime(2312, 232), "ECEF"
            ),
        ),
    ],
)
def test_from_vector(vec, expected):
    """Test creating UnitVector from Vector."""
    unVec = vectors.UnitVector.from_vector(vec)
    assert (unVec.coordinates == expected.coordinates).all()


@pytest.mark.parametrize(
    "vec", [vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 0.0), "ECI")]
)
def test_unit_switch_frame(vec):
    """Test switching frame for UnitVector."""
    vecECI = vec
    vecECEF = copy.copy(vecECI)
    vecECEF.switch_frame("ECEF")
    unVecECI = vectors.UnitVector.from_vector(vecECI)
    unVecECEF = vectors.UnitVector.from_vector(vecECEF)
    unVecECI.switch_frame("ECEF")
    assert np.all(unVecECI.coordinates == unVecECEF.coordinates)


def test_unit_switch_frame_LLA():
    """Test switch_frame not allowed for LLA on UnitVector."""
    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    with pytest.raises(ValueError):
        vecECI.switch_frame("LLA")


@pytest.mark.parametrize(
    "vec", [vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")]
)
def test_get_unit(vec):
    """Test get_unit_vector behavior."""
    vecECI = vec
    vecECEF = vecECI.get_vector("ECEF")
    unVecECI = vectors.UnitVector.from_vector(vecECI)
    unVecECEF = vectors.UnitVector.from_vector(vecECEF)

    assert np.allclose(
        unVecECI.coordinates, unVecECEF.get_unit_vector("ECI").coordinates
    )


def test_unit_get_vec_LLA():
    """Test get_vector raises ValueError for LLA for UnitVector (which inherits from Vector but conceptually similar)."""
    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    with pytest.raises(ValueError):
        vecECI.get_vector("LLA")


@pytest.mark.parametrize(
    "unVec, num",
    [
        (vectors.UnitVector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI"), 6),
        (
            vectors.UnitVector(
                np.array([9274, 83.122, -814.3]), GPSTime(42, 1232), "ECEF"
            ),
            -6.676,
        ),
    ],
)
def test_unit_mult(unVec, num):
    """Test scalar multiplication on UnitVector returns Vector (not UnitVector)."""
    vec = unVec * num
    assert (vec.coordinates == num * unVec.coordinates).all()


@pytest.mark.parametrize(
    "unVec, num",
    [
        (vectors.UnitVector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI"), 6),
        (
            vectors.UnitVector(
                np.array([9274, 83.122, -814.3]), GPSTime(42, 1232), "ECEF"
            ),
            -6.676,
        ),
    ],
)
def test_unit_mult_return_type(unVec, num):
    """Test scalar multiplication on UnitVector returns Vector object."""
    vec = unVec * num
    assert isinstance(vec, vectors.Vector)


def test_unit_mult_type_error():
    """Test scalar multiplication type checking."""
    with pytest.raises(TypeError):
        unVec = vectors.UnitVector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
        unVec * ("not a number")
