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
    vec = vectors.Vector([1, 2, 3], GPSTime(0, 0), "ECI")
    assert isinstance(vec.coordinates, np.ndarray)


def test_vec_post_init_no_LLA():
    with pytest.raises(ValueError):
        vec = vectors.Vector(np.array([1, 2, 3]), 0, "LLA")


def test_vec_post_init_randframe():
    with pytest.raises(ValueError):
        vec = vectors.Vector(np.array([1, 2, 3]), 0, "foo")


def test_vec_post_init_reshape():
    vec = vectors.Vector(np.array([[1], [2], [3]]), 0, "ECI")
    assert np.shape(vec.coordinates) == (3,)


def test_vec_post_init_too_many_dimensions():
    with pytest.raises(ValueError):
        vec = vectors.Vector(np.array([[[1]], [[2]], [[3]]]), 0, "ECI")


@pytest.mark.parametrize(
    "vec", [vectors.Vector(np.array([93, 1232, 2]), GPSTime(1232, 122), "ECEF")]
)
def test_magnitude(vec):
    assert vec.magnitude == np.linalg.norm(vec.coordinates)


@pytest.mark.parametrize("array", [np.array([1, 1, 1]), np.array([9, 73, 83.11])])
def test_neg(array):
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
    # not exactly zero
    assert np.isclose(vecOne.dot_product(vecTwo), expected)


def test_dot_type():
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
    assert (vecOne.cross_product(vecTwo).coordinates == expected).all()


def test_get_vec_coordinates():

    vecECI = vectors.Vector(
        np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECI"
    )
    assert np.allclose(
        trans.position_transform("ECI", "ECEF", vecECI.coordinates, vecECI.frame_time),
        vecECI.get_vector("ECEF").coordinates,
    )


def test_get_vec_frame():

    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    assert vecECI.get_vector("ECEF").frame == "ECEF"


def test_get_vec_LLA():

    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    with pytest.raises(ValueError):
        vecECI.get_vector("LLA")


def test_switch_frame_coordinates():

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

    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    vecECEF = copy.copy(vecECI)
    vecECEF.switch_frame("ECEF")
    assert vecECEF.frame == "ECEF"


def test_switch_frame_LLA():

    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    with pytest.raises(ValueError):
        vecECI.switch_frame("LLA")


def test_update_frame_time_ECI():
    vecECI = vectors.Vector(
        np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECI"
    )
    newVecECI = copy.copy(vecECI)
    newVecECI.update_frame_time(GPSTime(2111, 259200))
    assert np.array(
        (trans.add_weeks_eci(2, vecECI.coordinates)) == newVecECI.coordinates
    ).all()


def test_update_frame_time_ECEF():
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

    vecResult = vec * num

    assert (vecResult.coordinates == num * vec.coordinates).all()


def test_mult_type_error():

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

    unVec = vectors.UnitVector.from_vector(vec)
    assert (unVec.coordinates == expected.coordinates).all()


@pytest.mark.parametrize(
    "vec", [vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 0.0), "ECI")]
)
def test_unit_switch_frame(vec):

    vecECI = vec
    vecECEF = copy.copy(vecECI)
    vecECEF.switch_frame("ECEF")
    unVecECI = vectors.UnitVector.from_vector(vecECI)
    unVecECEF = vectors.UnitVector.from_vector(vecECEF)
    unVecECI.switch_frame("ECEF")
    assert np.all(unVecECI.coordinates == unVecECEF.coordinates)


def test_unit_switch_frame_LLA():

    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    with pytest.raises(ValueError):
        vecECI.switch_frame("LLA")


@pytest.mark.parametrize(
    "vec", [vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")]
)
def test_get_unit(vec):

    vecECI = vec
    vecECEF = vecECI.get_vector("ECEF")
    unVecECI = vectors.UnitVector.from_vector(vecECI)
    unVecECEF = vectors.UnitVector.from_vector(vecECEF)

    assert np.allclose(
        unVecECI.coordinates, unVecECEF.get_unit_vector("ECI").coordinates
    )


def test_unit_get_vec_LLA():

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

    vec = unVec * num
    assert isinstance(vec, vectors.Vector)


def test_unit_mult_type_error():

    with pytest.raises(TypeError):
        unVec = vectors.UnitVector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
        unVec * ("not a number")
