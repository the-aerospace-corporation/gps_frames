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

# Tests for Vector and UnitVector classes in gps_frames.vectors


def test_vec_post_init_convert_to_nparray():
    """Verify that coordinates passed as a standard Python list are converted to a NumPy array.
    
    Testing:
        Vector.__post_init__() type casting logic.
    Expected Result:
        vec.coordinates becomes an instance of np.ndarray with shape (3,) and values [1.0, 2.0, 3.0].
    """
    vec = vectors.Vector([1, 2, 3], GPSTime(0, 0), "ECI")
    assert isinstance(vec.coordinates, np.ndarray)


def test_vec_post_init_no_LLA():
    """Verify that initializing a Vector in the 'LLA' (curvilinear) frame raises ValueError.
    
    Testing:
        Vectors represent linear Euclidean vectors (displacements, velocities, or directions)
        and cannot be represented in Latitude/Longitude/Altitude curvilinear coordinates.
    Expected Result:
        ValueError is raised with message indicating vectors cannot be defined in LLA.
    """
    with pytest.raises(ValueError):
        vec = vectors.Vector(np.array([1, 2, 3]), 0, "LLA")


def test_vec_post_init_randframe():
    """Verify that initializing a Vector with an invalid or unsupported frame name raises ValueError.
    
    Testing:
        Frame validation in Vector.__post_init__(). Valid Cartesian frames are 'ECI' and 'ECEF'.
    Expected Result:
        ValueError is raised when providing an unknown frame string like 'foo'.
    """
    with pytest.raises(ValueError):
        vec = vectors.Vector(np.array([1, 2, 3]), 0, "foo")


def test_vec_post_init_reshape():
    """Verify that a 2D column array of shape (3, 1) is automatically flattened to (3,).
    
    Testing:
        Coordinate array reshaping in Vector.__post_init__().
    Expected Result:
        Input shape (3, 1) -> output shape (3,) with elements [1.0, 2.0, 3.0].
    """
    vec = vectors.Vector(np.array([[1], [2], [3]]), 0, "ECI")
    assert np.shape(vec.coordinates) == (3,)


def test_vec_post_init_too_many_dimensions():
    """Verify that coordinate arrays with dimensionality > 2 are rejected.
    
    Testing:
        Dimensionality bounds check in Vector.__post_init__().
    Expected Result:
        Input with 3 dimensions (shape (3, 1, 1)) raises ValueError("Too many dimensions for coordinates").
    """
    with pytest.raises(ValueError):
        vec = vectors.Vector(np.array([[[1]], [[2]], [[3]]]), 0, "ECI")


@pytest.mark.parametrize(
    "vec", [vectors.Vector(np.array([93, 1232, 2]), GPSTime(1232, 122), "ECEF")]
)
def test_magnitude(vec):
    """Verify calculation of vector magnitude (Euclidean 2-norm).
    
    Testing:
        vec.magnitude property computes sqrt(x^2 + y^2 + z^2).
    Calculation:
        coords = [93, 1232, 2]
        magnitude = sqrt(93^2 + 1232^2 + 2^2)
                  = sqrt(8649 + 1517824 + 4)
                  = sqrt(1526477)
                  ≈ 1235.5067786...
    Expected Result:
        vec.magnitude exactly matches np.linalg.norm(vec.coordinates).
    """
    assert vec.magnitude == np.linalg.norm(vec.coordinates)


@pytest.mark.parametrize("array", [np.array([1, 1, 1]), np.array([9, 73, 83.11])])
def test_neg(array):
    """Verify that vector negation operator (__neg__) flips the signs of all components.
    
    Testing:
        -Vector([x, y, z]) returns Vector([-x, -y, -z]).
    Calculation:
        Case 1: -[1.0, 1.0, 1.0] = [-1.0, -1.0, -1.0]
        Case 2: -[9.0, 73.0, 83.11] = [-9.0, -73.0, -83.11]
    Expected Result:
        Returned vector coordinates equal the element-wise negation of the input coordinates.
    """
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
    """Verify dot product calculation between a Vector and a raw NumPy array.
    
    Testing:
        Vector.dot_product() accepting a 1D np.ndarray (assumed already in the same frame).
    Calculation:
        u = [-1, 2, 5], v = [1, 1, 1]
        u . v = (-1 * 1) + (2 * 1) + (5 * 1) = -1 + 2 + 5 = 6.0
    Expected Result:
        Returned dot product is exactly 6.0.
    """
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
    """Verify dot product calculation between two Vector objects in the same frame and epoch.
    
    Testing:
        Vector.dot_product() with another Vector instance.
    Calculation:
        u = [-1, 2, 5], v = [1, 1, 1]
        u . v = (-1 * 1) + (2 * 1) + (5 * 1) = 6.0
    Expected Result:
        Returned dot product is exactly 6.0.
    """
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
    """Verify dot product between two vectors defined in different frames across Earth rotation.
    
    Testing:
        Vector.dot_product() automatically transforms vecTwo into vecOne's frame ('ECI')
        and frame time before calculating the inner product.
    Calculation:
        vecOne is along ECI X-axis: [1, 0, 0] at t = 0.
        vecTwo is along ECEF X-axis: [1, 0, 0] at t = 21541.024725943207 s (~6 hours).
        The Earth rotates at angular rate w_e = 7.2921151467e-5 rad/s.
        The rotation angle theta = w_e * t = (7.2921151467e-5) * 21541.024725943207 = pi / 2 rad (90 deg).
        Rotating ECEF [1, 0, 0] by 90 deg into ECI yields [0, 1, 0] (along ECI Y-axis).
        Dot product: [1, 0, 0] . [0, 1, 0] = (1*0) + (0*1) + (0*0) = 0.0 (orthogonal).
    Expected Result:
        Dot product is 0.0 within floating point tolerance.
    """
    # not exactly zero due to float precision, so np.isclose is used
    assert np.isclose(vecOne.dot_product(vecTwo), expected)


def test_dot_type():
    """Verify that passing an invalid type to dot_product raises TypeError.
    
    Testing:
        Type validation in Vector.dot_product().
    Expected Result:
        Passing a string raises TypeError("other must be a Vector of Numpy array").
    """
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
    """Verify 3D cross product between two Vector instances.
    
    Testing:
        Vector.cross_product() computes u x v.
    Calculation:
        u = [1, 2, 3], v = [4, 5, 6]
        u x v = [ (u_y * v_z - u_z * v_y),
                  (u_z * v_x - u_x * v_z),
                  (u_x * v_y - u_y * v_x) ]
              = [ (2 * 6 - 3 * 5),
                  (3 * 4 - 1 * 6),
                  (1 * 5 - 2 * 4) ]
              = [ (12 - 15), (12 - 6), (5 - 8) ]
              = [-3, 6, -3]
    Expected Result:
        Returned vector has coordinates exactly equal to [-3, 6, -3].
    """
    assert (vecOne.cross_product(vecTwo).coordinates == expected).all()


def test_get_vec_coordinates():
    """Verify get_vector returns a new Vector whose coordinates match position_transform.
    
    Testing:
        vec.get_vector('ECEF') converts an ECI vector to ECEF without mutating original vector.
    Expected Result:
        Coordinates of the returned Vector match trans.position_transform('ECI', 'ECEF', coords, time).
    """
    vecECI = vectors.Vector(
        np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECI"
    )
    assert np.allclose(
        trans.position_transform("ECI", "ECEF", vecECI.coordinates, vecECI.frame_time),
        vecECI.get_vector("ECEF").coordinates,
    )


def test_get_vec_frame():
    """Verify get_vector sets the target frame property on the returned Vector.
    
    Testing:
        The 'frame' attribute of the new Vector instance is updated to 'ECEF'.
    Expected Result:
        vecECI.get_vector('ECEF').frame == 'ECEF'.
    """
    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    assert vecECI.get_vector("ECEF").frame == "ECEF"


def test_get_vec_LLA():
    """Verify that calling get_vector with 'LLA' raises ValueError.
    
    Testing:
        Frame constraint check preventing vectors from being transformed into LLA.
    Expected Result:
        ValueError("Vectors cannot be defined in the LLA frame") is raised.
    """
    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    with pytest.raises(ValueError):
        vecECI.get_vector("LLA")


def test_switch_frame_coordinates():
    """Verify that switch_frame mutates the existing Vector coordinates in-place.
    
    Testing:
        Vector.switch_frame('ECEF') updates coordinates to the new frame representation.
    Expected Result:
        Mutated coordinates match trans.position_transform('ECI', 'ECEF', coords, time).
    """
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
    """Verify that switch_frame updates the frame attribute of the Vector instance in-place.
    
    Testing:
        vec.frame is changed from 'ECI' to 'ECEF'.
    Expected Result:
        vecECEF.frame == 'ECEF'.
    """
    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    vecECEF = copy.copy(vecECI)
    vecECEF.switch_frame("ECEF")
    assert vecECEF.frame == "ECEF"


def test_switch_frame_LLA():
    """Verify that switch_frame to 'LLA' raises ValueError.
    
    Testing:
        Prohibiting in-place frame transition to 'LLA' for Vector.
    Expected Result:
        ValueError is raised.
    """
    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    with pytest.raises(ValueError):
        vecECI.switch_frame("LLA")


def test_update_frame_time_ECI():
    """Verify update_frame_time on an ECI vector advances the coordinates across multiple GPS weeks.
    
    Testing:
        Vector.update_frame_time() for ECI frame rotates coordinates by the Earth rotation
        accumulated over delta_weeks (trans.add_weeks_eci).
    Calculation:
        Initial week: 2109, Target week: 2111 (delta_weeks = 2 weeks).
        Rotation angle: theta = w_e * (2 * 604800 s) rad.
    Expected Result:
        Coordinates match trans.add_weeks_eci(2, vec.coordinates).
    """
    vecECI = vectors.Vector(
        np.array([1, 2, 3], dtype=float), GPSTime(2109, 259200), "ECI"
    )
    newVecECI = copy.copy(vecECI)
    newVecECI.update_frame_time(GPSTime(2111, 259200))
    assert np.array(
        (trans.add_weeks_eci(2, vecECI.coordinates)) == newVecECI.coordinates
    ).all()


def test_update_frame_time_ECEF():
    """Verify update_frame_time on an ECEF vector rotates coordinates by Earth rotation angle.
    
    Testing:
        Vector.update_frame_time() for ECEF frame rotates coordinates around Z-axis by
        angle = w_e * delta_t via trans.rotate_ecef().
    Calculation:
        t1 = GPSTime(2109, 259200), t2 = GPSTime(2111, 259200) (delta_t = 2 weeks = 1,209,600 s).
        angle = w_e * delta_t.
    Expected Result:
        Coordinates match trans.rotate_ecef(t1, t2, vec.coordinates).
    """
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
    """Verify vector addition (__add__), including automatic frame and time alignment.
    
    Testing:
        v1 + v2 converts v2 into v1's frame and epoch before performing component-wise addition.
    Calculation:
        Case 1 (same frame): [1, 2, 3] + [4, 5, 6] = [5, 7, 9].
        Case 2 (different frame & time): vecTwo is transformed to ECI at t=0, then added.
    Expected Result:
        Resulting vector coordinates match vecOne.coordinates + transformed_vecTwo.coordinates.
    """
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
    """Verify vector subtraction (__sub__), including automatic frame and time alignment.
    
    Testing:
        v1 - v2 converts v2 into v1's frame and epoch before performing component-wise subtraction.
    Calculation:
        Case 1 (same frame): [1, 2, 3] - [4, 5, 6] = [-3, -3, -3].
        Case 2 (different frame & time): vecTwo is transformed to ECI at vecOne's time, then subtracted.
    Expected Result:
        Resulting vector coordinates match vecOne.coordinates - transformed_vecTwo.coordinates.
    """
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
    """Verify scalar multiplication (__mul__) on Vector.
    
    Testing:
        Vector * scalar scales each component by scalar: [x * num, y * num, z * num].
    Calculation:
        Case 1: [1, 2, 3] * 6 = [6, 12, 18]
        Case 2: [9274, 83.122, -814.3] * -6.676 = [-61913.224, -554.922472, 5436.2668]
    Expected Result:
        Returned vector coordinates match num * vec.coordinates element-wise.
    """
    vecResult = vec * num
    assert (vecResult.coordinates == num * vec.coordinates).all()


def test_mult_type_error():
    """Verify that multiplying a Vector by a non-numeric value raises TypeError.
    
    Testing:
        Type validation in Vector.__mul__().
    Expected Result:
        vec * "not a number" raises TypeError("other value must be a float").
    """
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
    """Verify that UnitVector automatically normalizes its coordinates to magnitude 1.0.
    
    Testing:
        UnitVector.__post_init__() normalizes input coordinates via coordinates / ||coordinates||.
    Calculation:
        Case 1: v = [7, 48, -17] -> ||v|| = sqrt(49 + 2304 + 289) = sqrt(2642) ≈ 51.400389...
                Normalized coordinates: [7/||v||, 48/||v||, -17/||v||].
                ||u|| = 1.0.
    Expected Result:
        | ||unVec.coordinates|| - 1.0 | < 1e-12.
    """
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
    """Verify classmethod UnitVector.from_vector(vec) constructs a normalized UnitVector.
    
    Testing:
        UnitVector.from_vector(v) copies frame and frame_time, and normalizes coordinates.
    Calculation:
        Input vector coordinates [7, 48, -17] are normalized to unit magnitude.
    Expected Result:
        Returned UnitVector coordinates exactly match UnitVector(coordinates, time, frame).coordinates.
    """
    unVec = vectors.UnitVector.from_vector(vec)
    assert (unVec.coordinates == expected.coordinates).all()


@pytest.mark.parametrize(
    "vec", [vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 0.0), "ECI")]
)
def test_unit_switch_frame(vec):
    """Verify that UnitVector.switch_frame() converts the frame and maintains unit magnitude.
    
    Testing:
        UnitVector.switch_frame('ECEF') transforms coordinates and re-normalizes them.
    Expected Result:
        unVecECI.switch_frame('ECEF') coordinates match unVecECEF.coordinates within float precision.
    """
    vecECI = vec
    vecECEF = copy.copy(vecECI)
    vecECEF.switch_frame("ECEF")
    unVecECI = vectors.UnitVector.from_vector(vecECI)
    unVecECEF = vectors.UnitVector.from_vector(vecECEF)
    unVecECI.switch_frame("ECEF")
    assert np.all(unVecECI.coordinates == unVecECEF.coordinates)


def test_unit_switch_frame_LLA():
    """Verify that switch_frame to 'LLA' is prohibited for UnitVector.
    
    Testing:
        Validation check against LLA frame on UnitVector.switch_frame().
    Expected Result:
        Raises ValueError("Vectors cannot be defined in the LLA frame").
    """
    vecECI = vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")
    with pytest.raises(ValueError):
        vecECI.switch_frame("LLA")


@pytest.mark.parametrize(
    "vec", [vectors.Vector(np.array([1, 2, 3]), GPSTime(2109, 259200), "ECI")]
)
def test_get_unit(vec):
    """Verify that get_unit_vector creates a new UnitVector in the requested destination frame.
    
    Testing:
        UnitVector.get_unit_vector('ECI') transforms and returns a UnitVector with magnitude 1.0.
    Expected Result:
        unVecECEF.get_unit_vector('ECI') coordinates match unVecECI coordinates.
    """
    vecECI = vec
    vecECEF = vecECI.get_vector("ECEF")
    unVecECI = vectors.UnitVector.from_vector(vecECI)
    unVecECEF = vectors.UnitVector.from_vector(vecECEF)

    assert np.allclose(
        unVecECI.coordinates, unVecECEF.get_unit_vector("ECI").coordinates
    )


def test_unit_get_vec_LLA():
    """Verify that calling get_vector with 'LLA' on a Vector/UnitVector raises ValueError.
    
    Testing:
        Frame constraint check for get_vector('LLA').
    Expected Result:
        Raises ValueError("Vectors cannot be defined in the LLA frame").
    """
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
    """Verify that multiplying a UnitVector by a scalar scales its coordinates correctly.
    
    Testing:
        UnitVector * scalar scales unit coordinates by num: [u_x * num, u_y * num, u_z * num].
    Calculation:
        Magnitude of resulting vector = |num| * ||unVec|| = |num| * 1.0 = |num|.
    Expected Result:
        Result coordinates equal num * unVec.coordinates element-wise.
    """
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
    """Verify that multiplying a UnitVector by a scalar returns a Vector instance, not a UnitVector.
    
    Testing:
        Type degradation: a scaled unit vector no longer has unit magnitude, so it must return
        an instance of Vector, not UnitVector.
    Expected Result:
        isinstance(unVec * num, vectors.Vector) is True and type is Vector.
    """
    vec = unVec * num
    assert isinstance(vec, vectors.Vector)


def test_unit_mult_type_error():
    """Verify that multiplying a UnitVector by a non-numeric type raises TypeError.
    
    Testing:
        Type validation in UnitVector.__mul__().
    Expected Result:
        unVec * "not a number" raises TypeError("other value must be a float").
    """
    with pytest.raises(TypeError):
        unVec = vectors.UnitVector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
        unVec * ("not a number")
