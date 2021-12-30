# Copyright (c) 2022 The Aerospace Corporation
"""Representation for Rotations.

The purpose of this submodule is to provide representations for rotations.
This is meant to be a low-level library that uses
[Numba](http://numba.pydata.org/) compilation to speed up execution. As such,
much of the code in this submodule should be considered strictly-typed.

The main externally used componentes are the Rotation class, which is a
non-Numba class that serves as a wrapper to Numba JIT compiled functions, and
the standard_rotation() function, which is a Numba JIT compiled function that
provides a means for rotating a vector around a principal axis.

"""

from __future__ import annotations

from numba import jit, float64
from numba.experimental import jitclass
import numpy as np

from typing import Tuple
from logging import getLogger


logger = getLogger(__name__)


class Rotation:
    r"""Represention of rotations.

    This class is used to represent a rotation. The rotation is internally
    represented by a _Rotator object, which is a Numba optimized class that is
    used to handle the actual rotations.

    Attributes
    ----------
    _rotator : _Rotator
        A numba optimized class used to describe rotations.

    Todo
    ----
    Consider adding to_yaml() and from_yaml() methods to enable storing as a
    YAML object.

    """

    _rotator: _Rotator

    def __init__(self, *args, **kwargs) -> None:
        r"""Construct a Rotation Object

        .. note:: Types
            Because this acts as an interface to a numba object (which is
            strictly typed), it is recommended that all inputs contain only
            floats. This should be corrected automatically, but success cannot
            be guarenteed.

        This constructor is able to take a various numbers of input arguments
        to create a _Rotator object. These can either be positional or keyword
        arguments. It is recommended to use the keyword arguments to avoid
        ambiguity.

        .. warning:: Duplicate Definitions
            There can only be one definition of the rotation. If multiple
            definitions are given, it will raise an error. This includes
            having both positional and keyword arguments.

        This constructor functions by taking the input arguments and
        converting the rotation representation to a direction cosine matrix,
        which is then used to instantiate a _Rotator object.

        .. tip:: Quaternion Ordering
            The order of the unit quaternion inputs are (\(q_{w}\), \(q_{i}\),
            \(q_{j}\), \(q_{k}\)) where
            \(\boldsymbol{q} = q_{w} + q_{i}i + q_{j}j + q_{k}k\).

        Parameters
        ----------
        *args
            If one positional argument is provided, it is assumed to be either
            a unit quaternion (if its length is 4) or a direction cosine
            matrix (otherwise). If two positional arguments are provided, then
            the first positional argument is assumed to be the axis of
            rotation and the second argument is assumed to be the angle of
            rotation. If four positional arguments are given, the values are
            assumed to be the components of the unit quaternion.
        **kwargs
            The rotation can be defined using keyword arguments. The valid
            keyword arguments are:

            - 'dcm': A direction cosine matrix (3x3 array)
            - 'axis', 'angle': The euler axis (3 element vector) and angle of
              rotation (float)
            - 'standard_axis', 'angle': The principal axis of rotation
              (int, 1,2,3) and angle of rotation (float)
            - 'quaternion': A unit quaternion (4 element vector)

        """

        if len(args) != 0:
            assert (
                len(kwargs) == 0
            ), "Only positional or keyword arguments accepted, not both"

            if len(args) == 1:
                _input = np.array(args[0])
                if len(_input) == 4:  # Infer Quaternion
                    _dcm = quaternion2direction_cosine_matrix(_input)
                else:  # Infer DCM
                    _dcm = _input
            elif len(args) == 2:  # Euler Axis Angle
                _dcm = euler_axis_angle2direction_cosine_matrix(args[0], args[1])
            elif len(args) == 4:  # Quaternions
                _dcm = quaternion2direction_cosine_matrix(np.array(args))
            else:
                raise ValueError(f"Cannot handle {len(args)} positional arguments")
        else:
            if "dcm" in kwargs:
                assert "axis" not in kwargs, "Duplicate rotation definitions"
                assert "angle" not in kwargs, "Duplicate rotation definitions"
                assert "quaternion" not in kwargs, "Duplicate rotation definitions"

                _dcm = kwargs["dcm"]

            elif "axis" in kwargs:
                assert "angle" in kwargs, "No angle provided"
                assert "dcm" not in kwargs, "Duplicate rotation definitions"
                assert "quaternion" not in kwargs, "Duplicate rotation definitions"
                assert "standard_axis" not in kwargs, "Duplicate rotation definitions"

                _dcm = euler_axis_angle2direction_cosine_matrix(
                    kwargs["axis"], kwargs["angle"]
                )

            elif "standard_axis" in kwargs:
                assert "angle" in kwargs, "No angle provided"
                assert "dcm" not in kwargs, "Duplicate rotation definitions"
                assert "quaternion" not in kwargs, "Duplicate rotation definitions"
                assert "axis" not in kwargs, "Duplicate rotation definitions"

                _standard_axis = {
                    1: np.array((1.0, 0.0, 0.0), dtype=float),
                    2: np.array((0.0, 1.0, 0.0), dtype=float),
                    3: np.array((0.0, 0.0, 1.0), dtype=float),
                }

                assert (
                    kwargs["standard_axis"] in _standard_axis
                ), f'Invalid standard axis {kwargs["standard_axis"]}'

                _dcm = euler_axis_angle2direction_cosine_matrix(
                    _standard_axis[kwargs["standard_axis"]], kwargs["angle"]
                )

            elif "quaternion" in kwargs:
                assert "dcm" not in kwargs, "Duplicate rotation definitions"
                assert "axis" not in kwargs, "Duplicate rotation definitions"
                assert "angle" not in kwargs, "Duplicate rotation definitions"

                _quaternion = np.array(kwargs["quaternion"])
                _dcm = quaternion2direction_cosine_matrix(_quaternion)

        self._check_dcm(_dcm)
        self._rotator = _Rotator(np.array(_dcm, dtype=float))

    @staticmethod
    def _check_dcm(dcm: np.ndarray) -> None:
        """Check to see if a Direction Cosine Matrix is valid.

        Specifically, this static method checks to ensure that the input
        direction cosine matrix is a 3x3 matrix that is both right-handed and
        orthonormal.

        Parameters
        ----------
        dcm : np.ndarray
            The Direction Cosine Matrix of interest.

        """
        assert np.shape(dcm) == (3, 3), "Direction Cosine Matrix must be 3x3"
        assert np.isclose(
            abs(np.linalg.det(dcm) - 1), 0
        ), "Direction Cosine Matrix must be right-handed and orthonormal"

    def rotate(self, vector: np.array) -> np.array:
        """Rotate a vector by this rotaion.

        .. note:: Vector Objects
            gps_frames.vectors includes an object called Vector. This method
            takes a numpy array that represents a vector. This is deliberate
            because rotations are dependant on a frame and the Vector object
            requires a frame

        Parameters
        ----------
        vector : np.array
            The 3-element 1D numpy array representing a vector

        Returns
        -------
        np.array
            The vector rotated into the new frame

        """
        return self._rotator.rotate(np.array(vector, dtype=float))


_rotator_spec = [
    ("dcm", float64[:, :]),
]
"""The numba specification for the _Rotator jitclass."""


@jitclass(_rotator_spec)
class _Rotator:
    """The numba-optimized rotation.

    This class is used to enable numba optimization on a rotation. It contains
    a direction cosine matrix that can be used to rotate a vector to a new
    frame.

    .. note:: Numba JIT Compiled
        This class is compiled using Numba. Use care when providing inputs as
        Numba is strictly typed. Unless otherwise stated, all inputs should be
        float arrays.

    Attributes
    ----------
    dcm : np.ndarray (float64[:, :])
        The direction cosine matrix

    """

    def __init__(self, dcm: np.ndarray) -> None:
        """The object constructor.

        Parameters
        ----------
        dcm : np.ndarray
            The direction cosine matrix.
        """
        self.dcm = dcm

    def rotate(self, vector: np.ndarray) -> np.ndarray:
        """Rotate a vector by the direction cosine matrix.

        Parameters
        ----------
        vector : np.ndarray
            The vector to be rotated.

        Returns
        -------
        np.ndarray
            The vector in the new frame.

        """
        return self.dcm @ vector
        # # If issues occur, switch to this, which is the expansion of matrix
        # # multiplication.
        # return [self.dcm[0, 0] * vector[0]
        #         + self.dcm[0, 1] * vector[1]
        #         + self.dcm[0, 2] * vector[2],
        #         self.dcm[1, 0] * vector[0]
        #         + self.dcm[1, 1] * vector[1]
        #         + self.dcm[1, 2] * vector[2],
        #         self.dcm[2, 0] * vector[0]
        #         + self.dcm[2, 1] * vector[1]
        #         + self.dcm[2, 2] * vector[2]]


@jit("float64[:](float64[:], float64)", nopython=True, cache=True)
def euler_axis_angle2quaternion(
    euler_axis: np.ndarray, euler_angle: float
) -> np.ndarray:
    r"""Convert an Euler axis and angle to a quaternion.

    Let the Euler axis and angle be \(\hat{\boldsymbol{e}}\) and
    \(\Phi\), respectively. The equivalent quaternion is
    $$
        \bar{\boldsymbol{q}} =
            \left[\begin{array}{c}
                q_{w} \\
                q_{i} \\
                q_{j} \\
                q_{k}
            \end{array}\right]
            =
            \left[\begin{array}{c}
                \cos\frac{\Phi}{2} \\
                \hat{\boldsymbol{e}}\sin\frac{\Phi}{2}
            \end{array}\right]
    $$

    .. note:: Numba JIT Compiled
        This function is compiled using Numba. Use care when providing inputs
        as Numba is strictly typed. Unless otherwise stated, all inputs should
        be float arrays.

    Parameters
    ----------
    euler_axis : np.ndarray
        The axis of rotation. Will be normalized to a magnitude of 1
    euler_angle : float
        The angle of rotation in radians

    Returns
    -------
    np.ndarray
        The quaternion equivalent to the the euler axis and angle

    """
    euler_axis = euler_axis / np.linalg.norm(euler_axis)  # Normialize length
    qijk = euler_axis * np.sin(euler_angle / 2) / np.linalg.norm(euler_axis)
    quaternion = np.append(np.cos(euler_angle / 2), qijk)

    return quaternion / np.linalg.norm(quaternion)


@jit("float64[:](float64[:,:])", nopython=True, cache=True)
def direction_cosine_matrix2quaternion(dcm: np.ndarray) -> np.array:
    r"""Create a rotation object from a Direction Cosine Matrix.

    Let the direction cosine matrix be \(\mathbf{R}\). The \(q_{w}\)
    element of the quaternion can be computed as
    $$
        q_{w} = \frac{
                \sqrt{1 + \text{trace}\mathbf{R}}
            }{
                2
            }
    $$
    The remaining elements of the quaternion are then computed as
    $$
        \begin{split}
            q_{i} = & \frac{
                \mathbf{R}_{23} - \mathbf{R}_{32}}{4q_{w}} \\
            q_{j} = & \frac{
                \mathbf{R}_{31} - \mathbf{R}_{13}}{4q_{w}} \\
            q_{k} = & \frac{
                \mathbf{R}_{12} - \mathbf{R}_{21}}{4q_{w}} \\
        \end{split}
    $$
    The quaternion is represented as
    $$
        \bar{\boldsymbol{q}} = \left[
            q_{w} \quad q_{i} \quad q_{j} \quad q_{k}
        \right]
    $$

    There a singulariy when the rotation is by an angle of \(\pi\).
    This occurs because \(q_{w} = \cos\left(\theta / 2\right)\), so
    \(q_{w}=0\) when \(\theta=\pi\). In the previous expression, the other
    quaternion elements were found by dividing the difference of various DCM
    elements by \(4q_{w}\), which cannot be done of \(q_{w}=0\). Thus, for
    these cases, an alternate approach is used. The angle of rotation is known
    (because \(\cos(\pi/2) = \cos(-\pi/2) = 0\) to be \(\pi\). The axis of
    rotation is found by find the eigenvector to the DCM corresponding the
    eigenvalue with a value of 1. This is done by calling
    euler_axis_angle2quaternion().

    .. note:: Numba JIT Compiled
        This function is compiled using Numba. Use care when providing inputs
        as Numba is strictly typed. Unless otherwise stated, all inputs should
        be float arrays.

    Parameters
    ----------
    dcm : np.ndarray
        The direction cosine matrix (a 3x3 matrix)

    Returns
    -------
    np.ndarray
        The corresponding unit quaternion

    """

    qw = np.sqrt(1 + np.trace(dcm)) / 2
    if qw == 0.0:
        euler_angle = np.pi

        eig_val, eig_axis = np.linalg.eigh(dcm)

        for _i in range(3):
            if eig_val[_i] == 1:
                euler_axis = eig_axis[:, _i]
                break
        else:
            raise ValueError("Encountered singular DCM")

        return euler_axis_angle2quaternion(euler_axis, euler_angle)

    else:
        qi = (dcm[1, 2] - dcm[2, 1]) / (4 * qw)
        qj = (dcm[2, 0] - dcm[0, 2]) / (4 * qw)
        qk = (dcm[0, 1] - dcm[1, 0]) / (4 * qw)

        return np.array([qw, qi, qj, qk])


@jit("float64[:,:](float64[:])", nopython=True, cache=True)
def quaternion2direction_cosine_matrix(quaternion: np.ndarray) -> np.ndarray:
    r"""Convert quaternions to a direction cosine matrix.

    A quaternions can be represented as
    $$
        \bar{\boldsymbol{q}} = \left[
            q_{w} \quad q_{i} \quad q_{j} \quad q_{k}
        \right]
    $$
    The equivalent direction cosine matrix can then be computed as
    $$
        \mathbf{R} =
            \left[\begin{array}{ccc}
                q_{w}^{2} + q_{i}^{2} - q_{j}^{2} - q_{k}^{2} &
                2 \left( q_{i}q_{j} - q_{k}q_{w} \right) &
                2 \left( q_{i}q_{k} + q_{j}q_{w} \right) \\
                2 \left( q_{i}q_{j} + q_{k}q_{w} \right) &
                q_{w}^{2} - q_{i}^{2} + q_{j}^{2} - q_{k}^{2} &
                2 \left( q_{j}q_{k} - q_{i}q_{w} \right) \\
                2 \left( q_{i}q_{k} - q_{j}q_{w} \right) &
                2 \left( q_{j}q_{k} + q_{i}q_{w} \right) &
                q_{w}^{2} - q_{i}^{2} - q_{j}^{2} + q_{k}^{2}
            \end{array}\right]
    $$


    .. note:: Numba JIT Compiled
        This function is compiled using Numba. Use care when providing inputs
        as Numba is strictly typed. Unless otherwise stated, all inputs should
        be float arrays.

    Parameters
    ----------
    quaternion : np.ndarray
        The unit quaternion to convert

    Returns
    -------
    np.ndarray
        The 3x3 Direction Cosine Matrix that represents the rotation

    """
    assert len(quaternion) == 4, "Quaternions must have 4 elements"

    # Normalize
    quaternion = quaternion / np.linalg.norm(quaternion)

    qw = quaternion[0]
    qi = quaternion[1]
    qj = quaternion[2]
    qk = quaternion[3]

    dcm = np.array(
        [
            [
                qw ** 2 + qi ** 2 - qj ** 2 - qk ** 2,
                2 * (qi * qj - qk * qw),
                2 * (qi * qk + qj * qw),
            ],
            [
                2 * (qi * qj + qk * qw),
                qw ** 2 - qi ** 2 + qj ** 2 - qk ** 2,
                2 * (qj * qk - qi * qw),
            ],
            [
                2 * (qi * qk - qj * qw),
                2 * (qj * qk + qi * qw),
                qw ** 2 - qi ** 2 - qj ** 2 + qk ** 2,
            ],
        ]
    )
    return np.transpose(dcm)


@jit("Tuple((float64[:], float64))(float64[:])", nopython=True, cache=True)
def quaternion2euler_axis_angle(quaternion: np.ndarray) -> Tuple[np.ndarray, float]:
    r"""Convert a quaternion to an euler axis and angle.

    .. warning:: Directional Ambiguity
        There is directional ambitguity when using this function. This occurs
        because rotating by an angle about a given axis is equivalent to
        rotating by the same angle in the opposite direction about the
        opposite axis. There is no way to resolve this ambitguity, but it
        should not effect anything in the simulation.

    A quaternions can be represented as
    $$
        \bar{\boldsymbol{q}} = \left[
            q_{w} \quad q_{i} \quad q_{j} \quad q_{k}
        \right]
    $$
    with \(\boldsymbol{q}=[q_{i}, q_{j}, q_{k}]^{T}\). The angle of
    rotation that this represents can be computed (using a 4-quadrant
    arctangent function) as
    $$
        \Phi = 2 \tan^{-1}\frac{
            ||\boldsymbol{q}||
        }{
            q_{w}
        }
    $$
    The axis of the rotation is then computed as
    $$
        \hat{\boldsymbol{e}} = \frac{
                \boldsymbol{q}
            }{
                ||\boldsymbol{q}||
            }
    $$

    .. note:: Numba JIT Compiled
        This function is compiled using Numba. Use care when providing inputs
        as Numba is strictly typed. Unless otherwise stated, all inputs should
        be float arrays.

    Parameters
    ----------
    quaternion : np.ndarray
        The unit quaternion to convert

    Returns
    -------
    Tuple[np.ndarray, float]
        A tuple where the first element is a 3-element 1D numpy array
        representing the rotation axis and the second element is the angle
        of rotation, in radians.

    """
    qw = quaternion[0]
    qijk = quaternion[1:]

    euler_angle = 2 * np.arctan2(np.linalg.norm(qijk), qw)
    euler_axis = qijk / np.linalg.norm(qijk)

    return euler_axis, euler_angle


@jit("Tuple((float64[:], float64))(float64[:,:])", nopython=True, cache=True)
def direction_cosine_matrix2euler_axis_angle(
    dcm: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Convert a Direction Cosine Matrix to an Euler axis and angle.

    This function simply calls direction_cosine_matrix2quaternion() to convert
    the input DCM to the corresponding unit quaternion. From there, the
    quaternion is converted to the corresponding Euler axis and angle using
    quaternion2euler_axis_angle().

    .. note:: Numba JIT Compiled
        This function is compiled using Numba. Use care when providing inputs
        as Numba is strictly typed. Unless otherwise stated, all inputs should
        be float arrays.

    Parameters
    ----------
    dcm : np.ndarray
        The direction cosine matrix

    Returns
    -------
    Tuple[np.ndarray, float]
        A tuple where the first element is a 3-element 1D numpy array
        representing the rotation axis and the second element is the angle
        of rotation, in radians.

    """
    quaternion = direction_cosine_matrix2quaternion(dcm)
    return quaternion2euler_axis_angle(quaternion)


@jit("float64[:,:](float64[:], float64)", nopython=True, cache=True)
def euler_axis_angle2direction_cosine_matrix(
    euler_axis: np.ndarray, euler_angle: float
) -> np.ndarray:
    """Convert an Euler axis and angle to a Direction Cosine Matrix.

    This function simply calls euler_axis_angle2quaternion() to convert
    the input Euler axis and angle to the corresponding unit quaternion. From
    there, the quaternion is converted to the corresponding DCM using
    quaternion2direction_cosine_matrix().


    .. note:: Numba JIT Compiled
        This function is compiled using Numba. Use care when providing inputs
        as Numba is strictly typed. Unless otherwise stated, all inputs should
        be float arrays.

    Parameters
    ----------
    euler_axis : np.ndarray
        The axis of rotation. Will be normalized to a magnitude of 1
    euler_angle : float
        The angle of rotation in radians
    Returns
    -------
    np.ndarray
        The direction cosine matrix (a 3x3 matrix)

    """
    return quaternion2direction_cosine_matrix(
        euler_axis_angle2quaternion(euler_axis, euler_angle)
    )


@jit("float64[:](int8, float64, float64[:])", nopython=True, cache=True)
def standard_rotation(
    rotation_axis: int, angle: float, vector: np.ndarray
) -> np.ndarray:
    """Rotate a vector about a principal axis.

    This function is an optimized way to rotate about one of the principal
    axes. The inputs are the axis to rotate around and the angle of rotation,
    along with the vector be rotated.

    For a single rotation, this class is faster to use than the Rotation class
    as it is signficantly simpler to instantiate. However, if a rotation needs
    to be performed repeatedly, the Rotation class may be faster.

    .. note:: Numba JIT Compiled
        This function is compiled using Numba. Use care when providing inputs
        as Numba is strictly typed. The inputs to this function are an int, a
        float, and a float array.

    Parameters
    ----------
    rotation_axis : int
        The axis of rotation. Must be 1, 2, or 3 (corresponding to x,y, and x)
    angle : float
        The angle of rotation in radians
    vector : np.ndarray
        The vector to rotate

    Returns
    -------
    np.ndarray
        The vector rotated to the new frame

    """
    _c = np.cos(angle)
    _s = np.sin(angle)

    assert rotation_axis in [1, 2, 3], "rotation_axis must be 1, 2, or 3"
    if rotation_axis == 1:
        dcm = np.array([[1.0, 0.0, 0.0], [0.0, _c, _s], [0.0, -_s, _c]])
    elif rotation_axis == 2:
        dcm = np.array([[_c, 0.0, -_s], [0.0, 1.0, 0.0], [_s, 0.0, _c]])
    elif rotation_axis == 3:
        dcm = np.array([[_c, _s, 0.0], [-_s, _c, 0.0], [0.0, 0.0, 1.0]])

    return dcm @ vector


@jit("float64[:,:](int8, float64)", nopython=True, cache=True)
def standard_rotation_matrix(rotation_axis: int, angle: float) -> np.ndarray:
    r"""Get a standard rotation matrix.

    This function is used to compute the rotation matrix (direction cosine
    matrix) for a rotation about one of the principle axes.

    That is, if the angle of rotation is \(\theta\), the returned rotation
    matrix is
    $$
        \begin{split}
            R_{1}(\theta) = &
                \left[\begin{array}{ccc}
                    1 & 0 & 0 \\
                    0 & \cos\theta & \sin\theta \\
                    0 & -\sin\theta & \cos\theta
                \end{array}\right] \\
            R_{2}(\theta) = &
                \left[\begin{array}{ccc}
                    \cos\theta & 0 & -\sin\theta \\
                    0 & 1 & 0 \\
                    \sin\theta & 0 & \cos\theta
                \end{array}\right] \\
            R_{3}(\theta) = &
                \left[\begin{array}{ccc}
                    \cos\theta & \sin\theta & 0 \\
                    -\sin\theta & \cos\theta & 0 \\
                    0 & 0 & 1
                \end{array}\right]
        \end{split}
    $$
    respectively for rotations about the 1, 2, or 3 axis.

    To simply the actual code, this is actually accomplished by expressing the
    rotation as a rotation about an Euler axis and simply converting to a
    direction cosine matrix using euler_axis_angle2direction_cosion_matrix().

    .. note:: Numba JIT Compiled
        This function is compiled using Numba. Use care when providing inputs
        as Numba is strictly typed. The inputs to this function are an int and
        a float.

    Parameters
    ----------
    rotation_axis : int
        The axis of rotation
    angle : float
        The angle of rotation in radians

    Returns
    -------
    np.ndarray
        The rotation matrix (DCM) for the axis and angle of rotation

    Raises
    ------
    ValueError
        If the rotation axis is not 1, 2, or 3

    """

    assert rotation_axis in [1, 2, 3], "rotation_axis must be 1, 2, or 3"
    euler_axis = np.zeros(3)
    euler_axis[rotation_axis - 1] = 1.0

    return euler_axis_angle2direction_cosine_matrix(euler_axis, angle)


@jit("float64[:,:](int8, float64, float64)", nopython=True, cache=True)
def standard_rotation_matrix_rates(
    rotation_axis: int, angle: float, rate: float
) -> np.ndarray:
    r"""Get the derivated of a standard rotation matrix.

    This function is used to compute the derivative of a rotation matrix
    (direction cosine matrix) for a rotation about one of the principle axes.

    That is, if the angle of rotation is \(\theta\), the returned matrix is
    $$
        \begin{split}
            R_{1}(\theta) = &
                \left[\begin{array}{ccc}
                    0 & 0 & 0 \\
                    0 &
                    -\dot{\theta}\sin\theta &
                    \dot{\theta}\cos\theta \\
                    0 &
                    -\dot{\theta}\cos\theta &
                    -\dot{\theta}\sin\theta
                \end{array}\right] \\
            R_{2}(\theta) = &
                \left[\begin{array}{ccc}
                    -\dot{\theta}\sin\theta &
                    0 &
                    -\dot{\theta}\cos\theta\\
                    0 &
                    0 &
                    0 \\
                    \dot{\theta}\cos\theta &
                    0 &
                    -\dot{\theta}\sin\theta
                \end{array}\right] \\
            R_{3}(\theta) = &
                \left[\begin{array}{ccc}
                    -\dot{\theta}\sin\theta
                    & \dot{\theta}\cos\theta &
                    0 \\
                    -\dot{\theta}\cos\theta &
                    -\dot{\theta}\sin\theta &
                    0 \\
                    0 & 0 & 0
                \end{array}\right]
        \end{split}
    $$
    respectively for rotations about the 1, 2, or 3 axis.

    .. note:: Numba JIT Compiled
        This function is compiled using Numba. Use care when providing inputs
        as Numba is strictly typed. The inputs to this function are an int and
        two floats.

    Parameters
    ----------
    rotation_axis : int
        The axis of rotation
    angle : float
        The angle of rotation in radians
    rate : float
        The angular rotation rate in radians per second

    Returns
    -------
    np.ndarray
        The rate of change of the rotation matrix (DCM) for the axis, angle,
        and rate of rotation

    Raises
    ------
    ValueError
        If the rotation axis is not 1, 2, or 3

    """
    c = np.cos(angle)
    s = np.sin(angle)

    if rotation_axis == 1:
        return np.array(
            [[0.0, 0.0, 0.0], [0.0, -s * rate, c * rate], [0.0, -c * rate, -s * rate]]
        )
    elif rotation_axis == 2:
        return np.array(
            [[-s * rate, 0.0, -c * rate], [0.0, 0.0, 0.0], [c * rate, 0.0, -s * rate]]
        )
    elif rotation_axis == 3:
        return np.array(
            [[-s * rate, c * rate, 0.0], [-c * rate, -s * rate, 0.0], [0.0, 0.0, 0.0]]
        )
    else:
        raise ValueError("rotation_axis must be 1, 2, or 3")


@jit("float64[:, :](float64, float64, float64)", nopython=True, cache=True)
def roll_pitch_yaw_matrix(
    roll_angle: float, pitch_angle: float, yaw_angle: float
) -> np.ndarray:
    r"""Create the Direction Cosine Matrix fo a roll-pitch-yaw sequence.

    This function is a way to generate the direction cosine matrix
    representing a roll-pitch-yaw sequence. This sequence is a rotation about
    the 1, 2, and 3 axes in that order, that is first it it rolled, then
    pitched, then yawed.

    Let \(\theta,\phi,\psi\) be the roll, pitch, and yaw angles, respectively.
    The corresponding direction cosine matrix is
    $$
        \boldsymbol{R}(\theta,\phi,\psi) =
            \left[\begin{array}{ccc}
                \cos\psi \cos\phi &
                \cos\psi \sin\phi \sin\theta - \sin\psi \cos\theta &
                \cos\psi \sin\phi \cos\theta + \sin\psi \sin\theta \\
                \sin\psi \cos\phi &
                \sin\psi \sin\phi \sin\theta + \cos\psi \cos\theta &
                \sin\psi \sin\phi \cos\theta - \cos\psi \sin\theta \\
                -\sin\phi &
                \cos\phi \sin\theta &
                \cos\phi \cos\theta
            \end{array}\right]
    $$

    .. note:: Numba JIT Compiled
        This function is compiled using Numba. Use care when providing inputs
        as Numba is strictly typed. The inputs to this function are three
        floats and a float array.

    Parameters
    ----------
    roll_angle : float
        Angle of rotation about the roll axis in radians
    pitch_angle : float
        Angle of rotation about the pitch axis in radians
    yaw_angle : float
        Angle of rotation about the yaw axis in radians

    Returns
    -------
    np.ndarray
        The direction cosine matrix representing the rotation.

    """
    _c_roll = np.cos(roll_angle)
    _s_roll = np.sin(roll_angle)

    _c_pitch = np.cos(pitch_angle)
    _s_pitch = np.sin(pitch_angle)

    _c_yaw = np.cos(yaw_angle)
    _s_yaw = np.sin(yaw_angle)

    dcm = np.array(
        [
            [
                _c_yaw * _c_pitch,
                _c_yaw * _s_pitch * _s_roll - _s_yaw * _c_roll,
                _c_yaw * _s_pitch * _c_roll + _s_yaw * _s_roll,
            ],
            [
                _s_yaw * _c_pitch,
                _s_yaw * _s_pitch * _s_roll + _c_yaw * _c_roll,
                _s_yaw * _s_pitch * _c_roll - _c_yaw * _s_roll,
            ],
            [-_s_pitch, _c_pitch * _s_roll, _c_pitch * _c_roll],
        ]
    )

    return dcm


@jit("float64[:](float64, float64, float64, float64[:])", nopython=True, cache=True)
def roll_pitch_yaw(
    roll_angle: float, pitch_angle: float, yaw_angle: float, vector: np.ndarray
) -> np.ndarray:
    """Rotate a vector through a roll-pitch-yaw sequence.

    This function is an optimized way to rotate a vector through a
    roll-ptich-yaw sequence. This sequence is a rotation about the 1, 2, and 3
    axes in that order, that is first it it rolled, then pitched, then yawed.

    For a single rotation, this class is faster to use than the Rotation class
    as it is signficantly simpler to instantiate. However, if a rotation needs
    to be performed repeatedly, the Rotation class may be faster.

    .. note:: Numba JIT Compiled
        This function is compiled using Numba. Use care when providing inputs
        as Numba is strictly typed. The inputs to this function are three
        floats and a float array.

    Parameters
    ----------
    roll_angle : float
        Angle of rotation about the roll axis in radians
    pitch_angle : float
        Angle of rotation about the pitch axis in radians
    yaw_angle : float
        Angle of rotation about the yaw axis in radians
    vector : np.ndarray
        The vector to rotate

    Returns
    -------
    np.ndarray
        The vector rotated to the new frame

    """

    return roll_pitch_yaw_matrix(roll_angle, pitch_angle, yaw_angle) @ vector
