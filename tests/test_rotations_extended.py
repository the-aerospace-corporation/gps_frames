# Copyright (c) 2022 The Aerospace Corporation
"""Extended unit tests for gps_frames.rotations.

Verifies:
- Flexible constructor arguments for Rotation (quaternion kwarg, positional array, DCM, 4-tuple, standard axis)
- Handling of 180-degree rotation singularities (trace = -1) in direction_cosine_matrix2quaternion
- Standard rotation matrix generation and time derivative rate matrices for axes 1, 2, and 3
- Roll-Pitch-Yaw 3-2-1 Euler sequence rotation matrices and gimbal lock warnings
- Branch coverage for standard_rotation vector transformations
"""

import logging
import numpy as np
import pytest

from gps_frames import rotations


def test_rotation_init_standard_axis():
    """Verify Rotation initialization with standard_axis parameter.

    Testing:
        Rotation(standard_axis=3, angle=pi/2) initializes a passive coordinate rotation
        about the Z-axis (standard axis 3) by 90 degrees.
        Rotation matrix R_3(pi/2):
            [[ 0,  1, 0],
             [-1,  0, 0],
             [ 0,  0, 1]]

    Expected Result:
        Rotating vector [1.0, 0.0, 0.0] yields:
            R_3 @ [1, 0, 0] = [0.0, -1.0, 0.0].
    """
    rot = rotations.Rotation(standard_axis=3, angle=np.pi / 2)
    vec = np.array([1.0, 0.0, 0.0])
    res = rot.rotate(vec)
    assert np.allclose(res, [0, -1, 0])


def test_rotation_init_quaternion_kwarg():
    """Verify Rotation initialization with quaternion keyword argument.

    Testing:
        Rotation(quaternion=[1, 0, 0, 0]) constructs an identity rotation.

    Expected Result:
        Rotating any vector [1.0, 2.0, 3.0] returns the identical vector.
    """
    q = np.array([1.0, 0.0, 0.0, 0.0])
    rot = rotations.Rotation(quaternion=q)
    vec = np.array([1.0, 2.0, 3.0])
    assert np.allclose(rot.rotate(vec), vec)


def test_rotation_init_positional_single_arg_quat():
    """Verify Rotation initialization with a single positional 4-element quaternion array.

    Testing:
        Positional dispatch identifying a 4-element 1D array as a unit quaternion [w, x, y, z].

    Expected Result:
        Identity quaternion produces identity rotation on [1.0, 2.0, 3.0].
    """
    q = np.array([1.0, 0.0, 0.0, 0.0])
    rot = rotations.Rotation(q)
    vec = np.array([1.0, 2.0, 3.0])
    assert np.allclose(rot.rotate(vec), vec)


def test_rotation_init_positional_single_arg_dcm():
    """Verify Rotation initialization with a single positional 3x3 DCM array.

    Testing:
        Positional dispatch identifying a (3, 3) 2D array as a Direction Cosine Matrix.

    Expected Result:
        Identity matrix np.eye(3) produces identity rotation on [1.0, 2.0, 3.0].
    """
    dcm = np.eye(3)
    rot = rotations.Rotation(dcm)
    vec = np.array([1.0, 2.0, 3.0])
    assert np.allclose(rot.rotate(vec), vec)


def test_rotation_init_positional_four_args():
    """Verify Rotation initialization with 4 separate scalar arguments (w, x, y, z).

    Testing:
        Constructor signature Rotation(w, x, y, z).

    Expected Result:
        Rotation(1.0, 0.0, 0.0, 0.0) produces identity rotation on [1.0, 2.0, 3.0].
    """
    rot = rotations.Rotation(1.0, 0.0, 0.0, 0.0)
    vec = np.array([1.0, 2.0, 3.0])
    assert np.allclose(rot.rotate(vec), vec)


def test_rotation_init_errors():
    """Verify constructor error handling for contradictory or malformed arguments.

    Testing:
        Validation in Rotation.__init__():
        - Providing both positional quaternion and keyword quaternion
        - Providing 3 positional arguments (neither 1, 2, nor 4 arguments)
        - Providing conflicting combinations of keyword arguments (e.g. dcm and axis)

    Expected Result:
        Raises AssertionError or ValueError appropriately.
    """
    with pytest.raises(AssertionError):
        rotations.Rotation([1, 0, 0, 0], quaternion=[1, 0, 0, 0])

    with pytest.raises(ValueError):
        rotations.Rotation(1, 2, 3)

    with pytest.raises(AssertionError):
        rotations.Rotation(dcm=np.eye(3), axis=[0, 0, 1])
    with pytest.raises(AssertionError):
        rotations.Rotation(axis=[0, 0, 1], angle=0, dcm=np.eye(3))
    with pytest.raises(AssertionError):
        rotations.Rotation(standard_axis=1, angle=0, dcm=np.eye(3))
    with pytest.raises(AssertionError):
        rotations.Rotation(quaternion=[1, 0, 0, 0], dcm=np.eye(3))


def test_rotation_singularity():
    """Verify handling of the 180-degree rotation singularity in DCM-to-quaternion extraction.

    Testing:
        When rotation angle is 180 degrees (pi rad), the matrix trace satisfies:
            trace(R) = 1 + 2*cos(pi) = -1
        yielding scalar quaternion component qw = 0.5 * sqrt(1 + trace(R)) = 0.
        The algorithm must fall back to resolving the eigenvector corresponding to eigenvalue +1.

    Expected Result:
        For DCM = diag([1.0, -1.0, -1.0]) (180 deg rotation about X):
            quat[0] (qw) == 0.0
            abs(quat[1]) (qx) == 1.0
            quat[2] (qy) == 0.0, quat[3] (qz) == 0.0.
    """
    dcm = np.diag([1.0, -1.0, -1.0])
    quat = rotations.direction_cosine_matrix2quaternion(dcm)

    assert np.isclose(quat[0], 0, atol=1e-8)
    assert np.isclose(np.abs(quat[1]), 1.0)


def test_standard_rotation_matrix_and_rates():
    """Verify analytical standard rotation matrices and rate matrices for 90-degree rotations.

    Testing:
        rotations.standard_rotation_matrix(axis, angle) and standard_rotation_matrix_rates(axis, angle, rate).

    Expected Result:
        At angle = pi/2:
        - Axis 1 (X): R1 == [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        - Axis 2 (Y): R2 == [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
        - Axis 3 (Z): R3 == [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        Rate matrices execute without error for all three axes.
    """
    angle = np.pi / 2
    rate = 1.0

    R1 = rotations.standard_rotation_matrix(1, angle)
    assert np.allclose(R1, [[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    R2 = rotations.standard_rotation_matrix(2, angle)
    assert np.allclose(R2, [[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    R3 = rotations.standard_rotation_matrix(3, angle)
    assert np.allclose(R3, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    rotations.standard_rotation_matrix_rates(1, angle, rate)
    rotations.standard_rotation_matrix_rates(2, angle, rate)
    rotations.standard_rotation_matrix_rates(3, angle, rate)


def test_rotation_init_axis_angle_positional():
    """Verify Rotation initialization with positional axis vector and scalar angle.

    Testing:
        Rotation(axis, angle) creates rotation corresponding to passive coordinate transformation:
            R_3(pi/2) @ [1.0, 0.0, 0.0] = [0.0, -1.0, 0.0].

    Expected Result:
        Output rotated vector is [0.0, -1.0, 0.0].
    """
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 2

    rot = rotations.Rotation(axis, angle)
    vec = np.array([1.0, 0.0, 0.0])
    res = rot.rotate(vec)

    expected = np.array([0.0, -1.0, 0.0])
    assert np.allclose(res, expected, atol=1e-15)


def test_dcm_singularity_error():
    """Verify that a malformed matrix with trace = -1 and no eigenvalue = 1 raises ValueError.

    Testing:
        Encountered singular DCM error in direction_cosine_matrix2quaternion.

    Expected Result:
        ValueError("Encountered singular DCM") is raised for non-rotation matrix diag([-1, 0, 0]).
    """
    bad_dcm = np.diag([-1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Encountered singular DCM"):
        rotations.direction_cosine_matrix2quaternion(bad_dcm)


def test_roll_pitch_yaw_matrix():
    """Verify 3-2-1 Euler sequence rotation matrix (Yaw -> Pitch -> Roll).

    Testing:
        rotations.roll_pitch_yaw_matrix(roll, pitch, yaw) constructs active composite rotation:
            R_rpy = R_x(roll)^T @ R_y(pitch)^T @ R_z(yaw)^T

    Expected Result:
        - Roll-only (roll = pi/2, pitch = 0, yaw = 0) matches transpose of passive R_1(pi/2).
        - Pitch-only (pitch = pi/2) matches transpose of passive R_2(pi/2).
        - Yaw-only (yaw = pi/2) matches transpose of passive R_3(pi/2).
        - Composite matrix satisfies det(R) == 1.0 and R @ R.T == I.
    """
    angle = np.pi / 2

    R_roll = rotations.roll_pitch_yaw_matrix(angle, 0, 0)
    R_std_1 = rotations.standard_rotation_matrix(1, angle)
    assert np.allclose(R_roll, R_std_1.T)

    R_pitch = rotations.roll_pitch_yaw_matrix(0, angle, 0)
    R_std_2 = rotations.standard_rotation_matrix(2, angle)
    assert np.allclose(R_pitch, R_std_2.T)

    R_yaw = rotations.roll_pitch_yaw_matrix(0, 0, angle)
    R_std_3 = rotations.standard_rotation_matrix(3, angle)
    assert np.allclose(R_yaw, R_std_3.T)

    R_composite = rotations.roll_pitch_yaw_matrix(angle, angle, angle)
    assert np.isclose(np.linalg.det(R_composite), 1.0)
    assert np.allclose(R_composite @ R_composite.T, np.eye(3))


def test_roll_pitch_yaw_vector_rotation():
    """Verify vector rotation using roll_pitch_yaw convenience function.

    Testing:
        rotations.roll_pitch_yaw(0, 0, pi/2, vec) applies an active 90-degree Yaw rotation about Z.

    Expected Result:
        Vector [1.0, 0.0, 0.0] on X-axis rotates into the Y-axis: [0.0, 1.0, 0.0].
    """
    res = rotations.roll_pitch_yaw(0, 0, np.pi / 2, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(res, [0, 1, 0])


def test_standard_rotation_function_coverage():
    """Verify functional vector rotation standard_rotation across all three standard axes.

    Testing:
        rotations.standard_rotation(axis, angle, vec) for axis in {1, 2, 3}:
        - Axis 1 (X-axis rotation): vector on X [1, 0, 0] is invariant -> [1, 0, 0].
        - Axis 2 (Y-axis rotation): R_2(pi/2) @ [1, 0, 0] = [0, 0, 1] (Z-axis).
        - Axis 3 (Z-axis rotation): R_3(pi/2) @ [1, 0, 0] = [0, -1, 0] (-Y-axis).

    Expected Result:
        - Axis 1: [1.0, 0.0, 0.0]
        - Axis 2: [0.0, 0.0, 1.0]
        - Axis 3: [0.0, -1.0, 0.0]
    """
    vec = np.array([1.0, 0.0, 0.0])
    angle = np.pi / 2

    res1 = rotations.standard_rotation(1, angle, vec)
    assert np.allclose(res1, vec)

    res2 = rotations.standard_rotation(2, angle, vec)
    assert np.allclose(res2, np.array([0.0, 0.0, 1.0]))

    res3 = rotations.standard_rotation(3, angle, vec)
    assert np.allclose(res3, np.array([0.0, -1.0, 0.0]))


def test_roll_pitch_yaw_singularity_warning(caplog):
    """Verify logging of gimbal lock warning when pitch angle is near 90 degrees (+/- pi/2).

    Testing:
        Pitch angle pitch = pi/2 causes pitch gimbal lock in 3-2-1 Euler sequence,
        triggering a logger warning.

    Expected Result:
        'Singular rotation (gimbal lock) detected' appears in log records.
    """
    phi = np.pi / 2

    with caplog.at_level(logging.WARNING):
        rotations.roll_pitch_yaw_matrix(0, phi, 0)

    assert "Singular rotation (gimbal lock) detected" in caplog.text


def test_standard_rotation_detailed():
    """Verify SO(3) Lie group properties and analytical entries of standard rotation matrices.

    Testing:
        Every standard rotation matrix R_i(pi/2) satisfies det(R) == 1 and R @ R.T == I.

    Expected Result:
        - Axis 1: [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        - Axis 2: [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
        - Axis 3: [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        Rate matrix R_dot shape is (3, 3).
    """
    for axis in [1, 2, 3]:
        R = rotations.standard_rotation_matrix(axis, np.pi / 2)

        det = np.linalg.det(R)
        assert np.isclose(det, 1.0)

        orth = R @ R.T
        assert np.allclose(orth, np.eye(3))

        if axis == 1:
            expected = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            assert np.allclose(R, expected)
        elif axis == 2:
            expected = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            assert np.allclose(R, expected)
        elif axis == 3:
            expected = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            assert np.allclose(R, expected)

    R_dot = rotations.standard_rotation_matrix_rates(1, np.pi / 2, 1.0)
    assert R_dot.shape == (3, 3)
