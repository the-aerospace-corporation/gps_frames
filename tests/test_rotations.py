# Copyright (c) 2022 The Aerospace Corporation
"""Unit tests for the Rotation class and rotation functions in gps_frames.rotations.

Verifies:
- Quaternion, Euler axis-angle, and Direction Cosine Matrix (DCM) representations
- Proper SO(3) properties: orthogonality (R @ R.T == I), determinant (+1 for right-handed), and dimension checks
- Forward and inverse conversions between Euler axis-angle and DCM representations
- Vector rotation calculations using DCM, axis-angle, and standard axis rotations
"""

import numpy as np
import pytest

from gps_frames import rotations

# Reference test cases for standard rotations
STANDARD_ROTATIONS_AXIS_ANGLE = {
    1: (np.array((1.0, 0.0, 0.0)), np.pi / 2),
    2: (np.array([0.0, 0.0, 1.0]), np.pi / 2),
    3: (np.array([1.0, 0.0, 0.0]), np.pi / 6),
    4: (np.array((1.0, 0.0, 0.0)), np.pi - 1e-8),
    5: (np.array([0.0, 0.0, 1.0]), -np.pi / 2),
}

STANDARD_ROTATIONS_DCM = {
    1: np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    2: np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    3: np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.sqrt(3.0) / 2.0, 0.5],
            [0.0, -0.5, np.sqrt(3.0) / 2.0],
        ]
    ),
    4: np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(np.pi - 1e-8), np.sin(np.pi - 1e-8)],
            [0.0, -np.sin(np.pi - 1e-8), np.cos(np.pi - 1e-8)],
        ]
    ),
    5: np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}

STANDARD_ROTATIONS_AXISNUM_ANGLE = {
    1: (1, np.pi / 2),
    2: (3, np.pi / 2),
    3: (1, np.pi / 6),
    4: (1, np.pi - 1e-8),
    5: (3, -np.pi / 2),
}

STANDARD_ROTATIONS_INPUT_VECTOR = {
    1: np.array([1.0, 2.0, 3.0]),
    2: np.array([1.0, 2.0, 3.0]),
    3: np.array([1.0, 2.0, 3.0]),
    4: np.array([1.0, 2.0, 3.0]),
    5: np.array([1.0, 2.0, 3.0]),
}

STANDARD_ROTATIONS_OUTPUT_VECTOR = {
    1: np.array([1.0, 3.0, -2.0]),
    2: np.array([2.0, -1.0, 3.0]),
    3: np.array([1.0, np.sqrt(3.0) + 1.5, -1.0 + 1.5 * np.sqrt(3)]),
    4: np.array([1.0, -2.0, -3.0]),
    5: np.array([-2.0, 1.0, 3.0]),
}


def test_init_not_four_elements():
    """Verify that initializing Rotation with a quaternion of length != 4 raises AssertionError.

    Testing:
        Input dimension check in Rotation.__init__(). A unit quaternion q = [w, x, y, z]
        must have exactly 4 elements.

    Expected Result:
        AssertionError is raised when providing a 3-element array [0, 0, 0].
    """
    _quaternion = np.array([0.0, 0.0, 0.0])
    with pytest.raises(AssertionError):
        rotations.Rotation(quaternion=_quaternion)


@pytest.mark.parametrize(
    "input_quaternion",
    [[1.0, 1.0, 1.0, 1.0], (1.0, 1.0, 1.0, 1.0), np.array((1.0, 1.0, 1.0, 1.0))],
)
def test_init_input_types(input_quaternion):
    """Verify that Rotation accepts list, tuple, and NumPy array quaternion inputs.

    Testing:
        Flexible container type handling in Rotation constructor.

    Expected Result:
        Instance created successfully for all valid 4-element sequence types.
    """
    rot = rotations.Rotation(input_quaternion)
    assert rot is not None


def test_from_direction_cosine_matrix_wrong_shape():
    """Verify that initializing Rotation with a DCM of invalid shape raises AssertionError.

    Testing:
        Shape validation for DCMs in Rotation(dcm=...). DCM must have shape (3, 3).

    Expected Result:
        AssertionError is raised for 1D array of shape (1,).
    """
    with pytest.raises(AssertionError):
        rotations.Rotation(dcm=np.array([0]))


def test_from_direction_cosine_matrix_right_handed():
    """Verify that a reflection matrix with determinant -1 is rejected.

    Testing:
        SO(3) proper rotation requirement: det(R) == +1.
        A matrix with a reflected column has det(R) = -1 and represents a left-handed coordinate system.

    Expected Result:
        AssertionError is raised for matrix [[1, 0, 0], [-1, 0, 0], [0, 0, 1]] (det = -1).
    """
    with pytest.raises(AssertionError):
        rotations.Rotation(
            dcm=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        )


def test_from_direction_cosine_matrix_orthonormal():
    """Verify that a non-orthogonal matrix is rejected.

    Testing:
        SO(3) orthogonality condition: R @ R.T == I.
        A matrix scaled along an axis (e.g. diag([1, 2, 1])) is not orthonormal.

    Expected Result:
        AssertionError is raised for matrix [[1, 0, 0], [0, 2, 0], [0, 0, 1]].
    """
    with pytest.raises(AssertionError):
        rotations.Rotation(
            dcm=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        )


def test_from_direction_cosine_matrix_converge():
    """Verify that near-singular DCMs (rotation angle near pi rad / 180 deg) converge stably.

    Testing:
        Numerical stability of DCM-to-quaternion extraction near the 180-degree singularity:
            trace(R) = 1 + 2*cos(theta) -> 1 - 2 = -1 when theta -> pi.

    Expected Result:
        Rotation instance is successfully created without division by zero or NaN coordinates.
    """
    _ = rotations.Rotation(
        dcm=np.array(
            [
                [1, 0, 0],
                [0, np.cos(np.pi - 1e-8), np.sin(np.pi - 1e-8)],
                [0, -np.sin(np.pi - 1e-8), np.cos(np.pi - 1e-8)],
            ]
        )
    )


@pytest.mark.parametrize(
    "axis_angle, dcm",
    [
        (STANDARD_ROTATIONS_AXIS_ANGLE[_n], STANDARD_ROTATIONS_DCM[_n])
        for _n in STANDARD_ROTATIONS_AXIS_ANGLE
    ],
)
def test_direction_cosine_matrix_to_axis_angle(axis_angle, dcm):
    """Verify conversion from Direction Cosine Matrix (DCM) to Euler axis-angle representation.

    Testing:
        rotations.direction_cosine_matrix2euler_axis_angle(dcm) extracts:
            cos(theta) = (trace(dcm) - 1) / 2
            e_axis = [dcm[1, 2] - dcm[2, 1], dcm[2, 0] - dcm[0, 2], dcm[0, 1] - dcm[1, 0]] / (2 * sin(theta))

    Expected Result:
        Extracted axis and angle match the ground truth axis and angle (accounting for dual representation (e, theta) == (-e, -theta)).
        - Case 1 (+90 deg about X): axis = [1, 0, 0], angle = pi/2 rad.
        - Case 2 (+90 deg about Z): axis = [0, 0, 1], angle = pi/2 rad.
        - Case 3 (+30 deg about X): axis = [1, 0, 0], angle = pi/6 rad.
        - Case 4 (~180 deg about X): axis = [1, 0, 0], angle = pi - 1e-8 rad.
        - Case 5 (-90 deg about Z): axis = [0, 0, 1], angle = -pi/2 rad.
    """
    dcm_axis, dcm_angle = rotations.direction_cosine_matrix2euler_axis_angle(dcm)

    if np.isclose(dcm_angle, -axis_angle[1]):
        dcm_axis = -dcm_axis
        dcm_angle = -dcm_angle

    assert np.allclose(
        dcm_axis, axis_angle[0]
    ), f"Inaccurate axis. IS: {dcm_axis} SB: {axis_angle[0]}"
    assert np.isclose(
        dcm_angle, axis_angle[1]
    ), f"Inaccurate angle. IS: {dcm_angle} SB: {axis_angle[1]}"


@pytest.mark.parametrize(
    "axis_angle, dcm",
    [
        (STANDARD_ROTATIONS_AXIS_ANGLE[_n], STANDARD_ROTATIONS_DCM[_n])
        for _n in STANDARD_ROTATIONS_AXIS_ANGLE
    ],
)
def test_axis_angle_to_dcm(axis_angle, dcm):
    """Verify conversion from Euler axis-angle representation to Direction Cosine Matrix (DCM).

    Testing:
        Rodrigues' rotation formula in passive convention:
            R = cos(theta) * I + (1 - cos(theta)) * (e (x) e) - sin(theta) * [e]_x

    Expected Result:
        Computed 3x3 matrix axis_angle_dcm equals the exact precomputed reference DCM.
    """
    axis_angle_dcm = rotations.euler_axis_angle2direction_cosine_matrix(
        axis_angle[0], axis_angle[1]
    )

    assert np.allclose(
        axis_angle_dcm, dcm
    ), f"Inaccurate DCM. IS: {axis_angle_dcm} SB: {dcm}"


@pytest.mark.parametrize(
    "axis_angle, dcm",
    [
        (STANDARD_ROTATIONS_AXIS_ANGLE[_n], STANDARD_ROTATIONS_DCM[_n])
        for _n in STANDARD_ROTATIONS_AXIS_ANGLE
    ],
)
def test_axis_angle_to_axis_angle(axis_angle, dcm):
    """Verify round-trip consistency: axis-angle -> Rotation -> axis-angle.

    Testing:
        End-to-end consistency through internal rotator representation.

    Expected Result:
        The recovered rotation axis and angle match the input axis and angle within machine precision.
    """
    rot_axis_angle = rotations.Rotation(axis=axis_angle[0], angle=axis_angle[1])

    axis, angle = rotations.direction_cosine_matrix2euler_axis_angle(
        rot_axis_angle._rotator.dcm
    )

    if np.isclose(angle, -axis_angle[1]):
        axis = -axis
        angle = -angle

    assert np.allclose(
        axis, axis_angle[0]
    ), f"Inaccurate axis. IS: {axis} SB: {axis_angle[0]}"
    assert np.isclose(
        angle, axis_angle[1]
    ), f"Inaccurate angle. IS: {angle} SB: {axis_angle[1]}"


@pytest.mark.parametrize("test_case", list(STANDARD_ROTATIONS_INPUT_VECTOR.keys()))
def test_rotate_dcm(test_case):
    """Verify vector rotation using a Direction Cosine Matrix.

    Testing:
        Rotation.rotate(v) computes matrix-vector product: v_rotated = DCM @ v.

    Expected Result:
        For input vector v = [1.0, 2.0, 3.0]:
        - Case 1 (R_1(pi/2)): [[1, 0, 0], [0, 0, 1], [0, -1, 0]] @ [1, 2, 3] = [1, 3, -2]
        - Case 2 (R_3(pi/2)): [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] @ [1, 2, 3] = [2, -1, 3]
        - Case 3 (R_1(pi/6)):
            x = 1
            y = 2*cos(30 deg) + 3*sin(30 deg) = 2*(sqrt(3)/2) + 3*(1/2) = sqrt(3) + 1.5
            z = -2*sin(30 deg) + 3*cos(30 deg) = -2*(1/2) + 3*(sqrt(3)/2) = -1 + 1.5*sqrt(3)
            v_rotated = [1.0, sqrt(3) + 1.5, -1.0 + 1.5*sqrt(3)]
        - Case 4 (R_1(pi)): [1, -2, -3]
        - Case 5 (R_3(-pi/2)): [[0, -1, 0], [1, 0, 0], [0, 0, 1]] @ [1, 2, 3] = [-2, 1, 3]
    """
    input_vec = STANDARD_ROTATIONS_INPUT_VECTOR[test_case]
    output_vec = STANDARD_ROTATIONS_OUTPUT_VECTOR[test_case]
    dcm = STANDARD_ROTATIONS_DCM[test_case]

    rot_dcm = rotations.Rotation(dcm=dcm)
    assert np.allclose(rot_dcm.rotate(input_vec), output_vec), "DCM Rotation Failed"


@pytest.mark.parametrize("test_case", list(STANDARD_ROTATIONS_INPUT_VECTOR.keys()))
def test_rotate_axis_angle(test_case):
    """Verify vector rotation using Euler axis-angle parameters.

    Testing:
        Rotation(axis=axis, angle=angle).rotate(input_vec) calculates rotation directly
        from axis-angle parameterization.

    Expected Result:
        v_rotated matches the exact expected output vector for each test case.
    """
    input_vec = STANDARD_ROTATIONS_INPUT_VECTOR[test_case]
    output_vec = STANDARD_ROTATIONS_OUTPUT_VECTOR[test_case]

    axis = STANDARD_ROTATIONS_AXIS_ANGLE[test_case][0]
    angle = STANDARD_ROTATIONS_AXIS_ANGLE[test_case][1]

    rot_axis_angle = rotations.Rotation(axis=axis, angle=angle)
    assert np.allclose(
        rot_axis_angle.rotate(input_vec), output_vec
    ), "Axis-Angle Rotation Failed"


@pytest.mark.parametrize("test_case", list(STANDARD_ROTATIONS_INPUT_VECTOR.keys()))
def test_rotate_standard_rotation(test_case):
    """Verify vector rotation using the functional standard_rotation(axis_number, angle, vector).

    Testing:
        rotations.standard_rotation(axis_num, angle, v) for standard axes 1 (X), 2 (Y), 3 (Z).

    Expected Result:
        Output vector matches the analytical rotation results for all test cases.
    """
    input_vec = STANDARD_ROTATIONS_INPUT_VECTOR[test_case]
    output_vec = STANDARD_ROTATIONS_OUTPUT_VECTOR[test_case]

    axis_number = STANDARD_ROTATIONS_AXISNUM_ANGLE[test_case][0]
    angle = STANDARD_ROTATIONS_AXISNUM_ANGLE[test_case][1]

    assert np.allclose(
        rotations.standard_rotation(axis_number, angle, input_vec), output_vec
    ), "Standard Rotation Failed"
