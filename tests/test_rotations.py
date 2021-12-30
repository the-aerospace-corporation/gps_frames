# Copyright (c) 2022 The Aerospace Corporation

import pytest
import numpy as np

# import quaternion
from gps_frames import rotations

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
    """Checks to make sure Rotation raises an error if 4 elements are not
    provided.

    """
    _quaternion = np.array([0.0, 0.0, 0.0])
    with pytest.raises(AssertionError):
        rotations.Rotation(quaternion=_quaternion)


# def test_init_normalize():
#     """Checks to ensure quaternion is normalized.
#     """
#     _quaternion = [1, 1, 1, 1]
#     rot = rotations.Rotation(quaternion=_quaternion)
#     assert(np.linalg.norm(rot.quaternion) == 1)


@pytest.mark.parametrize(
    "input_quaternion",
    [[1.0, 1.0, 1.0, 1.0], (1.0, 1.0, 1.0, 1.0), np.array((1.0, 1.0, 1.0, 1.0))],
)
def test_init_input_types(input_quaternion):
    """Check to ensure all of the valid input types work."""
    rot = rotations.Rotation(input_quaternion)

    # assert isinstance(rot._quaternion, np.quaternion), \
    #     f'{input_quaternion.__name__} did not produce a quaternion'


def test_from_direction_cosine_matrix_wrong_shape():
    """Check that the DCM conversion throws an error if wrong shape."""
    with pytest.raises(AssertionError):
        rotations.Rotation(dcm=np.array([0]))


def test_from_direction_cosine_matrix_right_handed():
    """Check that the DCM conversion throws an error if left handed."""
    with pytest.raises(AssertionError):
        rotations.Rotation(
            dcm=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        )


def test_from_direction_cosine_matrix_orthonormal():
    """Check that the DCM conversion throws an error if not orthonormal."""
    with pytest.raises(AssertionError):
        rotations.Rotation(
            dcm=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        )


def test_from_direction_cosine_matrix_converge():
    """Check to ensure near-singualr DCMs still work."""
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
    # rot_dcm = rotations.Rotation.from_direction_cosine_matrix(dcm)

    dcm_axis, dcm_angle = rotations.direction_cosine_matrix2euler_axis_angle(dcm)

    if np.isclose(dcm_angle, -axis_angle[1]):
        print("Flipping axis and angle to avoid ambiguity")
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
    # rot_axis_angle = rotations.Rotation.from_euler_axis_angle(
    #     axis_angle[0], axis_angle[1])

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
    rot_axis_angle = rotations.Rotation(axis=axis_angle[0], angle=axis_angle[1])

    axis, angle = rotations.direction_cosine_matrix2euler_axis_angle(
        rot_axis_angle._rotator.dcm
    )

    if np.isclose(angle, -axis_angle[1]):
        print("Flipping axis and angle to avoid ambiguity")
        axis = -axis
        angle = -angle

    assert np.allclose(
        axis, axis_angle[0]
    ), f"Inaccurate axis. IS: {axis} SB: {axis_angle[0]}"
    assert np.isclose(
        angle, axis_angle[1]
    ), f"Inaccurate angle. IS: {angle} SB: {axis_angle[1]}"


# @pytest.mark.parametrize(
#     "axis_angle, dcm",
#     [(STANDARD_ROTATIONS_AXIS_ANGLE[_n], STANDARD_ROTATIONS_DCM[_n])
#      for _n in STANDARD_ROTATIONS_AXIS_ANGLE])
# def test_dcm_to_dcm(axis_angle, dcm):
#     rot_dcm = rotations.Rotation.from_direction_cosine_matrix(dcm)

#     out_dcm = rot_dcm.to_direction_cosine_matrix()

#     assert np.allclose(out_dcm, dcm), \
#         f'Inaccurate DCM. IS: {out_dcm} SB: {dcm}'


@pytest.mark.parametrize("test_case", list(STANDARD_ROTATIONS_INPUT_VECTOR.keys()))
def test_rotate_dcm(test_case):
    input_vec = STANDARD_ROTATIONS_INPUT_VECTOR[test_case]
    output_vec = STANDARD_ROTATIONS_OUTPUT_VECTOR[test_case]

    dcm = STANDARD_ROTATIONS_DCM[test_case]

    rot_dcm = rotations.Rotation(dcm=dcm)

    print("*" * 10, rot_dcm.rotate(input_vec), output_vec)
    assert np.allclose(rot_dcm.rotate(input_vec), output_vec), "DCM Rotation Failed"


@pytest.mark.parametrize("test_case", list(STANDARD_ROTATIONS_INPUT_VECTOR.keys()))
def test_rotate_axis_angle(test_case):
    input_vec = STANDARD_ROTATIONS_INPUT_VECTOR[test_case]
    output_vec = STANDARD_ROTATIONS_OUTPUT_VECTOR[test_case]

    axis = STANDARD_ROTATIONS_AXIS_ANGLE[test_case][0]
    angle = STANDARD_ROTATIONS_AXIS_ANGLE[test_case][1]

    rot_axis_angle = rotations.Rotation(axis=axis, angle=angle)

    print("*" * 10, rot_axis_angle.rotate(input_vec), output_vec)
    assert np.allclose(
        rot_axis_angle.rotate(input_vec), output_vec
    ), "Axis-Angle Rotation Failed"


@pytest.mark.parametrize("test_case", list(STANDARD_ROTATIONS_INPUT_VECTOR.keys()))
def test_rotate_standard_rotation(test_case):
    input_vec = STANDARD_ROTATIONS_INPUT_VECTOR[test_case]
    output_vec = STANDARD_ROTATIONS_OUTPUT_VECTOR[test_case]

    axis_number = STANDARD_ROTATIONS_AXISNUM_ANGLE[test_case][0]
    angle = STANDARD_ROTATIONS_AXISNUM_ANGLE[test_case][1]

    # dcm = STANDARD_ROTATIONS_DCM[test_case]

    # rot_axis_angle = rotations.Rotation(axis=axis, angle=angle)
    # rot_dcm = rotations.Rotation(dcm=dcm)
    # print("*"*10, rot_axis_angle.rotate(input_vec), output_vec)
    # print("*"*10, rot_dcm.rotate(input_vec), output_vec)
    # assert np.allclose(rot_axis_angle.rotate(input_vec), output_vec), \
    #     "Axis-Angle Rotation Failed"
    assert np.allclose(
        rotations.standard_rotation(axis_number, angle, input_vec), output_vec
    ), "Standard Rotation Failed"
