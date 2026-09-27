# Copyright (c) 2022 The Aerospace Corporation
"""Regression and mathematical invariant tests for gps_frames.rotations.

Verifies SO(3) Lie group properties, Euler axis-angle / Quaternion / DCM round-trips,
norm and inner product preservation, rates, and edge cases.
"""

import numpy as np
import pytest
from gps_frames import rotations


class TestRotationSO3Invariants:
    """Verifies that all generated rotations satisfy the mathematical axioms of SO(3)."""

    @pytest.mark.parametrize(
        "axis",
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
            np.array([1.0, 2.0, -3.0]) / np.linalg.norm([1.0, 2.0, -3.0]),
            np.array([-0.5, 0.8, 0.3]) / np.linalg.norm([-0.5, 0.8, 0.3]),
        ],
    )
    @pytest.mark.parametrize(
        "angle",
        [
            0.0,
            1e-7,
            np.pi / 6,
            np.pi / 4,
            np.pi / 3,
            np.pi / 2,
            2 * np.pi / 3,
            np.pi - 1e-6,
            np.pi,
            -np.pi / 4,
            -np.pi / 2,
            2 * np.pi,
        ],
    )
    def test_dcm_orthogonality_and_determinant(self, axis, angle):
        """Every rotation matrix R must satisfy R @ R.T == I and det(R) == +1."""
        dcm = rotations.euler_axis_angle2direction_cosine_matrix(axis, angle)

        # Orthogonality: R * R.T = I
        identity = np.eye(3)
        assert np.allclose(dcm @ dcm.T, identity, atol=1e-12)
        assert np.allclose(dcm.T @ dcm, identity, atol=1e-12)

        # Proper rotation: det(R) = +1
        assert np.isclose(np.linalg.det(dcm), 1.0, atol=1e-12)

    @pytest.mark.parametrize(
        "axis",
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, -1.0, 2.0]) / np.linalg.norm([1.0, -1.0, 2.0]),
        ],
    )
    @pytest.mark.parametrize("angle", [np.pi / 5, np.pi / 2, 3 * np.pi / 4])
    def test_norm_and_inner_product_preservation(self, axis, angle):
        """Rotations must preserve vector lengths, dot products, and cross products."""
        rot = rotations.Rotation(axis=axis, angle=angle)

        v1 = np.array([3.0, -4.0, 12.0])  # norm = 13
        v2 = np.array([1.0, 2.0, 2.0])    # norm = 3

        v1_rot = rot.rotate(v1)
        v2_rot = rot.rotate(v2)

        # Norm preservation
        assert np.isclose(np.linalg.norm(v1_rot), np.linalg.norm(v1), atol=1e-12)
        assert np.isclose(np.linalg.norm(v2_rot), np.linalg.norm(v2), atol=1e-12)

        # Inner product preservation: (R*u) . (R*v) == u . v
        assert np.isclose(np.dot(v1_rot, v2_rot), np.dot(v1, v2), atol=1e-12)

        # Cross product equivariance: R*(u x v) == (R*u) x (R*v)
        cross_then_rot = rot.rotate(np.cross(v1, v2))
        rot_then_cross = np.cross(v1_rot, v2_rot)
        assert np.allclose(cross_then_rot, rot_then_cross, atol=1e-12)

    @pytest.mark.parametrize("std_axis", [1, 2, 3])
    @pytest.mark.parametrize("angle", [0.1, np.pi / 4, np.pi / 2, np.pi - 0.01])
    def test_inverse_rotation(self, std_axis, angle):
        """Rotating by angle then -angle must restore original vector."""
        rot_fwd = rotations.Rotation(standard_axis=std_axis, angle=angle)
        rot_inv = rotations.Rotation(standard_axis=std_axis, angle=-angle)

        test_vec = np.array([123.456, -789.012, 345.678])
        round_trip = rot_inv.rotate(rot_fwd.rotate(test_vec))
        assert np.allclose(round_trip, test_vec, atol=1e-12)


class TestRotationRoundTrips:
    """Tests conversions between Euler Axis/Angle, Quaternion, and DCM representations."""

    @pytest.mark.parametrize(
        "axis",
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
            np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0]),
        ],
    )
    @pytest.mark.parametrize(
        "angle",
        [
            0.1,
            np.pi / 4,
            np.pi / 2,
            2 * np.pi / 3,
            np.pi - 1e-4,
        ],
    )
    def test_euler_axis_angle_to_dcm_round_trip(self, axis, angle):
        """Converting axis-angle to DCM and back recovers equivalent axis and angle."""
        dcm = rotations.euler_axis_angle2direction_cosine_matrix(axis, angle)
        rec_axis, rec_angle = rotations.direction_cosine_matrix2euler_axis_angle(dcm)

        # Either (rec_axis, rec_angle) matches (axis, angle)
        # or antipodal (-axis, 2*pi - angle) or (-axis, -angle)
        case1 = np.allclose(rec_axis, axis, atol=1e-7) and np.isclose(rec_angle, angle, atol=1e-7)
        case2 = np.allclose(rec_axis, -axis, atol=1e-7) and np.isclose(rec_angle, -angle, atol=1e-7)
        assert case1 or case2

    @pytest.mark.parametrize(
        "quat",
        [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0]),
        ],
    )
    def test_quaternion_dcm_round_trip(self, quat):
        """Quaternion -> DCM -> Quaternion returns equivalent quaternion (q or -q)."""
        quat_normalized = quat / np.linalg.norm(quat)
        dcm = rotations.quaternion2direction_cosine_matrix(quat_normalized)
        rec_quat = rotations.direction_cosine_matrix2quaternion(dcm)

        # Quaternions q and -q represent the exact same rotation
        same = np.allclose(rec_quat, quat_normalized, atol=1e-7)
        neg_same = np.allclose(rec_quat, -quat_normalized, atol=1e-7)
        assert same or neg_same


class TestStandardRotationEquivalence:
    """Verifies standard_rotation (direct vector rotation) against standard_rotation_matrix."""

    @pytest.mark.parametrize("axis", [1, 2, 3])
    @pytest.mark.parametrize(
        "angle",
        [-np.pi, -np.pi / 2, -0.5, 0.0, 0.5, np.pi / 4, np.pi / 2, np.pi],
    )
    def test_standard_rotation_matches_matrix(self, axis, angle):
        vec = np.array([3.14, -2.71, 1.41])
        res_direct = rotations.standard_rotation(axis, angle, vec)
        dcm = rotations.standard_rotation_matrix(axis, angle)
        res_matrix = dcm @ vec
        assert np.allclose(res_direct, res_matrix, atol=1e-14)


class TestRateMatrixFiniteDifferences:
    """Validates standard_rotation_matrix_rates against numerical derivative."""

    @pytest.mark.parametrize("axis", [1, 2, 3])
    @pytest.mark.parametrize("angle", [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2.5])
    @pytest.mark.parametrize("rate", [1.0, -2.5, 7.292115e-5])
    def test_rate_matrix_finite_difference(self, axis, angle, rate):
        """Checks dR/dt ~ (R(t + dt) - R(t - dt)) / (2*dt)."""
        analytical_rate = rotations.standard_rotation_matrix_rates(axis, angle, rate)

        dt = 1e-6
        d_angle = rate * dt
        r_plus = rotations.standard_rotation_matrix(axis, angle + d_angle)
        r_minus = rotations.standard_rotation_matrix(axis, angle - d_angle)
        numerical_rate = (r_plus - r_minus) / (2.0 * dt)

        assert np.allclose(analytical_rate, numerical_rate, atol=1e-7)


class TestRollPitchYawRegression:
    """Tests properties of the 3-2-1 roll-pitch-yaw Euler sequence."""

    def test_zero_angles_gives_identity(self):
        dcm = rotations.roll_pitch_yaw_matrix(0.0, 0.0, 0.0)
        assert np.allclose(dcm, np.eye(3), atol=1e-15)

    @pytest.mark.parametrize("roll", [-np.pi / 3, 0.0, np.pi / 4])
    @pytest.mark.parametrize("pitch", [-np.pi / 6, 0.0, np.pi / 6])
    @pytest.mark.parametrize("yaw", [-np.pi / 2, 0.0, 3 * np.pi / 4])
    def test_roll_pitch_yaw_properties(self, roll, pitch, yaw):
        dcm = rotations.roll_pitch_yaw_matrix(roll, pitch, yaw)
        # Must be orthogonal
        assert np.allclose(dcm @ dcm.T, np.eye(3), atol=1e-12)
        # Determinant must be +1
        assert np.isclose(np.linalg.det(dcm), 1.0, atol=1e-12)

    def test_gimbal_lock_warning_negative_pi_over_2(self, caplog):
        """Test warning logged for pitch = -pi/2."""
        import logging
        with caplog.at_level(logging.WARNING):
            rotations.roll_pitch_yaw_matrix(0.0, -np.pi / 2, 0.0)
        assert "Singular rotation (gimbal lock) detected" in caplog.text
