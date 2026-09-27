# Copyright (c) 2022 The Aerospace Corporation
"""Regression and mathematical invariant tests for gps_frames.rotations.

Verifies:
- SO(3) Lie group axioms: orthogonality R @ R.T == I and det(R) == +1
- Metric space and vector algebra preservation: Euclidean norms, inner products, and cross products
- Inverse rotation round-trips: R(-theta) @ R(theta) == I
- Representation round-trips: Axis-Angle <-> DCM and Quaternion <-> DCM
- Functional equivalence between standard_rotation and standard_rotation_matrix
- Time-derivative rate matrices validated against central finite difference approximations
- Roll-Pitch-Yaw 3-2-1 Euler angles and gimbal lock singularity detection
"""

import logging
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
        """Verify that every generated DCM satisfies SO(3) orthogonality and proper rotation axioms.

        Testing:
            SO(3) Lie group definition:
            1. Orthogonality: R @ R.T == I and R.T @ R == I
            2. Orientation preservation (no reflection): det(R) == +1.0

        Expected Result:
            - R @ R.T matches identity matrix np.eye(3) within atol=1e-12.
            - det(R) == 1.0 within atol=1e-12.
        """
        dcm = rotations.euler_axis_angle2direction_cosine_matrix(axis, angle)

        identity = np.eye(3)
        assert np.allclose(dcm @ dcm.T, identity, atol=1e-12)
        assert np.allclose(dcm.T @ dcm, identity, atol=1e-12)
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
        """Verify that 3D rotations preserve Euclidean vector norms, inner products, and cross products.

        Testing:
            Rigid body isometry properties in Euclidean 3-space:
            1. ||R * v|| == ||v||
            2. (R * u) . (R * v) == u . v
            3. R * (u x v) == (R * u) x (R * v)

        Expected Result:
            - v1 = [3.0, -4.0, 12.0]: ||v1|| = sqrt(9 + 16 + 144) = sqrt(169) = 13.0
              ||R * v1|| == 13.0 within 1e-12.
            - v2 = [1.0, 2.0, 2.0]: ||v2|| = sqrt(1 + 4 + 4) = sqrt(9) = 3.0
              ||R * v2|| == 3.0 within 1e-12.
            - Inner product: v1 . v2 = 3(1) - 4(2) + 12(2) = 3 - 8 + 24 = 19.0
              (R * v1) . (R * v2) == 19.0 within 1e-12.
            - Cross product equivariance matches within 1e-12.
        """
        rot = rotations.Rotation(axis=axis, angle=angle)

        v1 = np.array([3.0, -4.0, 12.0])
        v2 = np.array([1.0, 2.0, 2.0])

        v1_rot = rot.rotate(v1)
        v2_rot = rot.rotate(v2)

        assert np.isclose(np.linalg.norm(v1_rot), 13.0, atol=1e-12)
        assert np.isclose(np.linalg.norm(v2_rot), 3.0, atol=1e-12)
        assert np.isclose(np.dot(v1_rot, v2_rot), 19.0, atol=1e-12)

        cross_then_rot = rot.rotate(np.cross(v1, v2))
        rot_then_cross = np.cross(v1_rot, v2_rot)
        assert np.allclose(cross_then_rot, rot_then_cross, atol=1e-12)

    @pytest.mark.parametrize("std_axis", [1, 2, 3])
    @pytest.mark.parametrize("angle", [0.1, np.pi / 4, np.pi / 2, np.pi - 0.01])
    def test_inverse_rotation(self, std_axis, angle):
        """Verify that rotating by angle theta followed by -theta restores the original vector.

        Testing:
            Inverse rotation axiom: R(-theta) @ R(theta) == I.

        Expected Result:
            For test vector [123.456, -789.012, 345.678], rot_inv.rotate(rot_fwd.rotate(v))
            restores the exact coordinates within atol=1e-12.
        """
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
        """Verify round-trip mapping: axis-angle -> DCM -> axis-angle.

        Testing:
            euler_axis_angle2direction_cosine_matrix followed by
            direction_cosine_matrix2euler_axis_angle.

        Expected Result:
            The recovered (axis, angle) matches the input (axis, angle) or antipodal (-axis, -angle)
            within atol=1e-7.
        """
        dcm = rotations.euler_axis_angle2direction_cosine_matrix(axis, angle)
        rec_axis, rec_angle = rotations.direction_cosine_matrix2euler_axis_angle(dcm)

        case1 = np.allclose(rec_axis, axis, atol=1e-7) and np.isclose(
            rec_angle, angle, atol=1e-7
        )
        case2 = np.allclose(rec_axis, -axis, atol=1e-7) and np.isclose(
            rec_angle, -angle, atol=1e-7
        )
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
        """Verify round-trip mapping: quaternion -> DCM -> quaternion.

        Testing:
            quaternion2direction_cosine_matrix followed by direction_cosine_matrix2quaternion.
            Because quaternions q and -q represent the identical SO(3) rotation (double cover),
            either rec_quat == q or rec_quat == -q is valid.

        Expected Result:
            rec_quat matches normalized quat or -quat within atol=1e-7.
        """
        quat_normalized = quat / np.linalg.norm(quat)
        dcm = rotations.quaternion2direction_cosine_matrix(quat_normalized)
        rec_quat = rotations.direction_cosine_matrix2quaternion(dcm)

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
        """Verify that direct vector rotation standard_rotation matches matrix multiplication.

        Testing:
            rotations.standard_rotation(axis, angle, vec) == rotations.standard_rotation_matrix(axis, angle) @ vec.

        Expected Result:
            Both routines produce identical rotated vectors within machine precision (atol=1e-14).
        """
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
        """Verify analytical rotation rate matrix against central finite differences.

        Testing:
            Analytical d/dt R(theta(t)) from standard_rotation_matrix_rates vs numerical derivative:
                dR/dt ~= (R(theta + rate*dt) - R(theta - rate*dt)) / (2*dt)
            with dt = 1e-6 s.

        Expected Result:
            Analytical derivative matches finite difference within atol=1e-7.
        """
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
        """Verify that zero roll, pitch, and yaw yields the 3x3 identity matrix.

        Testing:
            roll_pitch_yaw_matrix(0, 0, 0) == np.eye(3).

        Expected Result:
            Returns identity matrix np.eye(3).
        """
        dcm = rotations.roll_pitch_yaw_matrix(0.0, 0.0, 0.0)
        assert np.allclose(dcm, np.eye(3), atol=1e-15)

    @pytest.mark.parametrize("roll", [-np.pi / 3, 0.0, np.pi / 4])
    @pytest.mark.parametrize("pitch", [-np.pi / 6, 0.0, np.pi / 6])
    @pytest.mark.parametrize("yaw", [-np.pi / 2, 0.0, 3 * np.pi / 4])
    def test_roll_pitch_yaw_properties(self, roll, pitch, yaw):
        """Verify orthogonality and determinant of the composite roll-pitch-yaw matrix.

        Testing:
            Composite matrix R = R_x(roll)^T @ R_y(pitch)^T @ R_z(yaw)^T in SO(3).

        Expected Result:
            R @ R.T == eye(3) and det(R) == 1.0 within atol=1e-12.
        """
        dcm = rotations.roll_pitch_yaw_matrix(roll, pitch, yaw)
        assert np.allclose(dcm @ dcm.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(dcm), 1.0, atol=1e-12)

    def test_gimbal_lock_warning_negative_pi_over_2(self, caplog):
        """Verify that pitch = -pi/2 (-90 deg) triggers a gimbal lock warning.

        Testing:
            Singularity detection at pitch = -pi/2 where roll and yaw become collinear.

        Expected Result:
            Logger captures warning containing 'Singular rotation (gimbal lock) detected'.
        """
        with caplog.at_level(logging.WARNING):
            rotations.roll_pitch_yaw_matrix(0.0, -np.pi / 2, 0.0)
        assert "Singular rotation (gimbal lock) detected" in caplog.text
