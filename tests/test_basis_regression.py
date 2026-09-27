# Copyright (c) 2022 The Aerospace Corporation
"""Regression tests for Basis class and related operations in gps_frames.basis.

Verifies ENU basis orthonormality and orientation across global latitudes,
basis coordinate projection and exact reconstruction, rotate_basis invariants,
error checking, and YAML serialization.
"""

import numpy as np
import pytest
from io import StringIO
import ruamel.yaml

from gps_frames import basis, get_east_north_up_basis
from gps_frames.basis import Basis, coordinates_in_basis, rotate_basis
from gps_frames.position import Position
from gps_frames.vectors import UnitVector
from gps_frames.rotations import Rotation
from gps_time import GPSTime


class TestEnuBasisGlobalProperties:
    """Verifies that East-North-Up basis constructed at any global location is strictly orthonormal and right-handed."""

    @pytest.mark.parametrize("lat_deg", [-80.0, -45.0, 0.0, 34.0, 45.0, 80.0])
    @pytest.mark.parametrize("lon_deg", [-120.0, -45.0, 0.0, 60.0, 135.0])
    def test_enu_orthonormality_and_handedness(self, lat_deg, lon_deg):
        t = GPSTime(2100, 0.0)
        pos = Position(np.array([np.deg2rad(lat_deg), np.deg2rad(lon_deg), 100.0]), t, "LLA")

        enu = get_east_north_up_basis(pos)
        east = enu.axes[0].coordinates
        north = enu.axes[1].coordinates
        up = enu.axes[2].coordinates

        # 1. Unit lengths
        assert np.isclose(np.linalg.norm(east), 1.0, atol=1e-12)
        assert np.isclose(np.linalg.norm(north), 1.0, atol=1e-12)
        assert np.isclose(np.linalg.norm(up), 1.0, atol=1e-12)

        # 2. Mutual orthogonality
        assert np.isclose(np.dot(east, north), 0.0, atol=1e-12)
        assert np.isclose(np.dot(north, up), 0.0, atol=1e-12)
        assert np.isclose(np.dot(up, east), 0.0, atol=1e-12)

        # 3. Right-handedness: East x North == Up
        assert np.allclose(np.cross(east, north), up, atol=1e-12)
        assert np.allclose(np.cross(north, up), east, atol=1e-12)
        assert np.allclose(np.cross(up, east), north, atol=1e-12)


class TestCoordinatesInBasisReconstruction:
    """Verifies that coordinates in basis can accurately reconstruct the original position vector."""

    @pytest.mark.parametrize("basis_frame", ["ECEF", "ECI"])
    @pytest.mark.parametrize("target_frame", ["ECEF", "ECI", "LLA"])
    def test_projection_and_reconstruction(self, basis_frame, target_frame):
        t = GPSTime(2150, 1000.0)

        # Construct an arbitrary orthonormal right-handed basis
        origin = Position(np.array([1000.0, -2000.0, 3000.0]), t, basis_frame)
        # 30 deg rotation around Z
        rot = Rotation(standard_axis=3, angle=np.pi / 6)
        u1 = UnitVector(rot.rotate([1.0, 0.0, 0.0]), t, basis_frame)
        u2 = UnitVector(rot.rotate([0.0, 1.0, 0.0]), t, basis_frame)
        u3 = UnitVector(rot.rotate([0.0, 0.0, 1.0]), t, basis_frame)
        b = Basis(origin, u1, u2, u3)

        # Target position in potentially different frame
        if target_frame == "LLA":
            target = Position(np.array([0.5, -1.0, 2000.0]), t, "LLA")
        else:
            target = Position(np.array([5000.0, 8000.0, -1000.0]), t, target_frame)

        coords = coordinates_in_basis(target, b)
        assert len(coords) == 3

        # Reconstruct position in basis_frame
        reconstructed_vec = (
            origin.coordinates
            + coords[0] * u1.coordinates
            + coords[1] * u2.coordinates
            + coords[2] * u3.coordinates
        )
        reconstructed_pos = Position(reconstructed_vec, t, basis_frame)

        # Target and reconstructed position must be equal (distance < 1e-6)
        assert np.allclose(
            reconstructed_pos.get_position(basis_frame).coordinates,
            target.get_position(basis_frame).coordinates,
            atol=1e-6,
        )

    def test_target_at_origin(self):
        t = GPSTime(0, 0)
        b = basis.get_ecef_basis(t)
        target = Position(np.array([0.0, 0.0, 0.0]), t, "ECEF")
        coords = coordinates_in_basis(target, b)
        assert np.allclose(coords, [0.0, 0.0, 0.0], atol=1e-12)


class TestRotateBasisInvariants:
    """Verifies that rotating a basis preserves its geometric integrity."""

    @pytest.mark.parametrize("axis", [1, 2, 3])
    @pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2, np.pi])
    def test_rotate_basis_preserves_basis_properties(self, axis, angle):
        t = GPSTime(2000, 0.0)
        orig_basis = basis.get_eci_basis(t)
        rot = Rotation(standard_axis=axis, angle=angle)

        rotated_b = rotate_basis(rot, orig_basis)

        # Origin should be unchanged
        assert rotated_b.origin == orig_basis.origin

        # Axes should remain unit vectors, mutually orthogonal, right-handed
        a1 = rotated_b.axes[0].coordinates
        a2 = rotated_b.axes[1].coordinates
        a3 = rotated_b.axes[2].coordinates

        assert np.isclose(np.linalg.norm(a1), 1.0, atol=1e-12)
        assert np.isclose(np.linalg.norm(a2), 1.0, atol=1e-12)
        assert np.isclose(np.linalg.norm(a3), 1.0, atol=1e-12)

        assert np.isclose(np.dot(a1, a2), 0.0, atol=1e-12)
        assert np.isclose(np.dot(a2, a3), 0.0, atol=1e-12)
        assert np.isclose(np.dot(a3, a1), 0.0, atol=1e-12)

        assert np.allclose(np.cross(a1, a2), a3, atol=1e-12)

    def test_rotate_full_turn(self):
        t = GPSTime(0, 0)
        orig_basis = basis.get_ecef_basis(t)
        rot_360 = Rotation(standard_axis=3, angle=2 * np.pi)
        rotated_b = rotate_basis(rot_360, orig_basis)

        assert np.allclose(rotated_b.axes[0].coordinates, orig_basis.axes[0].coordinates, atol=1e-12)
        assert np.allclose(rotated_b.axes[1].coordinates, orig_basis.axes[1].coordinates, atol=1e-12)
        assert np.allclose(rotated_b.axes[2].coordinates, orig_basis.axes[2].coordinates, atol=1e-12)


class TestBasisSerialization:
    """Verifies YAML serialization and deserialization of Basis."""

    def test_yaml_round_trip(self):
        yaml = ruamel.yaml.YAML()
        yaml.register_class(Basis)
        yaml.register_class(Position)
        yaml.register_class(UnitVector)

        t = GPSTime(2100, 3600.0)
        orig = basis.get_ecef_basis(t)

        stream = StringIO()
        yaml.dump(orig, stream)
        yaml_str = stream.getvalue()

        assert "!Basis" in yaml_str
        loaded = yaml.load(yaml_str)

        assert isinstance(loaded, Basis)
        assert loaded.origin == orig.origin
        for i in range(3):
            assert np.allclose(loaded.axes[i].coordinates, orig.axes[i].coordinates, atol=1e-12)
            assert loaded.axes[i].frame == orig.axes[i].frame
