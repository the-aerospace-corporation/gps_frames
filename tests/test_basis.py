# Copyright (c) 2022 The Aerospace Corporation
"""Unit tests for the Basis class and basis operations in gps_frames.basis.

Verifies:
- Creation of canonical ECI and ECEF coordinate bases
- Basis validation rules: frame consistency, epoch consistency, orthogonality, and right-handed orientation
- Projection and resolution of 3D positions into a local basis
- Rigid body rotation of coordinate bases
- YAML serialization and object hashing
"""

from io import StringIO
import numpy as np
import pytest
import ruamel.yaml

from gps_frames import basis
from gps_frames.basis import Basis
from gps_frames.position import Position
from gps_frames.rotations import Rotation
from gps_frames.vectors import UnitVector
from gps_time import GPSTime


def test_get_eci_basis():
    """Verify construction of the canonical ECI basis.

    Testing:
        basis.get_eci_basis(t) initializes an orthogonal basis at the geocenter (0, 0, 0)
        with unit axes aligned with the standard Cartesian ECI directions:
            e_x = [1, 0, 0], e_y = [0, 1, 0], e_z = [0, 0, 1].

    Expected Result:
        Basis origin has frame 'ECI' and coordinates [0, 0, 0].
        Axis 0 is a UnitVector in 'ECI' with coordinates [1.0, 0.0, 0.0].
    """
    t = GPSTime(0, 0)
    b = basis.get_eci_basis(t)
    assert b.origin.frame == "ECI"
    assert b.axes[0].frame == "ECI"
    assert np.allclose(b.axes[0].coordinates, [1, 0, 0])


def test_get_ecef_basis():
    """Verify construction of the canonical ECEF basis.

    Testing:
        basis.get_ecef_basis(t) initializes an orthogonal basis at the geocenter (0, 0, 0)
        with unit axes aligned with the standard Cartesian ECEF directions.

    Expected Result:
        Basis origin has frame 'ECEF' and coordinates [0, 0, 0].
        Axis 1 is a UnitVector in 'ECEF' with coordinates [0.0, 1.0, 0.0].
    """
    t = GPSTime(0, 0)
    b = basis.get_ecef_basis(t)
    assert b.origin.frame == "ECEF"
    assert b.axes[1].frame == "ECEF"
    assert np.allclose(b.axes[1].coordinates, [0, 1, 0])


def test_basis_init_checks():
    """Verify validation rules enforced during Basis initialization.

    Testing:
        Basis.__init__() requires:
        1. All three axes and the origin must share the exact same frame (e.g. all ECI or all ECEF).
        2. All axes and the origin must have identical frame_time timestamps.
        3. The three axes must be pairwise orthogonal (u_i . u_j == 0 for i != j).
        4. The triad must be right-handed: det([u1, u2, u3]) > 0 and (u1 x u2) . u3 == 1.
        5. If origin is passed in LLA, it is automatically converted to ECEF.

    Expected Result:
        - Mismatched frames (origin ECEF, axes ECI) raises ValueError("Not all axes and origin in the same frame").
        - Mismatched epochs (axis at t2 != t) raises ValueError("Not all axes and origin have same frame time").
        - Non-orthogonal triad (u1 = [1, 1, 0]/sqrt(2), u2 = [1, 0, 0]; dot = 1/sqrt(2) != 0) raises ValueError("Axes are not orthogonal").
        - Left-handed triad (x, z, y where x x z = -y != +y) raises ValueError("Basis is not right-handed").
        - Origin in LLA [0, 0, 0] with ECEF axes converts origin frame to ECEF.
    """
    t = GPSTime(0, 0)
    p = Position([0, 0, 0], t, "ECI")
    x = UnitVector([1, 0, 0], t, "ECI")
    y = UnitVector([0, 1, 0], t, "ECI")
    z = UnitVector([0, 0, 1], t, "ECI")

    # 1. Valid orthogonal right-handed triad
    Basis(p, x, y, z)

    # 2. Frame mismatch error
    p_ecef = Position([0, 0, 0], t, "ECEF")
    with pytest.raises(ValueError, match="Not all axes and origin in the same frame"):
        Basis(p_ecef, x, y, z)

    # 3. Time mismatch error
    t2 = GPSTime(1, 0)
    x_t2 = UnitVector([1, 0, 0], t2, "ECI")
    with pytest.raises(ValueError, match="Not all axes and origin have same frame time"):
        Basis(p, x_t2, y, z)

    # 4. Non-orthogonal triad error
    u1 = UnitVector([1, 1, 0], t, "ECI")
    u2 = UnitVector([1, 0, 0], t, "ECI")
    u3 = UnitVector([0, 0, 1], t, "ECI")
    with pytest.raises(ValueError, match="Axes are not orthogonal"):
        Basis(p, u1, u2, u3)

    # 5. Left-handed triad error (swapping y and z creates left-handed system)
    with pytest.raises(ValueError, match="Basis is not right-handed"):
        Basis(p, x, z, y)

    # 6. Origin LLA conversion to ECEF
    p_lla = Position([0, 0, 0], t, "LLA")
    b_lla = Basis(
        p_lla,
        UnitVector([1, 0, 0], t, "ECEF"),
        UnitVector([0, 1, 0], t, "ECEF"),
        UnitVector([0, 0, 1], t, "ECEF"),
    )
    assert b_lla.origin.frame == "ECEF"


def test_coordinates_in_basis():
    """Verify projection of a 3D position vector into a local Basis coordinate system.

    Testing:
        basis.coordinates_in_basis(target, b) computes coordinates [c1, c2, c3] by projecting
        the displacement vector delta_r = target - origin onto the unit basis axes:
            c_i = delta_r . u_i

    Expected Result:
        - Origin at [10, 0, 0] ECI
        - Target at [15, 5, 2] ECI
        - Displacement delta_r = [15 - 10, 5 - 0, 2 - 0] = [5, 5, 2]
        - With axes [1, 0, 0], [0, 1, 0], [0, 0, 1]:
            c1 = [5, 5, 2] . [1, 0, 0] = 5
            c2 = [5, 5, 2] . [0, 1, 0] = 5
            c3 = [5, 5, 2] . [0, 0, 1] = 2
        Computed coordinates: [5.0, 5.0, 2.0].
    """
    t = GPSTime(0, 0)
    origin = Position([10, 0, 0], t, "ECI")
    b = Basis(
        origin,
        UnitVector([1, 0, 0], t, "ECI"),
        UnitVector([0, 1, 0], t, "ECI"),
        UnitVector([0, 0, 1], t, "ECI"),
    )

    target = Position([15, 5, 2], t, "ECI")
    coords = basis.coordinates_in_basis(target, b)
    assert np.allclose(coords, [5, 5, 2])


def test_rotate_basis():
    """Verify rigid rotation of a coordinate basis by a Rotation object.

    Testing:
        basis.rotate_basis(rot, b) applies a 3D rotation to each basis vector:
            u_i' = rot.rotate(u_i)
        Rotating an ECI basis by +90 deg (pi/2 rad) about standard axis 3 (Z-axis)
        using rotation matrix:
            R_3(pi/2) = [[ 0, 1, 0],
                         [-1, 0, 0],
                         [ 0, 0, 1]]

    Expected Result:
        - Axis 0 (X = [1, 0, 0]) rotates to: R_3 @ [1, 0, 0] = [0, -1, 0]
        - Axis 1 (Y = [0, 1, 0]) rotates to: R_3 @ [0, 1, 0] = [1, 0, 0]
        - Axis 2 (Z = [0, 0, 1]) rotates to: R_3 @ [0, 0, 1] = [0, 0, 1]
    """
    t = GPSTime(0, 0)
    b = basis.get_eci_basis(t)
    rot = Rotation(standard_axis=3, angle=np.pi / 2)

    b_rot = basis.rotate_basis(rot, b)

    assert np.allclose(b_rot.axes[0].coordinates, [0, -1, 0], atol=1e-15)
    assert np.allclose(b_rot.axes[1].coordinates, [1, 0, 0], atol=1e-15)
    assert np.allclose(b_rot.axes[2].coordinates, [0, 0, 1], atol=1e-15)


def test_basis_yaml():
    """Verify YAML serialization and deserialization of Basis objects using ruamel.yaml.

    Testing:
        Custom YAML representers and constructors registered for Basis and its constituent types.

    Expected Result:
        YAML dump contains '!Basis' tag, and yaml.load() reconstructs an identical Basis
        with matching origin and axes.
    """
    yaml = ruamel.yaml.YAML()
    yaml.register_class(Basis)
    yaml.register_class(Position)
    yaml.register_class(UnitVector)
    yaml.register_class(GPSTime)

    t = GPSTime(0, 0)
    b = basis.get_eci_basis(t)

    stream = StringIO()
    yaml.dump(b, stream)
    output = stream.getvalue()

    assert "!Basis" in output

    loaded_b = yaml.load(output)
    assert loaded_b.origin == b.origin
    assert np.allclose(loaded_b.axes[0].coordinates, b.axes[0].coordinates)


def test_basis_hash():
    """Verify that Basis instances are hashable.

    Testing:
        Basis.__hash__() method generating a stable integer hash based on origin and axes.

    Expected Result:
        hash(b) returns an integer without raising TypeError.
    """
    t = GPSTime(0, 0)
    b = basis.get_eci_basis(t)
    assert isinstance(hash(b), int)
