
import pytest
import numpy as np
import ruamel.yaml
from gps_frames import basis
from gps_frames.basis import Basis
from gps_frames.position import Position
from gps_frames.vectors import UnitVector
from gps_frames.rotations import Rotation
from gps_time import GPSTime

def test_get_eci_basis():
    t = GPSTime(0, 0)
    b = basis.get_eci_basis(t)
    assert b.origin.frame == "ECI"
    assert b.axes[0].frame == "ECI"
    assert np.allclose(b.axes[0].coordinates, [1, 0, 0])

def test_get_ecef_basis():
    t = GPSTime(0, 0)
    b = basis.get_ecef_basis(t)
    assert b.origin.frame == "ECEF"
    assert b.axes[1].frame == "ECEF"
    assert np.allclose(b.axes[1].coordinates, [0, 1, 0])

def test_basis_init_checks():
    t = GPSTime(0, 0)
    p = Position([0,0,0], t, "ECI")
    x = UnitVector([1,0,0], t, "ECI")
    y = UnitVector([0,1,0], t, "ECI")
    z = UnitVector([0,0,1], t, "ECI")
    
    # Valid
    Basis(p, x, y, z)
    
    # Frame mismatch
    p_ecef = Position([0,0,0], t, "ECEF")
    with pytest.raises(ValueError, match="Not all axes and origin in the same frame"):
        Basis(p_ecef, x, y, z)
        
    # Time mismatch
    t2 = GPSTime(1, 0)
    x_t2 = UnitVector([1,0,0], t2, "ECI")
    with pytest.raises(ValueError, match="Not all axes and origin have same frame time"):
        Basis(p, x_t2, y, z)
        
    # Non-orthogonal
    x_bad = UnitVector([1, 0.1, 0], t, "ECI") # Not normalized if just array, but UnitVector normalizes it
    # We need two vectors that are not orthogonal.
    # UnitVector normalizes [1, 1, 0] -> [0.707, 0.707, 0]
    # UnitVector [1, 0, 0]
    # Dot product is 0.707 != 0
    u1 = UnitVector([1,1,0], t, "ECI")
    u2 = UnitVector([1,0,0], t, "ECI")
    u3 = UnitVector([0,0,1], t, "ECI")
    with pytest.raises(ValueError, match="Axes are not orthogonal"):
        Basis(p, u1, u2, u3)
        
    # Left-handed
    # x, z, y is left handed
    with pytest.raises(ValueError, match="Basis is not right-handed"):
        Basis(p, x, z, y)

    # Origin LLA warning/conversion
    # This requires logging capture or just verifying it runs and converts
    # Position currently doesn't implement switch_frame fully properly for mocked objects or minimal dependencies?
    # Actually Position.switch_frame calls trans.position_transform.
    # We need a valid LLA position for this test to pass without Numba error if enable JIT?
    # But JIT is disabled.
    # LLA: [0,0,0]
    p_lla = Position([0,0,0], t, "LLA") 
    # This should trigger conversion to ECEF inside Basis init
    # But x,y,z are ECI. So it will fail "Not all axes and origin in the same frame" if it converts to ECEF and axes are ECI.
    # Let's make axes ECEF
    b_lla = Basis(p_lla, 
                  UnitVector([1,0,0], t, "ECEF"),
                  UnitVector([0,1,0], t, "ECEF"),
                  UnitVector([0,0,1], t, "ECEF"))
    assert b_lla.origin.frame == "ECEF"

def test_coordinates_in_basis():
    t = GPSTime(0, 0)
    origin = Position([10, 0, 0], t, "ECI")
    # Basis aligned with ECI but shifted
    b = Basis(origin, 
              UnitVector([1,0,0], t, "ECI"),
              UnitVector([0,1,0], t, "ECI"),
              UnitVector([0,0,1], t, "ECI"))
              
    target = Position([15, 5, 2], t, "ECI")
    coords = basis.coordinates_in_basis(target, b)
    # Relative is [5, 5, 2]
    # Projected onto X(1,0,0), Y(0,1,0), Z(0,0,1)
    assert np.allclose(coords, [5, 5, 2])

def test_rotate_basis():
    t = GPSTime(0, 0)
    b = basis.get_eci_basis(t)
    # Rotate 90 deg around Z
    rot = Rotation(standard_axis=3, angle=np.pi/2)
    
    b_rot = basis.rotate_basis(rot, b)
    
    # X axis becomes Y axis
    # Y axis becomes -X axis
    # Z axis stays Z axis
    assert np.allclose(b_rot.axes[0].coordinates, [0, -1, 0], atol=1e-15)
    assert np.allclose(b_rot.axes[1].coordinates, [1, 0, 0], atol=1e-15)
    assert np.allclose(b_rot.axes[2].coordinates, [0, 0, 1], atol=1e-15)

def test_basis_yaml():
    yaml = ruamel.yaml.YAML()
    yaml.register_class(Basis)
    yaml.register_class(Position)
    yaml.register_class(UnitVector)
    yaml.register_class(GPSTime)
    
    t = GPSTime(0, 0)
    b = basis.get_eci_basis(t)
    
    from io import StringIO
    stream = StringIO()
    yaml.dump(b, stream)
    output = stream.getvalue()
    
    assert "!Basis" in output
    
    loaded_b = yaml.load(output)
    # Basis doesn't have __eq__, so check components or hash
    assert loaded_b.origin == b.origin
    assert np.allclose(loaded_b.axes[0].coordinates, b.axes[0].coordinates)

def test_basis_hash():
    t = GPSTime(0, 0)
    b = basis.get_eci_basis(t)
    _ = hash(b)

