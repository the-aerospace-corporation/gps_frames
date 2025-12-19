
import pytest
import numpy as np
import ruamel.yaml
from gps_frames import vectors
from gps_frames.vectors import SerializeableVector, Vector, UnitVector
from gps_time import GPSTime

def test_serializable_vector_init():
    """Test SerializeableVector initialization."""
    # Test valid init
    vec = SerializeableVector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    assert vec.frame == "ECI"
    
    # Test list input
    vec = SerializeableVector([1, 2, 3], GPSTime(0, 0), "ECI")
    assert isinstance(vec.coordinates, np.ndarray)
    
    # Test reshape (2D to 1D)
    vec = SerializeableVector(np.array([[1], [2], [3]]), GPSTime(0, 0), "ECI")
    assert vec.coordinates.shape == (3,)
    
    # Test invalid shape
    with pytest.raises(ValueError):
        SerializeableVector(np.array([[[1]]]), GPSTime(0, 0), "ECI")

def test_serializable_vector_equality():
    """Test equality operator."""
    v1 = SerializeableVector([1, 2, 3], GPSTime(0, 0), "ECI")
    v2 = SerializeableVector([1, 2, 3], GPSTime(0, 0), "ECI")
    v3 = SerializeableVector([1, 2, 4], GPSTime(0, 0), "ECI")
    v4 = SerializeableVector([1, 2, 3], GPSTime(1, 0), "ECI")
    v5 = SerializeableVector([1, 2, 3], GPSTime(0, 0), "ECEF")
    
    assert v1 == v2
    assert v1 != v3
    assert v1 != v4
    assert v1 != v5

def test_serializable_vector_hash():
    """Test hash implementation."""
    v1 = SerializeableVector([1, 2, 3], GPSTime(0, 0), "ECI")
    v2 = SerializeableVector([1, 2, 3], GPSTime(0, 0), "ECI")
    
    assert hash(v1) == hash(v2)
    # Just verify it calculates a hash without error
    _ = hash(v1)

def test_unit_vector_hash():
    """Test UnitVector hash."""
    v1 = UnitVector([1, 0, 0], GPSTime(0, 0), "ECI")
    _ = hash(v1)

def test_vector_mul_int():
    """Test Vector multiplication by int."""
    v = Vector([1, 2, 3], GPSTime(0, 0), "ECI")
    res = v * 2
    assert np.allclose(res.coordinates, [2, 4, 6])
    assert isinstance(res, Vector)

def test_unit_vector_mul_int():
    """Test UnitVector multiplication by int."""
    v = UnitVector([1, 0, 0], GPSTime(0, 0), "ECI")
    res = v * 2
    assert np.allclose(res.coordinates, [2, 0, 0])
    # Should degrade to Vector
    assert isinstance(res, Vector)
    assert not isinstance(res, UnitVector)

def test_yaml_serialization():
    """Test to_yaml and from_yaml methods."""
    # Setup YAML
    yaml = ruamel.yaml.YAML()
    yaml.register_class(SerializeableVector)
    
    vec = SerializeableVector([1.0, 2.0, 3.0], GPSTime(1234, 567890.0), "ECI")
    
    # We can mock the representer/node interaction or just use the yaml library to dump/load
    # Using real dump/load is better integration test
    
    from io import StringIO
    stream = StringIO()
    yaml.dump(vec, stream)
    output = stream.getvalue()
    
    # Check if tag is present
    assert "!SerializeableVector" in output
    assert "frame: ECI" in output
    
    # Load back
    loaded_vec = yaml.load(output)
    
    assert loaded_vec == vec
    assert np.allclose(loaded_vec.coordinates, vec.coordinates)
    assert loaded_vec.frame_time == vec.frame_time
    assert loaded_vec.frame == vec.frame

