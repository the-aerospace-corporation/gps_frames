# Copyright (c) 2022 The Aerospace Corporation
"""Extended unit tests for Vector, UnitVector, and SerializeableVector in gps_frames.vectors.

Verifies:
- Type casting, coordinate reshaping, and dimension validation for SerializeableVector
- Component equality comparison (coordinates, frame, and epoch)
- Hash generation for vectors and unit vectors
- Integer scalar multiplication and type degradation from UnitVector to Vector
- YAML round-trip serialization and deserialization
"""

from io import StringIO
import numpy as np
import pytest
import ruamel.yaml

from gps_frames.vectors import SerializeableVector, UnitVector, Vector
from gps_time import GPSTime


def test_serializable_vector_init():
    """Verify SerializeableVector initialization, type conversion, and reshaping.

    Testing:
        SerializeableVector accepts NumPy arrays and lists, reshapes (3, 1) column vectors
        to (3,) 1D arrays, and rejects arrays with dimension != 1 or 2.

    Expected Result:
        - List [1, 2, 3] converts to np.ndarray.
        - 2D array [[1], [2], [3]] reshapes to (3,).
        - 3D array [[[1]]] raises ValueError.
    """
    vec = SerializeableVector(np.array([1, 2, 3]), GPSTime(0, 0), "ECI")
    assert vec.frame == "ECI"

    vec = SerializeableVector([1, 2, 3], GPSTime(0, 0), "ECI")
    assert isinstance(vec.coordinates, np.ndarray)

    vec = SerializeableVector(np.array([[1], [2], [3]]), GPSTime(0, 0), "ECI")
    assert vec.coordinates.shape == (3,)

    with pytest.raises(ValueError):
        SerializeableVector(np.array([[[1]]]), GPSTime(0, 0), "ECI")


def test_serializable_vector_equality():
    """Verify SerializeableVector equality operator across coordinates, frame, and time.

    Testing:
        Equality requiring identical coordinate values, matching frame, and matching GPSTime epoch.

    Expected Result:
        v1 == v2 (identical values) is True.
        v1 != v3 (coordinate mismatch: [1, 2, 3] vs [1, 2, 4]) is True.
        v1 != v4 (time mismatch: week 0 vs week 1) is True.
        v1 != v5 (frame mismatch: ECI vs ECEF) is True.
    """
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
    """Verify that SerializeableVector generates stable integer hashes.

    Testing:
        SerializeableVector.__hash__() method based on tuple(coordinates), frame, and frame_time.

    Expected Result:
        hash(v1) == hash(v2) for equal vectors, returning an integer.
    """
    v1 = SerializeableVector([1, 2, 3], GPSTime(0, 0), "ECI")
    v2 = SerializeableVector([1, 2, 3], GPSTime(0, 0), "ECI")

    assert hash(v1) == hash(v2)
    assert isinstance(hash(v1), int)


def test_unit_vector_hash():
    """Verify that UnitVector instances are hashable.

    Testing:
        UnitVector.__hash__() execution.

    Expected Result:
        Returns an integer hash without raising TypeError.
    """
    v1 = UnitVector([1, 0, 0], GPSTime(0, 0), "ECI")
    assert isinstance(hash(v1), int)


def test_vector_mul_int():
    """Verify scalar multiplication of Vector by an integer.

    Testing:
        Vector.__mul__(scalar) scales coordinates:
            v * 2 = [1*2, 2*2, 3*2] = [2, 4, 6]

    Expected Result:
        Coordinates are [2.0, 4.0, 6.0] and returned object is an instance of Vector.
    """
    v = Vector([1, 2, 3], GPSTime(0, 0), "ECI")
    res = v * 2
    assert np.allclose(res.coordinates, [2, 4, 6])
    assert isinstance(res, Vector)


def test_unit_vector_mul_int():
    """Verify that multiplying a UnitVector by an integer scales coordinates and degrades to Vector.

    Testing:
        UnitVector.__mul__(scalar): because the magnitude is no longer 1.0, the type degrades
        from UnitVector to Vector.

    Expected Result:
        u = [1, 0, 0] scaled by 2 yields coordinates [2.0, 0.0, 0.0].
        isinstance(res, Vector) is True and isinstance(res, UnitVector) is False.
    """
    v = UnitVector([1, 0, 0], GPSTime(0, 0), "ECI")
    res = v * 2
    assert np.allclose(res.coordinates, [2, 0, 0])
    assert isinstance(res, Vector)
    assert not isinstance(res, UnitVector)


def test_yaml_serialization():
    """Verify YAML serialization and deserialization for SerializeableVector.

    Testing:
        SerializeableVector to_yaml and from_yaml hooks registered with ruamel.yaml.

    Expected Result:
        Serialized YAML contains '!SerializeableVector' and 'frame: ECI'.
        Loaded object equals original vector with matching coordinates, frame, and frame_time.
    """
    yaml = ruamel.yaml.YAML()
    yaml.register_class(SerializeableVector)

    vec = SerializeableVector([1.0, 2.0, 3.0], GPSTime(1234, 567890.0), "ECI")

    stream = StringIO()
    yaml.dump(vec, stream)
    output = stream.getvalue()

    assert "!SerializeableVector" in output
    assert "frame: ECI" in output

    loaded_vec = yaml.load(output)
    assert loaded_vec == vec
    assert np.allclose(loaded_vec.coordinates, vec.coordinates)
    assert loaded_vec.frame_time == vec.frame_time
    assert loaded_vec.frame == vec.frame
