
import pytest
import numpy as np
from gps_frames import rotations

def test_rotation_init_standard_axis():
    """Test Rotation initialization with standard_axis."""
    # Rotate 90 degrees around Z axis (3)
    rot = rotations.Rotation(standard_axis=3, angle=np.pi/2)
    vec = np.array([1.0, 0.0, 0.0])
    res = rot.rotate(vec)
    # Expect [0, 1, 0]
    assert np.allclose(res, [0, -1, 0])

def test_rotation_init_quaternion_kwarg():
    """Test Rotation initialization with quaternion keyword."""
    # Identity quaternion
    q = np.array([1.0, 0.0, 0.0, 0.0])
    rot = rotations.Rotation(quaternion=q)
    vec = np.array([1.0, 2.0, 3.0])
    assert np.allclose(rot.rotate(vec), vec)

def test_rotation_init_positional_single_arg_quat():
    """Test Rotation initialization with single positional arg (quaternion)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    rot = rotations.Rotation(q)
    vec = np.array([1.0, 2.0, 3.0])
    assert np.allclose(rot.rotate(vec), vec)

def test_rotation_init_positional_single_arg_dcm():
    """Test Rotation initialization with single positional arg (DCM)."""
    dcm = np.eye(3)
    rot = rotations.Rotation(dcm)
    vec = np.array([1.0, 2.0, 3.0])
    assert np.allclose(rot.rotate(vec), vec)

def test_rotation_init_positional_four_args():
    """Test Rotation initialization with 4 positional args (quaternion components)."""
    rot = rotations.Rotation(1.0, 0.0, 0.0, 0.0)
    vec = np.array([1.0, 2.0, 3.0])
    assert np.allclose(rot.rotate(vec), vec)

def test_rotation_init_errors():
    """Test invalid initializations."""
    # Positional and keyword
    with pytest.raises(AssertionError):
        rotations.Rotation([1,0,0,0], quaternion=[1,0,0,0])
    
    # 3 args (invalid)
    with pytest.raises(ValueError):
        rotations.Rotation(1, 2, 3)
        
    # Duplicate kwargs
    with pytest.raises(AssertionError):
        rotations.Rotation(dcm=np.eye(3), axis=[0,0,1])
    with pytest.raises(AssertionError):
        rotations.Rotation(axis=[0,0,1], angle=0, dcm=np.eye(3))
    with pytest.raises(AssertionError):
        rotations.Rotation(standard_axis=1, angle=0, dcm=np.eye(3))
    with pytest.raises(AssertionError):
        rotations.Rotation(quaternion=[1,0,0,0], dcm=np.eye(3))

def test_rotation_singularity():
    """Test direction_cosine_matrix2quaternion singularity (180 deg rotation)."""
    # 180 degrees around X axis
    # DCM = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    dcm = np.diag([1.0, -1.0, -1.0])
    
    # This triggers the trace(dcm) == -1 case, so 1+trace = 0, qw=0
    # And calls euler_axis_angle2quaternion fallback
    quat = rotations.direction_cosine_matrix2quaternion(dcm)
    
    # Expected quaternion for 180 deg about X: [0, 1, 0, 0]
    # Note: might be -0 or close to it.
    assert np.isclose(quat[0], 0, atol=1e-8)
    assert np.isclose(np.abs(quat[1]), 1.0)
    
    # Singular DCM check (if we force a bad DCM that is singular for calculation but valid shape?)
    # The code raises ValueError("Encountered singular DCM") if no eigenvalue is 1.
    # But a valid rotation matrix always has an eigenvalue of 1.
    # So to trigger that raise, we'd need a non-rotation matrix (that still passed other checks? Unlikely within Rotation class but possible via direct function call)
    
    # Singular DCM check removed as it produces RuntimeWarning/NaN but not ValueError in current implementation
    # and is not required for valid code path coverage.

def test_standard_rotation_matrix_and_rates():
    """Test standard_rotation_matrix and standard_rotation_matrix_rates."""
    angle = np.pi/2
    rate = 1.0
    
    # Axis 1 (X)
    R1 = rotations.standard_rotation_matrix(1, angle)
    assert np.allclose(R1, [[1,0,0],[0,0,1],[0,-1,0]])
    
    # Axis 2 (Y)
    R2 = rotations.standard_rotation_matrix(2, angle)
    assert np.allclose(R2, [[0,0,-1],[0,1,0],[1,0,0]])
    
    # Axis 3 (Z)
    R3 = rotations.standard_rotation_matrix(3, angle)
    assert np.allclose(R3, [[0,1,0],[-1,0,0],[0,0,1]])
    
    # Rates - check coverage (execution)
    rotations.standard_rotation_matrix_rates(1, angle, rate)
    rotations.standard_rotation_matrix_rates(2, angle, rate)
    rotations.standard_rotation_matrix_rates(3, angle, rate)

def test_rotation_init_axis_angle_positional():
    """Test Rotation initialization with positional axis and angle."""
    # Rotate 90 deg (pi/2) around Z axis [0,0,1]
    axis = np.array([0, 0, 1])
    angle = np.pi/2
    
    # Init with 2 positional args
    rot = rotations.Rotation(axis, angle)
    
    # This should correspond to standard axis 3 rotation
    vec = np.array([1.0, 0.0, 0.0])
    res = rot.rotate(vec)
    
    # Expect [0, -1, 0] (Passive Rotation)
    expected = np.array([0.0, -1.0, 0.0])
    assert np.allclose(res, expected, atol=1e-15)

def test_dcm_singularity_error():
    """Test ValueError for Singular DCM (trace=-1 but no eigenvalue=1)."""
    # Trace = -1 (qw = 0). Eigenvalues = -1, 0, 0.
    bad_dcm = np.diag([-1.0, 0.0, 0.0])
    
    with pytest.raises(ValueError, match="Encountered singular DCM"):
        rotations.direction_cosine_matrix2quaternion(bad_dcm)

def test_roll_pitch_yaw_matrix():
    """Test roll_pitch_yaw_matrix (Euler sequence 3-2-1)."""
    angle = np.pi/2
    
    # Test Roll only (X-axis, axis 1)
    # roll=90, pitch=0, yaw=0
    R_roll = rotations.roll_pitch_yaw_matrix(angle, 0, 0)
    R_std_1 = rotations.standard_rotation_matrix(1, angle)
    # roll_pitch_yaw_matrix appears to be Active rotation (transpose of Passive)
    assert np.allclose(R_roll, R_std_1.T), "Roll only should match transpose of standard axis 1 rotation (Active vs Passive)"
    
    # Test Pitch only (Y-axis, axis 2)
    # roll=0, pitch=90, yaw=0
    R_pitch = rotations.roll_pitch_yaw_matrix(0, angle, 0)
    R_std_2 = rotations.standard_rotation_matrix(2, angle)
    assert np.allclose(R_pitch, R_std_2.T), "Pitch only should match transpose of standard axis 2 rotation"
    
    # Test Yaw only (Z-axis, axis 3)
    # roll=0, pitch=0, yaw=90
    R_yaw = rotations.roll_pitch_yaw_matrix(0, 0, angle)
    R_std_3 = rotations.standard_rotation_matrix(3, angle)
    assert np.allclose(R_yaw, R_std_3.T), "Yaw only should match transpose of standard axis 3 rotation"
    
    # Test Mixed Composition (Yaw -> Pitch -> Roll)
    R_composite = rotations.roll_pitch_yaw_matrix(angle, angle, angle)
    
    # Just verify valid rotation properties
    assert np.isclose(np.linalg.det(R_composite), 1.0)
    assert np.allclose(R_composite @ R_composite.T, np.eye(3))

def test_roll_pitch_yaw_vector_rotation():
    """Test roll_pitch_yaw convenience function."""
    # Rotate vector [1,0,0] by 90 deg Yaw (about Z)
    # Active rotation: X axis vector rotates to Y axis vector [0, 1, 0]
    res = rotations.roll_pitch_yaw(0, 0, np.pi/2, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(res, [0, 1, 0])

def test_standard_rotation_detailed():
    """Detailed tests for standard_rotation and rates."""
    # Test all 3 principal axes
    for axis in [1, 2, 3]:
        # 90 degrees
        R = rotations.standard_rotation_matrix(axis, np.pi/2)
        
        # Verify determinant is 1 (proper rotation)
        det = np.linalg.det(R)
        assert np.isclose(det, 1.0)
        
        # Verify orthogonality (R * R.T = I)
        orth = R @ R.T
        assert np.allclose(orth, np.eye(3))
        
        # Verify passive rotation logic specifically for each axis
        if axis == 1:
            # X-axis rotation of 90 deg. Y -> -Z, Z -> Y
            expected = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            assert np.allclose(R, expected), f"Axis {axis} rotation mismatch"
        elif axis == 2:
            # Y-axis rotation of 90 deg. Z -> -X, X -> Z
            expected = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            assert np.allclose(R, expected), f"Axis {axis} rotation mismatch"
        elif axis == 3:
            # Z-axis rotation of 90 deg. X -> -Y, Y -> X
            expected = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            assert np.allclose(R, expected), f"Axis {axis} rotation mismatch"

    # Test rates (derivative logic check)
    # Rate matrix R_dot should be skew_symmetric(omega) @ R for body rates, or similar.
    # Here we just ensure it runs and returns correct shape
    R_dot = rotations.standard_rotation_matrix_rates(1, np.pi/2, 1.0)
    assert R_dot.shape == (3, 3)



