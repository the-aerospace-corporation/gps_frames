
import pytest
import numpy as np
from gps_frames.velocity import Velocity
from gps_frames.position import Position
from gps_frames.vectors import Vector
from gps_time import GPSTime

def test_velocity_init():
    t = GPSTime(0, 0)
    pos = Position([0,0,6400e3], t, "ECEF")
    vel = Vector([0,0,0], t, "ECEF")
    
    v_obj = Velocity(pos, vel)
    assert v_obj.position.frame == "ECEF"
    assert v_obj.velocity.frame == "ECEF"

def test_velocity_init_frame_sync():
    """Test that position is converted to velocity's frame on init."""
    t = GPSTime(0, 0)
    pos = Position([1,0,0], t, "ECI")
    vel = Vector([0,0,0], t, "ECEF")
    
    # Init should convert pos to ECEF properly
    v_obj = Velocity(pos, vel)
    assert v_obj.position.frame == "ECEF"
    # Verify coordinates transformed (identity at t=0 but good to check frame)
    assert np.allclose(v_obj.position.coordinates, [1,0,0])

def test_get_velocity_transform():
    t = GPSTime(0, 0)
    # Stationary on rotating Earth (ECEF) -> Moving in Inertial (ECI)
    # Earth rotation rate approx 7.29e-5 rad/s
    r = 6400e3
    pos = Position([r, 0, 0], t, "ECEF")
    vel = Vector([0, 0, 0], t, "ECEF") # Stationary on surface
    
    v_ecef = Velocity(pos, vel)
    
    # Transform to ECI
    v_eci = v_ecef.get_velocity("ECI")
    
    # Position in ECI should be same at t=0
    assert np.allclose(v_eci.position.coordinates, [r, 0, 0])
    
    # Velocity in ECI should include rotation term: omega x r
    # omega is [0, 0, w], r is [X, 0, 0]
    # wxr = [0, wX, 0]
    # So Y velocity should be approx 466 m/s
    assert np.isclose(v_eci.velocity.coordinates[1], 466.0, rtol=0.1)
    
def test_update_frame_time_error():
    t = GPSTime(0, 0)
    pos = Position([0,0,0], t, "ECEF")
    vel = Vector([0,0,0], t, "ECEF")
    v = Velocity(pos, vel)
    
    with pytest.raises(NotImplementedError):
        v.update_frame_time(GPSTime(1, 0))
