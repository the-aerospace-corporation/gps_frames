# Copyright (c) 2022 The Aerospace Corporation
"""Extended unit tests for the Velocity class in gps_frames.velocity.

Verifies:
- Direct instantiation with matching ECEF position and velocity
- Automatic coordinate frame alignment during Velocity initialization
- Physical kinematic transformation from rotating ECEF to inertial ECI frame
- Proper exception raising for unsupported time updates
"""

import numpy as np
import pytest

from gps_frames.position import Position
from gps_frames.vectors import Vector
from gps_frames.velocity import Velocity
from gps_time import GPSTime


def test_velocity_init():
    """Verify standard initialization of a Velocity object with matching frames.

    Testing:
        Velocity(pos, vel) correctly stores position and velocity components.

    Expected Result:
        v_obj.position.frame == "ECEF" and v_obj.velocity.frame == "ECEF".
    """
    t = GPSTime(0, 0)
    pos = Position([0, 0, 6400e3], t, "ECEF")
    vel = Vector([0, 0, 0], t, "ECEF")

    v_obj = Velocity(pos, vel)
    assert v_obj.position.frame == "ECEF"
    assert v_obj.velocity.frame == "ECEF"


def test_velocity_init_frame_sync():
    """Verify that a position in ECI is automatically converted to match an ECEF velocity vector.

    Testing:
        Automatic coordinate frame transformation in Velocity.__post_init__().

    Expected Result:
        Input position [1, 0, 0] in ECI at t=0 is converted to ECEF frame (identity at t=0),
        yielding v_obj.position.frame == "ECEF" and coordinates [1.0, 0.0, 0.0].
    """
    t = GPSTime(0, 0)
    pos = Position([1, 0, 0], t, "ECI")
    vel = Vector([0, 0, 0], t, "ECEF")

    v_obj = Velocity(pos, vel)
    assert v_obj.position.frame == "ECEF"
    assert np.allclose(v_obj.position.coordinates, [1, 0, 0])


def test_get_velocity_transform():
    """Verify inertial velocity calculation for a surface-fixed point on the rotating Earth.

    Testing:
        Kinematic transformation from rotating ECEF to inertial ECI:
            v_ECI = R_z(-theta) * v_ECEF + omega_e x r_ECI
        For a point on the equator at radius r = 6400 km:
            r = [6400000, 0, 0] m, v_ECEF = [0, 0, 0] m/s (stationary on surface)
            omega_e = [0, 0, 7.292115e-5] rad/s
            omega_e x r = [0, omega_e * r, 0]

    Expected Result:
        Inertial velocity along Y-axis is:
            v_y = 7.292115e-5 rad/s * 6400000 m ~= 466.695 m/s
        v_eci.velocity.coordinates[1] matches ~466 m/s within 10% relative tolerance.
    """
    t = GPSTime(0, 0)
    r = 6400e3
    pos = Position([r, 0, 0], t, "ECEF")
    vel = Vector([0, 0, 0], t, "ECEF")

    v_ecef = Velocity(pos, vel)
    v_eci = v_ecef.get_velocity("ECI")

    assert np.allclose(v_eci.position.coordinates, [r, 0, 0])
    assert np.isclose(v_eci.velocity.coordinates[1], 466.0, rtol=0.1)


def test_update_frame_time_error():
    """Verify that calling update_frame_time on a Velocity instance raises NotImplementedError.

    Testing:
        Velocity.update_frame_time() raises NotImplementedError because propagating velocity
        requires full orbital or kinematic acceleration integration.

    Expected Result:
        NotImplementedError is raised when attempting to update frame time.
    """
    t = GPSTime(0, 0)
    pos = Position([0, 0, 0], t, "ECEF")
    vel = Vector([0, 0, 0], t, "ECEF")
    v = Velocity(pos, vel)

    with pytest.raises(NotImplementedError):
        v.update_frame_time(GPSTime(1, 0))
