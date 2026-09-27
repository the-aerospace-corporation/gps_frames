# Copyright (c) 2022 The Aerospace Corporation
"""Unit tests for the Velocity class in gps_frames.velocity.

Verifies:
- Frame and epoch synchronization between position and velocity vector on initialization
- Coordinate frame transformations for Velocity instances (e.g. ECI to ECEF)
"""

import numpy as np
import pytest

from gps_frames import position, transforms as trans, vectors, velocity
from gps_time import GPSTime

############### velocity tests ###############


def test_vel_post():
    """Verify synchronization of position frame and epoch to velocity vector in Velocity.__post_init__().

    Testing:
        When initializing Velocity(pos, vec) where pos and vec have different frames or epochs,
        the position is automatically converted to match the velocity vector's frame and frame_time.

    Expected Result:
        Given pos in ECI at week 0, sec 0 and vec in ECEF at week 6, sec 240:
        vel.position.frame becomes "ECEF" and vel.position.frame_time becomes GPSTime(6, 240).
    """
    pos = position.Position(np.array([1, 1, 1], dtype=float), GPSTime(0, 0), "ECI")
    vec = vectors.Vector(np.array([1, 2, 3], dtype=float), GPSTime(6, 240), "ECEF")
    vel = velocity.Velocity(pos, vec)

    assert vel.position.frame == "ECEF"
    assert vel.position.frame_time == vec.frame_time


def test_get_vel():
    """Verify coordinate frame transformation of a Velocity object via get_velocity().

    Testing:
        vel.get_velocity("ECEF") transforms both the position and velocity components to ECEF,
        accounting for the rotational velocity term omega_e x r via trans.velocity_transform().

    Expected Result:
        For pos = [1.0, 1.0, 1.0] and vel = [1.0, 2.0, 3.0] in ECI at week 2109, sec 259200:
        velTwo.velocity.coordinates matches trans.velocity_transform("ECI", "ECEF", pos.coords, vel.coords, t).
    """
    pos = position.Position(np.array([1.0, 1.0, 1.0]), GPSTime(2109, 259200), "ECI")
    vec = vectors.Vector(np.array([1.0, 2.0, 3.0]), GPSTime(2109, 259200), "ECI")
    vel = velocity.Velocity(pos, vec)

    velTwo = vel.get_velocity("ECEF")

    assert (
        velTwo.velocity.coordinates
        == trans.velocity_transform(
            vel.position.frame,
            "ECEF",
            vel.position.coordinates,
            vel.velocity.coordinates,
            vel.position.frame_time,
        )
    ).all()
