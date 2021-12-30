import numpy as np
import pytest
import copy
from gps_frames import vectors
from gps_frames import velocity
from gps_frames import position
from gps_frames import transforms as trans
from gps_time import GPSTime
from gps_frames.parameters import GeoidData

############### velocity tests ###############


def test_vel_post():

    # seVec = vectors.SerializeableVector(np.array([1, 1, 1]), GPSTime(0, 0), 'ECI')
    pos = position.Position(np.array([1, 1, 1], dtype=float), GPSTime(0, 0), "ECI")
    vec = vectors.Vector(np.array([1, 2, 3], dtype=float), GPSTime(6, 240), "ECEF")
    vel = velocity.Velocity(pos, vec)

    assert vel.position.frame == "ECEF"
    assert vel.position.frame_time == vec.frame_time


def test_get_vel():

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
