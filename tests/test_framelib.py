# Copyright (c) 2022 The Aerospace Corporation
"""Unit tests for top-level frame utilities in gps_frames.__init__.

Verifies:
- East-North-Up (ENU) topocentric local horizon basis generation
- Relative look angles (off-boresight angle and clock/reference angle)
- Radar/optical tracking coordinates: Range, Azimuth, and Elevation
- Geometric Earth obscuration checks along line-of-sight paths
"""

import numpy as np
import pytest

from gps_frames import (
    _get_spherical_radius,
    check_earth_obscuration,
    get_azimuth_elevation,
    get_east_north_up_basis,
    get_range_azimuth_elevation,
    get_relative_angles,
)
from gps_frames.basis import Basis, UnitVector
from gps_frames.parameters import EarthParam
from gps_frames.position import Position
from gps_time import GPSTime


def test_get_east_north_up_basis():
    """Verify East-North-Up (ENU) local tangent basis generation at equatorial prime meridian.

    Testing:
        get_east_north_up_basis(pos) constructs a right-handed orthonormal triad [East, North, Up]
        at the specified surface location:
            Up = normal to WGS84 ellipsoid (along +X at lat=0, lon=0)
            East = along parallel of latitude toward +lon (along +Y at lon=0)
            North = along meridian toward +lat (along +Z at lat=0)

    Expected Result:
        At LLA = [0.0 rad, 0.0 rad, 0.0 m] (intersection of Equator and Prime Meridian):
            - enu.axes[2] (Up)    == [1.0, 0.0, 0.0]
            - enu.axes[0] (East)  == [0.0, 1.0, 0.0]
            - enu.axes[1] (North) == [0.0, 0.0, 1.0]
    """
    t = GPSTime(0, 0)
    pos = Position([0, 0, 0], t, "LLA")

    enu = get_east_north_up_basis(pos)

    assert np.allclose(enu.axes[2].coordinates, [1, 0, 0])  # Up
    assert np.allclose(enu.axes[0].coordinates, [0, 1, 0])  # East
    assert np.allclose(enu.axes[1].coordinates, [0, 0, 1])  # North


def test_get_relative_angles():
    """Verify relative angle decomposition into off-boresight and clock angles.

    Testing:
        get_relative_angles(basis, target, look_axis, ref_axis) computes:
        1. Off-boresight angle theta: angle between the look axis and the line-of-sight vector:
            theta = arccos((r_los . u_look) / ||r_los||)
        2. Angle from reference phi: azimuthal angle of the projected LOS in the transverse plane,
           measured from ref_axis.

    Expected Result:
        - Origin at [0, 0, 0], Target at [1.0, 1.0, 0.0]
        - Look axis = 1 (X), Ref axis = 2 (Y)
        - LOS vector = [1.0, 1.0, 0.0], ||LOS|| = sqrt(2)
        - Off-boresight angle = arccos(1 / sqrt(2)) = pi/4 rad (45.0 deg)
        - Projection onto YZ plane is [1.0, 0.0], perfectly aligned with Ref axis (Y) -> phi = 0.0 rad.
        - Error handling: identical look and ref axis raises ValueError; out-of-range axis (4) raises ValueError.
        - Non-cyclic ordering: look = 1, ref = 3 with target at [1, 0, 1] yields off-bore = pi/4, angle = 0.
    """
    t = GPSTime(0, 0)
    origin = Position([0, 0, 0], t, "ECEF")
    u1 = UnitVector([1, 0, 0], t, "ECEF")
    u2 = UnitVector([0, 1, 0], t, "ECEF")
    u3 = UnitVector([0, 0, 1], t, "ECEF")
    b = Basis(origin, u1, u2, u3)

    target = Position([1, 1, 0], t, "ECEF")

    off_bore, from_ref = get_relative_angles(b, target, 1, 2)
    assert np.isclose(off_bore, np.pi / 4)
    assert np.isclose(from_ref, 0.0)

    # Input validation
    with pytest.raises(ValueError):
        get_relative_angles(b, target, 1, 1)  # Look and ref cannot be identical
    with pytest.raises(ValueError):
        get_relative_angles(b, target, 4, 1)  # Invalid axis index 4
    with pytest.raises(ValueError):
        get_relative_angles(b, target, 1, 4)

    # Non-cyclic look/ref pair: look 1 (X), ref 3 (Z)
    target_z = Position([1, 0, 1], t, "ECEF")
    off, ang = get_relative_angles(b, target_z, 1, 3)
    assert np.isclose(off, np.pi / 4)
    assert np.isclose(ang, 0.0)


def test_get_range_azimuth_elevation():
    """Verify spherical coordinate tracking angles (Range, Azimuth, Elevation).

    Testing:
        get_range_azimuth_elevation(basis, target) projects the relative displacement onto an ENU basis:
            Range = ||r_rel|| = sqrt(East^2 + North^2 + Up^2)
            Azimuth = arctan2(East, North) (clockwise from North toward East)
            Elevation = arcsin(Up / Range) (angle above local horizontal plane)

    Expected Result:
        For origin at geocenter with basis East=X, North=Y, Up=Z and target at [1.0, 1.0, 1.0]:
            East = 1.0, North = 1.0, Up = 1.0
            Range = sqrt(1^2 + 1^2 + 1^2) = sqrt(3) ~= 1.7320508 m
            Azimuth = arctan2(1.0, 1.0) = pi/4 rad (45.0 deg)
            Elevation = arcsin(1.0 / sqrt(3)) ~= 0.6154797 rad (35.264 deg)
        get_azimuth_elevation wrapper returns identical azimuth and elevation values.
    """
    t = GPSTime(0, 0)
    origin = Position([0, 0, 0], t, "ECEF")
    b = Basis(
        origin,
        UnitVector([1, 0, 0], t, "ECEF"),
        UnitVector([0, 1, 0], t, "ECEF"),
        UnitVector([0, 0, 1], t, "ECEF"),
    )

    target = Position([1, 1, 1], t, "ECEF")
    rng, az, el = get_range_azimuth_elevation(b, target)

    assert np.isclose(rng, np.sqrt(3))
    assert np.isclose(az, np.pi / 4)
    assert np.isclose(el, np.arcsin(1 / np.sqrt(3)))

    az2, el2 = get_azimuth_elevation(b, target)
    assert az2 == az
    assert el2 == el


def test_check_earth_obscuration():
    """Verify geometric line-of-sight visibility and Earth limb occultation.

    Testing:
        check_earth_obscuration(pos1, pos2) tests whether the straight line segment connecting
        two positions intersects the Earth ellipsoid or terrain mask. Returns True if visible (unobscured).

    Expected Result:
        - Case 1 (Radial overhead):
            sat = [2*r_e, 0, 0], gs = [r_e, 0, 0] (sat directly above gs)
            LOS is radial away from center -> unobscured -> returns True.
            Symmetric: check_earth_obscuration(gs, sat) == True.
        - Case 2 (Opposite hemispheres):
            sat = [2*r_e, 0, 0], gs_opp = [-r_e, 0, 0]
            LOS passes straight through Earth's core -> obscured -> returns False.
        - Case 3 (Transition altitude elevation mask):
            Specifying transition_altitude_m=1000 uses elevation mask angle logic.
        - Case 4 (Earth radius adjustment margin):
            Negative earth_adjustment_m=-1000 increases Earth sphere radius by 1000 m.
    """
    t = GPSTime(0, 0)
    sat = Position([2 * EarthParam.r_e, 0, 0], t, "ECEF")
    gs = Position([EarthParam.r_e, 0, 0], t, "ECEF")

    # Case 1: Directly overhead (unobscured)
    assert check_earth_obscuration(sat, gs)
    assert check_earth_obscuration(gs, sat)

    # Case 2: Opposite sides of Earth (obscured)
    gs_opp = Position([-EarthParam.r_e, 0, 0], t, "ECEF")
    assert not check_earth_obscuration(sat, gs_opp)

    # Case 3: Transition altitude configuration
    assert check_earth_obscuration(sat, gs, transition_altitude_m=1000)

    # Case 4: Negative Earth adjustment margin
    check_earth_obscuration(sat, gs, earth_adjustment_m=-1000)


def test_get_spherical_radius_deprecated():
    """Verify backward compatibility of deprecated _get_spherical_radius helper.

    Testing:
        _get_spherical_radius(pos, correct_lla=bool) returns geocentric radius.

    Expected Result:
        Returns radial norm 100.0 m for position [100.0, 0.0, 0.0].
    """
    t = GPSTime(0, 0)
    p = Position([100, 0, 0], t, "ECEF")
    r = _get_spherical_radius(p, correct_lla=False)
    assert r == 100.0
    r = _get_spherical_radius(p, correct_lla=True)
    assert r == 100.0
