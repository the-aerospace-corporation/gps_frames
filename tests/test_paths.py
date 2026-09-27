# Copyright (c) 2022 The Aerospace Corporation
"""Unit tests for geometric path analysis in gps_frames.paths.

Verifies:
- Pairwise Euclidean distance calculations between consecutive positions along a trajectory
- Ray/line intersection with spherical altitude shells
- Equidistant path discretization and point interpolation
- Point of closest approach (perigee / minimum distance to geocenter) along linear trajectories
"""

import numpy as np
import pytest

from gps_frames.parameters import EarthParam
from gps_frames.paths import (
    get_altitude_intersection_point,
    get_distance_between_points,
    get_point_closest_approach,
    get_points_along_path,
)
from gps_frames.position import Position
from gps_frames.vectors import Vector
from gps_time import GPSTime


def test_get_distance_between_points():
    """Verify pairwise Euclidean distance calculation along an ordered list of Positions.

    Testing:
        paths.get_distance_between_points(positions) calculates consecutive Euclidean segment lengths:
            d_i = ||r_{i+1} - r_i|| for i = 0, ..., N-2

    Expected Result:
        For positions on the X-axis:
            p1 = [1000.0, 0, 0] m
            p2 = [2000.0, 0, 0] m
            p3 = [4000.0, 0, 0] m
        Distances:
            d_0 = ||[2000, 0, 0] - [1000, 0, 0]|| = 1000.0 m
            d_1 = ||[4000, 0, 0] - [2000, 0, 0]|| = 2000.0 m
        Returned list: [1000.0, 2000.0].
    """
    pos1 = Position(np.array([1000, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    pos2 = Position(np.array([2000, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    pos3 = Position(np.array([4000, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")

    positions = [pos1, pos2, pos3]
    distances = get_distance_between_points(positions)

    assert len(distances) == 2
    assert distances[0] == 1000.0
    assert distances[1] == 2000.0


def test_get_altitude_intersection_point():
    """Verify calculating the intersection point of a linear path with a spherical altitude shell.

    Testing:
        paths.get_altitude_intersection_point(target_altitude, origin, target) solves for the point
        along the straight line segment between origin and target having radius:
            R_target = EarthParam.r_e + target_altitude

    Expected Result:
        - Radial upward path from surface to 2*r_e:
            origin = [r_e, 0, 0], target = [2*r_e, 0, 0], target_altitude = 1000.0 m
            Intersection occurs at x = r_e + 1000.0 = 6378137.0 + 1000.0 = 6379137.0 m
            y = 0.0, z = 0.0, and radius = 6379137.0 m.
        - Descending satellite path from GPS MEO altitude (r_e + 20000 km) to surface:
            Intersection at 1000 m altitude occurs at radius = r_e + 1000.0 m.
    """
    origin_radius = EarthParam.r_e
    origin = Position(
        np.array([origin_radius, 0, 0], dtype=float), GPSTime(0, 0), "ECEF"
    )
    target = Position(
        np.array([2 * origin_radius, 0, 0], dtype=float), GPSTime(0, 0), "ECEF"
    )

    target_altitude = 1000.0  # meters
    intersection = get_altitude_intersection_point(target_altitude, origin, target)

    expected_radius = EarthParam.r_e + target_altitude

    assert np.isclose(intersection.get_radius(), expected_radius)
    assert np.isclose(intersection.coordinates[0], expected_radius)
    assert np.isclose(intersection.coordinates[1], 0)
    assert np.isclose(intersection.coordinates[2], 0)

    # Descending path from high altitude to ground
    high_pos = Position(
        np.array([EarthParam.r_e + 20000e3, 0, 0], dtype=float),
        GPSTime(0, 0),
        "ECEF",
    )
    ground_pos = Position(
        np.array([EarthParam.r_e, 0, 0], dtype=float), GPSTime(0, 0), "ECEF"
    )

    intersection_low = get_altitude_intersection_point(1000.0, high_pos, ground_pos)
    assert np.isclose(intersection_low.get_radius(), EarthParam.r_e + 1000.0)


def test_get_points_along_path():
    """Verify linear interpolation of N equidistant points along a line segment.

    Testing:
        paths.get_points_along_path(start, end, num_points) computes:
            p_k = start + (k / (num_points - 1)) * (end - start) for k = 0, ..., num_points - 1

    Expected Result:
        For start = [0, 0, 0], end = [100, 0, 0], num_points = 5:
            Step size = 100 / (5 - 1) = 25.0 m
            p_0 = [0, 0, 0]
            p_1 = [25, 0, 0]
            p_2 = [50, 0, 0] (midpoint)
            p_3 = [75, 0, 0]
            p_4 = [100, 0, 0]
        num_points < 2 raises ValueError.
    """
    start = Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    end = Position(np.array([100, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")

    num_points = 5
    points = get_points_along_path(start, end, num_points)

    assert len(points) == num_points
    assert np.isclose(points[0].coordinates[0], 0.0)
    assert np.isclose(points[-1].coordinates[0], 100.0)
    assert np.isclose(points[2].coordinates[0], 50.0)

    with pytest.raises(ValueError):
        get_points_along_path(start, end, 1)


def test_get_point_closest_approach():
    """Verify determination of the point of closest approach along a directed trajectory.

    Testing:
        paths.get_point_closest_approach(start_point, path_vector, elevation, max_length=None)
        - If elevation >= 0 (path moves away from Earth center), closest approach is start_point.
        - If elevation < 0 (path moves toward Earth center), distance to closest approach along path is:
            d_c = start_radius * sin(-elevation)
            closest_point = start_point + path_unit_vector * min(d_c, max_length)

    Expected Result:
        - Outward path (elevation = pi/2): closest point is start_point.
        - Inward path (elevation = -pi/4 = -45 deg): closest point advances along path vector.
        - Clamping with max_length = 1.0 m: distance moved from start_point is clamped to exactly 1.0 m.
    """
    # Case 1: Outward path (Elevation >= 0)
    start = Position(
        np.array([EarthParam.r_e, 0, 0], dtype=float), GPSTime(0, 0), "ECEF"
    )
    end = Position(
        np.array([EarthParam.r_e + 1000, 0, 0], dtype=float), GPSTime(0, 0), "ECEF"
    )
    path_vec = end.to_vector() - start.to_vector()
    elevation = np.pi / 2

    closest = get_point_closest_approach(start, path_vec, elevation)
    assert closest == start

    # Case 2: Inward trajectory grazing atmosphere
    y_dist = EarthParam.r_e + 100000.0
    start_t = Position(
        np.array([-1000000.0, y_dist, 0], dtype=float), GPSTime(0, 0), "ECEF"
    )
    end_t = Position(
        np.array([1000000.0, y_dist, 0], dtype=float), GPSTime(0, 0), "ECEF"
    )
    path_vec_t = end_t.to_vector() - start_t.to_vector()

    closest_t = get_point_closest_approach(start_t, path_vec_t, -np.pi / 4)
    assert isinstance(closest_t, Position)
    assert closest_t != start_t

    # Case 3: max_length constraint
    closest_constrained = get_point_closest_approach(
        start_t, path_vec_t, -np.pi / 4, max_length=1.0
    )
    dist_moved = (closest_constrained.to_vector() - start_t.to_vector()).magnitude
    assert np.isclose(dist_moved, 1.0)
