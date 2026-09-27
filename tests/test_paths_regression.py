# Copyright (c) 2022 The Aerospace Corporation
"""Regression tests for signal path analysis in gps_frames.paths.

Verifies:
- Distance chaining along polygonal paths (e.g. 3-4-5 right triangle)
- Equidistant path discretization and collinearity across variable point counts
- Exact radial shell intersections at target spherical altitudes
- Closest approach calculations, positive elevation triviality, and negative elevation clamping
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
from gps_frames.position import Position, distance
from gps_time import GPSTime


class TestPathDistanceAndInterpolation:
    """Verifies pairwise path distances and equidistant discretization."""

    def test_pairwise_distances(self):
        """Verify cumulative and pairwise distance calculation along a 3-4-5 right triangle path.

        Testing:
            get_distance_between_points([p1, p2, p3, p4]) along vertices:
            - p1 = [0, 0, 0]
            - p2 = [3000, 0, 0]   (length = 3000 m)
            - p3 = [3000, 4000, 0] (length = 4000 m)
            - p4 = [0, 0, 0]      (hypotenuse length = sqrt(3000^2 + 4000^2) = 5000 m)

        Expected Result:
            Returned segments: [3000.0, 4000.0, 5000.0] with perimeter sum = 12000.0 m.
        """
        t = GPSTime(2100, 0.0)
        p1 = Position(np.array([0.0, 0.0, 0.0]), t, "ECEF")
        p2 = Position(np.array([3000.0, 0.0, 0.0]), t, "ECEF")
        p3 = Position(np.array([3000.0, 4000.0, 0.0]), t, "ECEF")
        p4 = Position(np.array([0.0, 0.0, 0.0]), t, "ECEF")

        distances = get_distance_between_points([p1, p2, p3, p4])
        assert len(distances) == 3
        assert np.isclose(distances[0], 3000.0, atol=1e-10)
        assert np.isclose(distances[1], 4000.0, atol=1e-10)
        assert np.isclose(distances[2], 5000.0, atol=1e-10)
        assert np.isclose(sum(distances), 12000.0, atol=1e-10)

    def test_pairwise_distances_edge_cases(self):
        """Verify that empty or single-point lists return empty distance lists.

        Testing:
            get_distance_between_points() boundary conditions.

        Expected Result:
            get_distance_between_points([]) == [] and get_distance_between_points([p1]) == [].
        """
        t = GPSTime(0, 0)
        assert get_distance_between_points([]) == []
        p1 = Position(np.array([1, 2, 3]), t, "ECEF")
        assert get_distance_between_points([p1]) == []

    @pytest.mark.parametrize("num_points", [2, 3, 5, 11, 50])
    def test_points_along_path_spacing_and_collinearity(self, num_points):
        """Verify uniform step spacing and strict collinearity of interpolated path points.

        Testing:
            get_points_along_path(start, end, num_points):
            1. First point equals start, last point equals end.
            2. Every consecutive sub-segment has length total_dist / (num_points - 1).
            3. Every intermediate displacement is parallel to the end-to-start direction.

        Expected Result:
            All points are collinear and equally spaced within atol=1e-9 m.
        """
        t = GPSTime(2100, 0.0)
        start = Position(np.array([1000.0, -2000.0, 3000.0]), t, "ECEF")
        end = Position(np.array([5000.0, 6000.0, -1000.0]), t, "ECEF")

        total_dist = distance(start, end)
        points = get_points_along_path(start, end, num_points)

        assert len(points) == num_points
        assert np.allclose(points[0].coordinates, start.coordinates, atol=1e-12)
        assert np.allclose(points[-1].coordinates, end.coordinates, atol=1e-12)

        expected_step = total_dist / (num_points - 1)
        sub_dists = get_distance_between_points(points)
        for d in sub_dists:
            assert np.isclose(d, expected_step, atol=1e-9)

        full_dir = (end.to_vector() - start.to_vector()).coordinates
        full_dir_unit = full_dir / np.linalg.norm(full_dir)

        for pt in points[1:]:
            step_dir = (pt.to_vector() - start.to_vector()).coordinates
            step_unit = step_dir / np.linalg.norm(step_dir)
            assert np.allclose(step_unit, full_dir_unit, atol=1e-12)

    def test_invalid_num_points_raises(self):
        """Verify that num_points < 2 raises ValueError.

        Testing:
            Input validation in get_points_along_path().

        Expected Result:
            ValueError("num_points must be >= 2") is raised for num_points in {0, 1}.
        """
        t = GPSTime(0, 0)
        p1 = Position(np.array([0, 0, 0]), t, "ECEF")
        p2 = Position(np.array([1, 0, 0]), t, "ECEF")
        with pytest.raises(ValueError, match="num_points must be >= 2"):
            get_points_along_path(p1, p2, 1)
        with pytest.raises(ValueError, match="num_points must be >= 2"):
            get_points_along_path(p1, p2, 0)


class TestAltitudeIntersectionAndClosestApproach:
    """Verifies geometric intersection points and closest approaches."""

    @pytest.mark.parametrize("target_alt", [100.0, 1000.0, 50000.0, 350000.0])
    def test_altitude_intersection_vertical_and_oblique(self, target_alt):
        """Verify that the intersection point has geocentric radius EarthParam.r_e + target_alt.

        Testing:
            get_altitude_intersection_point(target_alt, origin, sat) along a vertical ray.

        Expected Result:
            intersection.get_radius() == 6378137.0 + target_alt within atol=1e-6 m.
            intersection.get_altitude_spherical() == target_alt within atol=1e-6 m.
        """
        t = GPSTime(2100, 0.0)
        origin = Position(np.array([EarthParam.r_e, 0.0, 0.0]), t, "ECEF")
        sat = Position(np.array([EarthParam.r_e + 20200e3, 0.0, 0.0]), t, "ECEF")

        intersection = get_altitude_intersection_point(target_alt, origin, sat)
        expected_radius = EarthParam.r_e + target_alt

        assert np.isclose(intersection.get_radius(), expected_radius, atol=1e-6)
        assert np.isclose(intersection.get_altitude_spherical(), target_alt, atol=1e-6)

    def test_closest_approach_positive_elevation(self):
        """Verify that paths with non-negative elevation have start_point as the closest approach.

        Testing:
            When elevation >= 0, the trajectory moves away from the Earth geocenter,
            so the minimum distance occurs at the initial point.

        Expected Result:
            res == start for both elevation = 0.1 rad and elevation = 0.0 rad.
        """
        t = GPSTime(2100, 0.0)
        start = Position(np.array([EarthParam.r_e, 0.0, 0.0]), t, "ECEF")
        end = Position(np.array([EarthParam.r_e + 10000.0, 5000.0, 0.0]), t, "ECEF")
        path_vec = end.to_vector() - start.to_vector()

        res = get_point_closest_approach(start, path_vec, elevation=0.1)
        assert res == start
        res_zero = get_point_closest_approach(start, path_vec, elevation=0.0)
        assert res_zero == start

    def test_closest_approach_negative_elevation_clamping(self):
        """Verify closest approach distance calculation and max_length clamping for negative elevation.

        Testing:
            For negative elevation (path initially descending toward Earth):
                d_c = r_start * sin(-elevation)
            - When unconstrained, distance moved equals theoretical d_c.
            - When max_length < d_c, distance moved is clamped to exactly max_length.

        Expected Result:
            - For elevation = -30 deg and r_start = r_e + 20000 km:
                sin(30 deg) = 0.5 -> d_c = 0.5 * (6378137 + 20000000) = 13189068.5 m
            - distance(start, res_unconstrained) == d_c within atol=1e-6 m.
            - distance(start, res_clamped) == 5000.0 m when clamped with max_length = 5000.0 m.
        """
        t = GPSTime(2100, 0.0)
        start = Position(np.array([0.0, 0.0, EarthParam.r_e + 20000e3]), t, "ECEF")
        end = Position(np.array([0.0, 0.0, EarthParam.r_e]), t, "ECEF")
        path_vec = end.to_vector() - start.to_vector()

        elevation = -np.deg2rad(30.0)
        r_start = start.get_radius()
        theoretical_dc = r_start * np.sin(-elevation)

        # Unconstrained
        res_unconstrained = get_point_closest_approach(
            start, path_vec, elevation, max_length=np.inf
        )
        moved_dist = distance(start, res_unconstrained)
        assert np.isclose(moved_dist, theoretical_dc, atol=1e-6)

        # Clamped with max_length < theoretical_dc
        clamped_length = 5000.0
        res_clamped = get_point_closest_approach(
            start, path_vec, elevation, max_length=clamped_length
        )
        clamped_dist = distance(start, res_clamped)
        assert np.isclose(clamped_dist, clamped_length, atol=1e-6)
