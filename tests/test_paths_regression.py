# Copyright (c) 2022 The Aerospace Corporation
"""Regression tests for signal path analysis in gps_frames.paths.

Verifies distance chaining, equidistant path interpolation, spherical altitude
intersection geometry, and point of closest approach calculation.
"""

import numpy as np
import pytest

from gps_frames.paths import (
    get_distance_between_points,
    get_points_along_path,
    get_altitude_intersection_point,
    get_point_closest_approach,
)
from gps_frames.position import Position, distance
from gps_frames.parameters import EarthParam
from gps_time import GPSTime


class TestPathDistanceAndInterpolation:
    """Verifies pairwise path distances and equidistant discretization."""

    def test_pairwise_distances(self):
        t = GPSTime(2100, 0.0)
        # Form a 3-4-5 right triangle:
        # P1=(0,0,0), P2=(3,0,0), P3=(3,4,0), P4=(0,0,0)
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
        t = GPSTime(0, 0)
        assert get_distance_between_points([]) == []
        p1 = Position(np.array([1, 2, 3]), t, "ECEF")
        assert get_distance_between_points([p1]) == []

    @pytest.mark.parametrize("num_points", [2, 3, 5, 11, 50])
    def test_points_along_path_spacing_and_collinearity(self, num_points):
        t = GPSTime(2100, 0.0)
        start = Position(np.array([1000.0, -2000.0, 3000.0]), t, "ECEF")
        end = Position(np.array([5000.0, 6000.0, -1000.0]), t, "ECEF")

        total_dist = distance(start, end)
        points = get_points_along_path(start, end, num_points)

        assert len(points) == num_points
        # First point is start, last point is end
        assert np.allclose(points[0].coordinates, start.coordinates, atol=1e-12)
        assert np.allclose(points[-1].coordinates, end.coordinates, atol=1e-12)

        # Equal spacing
        expected_step = total_dist / (num_points - 1)
        sub_dists = get_distance_between_points(points)
        for d in sub_dists:
            assert np.isclose(d, expected_step, atol=1e-9)

        # Collinearity: (p_i - start) must be parallel to (end - start)
        full_dir = (end.to_vector() - start.to_vector()).coordinates
        full_dir_unit = full_dir / np.linalg.norm(full_dir)

        for pt in points[1:]:
            step_dir = (pt.to_vector() - start.to_vector()).coordinates
            step_unit = step_dir / np.linalg.norm(step_dir)
            assert np.allclose(step_unit, full_dir_unit, atol=1e-12)

    def test_invalid_num_points_raises(self):
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
        t = GPSTime(2100, 0.0)
        # Vertical ray from ground station to satellite
        origin = Position(np.array([EarthParam.r_e, 0.0, 0.0]), t, "ECEF")
        sat = Position(np.array([EarthParam.r_e + 20200e3, 0.0, 0.0]), t, "ECEF")

        intersection = get_altitude_intersection_point(target_alt, origin, sat)
        expected_radius = EarthParam.r_e + target_alt

        assert np.isclose(intersection.get_radius(), expected_radius, atol=1e-6)
        assert np.isclose(intersection.get_altitude_spherical(), target_alt, atol=1e-6)

    def test_closest_approach_positive_elevation(self):
        """When elevation >= 0, the path moves away from Earth, so start point is closest approach."""
        t = GPSTime(2100, 0.0)
        start = Position(np.array([EarthParam.r_e, 0.0, 0.0]), t, "ECEF")
        end = Position(np.array([EarthParam.r_e + 10000.0, 5000.0, 0.0]), t, "ECEF")
        path_vec = end.to_vector() - start.to_vector()

        res = get_point_closest_approach(start, path_vec, elevation=0.1)
        assert res == start
        res_zero = get_point_closest_approach(start, path_vec, elevation=0.0)
        assert res_zero == start

    def test_closest_approach_negative_elevation_clamping(self):
        """When elevation < 0, distance dc = r * sin(-elevation). max_length clamps dc."""
        t = GPSTime(2100, 0.0)
        start = Position(np.array([0.0, 0.0, EarthParam.r_e + 20000e3]), t, "ECEF")
        end = Position(np.array([0.0, 0.0, EarthParam.r_e]), t, "ECEF")
        path_vec = end.to_vector() - start.to_vector()

        elevation = -np.deg2rad(30.0)
        r_start = start.get_radius()
        theoretical_dc = r_start * np.sin(-elevation)

        # Unconstrained
        res_unconstrained = get_point_closest_approach(start, path_vec, elevation, max_length=np.inf)
        moved_dist = distance(start, res_unconstrained)
        assert np.isclose(moved_dist, theoretical_dc, atol=1e-6)

        # Clamped with max_length < theoretical_dc
        clamped_length = 5000.0
        res_clamped = get_point_closest_approach(start, path_vec, elevation, max_length=clamped_length)
        clamped_dist = distance(start, res_clamped)
        assert np.isclose(clamped_dist, clamped_length, atol=1e-6)
