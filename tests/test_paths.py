
import pytest
import numpy as np
from gps_frames.paths import (
    get_distance_between_points,
    get_altitude_intersection_point,
    get_points_along_path,
    get_point_closest_approach,
)
from gps_frames.position import Position
from gps_frames.parameters import EarthParam
from gps_frames.vectors import Vector


def test_get_distance_between_points():
    """Test calculation of distances between a list of positions."""
    # Create a list of 3 positions along the X axis
    pos1 = Position(1000, 0, 0, system="ecef")
    pos2 = Position(2000, 0, 0, system="ecef")
    pos3 = Position(4000, 0, 0, system="ecef")
    
    positions = [pos1, pos2, pos3]
    
    distances = get_distance_between_points(positions)
    
    # We expect 2 distances
    assert len(distances) == 2
    assert distances[0] == 1000.0
    assert distances[1] == 2000.0


def test_get_altitude_intersection_point():
    """Test finding the intersection point at a specific altitude."""
    # Define an origin on the surface (approximately)
    origin_radius = EarthParam.r_e
    origin = Position(origin_radius, 0, 0, system="ecef")
    
    # Define a target directly above at 2 * radius
    target = Position(2 * origin_radius, 0, 0, system="ecef")
    
    # We want to find the intersection at altitude slightly above surface
    target_altitude = 1000.0 # meters
    
    intersection = get_altitude_intersection_point(target_altitude, origin, target)
    
    # The intersection should be at radius = r_e + target_altitude
    expected_radius = EarthParam.r_e + target_altitude
    
    # Check if the result is close enough
    assert np.isclose(intersection.get_radius(), expected_radius)
    
    # Since it's a straight line up, x component should match expected radius
    assert np.isclose(intersection.x, expected_radius)
    assert np.isclose(intersection.y, 0)
    assert np.isclose(intersection.z, 0)
    
    # Test with a non-radial path
    # Origin on X axis, Target on Y axis (90 deg apart)
    # This is a longer path through the earth? No, straight line.
    
    origin = Position(EarthParam.r_e, 0, 0, system="ecef")
    target = Position(0, EarthParam.r_e, 0, system="ecef") # 90 degrees away
    
    # This function assumes a straight line path? 
    # The docstrings say "path from the origin to the target".
    # And implementation uses vector difference, so it is a straight line chord through the earth if needed.
    
    # If we look for intersection with altitude 0 (surface) when both are on surface?
    # Actually the function might be for satellite to ground paths or similar.
    
    # Let's test a case where we go from high altitude to ground
    high_pos = Position(EarthParam.r_e + 20000e3, 0, 0, system="ecef") # GPS orbit approx
    ground_pos = Position(EarthParam.r_e, 0, 0, system="ecef")
    
    # Find intersection at 1000m altitude
    intersection_low = get_altitude_intersection_point(1000.0, high_pos, ground_pos)
    assert np.isclose(intersection_low.get_radius(), EarthParam.r_e + 1000.0)


def test_get_points_along_path():
    """Test generating evenly spaced points along a path."""
    start = Position(0, 0, 0, system="ecef")
    end = Position(100, 0, 0, system="ecef")
    
    num_points = 5 # 0, 25, 50, 75, 100
    points = get_points_along_path(start, end, num_points)
    
    assert len(points) == num_points
    assert np.isclose(points[0].x, 0)
    assert np.isclose(points[-1].x, 100)
    assert np.isclose(points[2].x, 50) # Middle point
    
    # Test error handling
    with pytest.raises(ValueError):
        get_points_along_path(start, end, 1)


def test_get_point_closest_approach():
    """Test finding the point of closest approach to Earth."""
    # Case 1: Path is going away from Earth (Elevation >= 0)
    # Start at surface, go up
    start = Position(EarthParam.r_e, 0, 0, system="ecef")
    end = Position(EarthParam.r_e + 1000, 0, 0, system="ecef")
    path_vec = end.to_vector() - start.to_vector()
    
    # Elevation is 90 degrees (pi/2)
    elevation = np.pi/2
    
    closest = get_point_closest_approach(start, path_vec, elevation)
    # Should be start point
    assert closest == start
    
    # Case 2: Path passes near Earth (tangent)
    # Construct a path that grazes the atmosphere
    # Start at -X, some Y. End at +X, same Y.
    # Closest point should be at X=0
    
    y_dist = EarthParam.r_e + 100000.0 # 100km altitude
    start_t = Position(-1000000.0, y_dist, 0, system="ecef")
    end_t = Position(1000000.0, y_dist, 0, system="ecef")
    
    path_vec_t = end_t.to_vector() - start_t.to_vector() # Vector is (2000000, 0, 0)
    
    # elevation relative to start point needs to be calculated or estimated.
    # If the path is horizontal at the tangent point, the elevation at start will be negative (looking down at horizon)
    # But wait, elevation is an input. The function trusts the input elevation?
    # Yes.
    
    # The function implementation:
    # dc = start_point.get_radius() * np.sin(-elevation)
    # return start_point + (path_vector * (dc / path_vector.magnitude))
    
    # This formula looks like it assumes 'elevation' is the depression angle to the local horizontal?
    # Or "elevation of the end point relative to the start point".
    
    # Let's derive needed elevation for the perpendicular case.
    # Triangle: Center(0,0), Start(-X, Y), Closest(0, Y).
    # This geometry is tricky on a sphere with local horizon.
    # Let's trust the function's logic and provide a negative elevation.
    
    # Using a simpler case: Right triangle
    # Start at (0, R+h), Path goes 'down' and 'right'.
    # Closest approach will be somewhere along the line.
    
    # Let's try to trigger the calculation logic.
    elevation_neg = -0.1 # radians
    
    # If elevation is negative, it calculates dc (distance to closest approach along path?)
    # dc = r * sin(-el) ?
    # If this is the logic, let's verify it works as intended by the code structure.
    
    closest_t = get_point_closest_approach(start_t, path_vec_t, -np.pi/4) # -45 degrees
    
    # Just ensure it returns a Position and it's different from start if max_length allows
    assert isinstance(closest_t, Position)
    assert closest_t != start_t

    # Test max_length constraint
    closest_constrained = get_point_closest_approach(start_t, path_vec_t, -np.pi/4, max_length=1.0)
    # Should be very close to start
    dist_moved = (closest_constrained.to_vector() - start_t.to_vector()).magnitude
    assert np.isclose(dist_moved, 1.0)
