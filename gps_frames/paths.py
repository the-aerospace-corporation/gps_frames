"""Compute information about signal paths.

This submodule contains tools that can be used to determine information about a signal
path.

"""

import numpy as np

from typing import List
from logging import getLogger

from .parameters import EarthParam
from .position import Position, distance
from .vectors import Vector

logger = getLogger(__name__)


def get_distance_between_points(positions: List[Position]) -> List[float]:
    """Compute the distance between Positions in a list.

    .. note:: Length of Return Object
        Because it requires two points to compute a distance, the length of
        the returned list is one element shorter than the input list.

    Parameters
    ----------
    positions : List[Position]
        A list of positions

    Returns
    -------
    List[float]
        A list of distances between the positions. The first returned element
        is the distance between first and second positions, the second
        returned element is the distance between the second and third, and
        so on.

    """
    logger.debug(f"Calculating Distance between {len(positions)} points")
    return [
        distance(_pos1, _pos2) for _pos1, _pos2 in zip(positions[:-1], positions[1:])
    ]


def get_altitude_intersection_point(
    altitude: float, origin_position: Position, target_position: Position
) -> Position:
    """Get the point where a unit vector crosses an altitude

    The purpose of this function is to calculate the point at which a vector,
    starting at the origin, will cross a specific altitude.

    .. note:: Spherical Earth
        The function assumes a spherical Earth, not WGS84 ellipsoid

    Parameters
    ----------
    altitude : float
        The altitude in meters
    origin_position : Position
        The position of the origin
    target_position : Position
        The position of the target

    Returns
    -------
    Position
        The position where the path from the origin to the target passes
        through the specified altitude.

    """

    path_vector = target_position.to_vector() - origin_position.to_vector()
    path_length = path_vector.magnitude

    origin_radius = origin_position.get_radius()
    target_radius = target_position.get_radius()

    cos_e_p2 = (origin_radius ** 2 + path_length ** 2 - target_radius ** 2) / (
        2 * origin_radius * path_length
    )

    k = origin_radius * cos_e_p2 + np.sqrt(
        (EarthParam.r_e + altitude) ** 2 - origin_radius ** 2 * (1 - cos_e_p2 ** 2)
    )

    intersection_point = origin_position + (path_vector * (k / path_length))

    return intersection_point


def get_points_along_path(
    start_point: Position, end_point: Position, num_points: int
) -> List[Position]:
    """Get evenly space points along the path.

    Get a list of evenly spaced points along a path. This list includes the
    start and end points, so the number of points along the path must be at
    least 2 (the start and end points).

    Parameters
    ----------
    start_point : Position
        The starting position of the path
    end_point : Position
        The ending position of the path
    num_points : int
        The number of points to return along the path.

    Returns
    -------
    List[Position]
        A list of points along the path, including the start and end points.
        The length of the returned list is num_points.

    Raises
    ------
    ValueError
        If num_points < 2
    """
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    start_vector = start_point.to_vector()
    end_vector = end_point.to_vector()

    path_vector = end_vector - start_vector
    path_length = path_vector.magnitude

    return [
        start_point + (path_vector * (_d / path_length))
        for _d in np.linspace(0, path_length, num_points)
    ]


def get_point_closest_approach(
    start_point: Position,
    path_vector: Vector,
    elevation: float,
    max_length: float = np.infty,
) -> Position:
    """Get the point along the path that is is closest to Earth.

    Compute the point along the path that it is closest to the spherical Earth

    Parameters
    ----------
    start_point : Position
        The start position of the path
    path_vector : Vector
        The vector (not unit vector) describing the path from the start point
        to the end point, i.e. `end_point = start_point + path_vector` or
        `path_vector = end_point - start_point`.
    elevation : float
        The elevation of the end point of the path relative to the start point
    max_length : float, optional
        The maximum distance from the start point to the point of closest
        approach, by default infinity (np.infty). It is generally useful to
        specify that the length of the path is the maximum distance, otherwise
        this algorithm will find a point of closest approach beyond the end point.

    Returns
    -------
    Position
        The point of closest approach along the path.

    """
    if elevation >= 0:
        logger.warning(
            "Attempted to find closest approach when the end point was above "
            "the start point's horizon. Because the path starts at the start "
            "point and the path moves away from the Earth, the start point "
            "is by definition the point of closest approach."
        )
        return start_point
    else:
        dc = start_point.get_radius() * np.sin(-elevation)

        if dc > max_length:
            dc = max_length

        return start_point + (path_vector * (dc / path_vector.magnitude))
