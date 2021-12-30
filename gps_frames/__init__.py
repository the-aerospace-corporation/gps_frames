# Copyright (c) 2022 The Aerospace Corporation
"""
.. include::../README.md

gps_frames
----------
Define the reference frames for use throughout the model.

The purpose of this module is to provide tools for translating between the
different reference frames used for GPS. At first, only the WGS84 (LLA) frame,
the ECEF frame, and the GPS ECI frame (relative to start of week) will be
included.

Notes
-----
Like all internal modules, distances are stored here as meters and angles are
stored as radians.

"""
__version__ = "2.8.0"
__copyright__ = "Copyright (C) 2022 The Aerospace Corporation"
__license__ = "GNU AGPL v3"
__distribution_statement__ = (
    "UNCLASSIFIED // APPROVED FOR RELEASE ON OSS20-0011 // GNU AGPL v3"
)

import numpy as np
from numba import jit

from typing import Tuple
import logging
import functools

from gps_time.logutils import (
    AlignedColorFormatter,
    BasicColorTheme,
    display_distro_statement,
)

from .basis import Basis, coordinates_in_basis
from .vectors import UnitVector, Vector  # noqa: F401
from .position import Position, distance  # noqa: F401
from .velocity import Velocity  # noqa: F401

from .parameters import EarthParam

from gps_time import GPSTime  # noqa: F401
from gps_time.utilities import arange_gpstime  # noqa: F401
from gps_time.datetime import datetime2zcount, arange_datetime  # noqa: F401

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
theme = BasicColorTheme()
formatter = AlignedColorFormatter(theme)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.setLevel(logging.WARNING)

logger.debug("Running " + __name__ + " version " + __version__)
logger.debug(__copyright__)
logger.debug(__license__)

display_distro_statement(__distribution_statement__, logger, level="debug")


def get_east_north_up_basis(position: Position) -> Basis:
    """Get the East-North-Up Basis for the input position.

    Parameters
    ----------
    position : Position
        The position of interest

    Returns
    -------
    Basis
        The basis representing the East-North-Up frame

    """
    lla_pos = position.get_position("LLA")
    lla = lla_pos.coordinates
    upx = np.cos(lla[0]) * np.cos(lla[1])
    upy = np.cos(lla[0]) * np.sin(lla[1])
    upz = np.sin(lla[0])

    updirection = UnitVector(
        np.array([upx, upy, upz], dtype=float), position.frame_time, "ECEF"
    )

    localeast_coords = np.array(
        [-updirection.coordinates[1], updirection.coordinates[0], 0], dtype=float
    )
    localeast = UnitVector(localeast_coords, lla_pos.frame_time, "ECEF")

    # Local North is the cross between the up and east directions, i.e.
    # north = up cross east
    localnorth = UnitVector.from_vector(updirection.cross_product(localeast))

    return Basis(position.get_position("ECEF"), localeast, localnorth, updirection)


def get_relative_angles(
    basis: Basis, target_position: Position, look_axis: int, reference_axis: int
) -> Tuple[float, float]:
    """Get the angles from a basis to an object.

    Parameters
    ----------
    basis : Basis
        The basis to measure angles relative to
    target_position : Position
        The position of the target
    look_axis : int
        The axis that represents the look direction of the basis (i.e. the
        boresight)
    reference_axis : int
        The axis to measure the angle around the bore relative to (right-hand
        positive about the look_axis)

    Returns
    -------
    Tuple[float]
        The angle off of the look_axis and the angle about the look_axis, rad

    Raises
    ------
    ValueError
        If the look_axis and reference_axis have the same value or if they are
        not 1, 2, or 3

    """
    if look_axis == reference_axis:
        raise ValueError("look_axis and reference_axis must be different")

    if look_axis not in [1, 2, 3]:
        raise ValueError("look_axis must be 1, 2, or 3")

    if reference_axis not in [1, 2, 3]:
        raise ValueError("reference_axis must be 1, 2, or 3")

    # Get the position relative to the basis
    rp = coordinates_in_basis(target_position, basis)

    # Assign the components along the basis to the internal variables
    for axis in [1, 2, 3]:
        if axis == look_axis:
            look_component = rp[axis - 1]
        elif axis == reference_axis:
            reference_component = rp[axis - 1]
        else:
            other_componet = rp[axis - 1]

    # Compute the angle off of the boresight (look_axis)
    off_bore_component = np.sqrt(reference_component ** 2 + other_componet ** 2)
    angle_off_bore = np.arctan2(off_bore_component, look_component)

    # Determine if the reference_axis is next in the cyclic order after the
    # look_axis
    _cyclic_order = reference_axis == (look_axis % 3) + 1

    # If the order is cyclic, simply compute the angle directly, If not, flip
    # the value of the other axis and compute the angle about the boresight
    # relative to the reference axis
    if _cyclic_order:
        angle_from_ref = np.arctan2(other_componet, reference_component)
    else:
        angle_from_ref = np.arctan2(-other_componet, reference_component)

    return angle_off_bore, angle_from_ref


def get_range_azimuth_elevation(
    enu_basis: Basis, target_position: Position
) -> Tuple[float, float, float]:
    """Get the azimuth and elevation of a target relative to a basis.

    Parameters
    ----------
    enu_basis : Basis
        The East-North-Up basis
    target_position : Position
        The position of the target

    Returns
    Tuple[float, float, float]
        The (1) range, (2) azimuth, and (3) elevation of the target relative
        to ENU basis provided.

    """
    rp = coordinates_in_basis(target_position, enu_basis)
    range_ = np.linalg.norm(rp)

    elevation = (np.pi / 2.0) - np.arccos(rp[2] / range_)
    azimuth = np.arctan2(rp[0], rp[1])

    return range_, azimuth, elevation


def get_azimuth_elevation(
    enu_basis: Basis, target_position: Position
) -> Tuple[float, float]:
    """Get the azimuth and elevation of a target relative to a basis.

    Parameters
    ----------
    enu_basis : Basis
        The East-North-Up basis
    target_position : Position
        The position of the target

    Returns
    -------
    Tuple[float, float]
        The (1) azimuth and (2) elevation of the target relative to ENU
        basis provided.

    """
    _, azimuth, elevation = get_range_azimuth_elevation(enu_basis, target_position)

    return azimuth, elevation


def check_earth_obscuration(
    position1: Position,
    position2: Position,
    convert_hae_altitude: bool = False,
    earth_adjustment_m: float = 30e3,
    transition_altitude_m: float = -np.infty,
    elevation_mask_angle_rad: float = -0.1,
) -> bool:
    r"""Determine if position2 is visiable from position1.

    .. note:: Assumptions
        This function is meant to be a quick check of Earth obscuration. As
        such, it does not account for things like time delays, signal
        refraction, or a non-spherical Earth. All calculations are done under
        the assumptions of simultanity, geometric line-of-sight, and a
        spherical Earth. In fact, this includes the option to naively use the
        height above ellipsoid as the height above the spherical Earth.

    This function is meant to determine if an object at position 2 has a
    direct line of sight to the object at position 1. There are two ways that
    this function checks to see if an object is visible.

    First, it checks to see if the object at position 2 is above the limb of
    the Earth. This means that the horizon object at position 2 is above the
    horizon viewed from position 1. Let the position 1 be a distance \(r\) from
    the center of the Earth. For the purposes of determining the angle to the
    horizon, let the adjusted radius of the Earth be
    $$
        R_{\oplus}' = R_{\oplus} - \delta R
    $$
    where \(R_{\oplus}\) is the actual radius of the Earth and \(\delta R\)
    is a correction factor for the radius. Thus the angle between the nadir
    vector from position 1 to the limb is
    $$
        \lambda = \sin^{-1} \left(\frac{R_{\oplus}'}{r}\right)
    $$
    Let the angle from the anti-nadir vector of position 1 to position 2 be
    \(\psi\). Position 2 will be above the horizon if
    $$
        \pi - \lambda \geq \psi
    $$

    The first method works well for determining if a satellite in view from a
    terrestrial object, but is not useful for determining if a terrestrial
    object is in view of a satellite. This is because if a terrestrial object
    is between a satellite and the Earth, it will be definitionally below the
    horizon, but obviously in view of the satellite. For this reason a second
    test is implemented. An object at position 1 is defined to be in view of
    an object at position 1 if the distance from position 1 to 2 is less than
    the distance from position 1 to the horizon. This technically allows for a
    significant amount of the interior of the Earth to be considered in view,
    but would in practice only return objects in view because subterranian
    objects are not feasible. Let \(d\) be the distance from position 1 to
    position 2. This second condition is
    $$
        d < r \cos\lambda
    $$

    This function also includes a sort of safety factor; it allows for the
    radius of the Earth used for determining the horizon to be reduced. This
    factor can be used to account for the non-sphereical Earth and timing
    issues.

    Parameters
    ----------
    position1 : Position
        The position of object 1
    position2 : Position
        The position of object 2
    convert_hae_altitude : bool, optional
        If the position is in the LLA frame, then its altitude is in height
        above ellispoid (HAE). If this is true, actual spherical altitude will
        be calculated. However, it it is False, the HAE altitude will naively
        be used as a spherical altitude, which is less accurate but will
        provide a faster estimate and can be accounted for using the
        earth_adjustment_m parameter, by default False
    earth_adjustment_m : float, optional
        Effectively reduces the radius of the Earth for the purposes of the
        computing the obscuration. This distance (in meters) is subtracted
        from the radius of the Earth when calculating idealized mask angles,
        by default 100e3

    Returns
    -------
    bool
        True if the position2 is visible from position1

    """
    if earth_adjustment_m < 0:
        logger.debug(
            "Earth Obscuration: Earth Radius Adjustment is negative, which "
            "makes Earth larger."
        )

    elevation_mask_angle_rad = 5 * np.pi / 180

    position1_radius = position1.get_radius()
    position2_radius = position2.get_radius()

    if position1_radius < position2_radius:
        lower_position = position1
        lower_radius = position1_radius
        lower_altitude = position1.get_altitude_hae()

        upper_position = position2
        # upper_radius = position2_radius
    else:
        upper_position = position1
        # upper_radius = position1_radius

        lower_position = position2
        lower_radius = position2_radius
        lower_altitude = position2.get_altitude_hae()

    lower_enu_basis = get_east_north_up_basis(lower_position)
    _distance, _azimuth, _elevation = get_range_azimuth_elevation(
        lower_enu_basis, upper_position
    )

    if transition_altitude_m > lower_altitude:
        min_elevation = elevation_mask_angle_rad
    else:
        adjusted_earth_radius = EarthParam.r_e - earth_adjustment_m
        sin_limb_angle = adjusted_earth_radius / lower_radius
        limb_angle = np.arcsin(np.min([1.0, sin_limb_angle]))
        min_elevation = limb_angle - (np.pi / 2) + elevation_mask_angle_rad

    in_view = _elevation >= min_elevation

    return in_view


def _get_spherical_radius(position: Position, correct_lla: bool) -> float:
    """Get the spherical radius.

    .. deprectated:: Functionality moved to Position object
        This method will be deprecated in favor of the
        gps_frames.position.Position.get_altitude_spherical() and
        gps_frames.position.Position.get_radius() methods. This function is
        just an alias for the Position.get_radius() method.

    Get the spherical radius of the object. If the position is expressed
    in LLA, then the radius will either be the HAE plus the radius of the
    earth (if correct_lla is True) or the position will be converted to
    the ECEF frame. In the ECEF and ECI frame, the spherical radius is the
    norm of the position coordinates.

    Parameters
    ----------
    position : Position
        The position
    correct_lla : bool
        If True, accounts for HAE vs spherical Earth

    Returns
    -------
    float
        The spherical radius of the position

    .. todo:: Move into position object

    """
    logger.error("DEPRECATION: This function is superseded by methods in Position")
    if not correct_lla:
        logger.critical(
            "DEPRECATION: correct_lla argument is not mirrored in new method"
        )

    return position.get_radius()
