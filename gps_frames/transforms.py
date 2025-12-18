# Copyright (c) 2022 The Aerospace Corporation
"""Frame Transformations.

This submodule contains the mathematics used for translating between different
frames.

Todo
----
- Incorporate non-GPS standard frames, e.g. J2000
- Include velocity transformations into the LLA frame
- Add local coordinate systems, e.g. ENU
- Make GPSTime JIT-able
- Fix ECEF to LLA warning (currently a print for numba compatibility)

"""
import warnings
import numpy as np
import functools
from numba import jit

from logging import getLogger

from gps_time import GPSTime

from .parameters import EarthParam
from .rotations import (
    Rotation,  # noqa: F401
    standard_rotation,
    standard_rotation_matrix,
    standard_rotation_matrix_rates,
)

logger = getLogger(__name__)

VALID_FRAMES = ["ECI", "ECEF", "LLA"]
"""The frames implemented for this toolbox

This list contains the names of the frames available in this module. These
frames are

- ECI: The ECI frame aligned with the ECEF frame at the start of the week.
    When used, the coordinates refer to the X, Y, and Z positions in the
    frame in meters
- ECEF: The ECEF frame. When used, the coordinates refer to the X, Y, and
    Z positions in the frame in meters
- LLA: The WGS84 Latitude, Longitude, and Altitude. When used, the
    coordinates refer to the Latitude, Longitude, and Altitude in radians or
    meters as appropriate

Notes
-----
For consistency, all names must be in all caps.
"""

_wgs84a: float = EarthParam.wgs84a
_wgs84f: float = EarthParam.wgs84f
_wgs84ecc: float = EarthParam.wgs84ecc
_w_e: float = EarthParam.w_e


def velocity_transform(
    from_frame: str,
    to_frame: str,
    position_coordinates: np.ndarray,
    velocity_components: np.ndarray,
    time: GPSTime,
) -> np.ndarray:
    """Transform velocity from one frame to another.

    Parameters
    ----------
    from_frame : str
        The starting frame
    to_frame : str
        The desired frame
    position_coordinates : np.ndarray
        The position coordinates in the starting frame
    velocity_components : np.ndarray
        The velocity components in the starting from
    time : GPSTime
        The time that the transformation is taking place

    Returns
    -------
    np.ndarray
        The velocity components in the destination frame

    Raises
    ------
    ValueError
        Does not work with the 'LLA' frame, so raises an error if the to_frame
        or from_frame are specified as 'LLA'. Will also raise if the to_frame
        or from_frame are not valid frames.
    """
    position_coordinates = np.array(position_coordinates)
    velocity_components = np.array(velocity_components)
    if from_frame == "LLA" or to_frame == "LLA":
        raise ValueError("from_frame and to_frame cannot be LLA")

    if from_frame not in VALID_FRAMES:
        raise ValueError(f"from_frame ({from_frame}) not valid")
    if to_frame not in VALID_FRAMES:
        raise ValueError(f"to_frame ({to_frame}) not valid")

    if from_frame == "ECEF":
        if to_frame == "ECEF":
            angle_of_rotation = 0.0
            rotation_rate = 0.0
        elif to_frame == "ECI":
            angle_of_rotation = -_w_e * time.time_of_week
            rotation_rate = -_w_e
    elif from_frame == "ECI":
        if to_frame == "ECEF":
            angle_of_rotation = _w_e * time.time_of_week
            rotation_rate = _w_e
        elif to_frame == "ECI":
            angle_of_rotation = 0.0
            rotation_rate = 0.0

    rotation = Rotation(dcm=standard_rotation_matrix(3, angle_of_rotation))
    rate_matrix = standard_rotation_matrix_rates(3, angle_of_rotation, rotation_rate)

    return rotation.rotate(velocity_components) + rate_matrix @ position_coordinates


def position_transform(
    from_frame: str, to_frame: str, coordinates: np.array, time: GPSTime
) -> np.array:
    """Convert a position from one frame to another.

    The purpose of this function is to create a general tool to convert a
    position from one frame to another. The current frame is given as the
    `from_frame` and the desired new output frame is given as the `to_frame`.
    The position is defined in the `from_frame` as the `coordinates`. The
    `time` is used to define the time of the frame, as some conversions
    between frames vary with time, e.g. ECEF to ECI.

    Parameters
    ----------
    from_frame : str
        The frame in which the input coordinates are defined
    to_frame : str
        The frame that the coordinates should be provided in
    coordinates : np.ndarray
        The coordinates in the `from_frame`
    time : GPSTime
        The time at which the `from_frame` and `to_frame` are aligned.

    Returns
    -------
    np.ndarray
        The coordinates in the `to_frame`

    Raises
    ------
    NotImplementedError
        If the `from_frame` or `to_frame` are not valid frames

    See Also
    --------
    VALID_FRAMES: A list of the valid frames for the `from_frame` and
        `to_frame` args

    Notes
    -----
    This function is used to transform a position in one frame to another. The
    time is the same for both the `from_frame` and `to_frame`

    """

    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates, dtype=float)
    if from_frame not in VALID_FRAMES:
        raise NotImplementedError(f"(from_frame) Unknown Frame: {from_frame}")
    if to_frame not in VALID_FRAMES:
        raise NotImplementedError(f"(to_frame) Unknown Frame: {to_frame}")

    if from_frame == "LLA":
        if to_frame == "LLA":
            return coordinates
        elif to_frame == "ECEF":
            return lla2ecef(coordinates)
        elif to_frame == "ECI":
            return lla2eci(coordinates, time.time_of_week)
    elif from_frame == "ECEF":
        if to_frame == "LLA":
            return ecef2lla(coordinates)
        elif to_frame == "ECEF":
            return coordinates
        elif to_frame == "ECI":
            return ecef2eci(coordinates, time.time_of_week)
    elif from_frame == "ECI":
        if to_frame == "LLA":
            return eci2lla(coordinates, time.time_of_week)
        elif to_frame == "ECEF":
            return eci2ecef(coordinates, time.time_of_week)
        elif to_frame == "ECI":
            return coordinates

    logger.critical("Transformation Failed. Returning input")
    return coordinates


@jit("float64[:](float64[:])", nopython=True)
def lla2ecef(lla_coordinates: np.array) -> np.array:
    r"""Convert LLA to ECEF position.

    Computes the ECEF position based on the WGS84 ellipsoid latitude,
    longitude, and altitude. The inputs is are numpy arrays. Returns a
    numpy array with columns representing the ECEF positions of the
    elements of the inputs. From [1]_

    Let \(\lambda\), \(\phi\), \(h\), \(a\), and \(e\) be the latitude,
    longitude, altitude, WGS84 semi-major axis, and WGS84 eccentricity,
    repsectively. The effective radius of the Earth is
    $$
        n = \frac{a}{\sqrt{1 - (e\sin\lambda)^{2}}}
    $$
    From this, the ECEF \(x\), \(y\), and \(z\) positions can be computed as
    $$
        \begin{split}
            x = & (n + h)\cos\lambda\cos\phi \\
            y = & (n + h)\cos\lambda\sin\phi \\
            z = & \left(n\left(1 - e^{2}\right) + h\right) \sin\lambda
        \end{split}
    $$

    Parameters
    ----------
    lla_coordinates : np.ndarray
        The WGS 84 latitude, longitude, and altitude (in that order)

    Returns
    -------
    np.ndarray
        The ECEF position of the given WGS84 LLA

    References
    ----------
    .. [1] G. Xu and Y. Xu, "GPS: Theory, Algorithms and Applications" 3rd ed.
        https://doi.org/10.1007/978-3-662-50367-6

    """
    latitude = lla_coordinates[0]
    longitude = lla_coordinates[1]
    altitude = lla_coordinates[2]

    n = _wgs84a / np.sqrt(1 - (_wgs84ecc * np.sin(latitude)) ** 2)

    x = (n + altitude) * np.cos(latitude) * np.cos(longitude)
    y = (n + altitude) * np.cos(latitude) * np.sin(longitude)
    z = (n * (1 - _wgs84ecc ** 2) + altitude) * np.sin(latitude)

    return np.array((x, y, z))


@jit("float64[:](float64[:], float64)", nopython=True)
def ecef2eci(ecef_coordinates: np.ndarray, time_of_week: float) -> np.ndarray:
    r"""Rotate from the ECEF to the ECI frame.

    This function rotates the ECEF coordinates into the ECI frame. Note that
    the ECI and ECEF frames are aligned at the start of the week.

    Let \(\boldsymbol{r}^{\mathcal{F}}\) be the position in the ECEF frame
    and \(t\) be the time of week. The angle of rotation between the ECI and
    ECEF frames is \(\Theta = \omega_{\oplus}t\), with \(\omega_{\oplus}\)
    being the angular velocity of the Earth. Thus, the position in the ECI
    coordinates is
    $$
        \boldsymbol{r}^{\mathcal{N}} =
            \left[\begin{array}{ccc}
                \cos -\Theta & \sin -\Theta & 0 \\
                -\sin -\Theta & \cos -\Theta & 0 \\
                0 & 0 & 1
            \end{array}\right]\boldsymbol{r}^{\mathcal{F}}
        =
            \left[\begin{array}{ccc}
                \cos \Theta & -\sin \Theta & 0 \\
                \sin \Theta & \cos \Theta & 0 \\
                0 & 0 & 1
            \end{array}\right]\boldsymbol{r}^{\mathcal{F}}
    $$

    Parameters
    ----------
    ecef_coordinates : np.ndarray
        The coordinates in the ECEF frame
    time_of_week : float
        The time of week in seconds associated with the ECEF frame

    Returns
    -------
    np.ndarray
        The coordinates in the ECI frame

    See Also
    --------
    standard_rotation_matrix: Used to get the rotation matrix for the
        transformation

    """

    # Angle from the ECI to the ECEF frame
    angle_of_rotation = _w_e * time_of_week
    # rotation = Rotation(axis=np.array((0., 0., 1.)), angle=-angle_of_rotation)

    return standard_rotation(3, -angle_of_rotation, ecef_coordinates)


@jit("float64[:](float64[:], float64)", nopython=True)
def lla2eci(lla_coordinates: np.ndarray, time: float) -> np.ndarray:
    """Tranform LLA coordinates into ECI.

    This function is used to transform an LLA position into ECI coordinates.
    To do this, the coordinates are first converted to ECEF coordinates using
    `lla2ecef()`. Then, these new ECEF coordinates are transformed into ECI
    coordinates using `ecef2eci()`.

    Parameters
    ----------
    lla_coordinates : np.ndarray
        The Latitude, Longitude, Altitude position coordinates
    time : GPSTime
        The time at which the `lla_coordinates` are defined

    Returns
    -------
    np.ndarray
        The position in the ECI frame

    See Also
    --------
    lla2ecef: Convert LLA coordinates to ECEF coordinates
    ecef2eci: Convert ECEF coordinates into ECI coordinates

    """
    return ecef2eci(lla2ecef(lla_coordinates), time)


@jit("float64(float64)", nopython=True)
def _ecef2lla_beta_func(latitude: float) -> float:
    return np.arctan2((1 - _wgs84f) * np.sin(latitude), np.cos(latitude))


@jit("float64(float64, float64, float64)", nopython=True)
def _ecef2lla_latitude_func(s: float, z: float, beta: float) -> float:
    num = (
        z
        + (_wgs84ecc ** 2 * (1 - _wgs84f) / (1 - _wgs84ecc ** 2))
        * _wgs84a
        * np.sin(beta) ** 3
    )
    den = s - _wgs84ecc ** 2 * _wgs84a * np.cos(beta) ** 3

    return np.arctan2(num, den)


@jit("float64[:](float64[:])", nopython=True)
def ecef2lla(ecef_coordinates: np.ndarray) -> np.ndarray:
    r"""Transform an ECEF position to LLA position.

    The purpose of this function is to convert a position in the ECEF frame to
    the LLA frame relative to the WGS84 ellipsoid. Let the coordinates in the
    ECEF frame be
    \(\boldsymbol{r}^{\mathcal{F}} = [r_{x}, r_{y}, r_{z}]^{T}\). The
    longitude is calculated as
    $$
        \phi = \arctan\left(\frac{p_{y}}{p_{x}}\right)
    $$
    Note that a 4-quadrant arctangent function should be used to avoid
    ambiguity.

    The latitude cannot be solved analytically and must be solved numerically.
    The initial guesses for the geodetic latitude \(\lambda\) and reduced
    latitude \(\beta\) are
    $$
        \begin{split}
            \beta = & \arctan\left(\frac{p_{z}}{(1-f)s}\right) \\
            \lambda = & \arctan\left(
                    \frac{
                        p_{z} + \frac{e^{2}(1-f)}{1-e^{2}}a\sin^{3}\beta
                    }{
                        s - e^{2}a\cos^{3}\beta
                    }
                \right)
        \end{split}
    $$
    where \(a\) is the semi-major axis of the WGS84 ellipsoid, \(f\) is the
    flattening of the ellipsoid, \(e^{2}=1-(1-f)^{2}\), and
    \(s = \sqrt{p_{x}^{2} + p_{y}^{2}}\).

    Using the initial guesses, an update to the reduced latitude can be
    calculated as
    $$
        \beta = \arctan\left(
                \frac{(1 - f)\sin\lambda}{\cos\lambda}
            \right)
    $$
    which can in turn be used to update \(\lambda\). This cycle continues
    until \(\lambda\) converges ([1]_ claims this usually takes 2-3
    iterations).

    From here, it is possible to compute the altitude directly:
    $$
        h = s \cos\lambda + (p_{z} + e^{2}n\sin\lambda)\sin\lambda - n
    $$
    where \(n\) is the radius of curvature in the vertical prime
    $$
        n = \frac{a}{\sqrt{1 - (e\sin\lambda)^{2}}}
    $$

    This function is based on the development provided by [1]_

    .. todo:: 
        If the latitude does not converge, this function prints a warning 
        instead of using the logger. This is necessary because of use of
        Numba. In the future, this should be changed to logging a warning
        message.

    Parameters
    ----------
    ecef_coordinates : np.ndarray
        The coordinates in the ECEF frame

    Returns
    -------
    np.ndarray
        The latitude, longitude, altitude coordinates.

    References
    ----------
    .. [1] MathWorks Aerospace Blockset
        https://www.mathworks.com/help/aeroblks/ecefpositiontolla.html

    """
    x = ecef_coordinates[0]
    y = ecef_coordinates[1]
    z = ecef_coordinates[2]

    longitude = np.arctan2(y, x)

    max_iterations = 5
    desired_accuracy = 1e-15

    s = np.sqrt(x ** 2 + y ** 2)

    beta = np.arctan2(z, (1 - _wgs84f) * s)
    latitude = _ecef2lla_latitude_func(s, z, beta)

    for ii in range(max_iterations):
        old_latitude = latitude

        beta = _ecef2lla_beta_func(latitude)
        latitude = _ecef2lla_latitude_func(s, z, beta)

        if np.abs(latitude - old_latitude) <= desired_accuracy:
            break

    else:
        print(
            "ecef2lla >> WARNING: latitude did not converge: ",
            latitude - old_latitude,
        )

    # logger.debug(f"ecef2lla took {ii} iterations to converge.")

    n = _wgs84a / np.sqrt(1 - _wgs84ecc ** 2 * np.sin(latitude) ** 2)

    altitude = (
        s * np.cos(latitude)
        + (z + _wgs84ecc ** 2 * n * np.sin(latitude)) * np.sin(latitude)
        - n
    )

    return np.array((latitude, longitude, altitude))


@jit("float64[:](float64[:], float64)", nopython=True)
def eci2ecef(eci_coordinates: np.ndarray, time_of_week: float) -> np.ndarray:
    r"""Rotate from the ECI to the ECEF frame.

    This function rotates the ECI coordinates into the ECEF frame. Note that
    the ECI and ECEF frames are aligned at the start of the week.

    Let \(\boldsymbol{r}^{\mathcal{N}}\) be the position in the ECI frame
    and \(t\) be the time of week. The angle of rotation between the ECI and
    ECEF frames is \(\Theta = \omega_{\oplus}t\), with \(\omega_{\oplus}\)
    being the angular velocity of the Earth. Thus, the position in the ECEF
    coordinates is
    $$
        \boldsymbol{r}^{\mathcal{F}} =
            \left[\begin{array}{ccc}
                \cos \Theta & \sin \Theta & 0 \\
                -\sin \Theta & \cos \Theta & 0 \\
                0 & 0 & 1
            \end{array}\right]\boldsymbol{r}^{\mathcal{N}}
    $$

    Parameters
    ----------
    eci_coordinates : np.ndarray
        The coordinates in the ECI frame
    time_of_week : float
        The time of week in seconds associated with the ECEF frame

    Returns
    -------
    np.ndarray
        The coordinates in the ECI frame

    See Also
    --------
    standard_rotation_matrix: Used to get the rotation matrix for the
        transformation

    """
    # Angle from the ECI to the ECEF frame
    angle_of_rotation = _w_e * time_of_week
    # rotation = Rotation(axis=np.array((0., 0., 1.)), angle=angle_of_rotation)

    return standard_rotation(3, angle_of_rotation, eci_coordinates)


@jit("float64[:](float64[:], float64)", nopython=True)
def eci2lla(eci_coordinates: np.ndarray, time: GPSTime) -> np.ndarray:
    """Transform the ECI coordinates into an LLA position.

    The purpose of this function is to transform ECI coordinates into LLA
    coordinates. It does to by calling `eci2ecef()` and `ecef2lla()`

    Parameters
    ----------
    eci_coordinates : np.ndarray
        The ECI coordinates
    time : GPSTime
        The time for which to get the LLA

    Returns
    -------
    np.ndarray
        The Latitude, Longitude, Altitude coordinates (WGS84)

    See Also
    --------
    eci2ecef: Convert coordinates (and time) to ECEF frame
    ecef2lla: Convert ECEF coordinates to LLA position

    """
    return ecef2lla(eci2ecef(eci_coordinates, time))


@jit("float64[:](int64, float64[:])", nopython=True)
def add_weeks_eci(num_weeks: int, coordinates: np.ndarray) -> np.ndarray:
    r"""Move ECI frame to new week.

    Because the ECI frame is reference to the start of the week, if the frame
    is moved to a new week, the coordinates change. The purpose of this
    function is to update the ECI frame by moving forward `num_weeks` weeks
    (or backward is `num_weeks` is negative.

    This is accomplished by rotating the frame about the 3 axis by an angle of
    \(\omega_{\oplus}T\) where \(\omega_{\oplus}\) is the angular velocity
    of the Earth and \(T\) is the length of a week times the number of weeks
    to advance.

    Parameters
    ----------
    num_weeks : int
        The number of weeks to advance the ECI frame
    coordinates : np.ndarray
        The coordinates in the initial ECI frame

    Returns
    -------
    np.ndarray
        The coordinates in the new ECI frame

    See Also
    --------
    standard_rotation_matrix: Used to get the rotation matrix for the
        transformation

    """
    if num_weeks == 0:
        return coordinates
    else:
        # How far the Earth rotates in a week
        # coordinates = np.array(coordinates, dtype=float)
        weekly_eci_rotation = _w_e * 604800 * num_weeks

        # rotation = Rotation(
        #     dcm=standard_rotation_matrix(3, weekly_eci_rotation))

        return standard_rotation(3, weekly_eci_rotation, coordinates)


def rotate_ecef(
    old_time: GPSTime, new_time: GPSTime, coordinates: np.ndarray
) -> np.ndarray:
    """Rotate the coordinates in one ECEF frame to a new ECEF frame.

    Because the Earth is rotating, the ECEF frame is constantly moving and the
    coordinates in the ECEF frame at one time are different than the
    coordinates at some other time. This function uses the time between
    `old_time` and `new_time` to determine how much the ECEF frame rotates and
    updates the coordinates accordingly.

    Parameters
    ----------
    old_time : GPSTime
        The time for the ECEF frame where the `coordinates` are defined
    new_time : GPSTime
        The time of the new ECEF frame
    coordinates : np.ndarray
        The coordinates in the `old_time` ECEF frame

    Returns
    -------
    np.ndarray
        The coordinates in the `new_time` ECEF frame

    See Also
    --------
    standard_rotation_matrix: Used to get the rotation matrix for the
        transformation

    """
    time_delta = new_time - old_time
    assert isinstance(time_delta, float)

    angle_delta = _w_e * time_delta
    coordinates = np.array(coordinates, dtype=float)

    # rotation = Rotation(dcm=standard_rotation_matrix(3, angle_delta))

    return standard_rotation(3, angle_delta, coordinates)
