# Copyright (c) 2022 The Aerospace Corporation
"""Standard parameters used throughout the simulator.

This package contains physical parameters that are divided into several
classes. PhysicsParam contains standard physical parameters. EarthParam
contains parameters specific to the Earth. GPSparam includes parameters
specific to the GPS system.

References
----------
.. [1] IS-GPS-200 https://www.gps.gov/technical/icwg/

"""

# This file contains constant parameters related to the earth in a single class
#
# See IERS conventions (2010) Table 1.2 for reference

import os
import numpy as np

from dataclasses import dataclass
from datetime import datetime
from scipy import interpolate

from typing import Callable


@dataclass(init=False, repr=True, frozen=True)
class PhysicsParam:
    """Physical Constants.

    This class contains standard physical constants

    """

    c: float = 2.99792458e8  # m/s, speed of light (IS-GPS-200H)
    """The speed of light in meters per second

    The speed of light in meters per second as specified by IS-GPS-200H.
    """

    k_b: float = 1.38064852e-23  # m^2 kg / s^2 K
    r"""Boltzmann Constant in m^2 kg / s^2 K

    The Boltzmann constant, \(k_{B}\) or \(k\), is a physical
    constant relating the average kinetic energy of particles in a gas with
    the temperature of the gas and occurs in Planck's law of black-body
    radiation and in Boltzmann's entropy formula.
    """


@dataclass(init=False, repr=True, frozen=True)
class EarthParam:
    """Earth Parameters.

    This class contains parameters specific to the Earth. The values are
    specified by IS-GPS-200H or WGS-84

    """

    mu: float = 3.986005e14  # m^3/s^2, gravitational parameter
    r"""Standard Gravitational Parameter for the Earth. Units are
    \(\text{m}^{3}/\text{s}^{2}\). This value is specified by
    IS-GPS-200H.
    """

    w_e: float = 7.2921151467e-5  # rad/s, angular velocity of Earth
    r"""Angular velocity of the Earth. Units are \(\text{rad}/\text{s}\). This
    value is specified by IS-GPS-200H.
    """

    r_e: float = 6378137.0  # m, radius of Earth
    r"""Mean radius of the Earth. Units are \(\text{m}\). This value is
    specified by IS-GPS-200H.
    """

    wgs84a: float = 6378137.0  # m, semimajor axis of Earth from WGS 84
    r"""The semimajor axis of the ellipsoidal Earth. Units are \(\text{m}\).
    This value is specified by WGS-84.
    """

    wgs84f: float = 1 / 298.257223563
    """The WGS84 ellipsoid flattening value, specified by WGS-84
    """

    # m, semiminor axis of Earth from WGS 84
    wgs84b: float = wgs84a * (1 - wgs84f)
    r"""The semiminor axis of the ellipsoidal Earth. Units are \(\text{m}\).
    This value is derived from values specified by WGS-84.
    """

    wgs84ecc_squared: float = 1 - (1 - wgs84f) ** 2
    """The square of the eccentricity of the WGS-84 ellipsoid.
    This valuse is derived from values specified by WGS-84
    """

    wgs84ecc: float = np.sqrt(wgs84ecc_squared)
    """The eccentricity of the ellipsoidal Earth. Dimensionless.
    This value is derived from values specified by WGS-84.
    """

    F: float = -4.442807633e-10  # -2sqrt(mu)/c^2, relativity parameter
    r"""Parameter used to determine the relativistic clock bias of each
    satellite. This values is specified by IS-GPS-200.

    Equal to \(\frac{-2\sqrt{\mu}}{c^{2}}\), but the value here is the one
    specified by IS-GPS-200, not calculated from standard values.
    """


@dataclass(init=False, repr=True, frozen=True)
class GPSparam:
    """GPS Parameters.

    This class contains parameters specific to the GPS system. These
    parameters are specified by IS-GPS-200H.

    """

    epoch_datetime: datetime = datetime(1980, 1, 6)
    """The start time of the GPS Epoch, which is defined to be 00:00:00
    on 6 January 1980.
    """

    pi: float = 3.1415926535898  # Mathematical constant pi (IS-GPS-200H)
    r"""The mathematical constant \(\pi\). As many of the parameters in the
    broadcast ephemeris are transmitted in units of semicircles, it is
    important that the value of \(\pi\) used by the simulator is exactly
    the value used by receivers and the GPS constellation. As such this
    value is specified in IS-GPS-200H.
    """

    gamma: float = (1575.42 / 1227.6) ** 2
    r"""Square of L1-L2 center frequency ratio \(\gamma\)

    This parameter is used in determining the difference in the pseudorange
    between L1 and L2. Specifically,
    $$
        \gamma
            = \left(\frac{f_{\mathrm{L1}}}{f_{\mathrm{L2}}}\right)^{2}
            = \left(\frac{1575.42}{1227.6}\right)^{2}
    $$
    where \(f_{\mathrm{L1}} = 1575.42 \text{MHz}\) is the L1 center
    frequency and \(f_{\mathrm{L2}} = 1227.6 \text{MHz}\) is the L2
    center frequency.
    """

    lnav_t_oc_epoch: int = 2 ** 4
    r"""The LNAV Time of Clock Epoch

    This is the resolution of the time of clock,
    \(t_{\mathrm{OC}}\), as reported in the ephemeris of the LNAV
    messages.
    """

    lnav_t_oe_epoch: int = 2 ** 4
    r"""The LNAV Time of Ephemeris Epoch

    This is the resolution of the time of ephemeris,
    \(t_{\mathrm{OE}}\), as reported in the ephemeris of the LNAV
    messages.
    """

    lnav_t_oa_epoch: int = 2 ** 12
    r"""The LNAV Time of Applicability Epoch

    This is the resolution of the time of applicability,
    \(t_{\mathrm{OA}}\), as reported in the almanac of the LNAV
    messages.
    """

    L1_CENTER_FREQ_Hz: float = 1575.42e6
    """The center frequency of the GPS L1 signal in Hz
    """

    L2_CENTER_FREQ_Hz: float = 1227.6e6
    """The center frequency of the GPS L2 signal in Hz
    """

    L5_CENTER_FREQ_Hz: float = 1176.45e6
    """The center frequency of the GPS L5 signal in Hz
    """


@dataclass(init=False, repr=True, frozen=True)
class GeoidData:
    """EGM-96 Geoid Model.

    This class is used to store the information related to the shape of the
    Earth, relative to the WGS-84 ellipsoid.

    This class is based on the data from [1]_ under the MIT License

    References
    ----------
    .. [1] https://github.com/vectorstofinal/geoid_heights

    License
    -------
    The MIT License (MIT)

    Copyright (c) 2015 vectorstofinal

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

    """

    _this_dir, _ = os.path.split(__file__)
    _data_path = os.path.join(_this_dir, "geoid_data_1deg.npz")

    _geoid_data_npz = np.load(_data_path)

    latitudes: np.ndarray = _geoid_data_npz["latitudes"]
    """Geoid Grid latitudes (deg)"""

    longitudes: np.ndarray = _geoid_data_npz["longitudes"]
    """Geoid Grid longitudes (deg)"""

    geoid_heights: np.ndarray = _geoid_data_npz["geoid_heights"]
    """Geoid Heights in a grid (m)"""

    _geoid_height_interpolator: Callable = interpolate.RectBivariateSpline(
        latitudes, longitudes, geoid_heights
    )
    """Interpolant for geoid height, arguments are (Lat,Long) in degrees"""

    @staticmethod
    def get_geoid_height(
        latitude: float, longitude: float, units: str = "rad"
    ) -> float:
        """Get the geoid height.

        This function is used to return the geoid height at the provided
        location. This is done using the data stored in the geoiddata folder.

        Parameters
        ----------
        latitude : float
            The latitude of interest
        longitude : float
            The longitude of interest
        units : str, optional
            The units for latitude and longitude. Choices are either
            'rad' or 'deg' for radians or degrees, by default 'rad'

        Returns
        -------
        float
            The geoid height in meters for the given latitude and longitude

        Raises
        ------
        NotImplementedError
            If units other than 'rad' or 'deg' is requested

        """
        if units.lower() == "rad":
            latitude = latitude * 180.0 / np.pi
            longitude = longitude * 180.0 / np.pi
        elif units.lower() == "deg":
            pass
        else:
            raise NotImplementedError("Only rad and deg are available units")

        geoid_height = GeoidData._geoid_height_interpolator(latitude, longitude)

        return geoid_height[0, 0]
