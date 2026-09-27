# Copyright (c) 2022 The Aerospace Corporation
"""Unit tests for GeoidData and geoid height interpolation in gps_frames.parameters.

Verifies:
- Interpolation of EGM-96 geoid undulation N(lat, lon) in both degrees and radians
- Error handling for unsupported angle unit strings
"""

import numpy as np
import pytest

import gps_frames.parameters as parameters


@pytest.mark.parametrize(
    "lat, lon",
    [
        (0, 0),
        (-90, -180),
        (90, 180),
        (47, 93),
    ],
)
def test_get_geoid_height(lat, lon):
    """Verify EGM-96 geoid undulation interpolation in degrees and radians.

    Testing:
        parameters.GeoidData.get_geoid_height(lat, lon, unit) queries the 2D regular grid interpolator
        derived from the EGM-96 geoid height model.

    Expected Result:
        - Calling with unit='deg' matches raw interpolator output height[0, 0].
        - Converting lat and lon to radians (lat * pi / 180, lon * pi / 180) and calling with unit='rad'
          returns the identical height within tolerance eps = 1e-8 meters.
        - Calling with an unsupported unit string like 'foo' raises NotImplementedError.
    """
    eps = 1e-8
    gHeight = parameters.GeoidData.get_geoid_height(lat, lon, "deg")
    height = parameters.GeoidData._geoid_height_interpolator(lat, lon)

    assert gHeight == height[0, 0]

    lat_rad = lat * np.pi / 180
    lon_rad = lon * np.pi / 180

    gHeight_rad = parameters.GeoidData.get_geoid_height(lat_rad, lon_rad, "rad")
    assert abs(gHeight_rad - height[0, 0]) <= eps

    with pytest.raises(NotImplementedError):
        parameters.GeoidData.get_geoid_height(lat, lon, "foo")
