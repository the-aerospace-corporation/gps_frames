import numpy as np
import pytest
import copy
import gps_frames.parameters as parameters


@pytest.mark.parametrize(
    "lat, lon",
    [
        (
            0,
            0,
        ),
        (-90, -180),
        (90, 180),
        (47, 93),
    ],
)
def test_get_geoid_height(lat, lon):

    eps = 1e-8
    gHeight = parameters.GeoidData.get_geoid_height(lat, lon, "deg")
    height = parameters.GeoidData._geoid_height_interpolator(lat, lon)

    assert gHeight == height[0, 0]

    lat = lat * np.pi / 180
    lon = lon * np.pi / 180

    gHeight = parameters.GeoidData.get_geoid_height(lat, lon, "rad")
    assert gHeight >= height[0, 0] - eps and gHeight <= height[0, 0] + eps

    try:
        parameters.GeoidData.get_geoid_height(lat, lon, "foo")
        assert -1 == 1
    except NotImplementedError:
        pass
