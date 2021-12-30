import numpy as np
import pytest
import copy
from gps_frames import basis
from gps_frames import vectors
from gps_frames import position
from gps_frames import rotations
from gps_frames import transforms as trans
from gps_time import GPSTime
from gps_frames.parameters import GeoidData, EarthParam


@pytest.mark.parametrize(
    "test_time",
    [
        (GPSTime(0, 0)),
        (GPSTime(2109, 259200)),
        (GPSTime(2111, 259200)),
    ],
)
def test_get_eci_basis(test_time):
    b = basis.get_eci_basis(test_time)
    assert np.linalg.norm(b.origin.coordinates) == 0


@pytest.mark.parametrize(
    "test_time",
    [
        (GPSTime(0, 0)),
        (GPSTime(2109, 259200)),
        (GPSTime(2111, 259200)),
    ],
)
def test_get_ecef_basis(test_time):
    b = basis.get_ecef_basis(test_time)
    assert np.linalg.norm(b.origin.coordinates) == 0


@pytest.mark.parametrize(
    "pos, expected",
    [
        (
            position.Position(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECI"),
            position.Position(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECI"),
        ),
        (
            position.Position(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECI"),
            position.Position(
                np.array([1, 0, 0], dtype=float),
                GPSTime(0, np.pi / 2 / EarthParam.w_e),
                "ECEF",
            ),
        ),
    ],
)
def test_crd_in_basis(pos, expected):

    _basis = basis.get_eci_basis(GPSTime(0, 0))
    assert (basis.coordinates_in_basis(pos, _basis) == expected.coordinates).all()


def test_init_right_handedness():

    pos = position.Position(np.array([0, 0, 0]), GPSTime(0, 0), "ECI")
    vec1 = vectors.UnitVector(np.array([1, 0, 0]), GPSTime(0, 0), "ECI")
    vec2 = vectors.UnitVector(np.array([0, 1, 0]), GPSTime(0, 0), "ECI")
    vec3 = vectors.UnitVector(np.array([0, 0, 1]), GPSTime(0, 0), "ECI")

    with pytest.raises(ValueError):
        basis.Basis(pos, vec1, vec3, vec2)


def test_init_diff_frames():

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "ECI")
    vec1 = vectors.UnitVector(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECI")
    vec2 = vectors.UnitVector(np.array([0, 1, 0], dtype=float), GPSTime(0, 0), "ECI")
    vec3 = vectors.UnitVector(np.array([0, 0, 1], dtype=float), GPSTime(0, 0), "ECI")

    posECEF = pos.get_position("ECEF")
    with pytest.raises(ValueError):
        basis.Basis(posECEF, vec1, vec2, vec3)


def test_init_orthgonality():

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "ECI")
    vec1 = vectors.UnitVector(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECI")
    vec2 = vectors.UnitVector(np.array([0, 1, 0], dtype=float), GPSTime(0, 0), "ECI")
    vec3 = vectors.UnitVector(np.array([0, 1, 1], dtype=float), GPSTime(0, 0), "ECI")

    with pytest.raises(ValueError):
        _basis = basis.Basis(pos, vec1, vec2, vec3)


# #def test_to yaml()

# #def test_from_yaml()


def test_check_frames_LLA():

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec1 = vectors.UnitVector(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec2 = vectors.UnitVector(np.array([0, 1, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec3 = vectors.UnitVector(np.array([0, 0, 1], dtype=float), GPSTime(0, 0), "ECEF")

    _basis = basis.Basis(pos, vec1, vec2, vec3)
    _basis.origin.switch_frame("LLA")
    _basis.check_frames()
    assert _basis.origin.frame == "ECEF"


def test_check_frames_diff_frames():

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec1 = vectors.UnitVector(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec2 = vectors.UnitVector(np.array([0, 1, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec3 = vectors.UnitVector(np.array([0, 0, 1], dtype=float), GPSTime(0, 0), "ECEF")

    _basis = basis.Basis(pos, vec1, vec2, vec3)
    _basis.axes[1].switch_frame("ECI")
    with pytest.raises(ValueError):
        _basis.check_frames()


def test_check_frames_right_handed():

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec1 = vectors.UnitVector(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec2 = vectors.UnitVector(np.array([0, 1, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec3 = vectors.UnitVector(np.array([0, 0, 1], dtype=float), GPSTime(0, 0), "ECEF")

    _basis1 = basis.Basis(pos, vec1, vec2, vec3)

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec1 = vectors.UnitVector(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec3 = vectors.UnitVector(np.array([0, 1, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec2 = vectors.UnitVector(np.array([0, 0, 1], dtype=float), GPSTime(0, 0), "ECEF")

    with pytest.raises(ValueError):
        _basis2 = basis.Basis(pos, vec1, vec2, vec3)


def test_check_frames_diff_times():

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec1 = vectors.UnitVector(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec2 = vectors.UnitVector(np.array([0, 1, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec3 = vectors.UnitVector(np.array([0, 0, 1], dtype=float), GPSTime(0, 0), "ECEF")

    _basis = basis.Basis(pos, vec1, vec2, vec3)
    _basis.axes[2].update_frame_time(GPSTime(200, 0))
    with pytest.raises(ValueError):
        _basis.check_frames()


def test_orthogonality():

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec1 = vectors.UnitVector(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec2 = vectors.UnitVector(np.array([0, 1, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec3 = vectors.UnitVector(np.array([0, 0, 1], dtype=float), GPSTime(0, 0), "ECEF")

    _basis = basis.Basis(pos, vec1, vec2, vec3)

    vec4 = vectors.UnitVector(np.array([0, 1, 1]), GPSTime(0, 0), "ECEF")
    _basis.axes[2].coordinates = vec4.coordinates

    with pytest.raises(ValueError):
        _basis.check_orthogonality()


def test_right_hand():

    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "ECI")
    vec1 = vectors.UnitVector(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECI")
    vec2 = vectors.UnitVector(np.array([0, 1, 0], dtype=float), GPSTime(0, 0), "ECI")
    vec3 = vectors.UnitVector(np.array([0, 0, 1], dtype=float), GPSTime(0, 0), "ECI")

    with pytest.raises(ValueError):
        _basis = basis.Basis(pos, vec1, vec3, vec2)


@pytest.mark.parametrize(
    "angle, axis",
    [
        (np.pi / 2, np.array([0.0, 0.0, 1.0])),
        (np.pi / 3, np.array([0.0, 1.0, 1.0])),
        (np.pi / 6, np.array([4.0, 0.0, 1.0])),
    ],
)
def test_rotate_basis(angle, axis):

    rotation = rotations.Rotation(axis=axis, angle=angle)
    pos = position.Position(np.array([0, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec1 = vectors.UnitVector(np.array([1, 0, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec2 = vectors.UnitVector(np.array([0, 1, 0], dtype=float), GPSTime(0, 0), "ECEF")
    vec3 = vectors.UnitVector(np.array([0, 0, 1], dtype=float), GPSTime(0, 0), "ECEF")

    _basis = basis.Basis(pos, vec1, vec2, vec3)

    v1 = vectors.UnitVector(
        rotation.rotate(_basis.axes[0].coordinates),
        _basis.origin.frame_time,
        _basis.origin.frame,
    )
    v2 = vectors.UnitVector(
        rotation.rotate(_basis.axes[1].coordinates),
        _basis.origin.frame_time,
        _basis.origin.frame,
    )
    v3 = vectors.UnitVector(
        rotation.rotate(_basis.axes[2].coordinates),
        _basis.origin.frame_time,
        _basis.origin.frame,
    )

    _basis = basis.rotate_basis(rotation, _basis)

    assert (_basis.axes[0].coordinates == v1.coordinates).all()
    assert (_basis.axes[1].coordinates == v2.coordinates).all()
    assert (_basis.axes[2].coordinates == v3.coordinates).all()


def test_hash():

    # unsure how to properly test this
    _basis = basis.get_ecef_basis(GPSTime(1100, 1.0))
    print("*" * 50, _basis.axes)
    _basis.__hash__()
