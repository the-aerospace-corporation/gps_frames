"""Reference Frame Basis.

This submodule is used to define the basis for reference frames.

"""
from __future__ import annotations

import numpy as np
import ruamel.yaml
from numba import jit

from logging import getLogger
from typing import List

from .position import Position
from .vectors import UnitVector, SerializeableVector
from .rotations import Rotation

from gps_time import GPSTime

logger = getLogger(__name__)


def get_eci_basis(time: GPSTime) -> Basis:
    """Get the ECI basis.

    Parameters
    ----------
    time : GPSTime
        The time to define the ECI basis

    Returns
    -------
    Basis
        The ECI basis

    """
    origin = Position(
        coordinates=np.array([0.0, 0.0, 0.0]), frame_time=time, frame="ECI"
    )
    x_axis = UnitVector(
        coordinates=np.array([1.0, 0.0, 0.0]), frame_time=time, frame="ECI"
    )
    y_axis = UnitVector(
        coordinates=np.array([0.0, 1.0, 0.0]), frame_time=time, frame="ECI"
    )
    z_axis = UnitVector(
        coordinates=np.array([0.0, 0.0, 1.0]), frame_time=time, frame="ECI"
    )

    return Basis(origin=origin, axis1=x_axis, axis2=y_axis, axis3=z_axis)


def get_ecef_basis(time: GPSTime) -> Basis:
    """Get the ECE basis.

    Parameters
    ----------
    time : GPSTime
        The time to define the ECEF basis

    Returns
    -------
    Basis
        The ECEF basis

    """
    origin = Position(
        coordinates=np.array([0.0, 0.0, 0.0]), frame_time=time, frame="ECEF"
    )
    x_axis = UnitVector(
        coordinates=np.array([1.0, 0.0, 0.0]), frame_time=time, frame="ECEF"
    )
    y_axis = UnitVector(
        coordinates=np.array([0.0, 1.0, 0.0]), frame_time=time, frame="ECEF"
    )
    z_axis = UnitVector(
        coordinates=np.array([0.0, 0.0, 1.0]), frame_time=time, frame="ECEF"
    )

    return Basis(origin=origin, axis1=x_axis, axis2=y_axis, axis3=z_axis)


def coordinates_in_basis(position: Position, basis: Basis) -> np.ndarray:
    """Get the coordinates of a position in a basis.

    Parameters
    ----------
    position : Position
        The position
    basis : Basis
        The basis to express the position in

    Returns
    -------
    np.ndarray
        A three element array reprenting the coordinates of the position in
        the basis.

    """
    origin_frame = basis.origin.frame
    origin_frame_time = basis.origin.frame_time

    # Move the position ot the same frame as the basis
    _position = position.get_position(origin_frame)
    _position.update_frame_time(origin_frame_time)

    # Find the relative position
    rp = _position.coordinates - basis.origin.coordinates

    # Project the relative position onto the basis' axes
    v1 = basis.axes[0].dot_product(rp)
    v2 = basis.axes[1].dot_product(rp)
    v3 = basis.axes[2].dot_product(rp)

    return [v1, v2, v3]


class Basis:
    """Basis of a reference frame.

    Attributes
    ----------
    origin : Position
        The origin of the basis
    axes : list
        This is a three element array whose elements are of the `UnitVector`
        type. This represent the axes that make up the basis.

    Raises
    ------
    ValueError
        If any of the following:

        - The origin is in the LLA frame
        - The axes and origin are defined in different frames
        - The axes are not orthogonal
        - The basis is not right-handed

    """

    UNSAFE = False
    yaml_tag = "!Basis"

    def __init__(
        self, origin: Position, axis1: UnitVector, axis2: UnitVector, axis3: UnitVector
    ) -> None:
        """Object constructor.

        Parameters
        ----------
        origin : Position
            The position of the origin
        axis1 : UnitVector
            The 1-axis (e.g. X)
        axis2 : UnitVector
            The 2-axis (e.g. Y)
        axis3 : UnitVector
            The 3-axis (e.g. Z)

        """
        self.origin: Position = origin
        self.axes: List[UnitVector] = [axis1, axis2, axis3]

        if not Basis.UNSAFE:
            self.check_frames()
            self.check_orthogonality()
            self.check_right_handedness()

    @classmethod
    def to_yaml(
        cls, representer: ruamel.yaml.Representer, node: SerializeableVector
    ) -> ruamel.yaml.MappingNode:
        """Convert the class into a mapping node for serialzation.

        Takes the attributes of the class and converts them to ScalarNode
        used in a ruamel.yaml MappingNode to dump the object in a specific
        format. This method is called by the ruamel.yaml.YAML object
        when passed to register_class

        Parameters
        ----------
        representer : ruamel.yaml.Representer
            Yaml representer
        node : SerializeableVector
            The instance of the class to serialize

        Returns
        -------
        ruamel.yaml.MappingNode
            A mapping node which describes to the yaml dumper how to serialize
            the object

        """
        return representer.represent_mapping(
            cls.yaml_tag,
            {
                "origin": node.origin,
                "axis1": node.axes[0],
                "axis2": node.axes[1],
                "axis3": node.axes[2],
            },
        )

    @classmethod
    def from_yaml(
        cls: type, constructor: ruamel.yaml.Constructor, node: ruamel.yaml.MappingNode
    ) -> SerializeableVector:
        """Construct a class from a MappingNode.

        This method is called when using yaml.load if the
        SerializeableVector class has been registered to the
        ruamel.yaml.YAML object

        Parameters
        ----------
        cls : type
            The type of the class to deserialize
        constructor : ruamel.yaml.Constructor
            The constructor object
        node : ruamel.yaml.MappingNode
            Node created from the yaml parser

        Returns
        -------
        SerializeableVector
            Instance of serialized class

        """
        nodes = node.value
        origin = None
        axes = [None, None, None]
        for i in range(len(nodes)):
            node_name = nodes[i][0].value
            if node_name == "axis1":
                axes[0] = constructor.construct_object(nodes[i][1])
            elif node_name == "axis2":
                axes[1] = constructor.construct_object(nodes[i][1])
            elif node_name == "axis3":
                axes[2] = constructor.construct_object(nodes[i][1])
            elif node_name == "origin":
                origin = constructor.construct_object(nodes[i][1])
        return cls(origin, *axes)

    def check_frames(self) -> None:
        """Ensure that the frames are consistent.

        .. note:: LLA Frame
            If the LLA frame is provided for the origin, will attempt to
            convert to the ECEF frame. Will log an error

        Raises
        ------
        ValueError
            If the axes are not all in the same frame as the origin

        """
        if self.origin.frame == "LLA":
            logger.warning("LLA frame used for basis of origin, ECEF will be used")
            self.origin.switch_frame(to_frame="ECEF")

        origin_frame = self.origin.frame
        origin_frame_time = self.origin.frame_time

        if any([axis.frame != origin_frame for axis in self.axes]):
            raise ValueError("Not all axes and origin in the same frame")

        if any([axis.frame_time != origin_frame_time for axis in self.axes]):
            raise ValueError("Not all axes and origin have same frame time")

    def check_orthogonality(self, eps: float = 1e-12) -> None:
        """Check that the axes are orthogonal.

        Parameters
        ----------
        eps : float, optional
            The amount of allowable numerical error, by default 1e-12

        Raises
        ------
        ValueError
            If the axes are not mutually orthogonal

        """
        v1 = self.axes[0].coordinates
        v2 = self.axes[1].coordinates
        v3 = self.axes[2].coordinates

        checks = np.array(
            [np.abs(np.dot(v1, v2)), np.abs(np.dot(v2, v3)), np.abs(np.dot(v3, v1))]
        )
        if any(checks > eps):
            raise ValueError("Axes are not orthogonal")

    def check_right_handedness(self, eps: float = 1e-12) -> None:
        """Check that the axes are right-handed.

        Parameters
        ----------
        eps : float, optional
            The amount of allowable numerical error, by default 1e-12

        Raises
        ------
        ValueError
            If the basis is not right-handed

        """
        Basis.right_hand_numba(
            np.array(
                [
                    self.axes[0].coordinates,
                    self.axes[1].coordinates,
                    self.axes[2].coordinates,
                ]
            )
        )

    @staticmethod
    @jit(nopython=True)
    def right_hand_numba(axes: np.array, eps: float = 1e-12) -> None:

        v1 = axes[0]
        v2 = axes[1]
        v3 = axes[2]

        check1 = np.linalg.norm(np.cross(v1, v2) - v3) > eps
        check2 = np.linalg.norm(np.cross(v2, v3) - v1) > eps
        check3 = np.linalg.norm(np.cross(v3, v1) - v2) > eps

        if np.any(np.array([check1, check2, check3])):
            raise ValueError("Basis is not right-handed")

    def __hash__(self):
        """Make basis hashable."""
        return hash(
            str(hash(self.origin))
            + str(hash(self.axes[0]))
            + str(hash(self.axes[1]))
            + str(hash(self.axes[2]))
        )


def rotate_basis(rotation: Rotation, basis: Basis) -> Basis:
    """Rotate a basis about its origin.

    Parameters
    ----------
    rotation : Rotation
        A rotation object
    basis : Basis
        A basis

    Returns
    -------
    Basis
        A basis with the same origin as the input basis, but whose axes are
        rotated according to the given rotation.

    """
    v1 = UnitVector(
        rotation.rotate(basis.axes[0].coordinates),
        basis.origin.frame_time,
        basis.origin.frame,
    )
    v2 = UnitVector(
        rotation.rotate(basis.axes[1].coordinates),
        basis.origin.frame_time,
        basis.origin.frame,
    )
    v3 = UnitVector(
        rotation.rotate(basis.axes[2].coordinates),
        basis.origin.frame_time,
        basis.origin.frame,
    )

    return Basis(origin=basis.origin, axis1=v1, axis2=v2, axis3=v3)
