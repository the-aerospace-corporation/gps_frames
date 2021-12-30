# Copyright (c) 2022 The Aerospace Corporation
"""Representation for Vectors.

The purpose of this submodule is to provide for the representation of vectors
in various reference frames.

"""
from __future__ import annotations

import numpy as np
from numba import jit
from numba import int32, float32, int64, float64  # import the types
from numba.experimental import jitclass
import ruamel.yaml

from dataclasses import dataclass
from typing import Union

from . import transforms as trans
from gps_time import GPSTime


@dataclass
class SerializeableVector:
    """Vector to serialize numpy arrays human readable.

    Attributes
    ----------
    coordinates : np.ndarray
        A three element array representing the components of the vector
    frame_time : GPSTime
        The time at which the frame is defined
    frame : str
        The frame in which the vector is defined

    """

    coordinates: np.ndarray
    frame_time: GPSTime
    frame: str

    yaml_tag: str = "!SerializeableVector"

    def __post_init__(self):
        """Nothing."""
        if not isinstance(self.coordinates, np.ndarray):
            self.coordinates = np.array(self.coordinates)

        self.coordinates = np.array(self.coordinates, dtype=float)

        if len(np.shape(self.coordinates)) == 2:
            self.coordinates = np.reshape(self.coordinates, 3)
        elif len(np.shape(self.coordinates)) > 2:
            raise ValueError("Too many dimensions for coordinates")

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
        coords = [float(d) for d in node.coordinates]
        return representer.represent_mapping(
            cls.yaml_tag,
            {
                "frame": node.frame,
                "frame_time": node.frame_time.to_datetime(),
                "coordinates": coords,
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
        coordinates = None
        frame_time = None
        frame = None
        for i in range(0, len(nodes)):
            node_name = nodes[i][0].value
            if node_name == "coordinates":
                coordinates = np.array(constructor.construct_sequence(nodes[i][1]))
            elif node_name == "frame":
                frame = constructor.construct_scalar(nodes[i][1])
            elif node_name == "frame_time":
                frame_time = GPSTime.from_datetime(
                    constructor.construct_object(nodes[i][1])
                )
        return cls(np.array(coordinates, dtype=float), frame_time, frame)

    def __eq__(self, other: SerializeableVector) -> bool:
        """Check equality.

        Parameters
        ----------
        other : SerializeableVector
            Vector to check equality against

        Returns
        -------
        bool
            True if the two are equal
        """
        return (
            self.coordinates == other.coordinates
            and self.frame_time == other.frame_time
            and self.frame == other.frame
        )

    def __hash__(self):
        """Create hash."""
        return hash(
            self.frame
            + str(hash(self.frame_time))
            + "".join([str(_coord) for _coord in self.coordinates])
        )


@dataclass
class Vector(SerializeableVector):
    """Representation of a vector."""

    yaml_tag: str = "!SerializeableVector.Vector"

    def __post_init__(self):
        """Run post-init checks.

        Ensure that the coordinates are a numpy array and is not in the LLA
        frame to avoid numerical issues

        """
        super().__post_init__()

        if self.frame == "LLA":
            raise ValueError("Vectors cannot be defined in the LLA frame")

        if self.frame not in ["ECEF", "ECI"]:
            raise ValueError("Vectors must be defined in either the ECI or ECEF frame")

    def get_vector(self, out_frame: str) -> Vector:
        """Get a vector object in new frame.

        .. note:: New Object Created on Output
            This function returns a new `Vector` object. It does not change
            the reference frame used in this object. Use `switch_frame()` to
            change the frame of the object.

        Parameters
        ----------
        out_frame : str
            The frame to output the vector in

        Returns
        -------
        Vector
            The vector in the desired frame

        See Also
        --------
        switch_frame: Change the frame that the vector is interally
            expressed in

        """
        if out_frame == "LLA":
            raise ValueError("Vectors cannot be defined in the LLA frame")

        new_coordinates = trans.position_transform(
            self.frame, out_frame, self.coordinates, self.frame_time
        )

        new_vector = self.__class__(new_coordinates, self.frame_time, out_frame)

        return new_vector

    def switch_frame(self, to_frame: str) -> None:
        """Change the frame used to store this vector internally.

        Parameters
        ----------
        to_frame : str
            The desired new frame

        """
        if to_frame == "LLA":
            raise ValueError("Vectors cannot be defined in the LLA frame")

        new_coordinates = trans.position_transform(
            self.frame, to_frame, self.coordinates, self.frame_time
        )

        self.coordinates = new_coordinates
        self.frame = to_frame

    def update_frame_time(self, new_frame_time: GPSTime) -> None:
        """Change the reference time for the frame.

        Reference frames can be time dependant. This method is used to change
        the time reference for a reference frame.

        This method updates the internal representation of the vector to
        the frame at the new reference time.

        Parameters
        ----------
        new_frame_time : GPSTime
            The new frame reference time

        """
        if self.frame == "ECI":
            delta_weeks = new_frame_time.week_number - self.frame_time.week_number

            self.coordinates = trans.add_weeks_eci(delta_weeks, self.coordinates)
            self.frame_time = new_frame_time

        elif self.frame == "ECEF":
            self.coordinates = trans.rotate_ecef(
                self.frame_time, new_frame_time, self.coordinates
            )
            self.frame_time = new_frame_time

    def cross_product(self, other: Vector) -> Vector:
        """Cross Product.

        Take the cross product of one vector with respect to annother

        Parameters
        ----------
        other : Vector
            Another vector

        Returns
        -------
        Vector
            self cross other

        """
        other_vector = other.get_vector(self.frame)
        other_vector.update_frame_time(self.frame_time)

        new_coordinates = _cross_numba(self.coordinates, other_vector.coordinates)

        return self.__class__(new_coordinates, self.frame_time, self.frame)

    def dot_product(self, other: Union[Vector, np.ndarray]) -> float:
        """Dot Product.

        Take the dot product of one vector with repsect to another

        Parameters
        ----------
        other : Union[Vector, np.ndarray]
            If other is a `Vector`, then rotates into the correct frame before
            taking the dot product. If the other is a `np.ndarray`, it is
            assumed to already be in the correct frame.

        Returns
        -------
        float
            The dot product

        Raises
        ------
        TypeError
            If the other is not a `Vector` or `np.ndarray`

        """
        if isinstance(other, Vector):
            other_vector = other.get_vector(self.frame)
            other_vector.update_frame_time(self.frame_time)

            _coordinates = other_vector.coordinates
        elif isinstance(other, np.ndarray):
            _coordinates = other
        else:
            raise TypeError("other must be a Vector of Numpy array")

        return np.dot(self.coordinates, _coordinates)

    @property
    def magnitude(self) -> float:
        """Get the magnitude of the vector.

        Returns
        -------
        float
            The magnitude of the vector

        """
        return np.linalg.norm(self.coordinates)

    def __neg__(self) -> Vector:
        """Negation Operator.

        This flips the direction of the vector

        Returns
        -------
        Vector
            A new vector with the coordinates flipped

        """
        return self.__class__(-self.coordinates, self.frame_time, self.frame)

    def __add__(self, other: Vector) -> Vector:
        """Add two vectors.

        Parameters
        ----------
        other : Vector
            Vector to add

        Returns
        -------
        Vector
            New vector

        """
        other_vector = other.get_vector(self.frame)
        other_vector.update_frame_time(self.frame_time)

        new_coordinates = self.coordinates + other_vector.coordinates

        return self.__class__(new_coordinates, self.frame_time, self.frame)

    def __sub__(self, other: Vector) -> Vector:
        """Subtract two vectors.

        Parameters
        ----------
        other : Vector
            Vector to subtract

        Returns
        -------
        Vector
            New vector

        """
        other_vector = other.get_vector(self.frame)
        other_vector.update_frame_time(self.frame_time)

        new_coordinates = self.coordinates - other_vector.coordinates

        return self.__class__(new_coordinates, self.frame_time, self.frame)

    def __mul__(self, other: float) -> Vector:
        """Multiply vector by an integer (scale it).

        Parameters
        ----------
        other : float
            integer to multiply

        Returns
        -------
        Vector
            New vector

        Raises
        ------
        TypeError
            If other is not a float

        """
        if isinstance(other, int):
            other = float(other)

        if not isinstance(other, float):
            raise TypeError(f"other value must be a float. IS: {type(other)}")

        new_coordinates = self.coordinates * other
        return self.__class__(new_coordinates, self.frame_time, self.frame)


class UnitVector(Vector):
    """Represention of a unit vector.

    Attributes
    ----------
    coordinates : np.ndarray
        The three coordinates that define the position.
    frame_time : GPSTime
        The time for which the reference frame is defined
    frame : str
        The name of the frame

    """

    yaml_tag: str = "!SerializeableVector.Vector.UnitVector"

    def __post_init__(self):
        """Run post-init checks.

        Ensure vector is normalized

        """
        super().__post_init__()
        self.normalize()

    @classmethod
    def from_vector(cls, vector: Vector) -> UnitVector:
        """Recast a vector as a unit vector.

        Parameters
        ----------
        vector : Vector
            A vector

        Returns
        -------
        UnitVector
            The vector normalized to a unit vector

        """
        return cls(vector.coordinates, vector.frame_time, vector.frame)

    def normalize(self) -> None:
        """Normalize the Unit Vector to a magnitude of 1."""
        self.coordinates = _norm_numba(self.coordinates)

    def get_unit_vector(self, out_frame: str) -> UnitVector:
        """Get a unit vector object in new frame.

        .. note:: New Object Created on Output
            This function returns a new `UnitVector` object. It does not change
            the reference frame used in this object. Use `switch_frame()` to
            change the frame of the object.

        Parameters
        ----------
        out_frame : str
            The frame to output the unit vector in

        Returns
        -------
        UnitVector
            The unit vector in the desired frame

        See Also
        --------
        switch_frame: Change the frame that the unit vector is interally
            expressed in

        """
        return self.from_vector(super().get_vector(out_frame))

    def switch_frame(self, to_frame: str) -> None:
        """Change the frame used to store this unit vector internally.

        Parameters
        ----------
        to_frame : str
            The desired new frame

        """
        super().switch_frame(to_frame)
        self.normalize()

    def __mul__(self, other: float) -> Vector:
        """Multiply UnitVector by a float (scale it to a Vector).

        Parameters
        ----------
        other : float
            integer to multiply

        Returns
        -------
        Vector
            New vector

        Raises
        ------
        TypeError
            If other is not a float

        """
        if isinstance(other, int):
            other = float(other)

        if not isinstance(other, float):
            raise TypeError(f"other value must be a float. IS: {type(other)}")

        new_coordinates = self.coordinates * other
        return Vector(new_coordinates, self.frame_time, self.frame)

    def __hash__(self):
        """Create hash."""
        return hash(
            self.frame
            + str(hash(self.frame_time))
            + "".join([str(_coord) for _coord in self.coordinates])
        )


@jit(float64[:](float64[:], float64[:]), nopython=True, cache=True)
def _cross_numba(vec1: np.array, vec2: np.array) -> np.array:
    return np.cross(vec1, vec2)


@jit("float64[:](float64[:])", nopython=True, cache=True)
def _norm_numba(coordinates: np.array) -> np.array:
    return coordinates / np.linalg.norm(coordinates)
