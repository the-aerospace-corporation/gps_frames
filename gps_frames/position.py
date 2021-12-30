"""Positions in reference frames.

The purpose of this submodule is to provide a representation for the position
of objects and tools to manipulate the positions.

"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from . import transforms as trans
from .parameters import GeoidData, EarthParam
from .vectors import Vector, SerializeableVector

from gps_time import GPSTime


def distance(position1: Position, position2: Position) -> float:
    """Compute the distance between two positions.

    This function is used to calculate the distance between two positions. To
    do this, it first computes the positions in their ECEF reference frames
    (for their frame refer times). It then rotates the ECEF frame of
    `position2` into the ECEF frame of `position1`. From there, it simply
    takes the 2-norm of the difference of the position coordinates.

    Although the calculation of distance is done in the ECEF frame, the
    distance between two points is the same regardless of which frames are
    used.

    Parameters
    ----------
    position1 : Position
        A Position
    position2 : Position
        Another Position

    Returns
    -------
    float
        The distance between `position1` and `position2`

    """
    # Get both positions in the ECEF frame
    eci_position1 = position1.get_position("ECI")
    eci_position2 = position2.get_position("ECI")

    # Rotate the second into the frame of the first
    eci_position2.update_frame_time(eci_position1.frame_time)

    distance_vector = eci_position1.coordinates - eci_position2.coordinates

    return np.linalg.norm(distance_vector)


@dataclass(eq=False)
class Position(SerializeableVector):
    """Represent a position in multiple frames."""

    yaml_tag = "!SerializeableVector.Position"

    def __post_init__(self) -> None:
        """Check for proper coordinate shape."""
        if len(np.shape(self.coordinates)) == 2:
            self.coordinates = np.reshape(self.coordinates, 3)
        elif len(np.shape(self.coordinates)) > 2:
            raise ValueError("Too many dimensions for position")

    def get_position(self, out_frame: str) -> Position:
        """Get a position object in new frame.

        .. note:: New Object Created on Output
            This function returns a new position object. It does not change
            the reference frame used in this object. Use `switch_frame()` to
            change the frame of the object.

        Parameters
        ----------
        out_frame : str
            The frame to output the position in

        Returns
        -------
        Position
            The position expressed in a new frame

        See Also
        --------
        switch_frame: Change the frame that the position is interally
            expressed in

        """
        new_coordinates = trans.position_transform(
            self.frame, out_frame, self.coordinates, self.frame_time
        )

        return self.__class__(new_coordinates, self.frame_time, out_frame)

    def switch_frame(self, to_frame: str) -> None:
        """Change the frame used to store this position internally.

        Parameters
        ----------
        to_frame : str
            The desired new frame

        """
        new_coordinates = trans.position_transform(
            self.frame, to_frame, self.coordinates, self.frame_time
        )

        self.coordinates = new_coordinates
        self.frame = to_frame

    def update_frame_time(self, new_frame_time: GPSTime) -> None:
        """Change the reference time for the frame.

        Reference frames can be time dependant. This method is used to change
        the time reference for a reference frame.

        This method updates the internal representation of the position to the
        frame at the new reference time.

        Parameters
        ----------
        new_frame_time : GPSTime
            The new frame reference time

        """
        coordinates = self.coordinates
        if self.frame == "ECI":
            delta_weeks = new_frame_time.week_number - self.frame_time.week_number

            self.coordinates = trans.add_weeks_eci(delta_weeks, coordinates)
            self.frame_time = new_frame_time

        elif self.frame == "ECEF":
            self.coordinates = trans.rotate_ecef(
                self.frame_time, new_frame_time, coordinates
            )
            self.frame_time = new_frame_time

        elif self.frame == "LLA":
            self.switch_frame("ECEF")
            coordinates = self.coordinates
            self.coordinates = trans.rotate_ecef(
                self.frame_time, new_frame_time, coordinates
            )
            self.frame_time = new_frame_time

            self.switch_frame("LLA")

    def get_altitude_msl(self) -> float:
        """Get the Altitude above Mean Sea Level.

        Returns
        -------
        float
            The altitude above mean sea level

        """
        lla = trans.position_transform(
            self.frame, "LLA", self.coordinates, self.frame_time
        )

        geoid_height = GeoidData.get_geoid_height(lla[0], lla[1])

        return lla[2] - geoid_height

    def get_altitude_hae(self) -> float:
        """Get the altitude, height above ellipsoid.

        Returns
        -------
        float
            The height above ellipsoid

        """
        lla = trans.position_transform(
            self.frame, "LLA", self.coordinates, self.frame_time
        )

        return lla[2]

    def get_radius(self) -> float:
        """Get the radius of the position.

        Get the radius of the position, i.e. the distance from the position to
        the center of the Earth

        Returns
        -------
        float
            The distance to the center of the Earth

        """
        ecef = trans.position_transform(
            self.frame, "ECEF", self.coordinates, self.frame_time
        )

        return np.linalg.norm(ecef)

    def get_altitude_spherical(self) -> float:
        """Get the altitude above the spherical Earth.

        Get the altitude for the position if the Earth was a perfect sphere.
        This is done by subtracting the radius of the Earth from the value
        returned by get_radius() method.

        Returns
        -------
        float
            The altitude above the spherical Earth

        """
        return self.get_radius() - EarthParam.r_e

    def __hash__(self):
        """Make position hashable."""
        return super().__hash__()

    @classmethod
    def from_vector(cls, vector: Vector) -> Position:
        """Create a position from a vector.

        Parameters
        ----------
        vector : Vector
            The vector

        Returns
        -------
        Position
            The vector represented as a position

        """
        return cls(
            coordinates=vector.coordinates,
            frame_time=vector.frame_time,
            frame=vector.frame,
        )

    def to_vector(self) -> Vector:
        """Recast Position as a Vector.

        Returns
        -------
        Vector
            New vector

        """
        return Vector(
            coordinates=self.coordinates, frame_time=self.frame_time, frame=self.frame
        )

    def __eq__(self, other: object) -> bool:
        """Equality Comparision.

        Parameters
        ----------
        other : Position
            Another position

        Returns
        -------
        bool
            If the distance between the positions is less than 1e-6

        Raises
        ------
        TypeError
            If the other value is not a Position

        """
        if not isinstance(other, self.__class__):
            raise TypeError("Equality only defined between two Positions")

        eps = 1e-6

        return distance(self, other) < eps

    def __add__(self, other: Vector) -> Position:
        """Add a vector to a position.

        Parameters
        ----------
        other : Vector
            Vector to add

        Returns
        -------
        Position
            New position

        Raises
        ------
        TypeError
            Other is not a vector

        """
        if not isinstance(other, Vector):
            raise TypeError("other must be a vector")

        vec1 = self.to_vector()

        new_vec = vec1 + other

        return self.from_vector(new_vec)

    def __sub__(self, other: Vector) -> Position:
        """Subtract a vector from a position.

        Parameters
        ----------
        other : Vector
            Vector to subtract

        Returns
        -------
        Position
            New position

        Raises
        ------
        TypeError
            If other is not a vector
        """
        if not isinstance(other, Vector):
            raise TypeError("other must be a vector")

        vec1 = self.to_vector()

        new_vec = vec1 - other

        return self.from_vector(new_vec)
