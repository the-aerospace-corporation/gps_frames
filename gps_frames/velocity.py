# Copyright (c) 2022 The Aerospace Corporation
"""Positions in reference frames.

The purpose of this submodule is to provide a representation for the velocity
of objects and tools to manipulate the velocities.
"""
from __future__ import annotations


from dataclasses import dataclass

from .position import Position
from .vectors import Vector
from .transforms import velocity_transform

from gps_time import GPSTime


@dataclass
class Velocity:
    """Dfines the velocity of an object.

    Attributes
    ----------
    position : Position
        The position of the object
    velocity : Vector
        The velocity of the object

    Notes
    -----
    The position is required in order to translate velocity between inertial
    and non-inertial frames

    Raises
    ------
    NotImplementedError
        The `update_frame_time()` is currently only a placeholder, so it
        raises an error if called

    """

    position: Position
    velocity: Vector

    def __post_init__(self) -> None:
        """Initialize positions."""
        self.position = self.position.get_position(self.velocity.frame)
        self.position.update_frame_time(self.velocity.frame_time)

    def get_velocity(self, out_frame: str) -> Velocity:
        """Get the velocity in a specified frame.

        Parameters
        ----------
        out_frame : str
            Frame to output velocity in

        Returns
        -------
        Velocity
            Velocity in specified frame

        """
        components = self.velocity
        new_components = velocity_transform(
            self.position.frame,
            out_frame,
            self.position.coordinates,
            components.coordinates,
            self.position.frame_time,
        )
        new_velocity = Vector(
            coordinates=new_components,
            frame_time=self.velocity.frame_time,
            frame=out_frame,
        )
        new_position = self.position.get_position(out_frame)

        return self.__class__(position=new_position, velocity=new_velocity)

    def update_frame_time(self, new_frame_time: GPSTime) -> None:
        """Change the reference time for the frame.

        !!! quote "Todo"
            Placeholder
            This is a placeholder that raises an error. It is not clear if
            this is needed in the future.

        Reference frames can be time dependant. This method is used to change
        the time reference for a reference frame.

        This method updates the internal representation of the velocity to the
        frame at the new reference time.

        Parameters
        ----------
        new_frame_time : GPSTime
            The new frame reference time

        """
        raise NotImplementedError("Placeholder for to mark future needs")
