# Copyright (c) 2022 The Aerospace Corporation

# Example: Ranging from a ground antenna to spacecraft

# Import a few required tools
import numpy as np

from datetime import datetime
from gps_time import GPSTime

# Import Components that will be shown in the example
from gps_frames import (
    Position,
    distance,
    get_range_azimuth_elevation,
    get_east_north_up_basis,
    check_earth_obscuration,
)

# Set the date and time for the scenario
time = GPSTime.from_datetime(datetime(2017, 9, 2, 12, 0, 0))

# Location of ground antenna
ground_antenna_latitude_deg = 33
ground_antenna_longitude_deg = -118
ground_antenna_altitude_m = 30

ground_antenna_position = Position(
    np.array(
        [
            ground_antenna_latitude_deg * np.pi / 180,
            ground_antenna_longitude_deg * np.pi / 180,
            ground_antenna_altitude_m,
        ]
    ),
    time,
    "LLA",
)

# Location of Satellites (LLA for simplicity). Satellite 1 is meant to be in
# view of the ground antenna. Satellite 2 is not in view of the antenna, but
# is in view of Satellite 1.
satellite1_latitude_deg = 45
satellite1_longitude_deg = -110
satellite1_altitude_m = 600e3

satellite1_position = Position(
    np.array(
        [
            satellite1_latitude_deg * np.pi / 180,
            satellite1_longitude_deg * np.pi / 180,
            satellite1_altitude_m,
        ]
    ),
    time,
    "LLA",
)

satellite2_latitude_deg = -10
satellite2_longitude_deg = -30
satellite2_altitude_m = 30000e3

satellite2_position = Position(
    np.array(
        [
            satellite2_latitude_deg * np.pi / 180,
            satellite2_longitude_deg * np.pi / 180,
            satellite2_altitude_m,
        ]
    ),
    time,
    "LLA",
)

# Get Coordinates of Satellite 1 in different frames
satellite1_ecef_coords = satellite1_position.get_position("ECEF").coordinates
satellite1_eci_coords = satellite1_position.get_position("ECI").coordinates

# Determine if different locations are in view of each other
ga_to_sat1_in_view = check_earth_obscuration(
    ground_antenna_position, satellite1_position, earth_adjustment_m=0
)
ga_to_sat2_in_view = check_earth_obscuration(
    ground_antenna_position, satellite2_position, earth_adjustment_m=0
)
sat1_to_sat2_in_view = check_earth_obscuration(
    satellite1_position, satellite2_position, earth_adjustment_m=0
)

# Find the distances between the locations
ga_to_sat1_distance = distance(ground_antenna_position, satellite1_position)
ga_to_sat2_distance = distance(ground_antenna_position, satellite2_position)
sat1_to_sat2_distance = distance(satellite1_position, satellite2_position)

# Get the azimuth and elevation of the satellites relative to each other. This
# requires finding the East-North-Up basis
_, ga_to_sat1_azimuth, ga_to_sat1_elevation = get_range_azimuth_elevation(
    get_east_north_up_basis(ground_antenna_position), satellite1_position
)
_, ga_to_sat2_azimuth, ga_to_sat2_elevation = get_range_azimuth_elevation(
    get_east_north_up_basis(ground_antenna_position), satellite2_position
)
_, sat1_to_sat2_azimuth, sat1_to_sat2_elevation = get_range_azimuth_elevation(
    get_east_north_up_basis(satellite1_position), satellite2_position
)


# Report Results
print("\n**** Ground Antenna Information *************************")
print(
    "Latitude (deg):  {:.2f}".format(
        ground_antenna_position.coordinates[0] * 180.0 / np.pi
    )
)
print(
    "Longitude (deg): {:.2f}".format(
        ground_antenna_position.coordinates[1] * 180.0 / np.pi
    )
)
print("Altitude (km):  {:.2f}".format(ground_antenna_position.coordinates[2] / 1000.0))

print("\n**** Satellite 1 Information (w/ other frames) **********")
print(
    "Latitude (deg):  {:.2f}".format(satellite1_position.coordinates[0] * 180.0 / np.pi)
)
print(
    "Longitude (deg): {:.2f}".format(satellite1_position.coordinates[1] * 180.0 / np.pi)
)
print("Altitude (km):   {:.2f}".format(satellite1_position.coordinates[2] / 1000.0))
print("-> ECEF X (km):  {:.2f}".format(satellite1_ecef_coords[0] / 1000.0))
print("-> ECEF Y (km):  {:.2f}".format(satellite1_ecef_coords[1] / 1000.0))
print("-> ECEF Z (km):  {:.2f}".format(satellite1_ecef_coords[2] / 1000.0))
print("-> ECI X (km):   {:.2f}".format(satellite1_eci_coords[0] / 1000.0))
print("-> ECI Y (km):   {:.2f}".format(satellite1_eci_coords[1] / 1000.0))
print("-> ECI Z (km):   {:.2f}".format(satellite1_eci_coords[2] / 1000.0))

print("\n**** Satellite 2 Information ****************************")
print(
    "Latitude (deg):  {:.2f}".format(satellite2_position.coordinates[0] * 180.0 / np.pi)
)
print(
    "Longitude (deg): {:.2f}".format(satellite2_position.coordinates[1] * 180.0 / np.pi)
)
print("Altitude (km):   {:.2f}".format(satellite2_position.coordinates[2] / 1000.0))


print("\n**** Line of Sight **************************************")
print(f"Ground Antenna to Satellite 1: {ga_to_sat1_in_view}")
print(f"Ground Antenna to Satellite 2: {ga_to_sat2_in_view}")
print(f"Satellite 1 to Satellite 2:    {sat1_to_sat2_in_view}")


print("\n**** Distances ******************************************")
print("Ground Antenna to Satellite 1 (km): {:.2f}".format(ga_to_sat1_distance / 1000.0))
print("Ground Antenna to Satellite 2 (km): {:.2f}".format(ga_to_sat2_distance / 1000.0))
print(
    "Satellite 1 to Satellite 2 (km):    {:.2f}".format(sat1_to_sat2_distance / 1000.0)
)

print("\n**** Azimuth/Elevation **********************************")
print(
    "Ground Antenna to Satellite 1 (deg): {:.2f}, {:.2f}".format(
        ga_to_sat1_azimuth * 180.0 / np.pi, ga_to_sat1_elevation * 180.0 / np.pi
    )
)
print(
    "Ground Antenna to Satellite 2 (deg): {:.2f}, {:.2f}".format(
        ga_to_sat2_azimuth * 180.0 / np.pi, ga_to_sat2_elevation * 180.0 / np.pi
    )
)
print(
    "Satellite 1 to Satellite 2 (deg):    {:.2f}, {:.2f}".format(
        sat1_to_sat2_azimuth * 180.0 / np.pi, sat1_to_sat2_elevation * 180.0 / np.pi
    )
)

print("")
