# Copyright (c) 2022 The Aerospace Corporation

# Example: Ranging from a ground antenna to spacecraft
# ====================================================
# This example demonstrates how to use the 'gps_frames' library to perform
# common astrodynamics calculations, such as frame conversions, visibility
# checks, and range/azimuth/elevation computations.

# 1. Imports
# ----------
# We start by importing the necessary libraries. 'numpy' is used for array
# manipulations, which is central to the 'gps_frames' library. 'gps_time' is
# used for precise time handling, which is crucial for frame transformations
# (e.g., ECEF to ECI) that depend on the Earth's rotation.
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

# 2. Setup Scenario Time
# ----------------------
# Define the specific date and time for the scenario. This time is used for
# all subsequent calculations to ensure consistency, especially for
# time-dependent coordinate frame transformations.
# We create a GPSTime object from a standard Python datetime object.
time = GPSTime.from_datetime(datetime(2017, 9, 2, 12, 0, 0))

# 3. Define Ground Station
# ------------------------
# Define the location of a ground antenna. We start with Geodetic coordinates
# (Latitude, Longitude, Altitude/LLA) as these are most intuitive for ground
# locations.
ground_antenna_latitude_deg = 33
ground_antenna_longitude_deg = -118
ground_antenna_altitude_m = 30

# Create a Position object for the ground antenna.
# Note:
#   - Input coordinates must be converted to radians for the Position object.
#   - We specify 'LLA' as the frame type.
#   - The 'time' object associates this position with our scenario time.
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

# 4. Define Satellites
# --------------------
# We define two satellites in this scenario.
#   - Satellite 1: Low Earth Orbit (LEO). Positioned to be in view of the ground antenna.
#   - Satellite 2: Geostationary Orbit (GEO). Positioned to be NOT in view of the ground antenna,
#     but in view of Satellite 1.

# Define Satellite 1 (LEO)
satellite1_latitude_deg = 45
satellite1_longitude_deg = -110
satellite1_altitude_m = 600e3  # 600 km altitude

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

# Define Satellite 2 (GEO)
satellite2_latitude_deg = -10
satellite2_longitude_deg = -30
satellite2_altitude_m = 30000e3  # 30,000 km altitude (approx GEO/MEO)

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

# 5. Frame Conversions
# --------------------
# The Position object allows for easy conversion between coordinate frames.
# Here we extract the coordinates of Satellite 1 in both Earth-Centered, Earth-Fixed (ECEF)
# and Earth-Centered Inertial (ECI) frames. The library handles the necessary rotations
# based on the time associated with the Position object.
satellite1_ecef_coords = satellite1_position.get_position("ECEF").coordinates
satellite1_eci_coords = satellite1_position.get_position("ECI").coordinates

# 6. Visibility Checks
# --------------------
# We use 'check_earth_obscuration' to determine if there is a direct line of sight
# between two objects, blocked by the Earth.
#   - earth_adjustment_m: Optional parameter to adjust the effective Earth radius
#     (e.g., to account for atmosphere or terrain). Here set to 0.

# Check: Ground Antenna <-> Satellite 1
ga_to_sat1_in_view = check_earth_obscuration(
    ground_antenna_position, satellite1_position, earth_adjustment_m=0
)

# Check: Ground Antenna <-> Satellite 2
ga_to_sat2_in_view = check_earth_obscuration(
    ground_antenna_position, satellite2_position, earth_adjustment_m=0
)

# Check: Satellite 1 <-> Satellite 2 (Inter-satellite link)
sat1_to_sat2_in_view = check_earth_obscuration(
    satellite1_position, satellite2_position, earth_adjustment_m=0
)

# 7. Distance Calculations
# ------------------------
# Compute the straight-line Euclidean distance between the objects.
# The 'distance' function handles frame harmonization automatically (converting
# both to ECEF before calculation).
ga_to_sat1_distance = distance(ground_antenna_position, satellite1_position)
ga_to_sat2_distance = distance(ground_antenna_position, satellite2_position)
sat1_to_sat2_distance = distance(satellite1_position, satellite2_position)

# 8. Azimuth and Elevation
# ------------------------
# Calculate the look angles (Azimuth and Elevation) from an observer to a target.
# This requires defining a local reference frame (East-North-Up or ENU) at the
# observer's location.

# From Ground Antenna to Satellite 1
# First, get the basis vectors for the local ENU frame centered at the ground antenna.
# Then compute range, azimuth, and elevation.
_, ga_to_sat1_azimuth, ga_to_sat1_elevation = get_range_azimuth_elevation(
    get_east_north_up_basis(ground_antenna_position), satellite1_position
)

# From Ground Antenna to Satellite 2
_, ga_to_sat2_azimuth, ga_to_sat2_elevation = get_range_azimuth_elevation(
    get_east_north_up_basis(ground_antenna_position), satellite2_position
)

# From Satellite 1 to Satellite 2
# Note: Defining an ENU frame on a satellite is mathematically valid (based on its
# sub-satellite point) and allows us to compute relative look angles.
_, sat1_to_sat2_azimuth, sat1_to_sat2_elevation = get_range_azimuth_elevation(
    get_east_north_up_basis(satellite1_position), satellite2_position
)


# 9. Report Results
# -----------------
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
