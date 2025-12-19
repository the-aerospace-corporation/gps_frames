"""
Satellite Tracking Case Study (Refactored)
==========================================
This script demonstrates the usage of `gps_frames` and `gps_time` to solve a
satellite tracking problem, leveraging native library features like Rotation,
Basis, and Position objects.

Problem Statement:
------------------
A GPS satellite is in a nominal MEO orbit. A ground station in El Segundo, CA
needs to track it. We want to find:
1. Contact windows (intervals where Elevation > 5 deg).
2. The time and value of the maximum elevation over 3 days.
3. The off-boresight angle of the ground station relative to the satellite's
   nadir vector at the time of maximum elevation.

Library Features Demonstrated:
- GPSTime for high-precision time and direct arithmetic.
- Position for coordinate frame encapsulation and automatic transforms.
- Rotation for composing orbital orientation from standard axes.
- Basis for defining a local topocentric (ENU) frame and projecting vectors.

Author: Gemini 2.0 (and gps_frames contributors)
"""

from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

from gps_time import GPSTime
from gps_frames.position import Position
from gps_frames.vectors import UnitVector
from gps_frames.rotations import Rotation, standard_rotation_matrix
from gps_frames.basis import Basis, coordinates_in_basis

# Constants
MU_EARTH = 3.986004418e14  # m^3/s^2 (Standard Gravitational Parameter)
RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0


def kepler_to_eci(
    semimajor_axis: float,
    eccentricity: float,
    inclination: float,
    raan: float,
    arg_perigee: float,
    mean_anomaly: float,
    time: GPSTime,
) -> Position:
    """
    Convert Keplerian orbital elements to ECI Position vector using Rotation objects.

    Parameters
    ----------
    semimajor_axis : float
        Semi-major axis in meters.
    eccentricity : float
        Eccentricity (unitless).
    inclination : float
        Inclination in radians.
    raan : float
        RAAN in radians.
    arg_perigee : float
        Argument of Perigee in radians.
    mean_anomaly : float
        Mean Anomaly in radians.
    time : GPSTime
        The time for the position calculation.

    Returns
    -------
    Position
        The satellite position in the ECI frame.
    """
    # 1. Solve Kepler's Equation for Eccentric Anomaly (E)
    E = mean_anomaly
    if eccentricity > 1e-10:
        for _ in range(15):
            E_new = mean_anomaly + eccentricity * np.sin(E)
            if abs(E_new - E) < 1e-12:
                break
            E = E_new

    # 2. Compute Position in Perifocal Frame (PQW)
    cE = np.cos(E)
    sE = np.sin(E)
    fac = np.sqrt(1 - eccentricity**2)

    r_pqw = np.array(
        [semimajor_axis * (cE - eccentricity), semimajor_axis * fac * sE, 0.0]
    )

    # 3. Rotate from Perifocal to ECI using gps_frames.rotations.Rotation
    # The transformation is R = R_z(raan) * R_x(inc) * R_z(arg_p)
    dcm_argp = standard_rotation_matrix(3, -arg_perigee)
    dcm_inc = standard_rotation_matrix(1, -inclination)
    dcm_raan = standard_rotation_matrix(3, -raan)

    # Combined DCM for vector transformation
    dcm_vec = dcm_raan.T @ dcm_inc.T @ dcm_argp.T

    orbit_rotation = Rotation(dcm=dcm_vec)
    r_eci_coords = orbit_rotation.rotate(r_pqw)

    return Position(r_eci_coords, time, "ECI")


def get_enu_basis(station_pos: Position) -> Basis:
    """
    Create an ENU (East-North-Up) Basis object at the station location.

    Parameters
    ----------
    station_pos : Position
        The position of the ground station (expected in LLA or ECEF).

    Returns
    -------
    Basis
        The local ENU basis at the station.
    """
    # Ensure we have ECEF for basis construction
    pos_lla = station_pos.get_position("LLA")
    lat, lon, _ = pos_lla.coordinates
    time = station_pos.frame_time

    # 1. Define unit vectors in ECEF frame
    up_coords = np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )
    east_coords = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north_coords = np.cross(up_coords, east_coords)

    # 2. Package into gps_frames objects
    origin = station_pos.get_position("ECEF")
    u_east = UnitVector(east_coords, time, "ECEF")
    u_north = UnitVector(north_coords, time, "ECEF")
    u_up = UnitVector(up_coords, time, "ECEF")

    return Basis(origin, u_east, u_north, u_up)


def get_look_angles(rho_enu: np.ndarray) -> Tuple[float, float]:
    """
    Compute Azimuth and Elevation from ENU components.

    Parameters
    ----------
    rho_enu : np.ndarray
        Relative vector in ENU coordinates.

    Returns
    -------
    Tuple[float, float]
        Azimuth and Elevation in degrees.
    """
    e, n, u = rho_enu
    slant_range = np.linalg.norm(rho_enu)
    el_rad = np.arcsin(u / slant_range)
    az_rad = np.arctan2(e, n)
    # Return Azimuth in -180 to 180 range
    return az_rad * RAD2DEG, el_rad * RAD2DEG


def main() -> None:
    print("================================================================")
    print("            SATELLITE TRACKING CASE STUDY (OBJECT-ORIENTED)     ")
    print("================================================================")

    # 1. Setup Parameters
    a = 26559700.0  # m (GPS Semi-major axis)
    e = 0.0  # Circular
    i_inc = 55.0 * DEG2RAD
    raan = 0.0
    w = 0.0
    M0 = 0.0

    start_week = 2294
    start_sec = 0.0
    duration_hours = 72
    dt_sec = 60.0  # 1 minute steps

    # Ground Station: The Aerospace Corporation, El Segundo
    sta_lat = 33.9167 * DEG2RAD
    sta_lon = -118.4167 * DEG2RAD
    sta_alt = 0.0

    base_time = GPSTime(start_week, start_sec)
    station_pos = Position(np.array([sta_lat, sta_lon, sta_alt]), base_time, "LLA")

    # Pre-calculate the ENU basis (it's fixed in the ECEF frame)
    enu_basis = get_enu_basis(station_pos)

    # 2. Simulation Loop
    times_hr: List[float] = []
    elevations: List[float] = []
    azimuths: List[float] = []
    contacts: List[Tuple[float, float]] = []
    in_contact = False
    contact_start_hr = 0.0

    max_el_val = -np.inf
    max_el_time_hr = 0.0
    max_el_sat_pos: Position = None

    print(f"Simulating {duration_hours} hours...")

    n_steps = int(duration_hours * 3600 / dt_sec)
    for step in range(n_steps):
        t_sec = float(step * dt_sec)
        # Use direct GPSTime arithmetic
        current_time = base_time + t_sec

        # A. Propagate (returns Position in ECI)
        n = np.sqrt(MU_EARTH / a**3)
        M_curr = M0 + n * t_sec
        sat_pos_eci = kepler_to_eci(a, e, i_inc, raan, w, M_curr, current_time)

        # B. ECI -> ECEF (Native Position method)
        sat_pos_ecef = sat_pos_eci.get_position("ECEF")

        # C. ECEF -> ENU (Using Basis & coordinates_in_basis)
        # rho_enu is the relative vector projected into the local basis
        rho_enu = np.array(coordinates_in_basis(sat_pos_ecef, enu_basis))

        # D. Get Angles
        az, el = get_look_angles(rho_enu)

        # Store for plotting
        t_hr = t_sec / 3600.0
        times_hr.append(t_hr)
        elevations.append(el)
        azimuths.append(az)

        # E. Analysis
        if el > 5.0:
            if not in_contact:
                in_contact = True
                contact_start_hr = t_hr
            if el > max_el_val:
                max_el_val = el
                max_el_time_hr = t_hr
                max_el_sat_pos = sat_pos_ecef
        else:
            if in_contact:
                in_contact = False
                contacts.append((contact_start_hr, t_hr))

    # 3. Boresight Analysis at Peak
    # Nadir vector is opposite to satellite position in ECEF (points to Earth center)
    sat_ecef_coords = max_el_sat_pos.coordinates
    station_ecef_coords = station_pos.get_position("ECEF").coordinates

    vec_nadir = -sat_ecef_coords
    vec_to_station = station_ecef_coords - sat_ecef_coords

    # Calculate angle using dot product
    cos_theta = np.dot(vec_nadir, vec_to_station) / (
        np.linalg.norm(vec_nadir) * np.linalg.norm(vec_to_station)
    )
    off_boresight_deg = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * RAD2DEG

    # 4. Results Output
    print(f"\nSimulation Results:")
    print(f"Total passes observed: {len(contacts)}")
    print(f"Max Elevation: {max_el_val:.2f} deg at T+{max_el_time_hr:.2f}h")
    print(f"Off-Boresight Angle at Peak: {off_boresight_deg:.4f} deg")

    print("\nContact Windows (>5 deg):")
    for _ii, (start, end) in enumerate(contacts):
        print(
            f"  Pass {_ii+1:02}: {start:5.2f}h to {end:5.2f}h (Duration: {(end-start)*60:.1f} min)"
        )

    # 5. Plotting
    print("\nLaunching Plot...")
    vis_mask = np.array(elevations) > 0
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(
        np.array(times_hr)[vis_mask],
        np.array(elevations)[vis_mask],
        "b.",
        markersize=2,
    )
    plt.axhline(5, color="r", linestyle="--", alpha=0.5, label="5° Mask")
    plt.ylabel("Elevation [deg]")
    plt.title("Satellite Visibility - Topocentric Angles")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(
        np.array(times_hr)[vis_mask],
        np.array(azimuths)[vis_mask],
        "g.",
        markersize=2,
    )
    plt.ylabel("Azimuth [deg]")
    plt.xlabel("Time since Epoch [hours]")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
