# Case Study: The Lost Signal

## Problem Statement
A nominal GPS satellite (Slot A1) has been launched into a Medium Earth Orbit (MEO). Your mission is to plan ground station contacts for **The Aerospace Corporation** (El Segundo, CA) over the next 72 hours.

You need to answer the following questions:
1.  **When can we talk to it?** Identify all contact windows where the satellite is at least 5 degrees above the horizon.
2.  **When is the signal strongest?** Find the time and value of the maximum elevation angle.
3.  **How far off-axis are we?** At the moment of maximum elevation, assuming the satellite is pointing directly at the center of the Earth (Nadir), what is the angle between the satellite's boresight and the ground station?

### Simulation Parameters
*   **Orbit**: GPS Nominal
    *   Semi-Major Axis: 26,559.7 km
    *   Inclination: 55.0 deg
    *   Eccentricity: 0.0 (Circular)
    *   Start Time: GPS Week 2294, 0.0 s
*   **Ground Station**:
    *   Latitude: 33.9167° N
    *   Longitude: 118.4167° W

## Mathematical Background
To solve this, we rely on coordinate frames and transformations provided by `gps_frames`.

### 1. Orbit Propagation (Kepler -> ECI)
We start with Keplerian Orbital Elements ($a, e, i, \Omega, \omega, M$). While the physics of propagation are standard, we wrap the result in a `gps_frames.position.Position` object. This ties the coordinates to a specific frame ("ECI") and time (`GPSTime`).

```python
from gps_frames.position import Position
from gps_frames.rotations import Rotation

# ... calculate r_pqw ...
orbit_rotation = Rotation(dcm=dcm_combined)
r_eci_coords = orbit_rotation.rotate(r_pqw)

# Wrap in a Position object
sat_pos_eci = Position(r_eci_coords, current_time, "ECI")
```

### 2. Time Control (GPSTime Arithmetic)
The `gps_time.GPSTime` objects support direct addition of seconds (as `float` or `int`), making simulation loops extremely clean:

```python
# Direct arithmetic with seconds
current_time = base_time + t_sec
```

### 3. Automatic Frame Transformations
Once the satellite is in a `Position` object, we can transform it to the Earth-Fixed (ECEF) frame without manually calculating Sidereal Time. The library handles the time-dependent rotation internally through the `get_position` method.

```python
# The library handles the ECI -> ECEF rotation for current_time
sat_pos_ecef = sat_pos_eci.get_position("ECEF")
```

### 4. Topocentric Horizon Frame (ENU Basis)
To compute Azimuth and Elevation, we project the satellite position into the station's local **East-North-Up (ENU)** frame using the `gps_frames.basis.Basis` object.

```python
from gps_frames.basis import Basis, coordinates_in_basis

# Define the local ENU basis at the station
enu_basis = Basis(station_ecef, u_east, u_north, u_up)

# Project the satellite position into this basis directly!
rho_enu = coordinates_in_basis(sat_pos_ecef, enu_basis)
```

**Note**: Azimuth is calculated using `arctan2(East, North)`, resulting in a range of **-180° to +180°**, which avoids wrapping discontinuities on plots.

## Solution Script
The refactored solution is implemented in `examples/case_study_satellite_tracking.py`. It demonstrates how to leverage high-level objects to minimize manual coordinate bookkeeping.

### Running the Code
Ensure you have the optional dependencies installed:
```bash
pip install .[examples]
python examples/case_study_satellite_tracking.py
```

### Expected Output
The script will output contact times and angles. Example snippet:
```text
Pass 1: T+2.35h to T+6.10h (Duration: 225.0 min)
...
Maximum Elevation:
  Value: 82.45 deg
Off-Boresight Angle at Max Elevation:
  Angle: 0.9841 deg
```

visualization plots will also be generated showing the Azimuth and Elevation profiles for all visible passes.
