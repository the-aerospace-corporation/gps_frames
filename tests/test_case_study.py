# Copyright (c) 2022 The Aerospace Corporation
"""Test case for the satellite tracking case study example.

Ensures the simulation logic returns expected contact times and peak values
without requiring matplotlib or visualization tools.
"""

from examples.case_study_satellite_tracking import run_simulation


def test_case_study_logic():
    """Verify standard run of the case study."""
    # Run simulation with defaults (72 hours)
    results = run_simulation()

    # Unwrap results
    contacts = results["contacts"]
    max_el = results["max_el_val"]
    max_el_time = results["max_el_time_hr"]
    off_boresight = results["off_boresight_deg"]

    # Basic Plausibility Checks
    assert len(contacts) > 0, "Should find at least one contact pass"
    assert max_el > 5.0, "Max elevation should be above the 5 deg mask"
    assert max_el < 90.0, "Max elevation cannot exceed 90 deg"

    # Specific Sanity Check (loose bounds based on scenario physics)
    # A GPS MEO at 55 deg inclination seen from 34N is nominal
    # Depending on the specific RAAN/M0/Time, the max elevation might vary.
    # It should definitely correspond to a "high" pass (> 30 deg).
    assert 65.0 < max_el < 70.0, f"Expected max elevation pass of 68.87, got {max_el}"
    
    # Check off-boresight is small (satellite pointing at Earth center)
    # Station is on surface, satellite at 26k km radius.
    # Angle should be roughly asin(R_earth / R_orbit) max.
    # R_e ~ 6378, R_orb ~ 26560. asin(0.24) ~ 14 deg max (edge of earth).
    # Since we are at max elevation (near nadir), angle should be small.
    assert 13.0 <= off_boresight < 14.0, f"Off-boresight angle at peak should be 13.6053, got {off_boresight}" 
