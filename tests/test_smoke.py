"""Smoke tests: each stage runs and pins one known case from its source."""

import math

from orbitwear.atmosphere import Orbit, baseline_flux, exponential_density
from orbitwear.erosion import erode
from orbitwear.pipeline import Mission, Panel, run
from orbitwear.structure import Section, baseline_structure
from orbitwear.surface import maxwell_kernel


def test_density_table_anchor():
    # US Standard Atmosphere 1976 anchor at 400 km.
    assert math.isclose(exponential_density(400e3), 3.725e-12, rel_tol=1e-6)
    # Falls with altitude.
    assert exponential_density(500e3) < exponential_density(400e3)


def test_orbital_speed_400km():
    assert math.isclose(Orbit(400e3).speed, 7668.0, rel_tol=2e-3)


def test_flux_and_drag_coefficient_flat_plate():
    flux = baseline_flux(Orbit(400e3), incidence_cos=1.0)
    r = maxwell_kernel(flux, wall_temperature=300.0, alpha=0.9)
    # Free-molecular flat plate at normal incidence: Cd between 2 and 2.4.
    assert 2.0 <= r.drag_coefficient <= 2.4
    assert r.heat_flux > 0.0
    # Edge-on: nothing arrives.
    edge = maxwell_kernel(baseline_flux(Orbit(400e3), incidence_cos=0.0), 300.0, 0.9)
    assert edge.pressure == 0.0 and edge.drag_coefficient == 0.0


def test_kapton_erosion_order_of_magnitude():
    # Ram-facing Kapton at 400 km loses tens to a few hundred micrometres a
    # year depending on solar activity (MISSE, de Groh et al. 2008). The
    # baseline must land in that range with its mean-density table.
    flux = baseline_flux(Orbit(400e3), incidence_cos=1.0)
    year = 365.25 * 86400.0
    e = erode("kapton_H", flux, year)
    assert 20e-6 <= e.thickness_lost <= 500e-6


def test_structure_stress_rises_with_loss():
    sec = Section(thickness=100e-6, membrane_load=50.0, yield_strength=70e6, youngs_modulus=2.5e9)
    flux = baseline_flux(Orbit(400e3))
    r = maxwell_kernel(flux)
    s0 = baseline_structure(sec, 0.0, r)
    s1 = baseline_structure(sec, 50e-6, r)
    assert math.isclose(s0.stress, 50.0 / 100e-6)
    assert math.isclose(s1.stress, 2.0 * s0.stress)
    assert s1.margin < s0.margin


def test_pipeline_runs_end_to_end():
    panel = Panel("kapton_H", Section(thickness=125e-6, membrane_load=20.0, yield_strength=70e6, youngs_modulus=2.5e9))
    mission = Mission(Orbit(400e3), duration_s=365.25 * 86400.0, steps=12)
    h = run(panel, mission)
    assert len(h.time_s) == 13
    assert h.thickness_remaining[-1] < h.thickness_remaining[0]
    assert h.stress[-1] > h.stress[0]
    assert h.fluence[-1] > 0
