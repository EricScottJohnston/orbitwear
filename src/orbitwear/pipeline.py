"""The chain: atmosphere -> surface -> erosion -> structure, over a mission.

A :class:`Panel` is one surface element with a material and a section. A
:class:`Mission` is an orbit and a duration. :func:`run` steps the mission in
time, so that as the panel thins, the next step sees the thinner section. In
the baseline the flux does not depend on the thickness, so the result is
linear in time; with a real geometry and DSMC it will not be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .atmosphere import Orbit, baseline_flux
from .erosion import baseline_yield, erode
from .structure import Section, baseline_structure
from .surface import maxwell_kernel

SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class Panel:
    material: str
    section: Section
    incidence_cos: float = 1.0       # 1.0 = facing straight into the wind
    wall_temperature: float = 300.0  # K
    accommodation: float = 0.9


@dataclass(frozen=True)
class Mission:
    orbit: Orbit
    duration_s: float
    steps: int = 12


@dataclass
class History:
    """Per-step record. Every list has ``steps + 1`` entries, time 0 first."""
    time_s: list[float] = field(default_factory=list)
    drag_pressure: list[float] = field(default_factory=list)
    drag_coefficient: list[float] = field(default_factory=list)
    heat_flux: list[float] = field(default_factory=list)
    fluence: list[float] = field(default_factory=list)
    thickness_remaining: list[float] = field(default_factory=list)
    stress: list[float] = field(default_factory=list)
    margin: list[float] = field(default_factory=list)

    def last(self) -> dict[str, float]:
        return {k: v[-1] for k, v in self.__dict__.items() if isinstance(v, list) and v}


def run(panel: Panel, mission: Mission, *, flux_model=baseline_flux, kernel=maxwell_kernel,
        yield_model=baseline_yield, structural_model=baseline_structure) -> History:
    """Run the four stages over the mission and return the history.

    Each keyword lets a full engine replace a baseline stage without touching
    the others. The signatures are the ``Protocol`` classes in each module.
    """
    h = History()
    dt = mission.duration_s / mission.steps
    lost = 0.0
    fluence = 0.0

    def record(t: float, response, lost_now: float, fluence_now: float) -> None:
        s = structural_model(panel.section, lost_now, response)
        h.time_s.append(t)
        h.drag_pressure.append(response.pressure)
        h.drag_coefficient.append(response.drag_coefficient)
        h.heat_flux.append(response.heat_flux)
        h.fluence.append(fluence_now)
        h.thickness_remaining.append(s.thickness_remaining)
        h.stress.append(s.stress)
        h.margin.append(s.margin)

    flux = flux_model(mission.orbit, panel.incidence_cos)
    response = kernel(flux, panel.wall_temperature, panel.accommodation)
    record(0.0, response, lost, fluence)

    for i in range(1, mission.steps + 1):
        flux = flux_model(mission.orbit, panel.incidence_cos)
        response = kernel(flux, panel.wall_temperature, panel.accommodation)
        e = erode(panel.material, flux, dt, yield_model=yield_model, surface_temperature=panel.wall_temperature)
        lost += e.thickness_lost
        fluence += e.fluence
        record(i * dt, response, lost, fluence)
        if lost >= panel.section.thickness:
            break
    return h


def summary(h: History) -> str:
    """One block of text with the end-of-mission numbers in engineering units."""
    L = h.last()
    years = L["time_s"] / (365.25 * SECONDS_PER_DAY)
    lines = [
        f"after {years:.2f} years",
        f"  drag pressure        {L['drag_pressure']:.3e} Pa   (Cd {L['drag_coefficient']:.2f})",
        f"  heat flux absorbed   {L['heat_flux']:.3f} W/m^2",
        f"  AO fluence           {L['fluence'] / 1e4:.3e} atoms/cm^2",
        f"  thickness remaining  {L['thickness_remaining'] * 1e6:.1f} um",
        f"  membrane stress      {L['stress'] / 1e6:.2f} MPa   (margin to yield {L['margin']:.2f})",
    ]
    return "\n".join(lines)
