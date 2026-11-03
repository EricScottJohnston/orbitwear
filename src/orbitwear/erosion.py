"""Stage 3: atomic oxygen erosion.

Baseline: thickness lost = fluence x erosion yield. Fluence is the atomic
oxygen number flux integrated over exposure time. Erosion yields are the
ASTM E2089 ground-test values and the Materials International Space Station
Experiment (MISSE) flight values, per material, at normal incidence.

The full engine adds angle and temperature dependence and synergy with
ultraviolet exposure through :class:`YieldModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .atmosphere import Flux

# Erosion yield, m^3 per incident atom, normal incidence.
# Kapton H: 3.0e-24 cm^3/atom is the ASTM E2089 reference value.
# Others: MISSE 2 PEACE results (de Groh et al., NASA/TM-2008-215482).
EROSION_YIELD_M3_PER_ATOM = {
    "kapton_H":        3.00e-30,   # 3.00e-24 cm^3/atom, ASTM E2089 reference
    "kapton_HN":       2.81e-30,   # MISSE 2
    "teflon_FEP":      2.00e-31,   # MISSE 2 (Teflon FEP, 0.200e-24 cm^3/atom)
    "polyethylene":    3.74e-30,   # MISSE 2
    "epoxy":           4.21e-30,   # MISSE 2 (epoxy, unfilled)
    "carbon_fiber_epoxy": 1.0e-30, # order of magnitude; the epoxy erodes, the fiber slower; needs a source per laminate
    "aluminum":        0.0,        # forms a stable oxide
}


@dataclass(frozen=True)
class ErosionResult:
    fluence: float          # atoms/m^2 over the exposure
    thickness_lost: float   # m
    rate: float             # m/s of thickness loss


class YieldModel(Protocol):
    """Hook for the full engine."""
    def __call__(self, material: str, flux: Flux, surface_temperature: float) -> float: ...


def baseline_yield(material: str, flux: Flux | None = None, surface_temperature: float = 300.0) -> float:
    """Erosion yield, m^3/atom, at normal incidence, from the table."""
    try:
        return EROSION_YIELD_M3_PER_ATOM[material]
    except KeyError:
        raise KeyError(f"no erosion yield for {material!r}; known: {sorted(EROSION_YIELD_M3_PER_ATOM)}") from None


def erode(material: str, flux: Flux, exposure_s: float, yield_model=baseline_yield, surface_temperature: float = 300.0) -> ErosionResult:
    """Thickness lost over ``exposure_s`` seconds under ``flux``.

    thickness = fluence x yield, with fluence = number_flux_O x time. The
    number flux already carries the cosine of incidence, so a surface edge-on
    to the wind erodes at zero rate in the baseline.
    """
    ey = yield_model(material, flux, surface_temperature)
    rate = flux.number_flux_O * ey
    fluence = flux.number_flux_O * exposure_s
    return ErosionResult(fluence=fluence, thickness_lost=rate * exposure_s, rate=rate)
