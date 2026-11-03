"""Stage 4: structural consequence of the thinned, loaded section.

Baseline: a thin-walled element carrying a membrane load per unit width.
Erosion removes thickness from the wind-facing face; the remaining section
carries the same load at higher stress. Drag pressure is applied as a
uniform lateral pressure and reported, which for a small panel is a check,
not a driver: at 400 km the drag pressure is of order 1e-4 Pa.

The full engine is a finite element model (CalculiX, FEniCS) that takes the
per-element pressure, heat flux and thickness fields through
:class:`StructuralModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .surface import SurfaceResponse


@dataclass(frozen=True)
class Section:
    thickness: float            # m, as built
    membrane_load: float        # N per metre of width carried by the element
    yield_strength: float       # Pa
    youngs_modulus: float       # Pa


@dataclass(frozen=True)
class StructuralResult:
    thickness_remaining: float  # m
    stress: float               # Pa, membrane stress on the remaining section
    stress_as_built: float      # Pa
    margin: float               # yield_strength / stress - 1
    lateral_pressure: float     # Pa, drag pressure applied to the face


class StructuralModel(Protocol):
    """Hook for the full engine."""
    def __call__(self, section: Section, thickness_lost: float, response: SurfaceResponse) -> StructuralResult: ...


def baseline_structure(section: Section, thickness_lost: float, response: SurfaceResponse) -> StructuralResult:
    """Membrane stress on the section that remains after erosion."""
    t_rem = max(section.thickness - thickness_lost, 0.0)
    if t_rem <= 0.0:
        return StructuralResult(0.0, float("inf"), section.membrane_load / section.thickness, -1.0, response.pressure)
    stress = section.membrane_load / t_rem
    stress0 = section.membrane_load / section.thickness
    margin = section.yield_strength / stress - 1.0 if stress > 0 else float("inf")
    return StructuralResult(
        thickness_remaining=t_rem,
        stress=stress,
        stress_as_built=stress0,
        margin=margin,
        lateral_pressure=response.pressure,
    )
