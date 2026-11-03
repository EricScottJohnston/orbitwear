"""Stage 1: flux of atmospheric species onto a surface element.

Baseline: free-molecular flux from an exponential density model. The full
engine for this stage is a DSMC code (SPARTA) driven by NRLMSIS densities;
it plugs in through :class:`FluxModel`.

Units are SI throughout: metres, seconds, kilograms, kelvin, per cubic metre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

MU_EARTH = 3.986004418e14   # m^3/s^2
R_EARTH = 6.371e6           # m
K_B = 1.380649e-23          # J/K
M_O = 2.6567e-26            # kg, atomic oxygen
M_AIR = 2.66e-26            # kg, mean particle mass at 400 km (mostly O)

# Exponential density anchors, US Standard Atmosphere 1976 (Vallado, Fundamentals
# of Astrodynamics, table 8-4): base altitude, nominal density, scale height.
# These are mean values; real thermospheric density varies by a factor of ten
# with solar activity, which is why the full engine takes NRLMSIS.
_DENSITY_TABLE = [
    # alt_km, rho0 kg/m^3, H km
    (150, 2.070e-9, 22.523),
    (180, 5.464e-10, 29.740),
    (200, 2.789e-10, 37.105),
    (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628),
    (350, 9.518e-12, 53.298),
    (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828),
    (500, 6.967e-13, 63.822),
    (600, 1.454e-13, 71.835),
    (700, 3.614e-14, 88.667),
    (800, 1.170e-14, 124.64),
    (900, 5.245e-15, 181.05),
    (1000, 3.019e-15, 268.00),
]

# Atomic oxygen share of particles by number near 400 km, solar mean. The
# thermosphere above 200 km is mostly atomic oxygen (NRLMSIS 2.0 climatology);
# a single fraction is a baseline, not a model.
O_NUMBER_FRACTION = 0.9


@dataclass(frozen=True)
class Orbit:
    """Circular orbit by altitude."""
    altitude_m: float

    @property
    def speed(self) -> float:
        """Orbital speed, m/s, for a circular orbit."""
        return float(np.sqrt(MU_EARTH / (R_EARTH + self.altitude_m)))


@dataclass(frozen=True)
class Flux:
    """What arrives at one surface element, per unit area per second."""
    mass_density: float          # kg/m^3 at the orbit
    number_density_O: float      # atomic oxygen, 1/m^3
    speed: float                 # m/s, relative wind speed
    incidence_cos: float         # cos of angle between wind and surface normal
    temperature: float           # K, gas kinetic temperature

    @property
    def number_flux_O(self) -> float:
        """Atomic oxygen arriving per m^2 per s, free-molecular ram flux."""
        return self.number_density_O * self.speed * max(self.incidence_cos, 0.0)

    @property
    def mass_flux(self) -> float:
        """All species, kg per m^2 per s."""
        return self.mass_density * self.speed * max(self.incidence_cos, 0.0)


class FluxModel(Protocol):
    """Hook for the full engine. A DSMC run implements this signature."""
    def __call__(self, orbit: Orbit, incidence_cos: float) -> Flux: ...


def exponential_density(altitude_m: float) -> float:
    """Mass density, kg/m^3, by piecewise exponential interpolation."""
    alt_km = altitude_m / 1000.0
    if alt_km < _DENSITY_TABLE[0][0] or alt_km > _DENSITY_TABLE[-1][0]:
        raise ValueError(f"altitude {alt_km:.0f} km outside the table (150 to 1000 km)")
    base = _DENSITY_TABLE[0]
    for row in _DENSITY_TABLE:
        if row[0] <= alt_km:
            base = row
    alt0, rho0, h = base
    return rho0 * float(np.exp(-(alt_km - alt0) / h))


def baseline_flux(orbit: Orbit, incidence_cos: float = 1.0, temperature: float = 1000.0) -> Flux:
    """Free-molecular flux from the exponential density model.

    The gas is taken as moving at orbital speed relative to the spacecraft,
    which holds when the speed ratio (orbital speed over thermal speed) is
    large. At 400 km it is about 8, so thermal spread adds a few percent
    the baseline ignores.
    """
    rho = exponential_density(orbit.altitude_m)
    n_total = rho / M_AIR
    return Flux(
        mass_density=rho,
        number_density_O=O_NUMBER_FRACTION * n_total,
        speed=orbit.speed,
        incidence_cos=incidence_cos,
        temperature=temperature,
    )
