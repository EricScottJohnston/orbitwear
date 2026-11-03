"""Stage 2: what each arriving particle does at the surface.

Baseline: Maxwell gas-surface kernel. A fraction ``alpha`` of arrivals
accommodates fully to the wall (diffuse re-emission at wall temperature) and
the rest reflects specularly. From that, momentum and energy transfer per
element: drag pressure and heating.

The full engine replaces the kernel with energy-dependent scattering
(Cercignani-Lampis-Lord or measured kernels) through :class:`ScatteringKernel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .atmosphere import Flux, K_B, M_AIR


@dataclass(frozen=True)
class SurfaceResponse:
    """Per unit area of one element."""
    pressure: float          # Pa, normal force per area from the wind
    shear: float             # Pa, tangential force per area
    heat_flux: float         # W/m^2 absorbed from the incident stream
    drag_coefficient: float  # referenced to the element area and 0.5 rho V^2


class ScatteringKernel(Protocol):
    """Hook for the full engine."""
    def __call__(self, flux: Flux, wall_temperature: float) -> SurfaceResponse: ...


def maxwell_kernel(flux: Flux, wall_temperature: float = 300.0, alpha: float = 0.9) -> SurfaceResponse:
    """Maxwell accommodation model in the hyperthermal limit.

    For a flat element with the wind at incidence angle theta (cos theta =
    ``flux.incidence_cos``), with incident speed V and mass flux m_dot:

    * specular fraction (1 - alpha): normal momentum 2 m_dot V cos(theta), no shear
    * diffuse fraction alpha: normal momentum m_dot V cos(theta) plus the
      re-emission term at wall temperature, shear m_dot V sin(theta)

    Heating: the diffuse fraction deposits the incident kinetic energy less what
    the re-emitted gas carries away at wall temperature. The specular fraction
    deposits nothing. This is the standard free-molecular flat-plate result
    (Schaaf and Chambre, "Flow of Rarefied Gases", 1961) with the thermal
    velocity terms of the incident stream dropped, which is accurate to a few
    percent at LEO speed ratios.

    ``alpha`` near 0.9 to 1.0 is the usual engineering value for engineering
    surfaces in LEO (Moe and Moe, Planetary and Space Science 53, 2005).
    """
    c = max(flux.incidence_cos, 0.0)
    s = float(np.sqrt(max(0.0, 1.0 - c * c)))
    v = flux.speed
    m_dot = flux.mass_flux                       # kg/m^2/s, already includes cos
    if m_dot == 0.0:
        return SurfaceResponse(0.0, 0.0, 0.0, 0.0)

    # Re-emission speed of diffusely scattered gas at wall temperature.
    v_wall = float(np.sqrt(np.pi * K_B * wall_temperature / (2.0 * M_AIR)))

    p_specular = 2.0 * m_dot * v * c
    p_diffuse = m_dot * (v * c + v_wall)
    pressure = (1.0 - alpha) * p_specular + alpha * p_diffuse
    shear = alpha * m_dot * v * s

    e_in = 0.5 * m_dot * v * v                   # W/m^2 incident kinetic energy
    e_out = alpha * m_dot * 2.0 * K_B * wall_temperature / M_AIR
    heat = alpha * e_in - e_out

    q = 0.5 * flux.mass_density * v * v
    cd = (pressure * c + shear * s) / q if q > 0 else 0.0
    return SurfaceResponse(pressure=pressure, shear=shear, heat_flux=max(heat, 0.0), drag_coefficient=cd)
