"""Read a case from a TOML file. See ``examples/kapton_400km.toml`` for every key."""

from __future__ import annotations

import tomllib
from pathlib import Path

from .atmosphere import DEFAULT_COMPOSITION, Orbit
from .calculix import CalculiXRunner, ShellMaterial
from .geometry import PanelGeometry
from .openfoam import DSMCSettings, FlowConditions, OpenFOAMRunner
from .simulate import Case


def _get(d: dict, key: str, default=None, required: bool = False):
    if key in d:
        return d[key]
    if required:
        raise KeyError(f"missing required key: {key}")
    return default


def load(path: str | Path) -> Case:
    with open(path, "rb") as f:
        c = tomllib.load(f)
    orbit = c.get("orbit", {})
    mission = c.get("mission", {})
    panel = c.get("panel", {})
    struct = c.get("structure", {})
    atm = c.get("atmosphere", {})
    of = c.get("openfoam", {})
    ccx = c.get("calculix", {})

    geom = PanelGeometry(
        width=float(_get(panel, "width_m", 1.0)),
        height=float(_get(panel, "height_m", 1.0)),
        thickness=float(_get(panel, "thickness_um", required=True)) * 1e-6,
        incidence_deg=float(_get(panel, "incidence_deg", 0.0)),
        box_depth=float(_get(panel, "flow_box_depth_m", 0.02)),
        nx=int(_get(struct, "mesh_nx", 8)),
        ny=int(_get(struct, "mesh_ny", 8)),
    )
    expansion = _get(struct, "expansion_per_K")
    shell = ShellMaterial(
        youngs_modulus=float(_get(struct, "youngs_modulus_GPa", required=True)) * 1e9,
        poisson_ratio=float(_get(struct, "poisson_ratio", 0.3)),
        yield_strength=float(_get(struct, "yield_strength_MPa", required=True)) * 1e6,
        expansion=float(expansion) if expansion is not None else None,
    )
    flow = FlowConditions(
        orbit=Orbit(float(_get(orbit, "altitude_km", required=True)) * 1e3),
        temperature=float(_get(atm, "temperature_K", 1000.0)),
        wall_temperature=float(_get(panel, "wall_temperature_K", 300.0)),
        accommodation=float(_get(panel, "accommodation", 0.9)),
        composition=dict(_get(atm, "composition", DEFAULT_COMPOSITION)),
    )
    dsmc = DSMCSettings(**{k: v for k, v in of.items() if k in DSMCSettings.__dataclass_fields__})
    of_runner = OpenFOAMRunner(template=of["command"]) if "command" in of else OpenFOAMRunner.from_env()
    ccx_runner = CalculiXRunner(executable=ccx["command"]) if "command" in ccx else CalculiXRunner.from_env()

    return Case(
        orbit=flow.orbit,
        duration_s=float(_get(mission, "duration_days", 365.25)) * 86400.0,
        steps=int(_get(mission, "steps", 12)),
        geometry=geom,
        material=str(_get(panel, "material", required=True)),
        shell=shell,
        membrane_load=float(_get(struct, "membrane_load_N_per_m", 0.0)),
        delta_t=float(_get(struct, "delta_T_K", 0.0)),
        flow=flow,
        dsmc=dsmc,
        openfoam_mode=str(_get(of, "enabled", "auto")),
        calculix_mode=str(_get(ccx, "enabled", "auto")),
        openfoam=of_runner,
        calculix=ccx_runner,
    )
