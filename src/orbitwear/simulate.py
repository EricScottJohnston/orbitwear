"""The coupled simulator: OpenFOAM flow, atomic oxygen erosion, CalculiX structure.

One call, :func:`simulate`, runs a mission:

1. **Flow.** dsmcFoam on the panel in the free stream gives drag pressure,
   shear, heat flux and drag coefficient on the ram face. Without OpenFOAM,
   the free-molecular Maxwell model gives the same quantities.
2. **Erosion.** The atomic oxygen flux on the ram face, times the material's
   erosion yield, gives thickness lost per step.
3. **Structure.** At each step, CalculiX solves the panel shell at its
   remaining thickness under the membrane load, the flow pressure and any
   temperature change. Without CalculiX, membrane stress is load over
   thickness.

Each engine runs in ``auto`` (use it when installed), ``on`` (require it) or
``off`` mode. The report says which engine produced each number.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .atmosphere import Flux, Orbit, baseline_flux, exponential_density
from .calculix import CalculiXRunner, ShellLoads, ShellMaterial, read_dat, write_inp
from .erosion import erode
from .geometry import PanelGeometry
from .openfoam import DSMCSettings, FlowConditions, OpenFOAMRunner, layout, read_wall_loads, write_case
from .surface import SurfaceResponse, maxwell_kernel

MODES = ("auto", "on", "off")


@dataclass
class Case:
    """Everything one simulation needs."""
    orbit: Orbit
    duration_s: float
    geometry: PanelGeometry
    material: str                    # erosion yield table key, e.g. "kapton_H"
    shell: ShellMaterial
    membrane_load: float             # N/m
    flow: FlowConditions
    steps: int = 12
    delta_t: float = 0.0             # K, uniform temperature change for the structure
    dsmc: DSMCSettings = field(default_factory=DSMCSettings)
    openfoam_mode: str = "auto"
    calculix_mode: str = "auto"
    openfoam: OpenFOAMRunner = field(default_factory=OpenFOAMRunner.from_env)
    calculix: CalculiXRunner = field(default_factory=CalculiXRunner.from_env)

    def __post_init__(self) -> None:
        for name in ("openfoam_mode", "calculix_mode"):
            if getattr(self, name) not in MODES:
                raise ValueError(f"{name} must be one of {MODES}")


@dataclass
class SimulationResult:
    engines: dict[str, str]
    flow: dict[str, float]
    history: dict[str, list]
    dsmc_layout: dict | None = None

    def final(self) -> dict[str, float]:
        return {k: v[-1] for k, v in self.history.items() if v}


def _use(mode: str, available: bool, name: str) -> bool:
    if mode == "off":
        return False
    if mode == "on" and not available:
        raise RuntimeError(f"{name} is set to 'on' but was not found; run 'orbitwear check'")
    return available


def ram_flux(case: Case) -> Flux:
    """Free-stream flux onto the ram face, from the density table and composition."""
    return baseline_flux(case.orbit, case.geometry.incidence_cos(), case.flow.temperature, case.flow.composition)


def solve_flow(case: Case, out_dir: Path) -> tuple[SurfaceResponse, str, dict | None]:
    """Stage 1 and 2. Returns the response, the engine name, and the DSMC layout if used."""
    if _use(case.openfoam_mode, case.openfoam.available(), "OpenFOAM"):
        cdir = out_dir / "openfoam"
        L = write_case(cdir, case.geometry, case.flow, case.dsmc)
        case.openfoam.run(cdir)
        resp = read_wall_loads(cdir, case.geometry, case.flow)
        lay = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(L).items()}
        return resp, "openfoam dsmcFoam", lay
    resp = maxwell_kernel(ram_flux(case), case.flow.wall_temperature, case.flow.accommodation)
    return resp, "baseline free-molecular Maxwell", None


def solve_structure(case: Case, out_dir: Path, step: int, thickness: float,
                    pressure: float, use_ccx: bool) -> tuple[float, float, float]:
    """Stage 4 at one step: (max von Mises Pa, max displacement m, margin)."""
    if thickness <= 0:
        return math.inf, math.nan, -1.0
    if use_ccx:
        inp = write_inp(out_dir / "calculix" / f"step_{step:03d}" / "panel.inp", case.geometry, thickness,
                        case.shell, ShellLoads(case.membrane_load, pressure, case.delta_t))
        r = read_dat(case.calculix.run(inp), case.shell.yield_strength)
        return r.max_von_mises, r.max_displacement, r.margin
    stress = case.membrane_load / thickness
    margin = case.shell.yield_strength / stress - 1.0 if stress > 0 else math.inf
    return stress, math.nan, margin


def simulate(case: Case, out_dir: str | Path = "orbitwear-out") -> SimulationResult:
    """Run the mission and write ``report.json`` and ``history.csv`` to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    response, flow_engine, lay = solve_flow(case, out)
    use_ccx = _use(case.calculix_mode, case.calculix.available(), "CalculiX")
    struct_engine = "calculix shell" if use_ccx else "baseline membrane stress"

    area = case.geometry.width * case.geometry.height
    q_dyn = 0.5 * exponential_density(case.orbit.altitude_m) * case.orbit.speed ** 2
    flow = {
        "drag_pressure_Pa": response.pressure,
        "shear_Pa": response.shear,
        "heat_flux_W_m2": response.heat_flux,
        "drag_coefficient": response.drag_coefficient,
        "drag_force_N": response.drag_coefficient * q_dyn * area,
    }

    flux = ram_flux(case)
    t0 = case.geometry.thickness
    dt = case.duration_s / case.steps
    h = {k: [] for k in ("time_s", "fluence_atoms_m2", "thickness_m", "von_mises_Pa", "displacement_m", "margin")}
    lost, fluence = 0.0, 0.0
    for step in range(case.steps + 1):
        if step > 0:
            e = erode(case.material, flux, dt, surface_temperature=case.flow.wall_temperature)
            lost += e.thickness_lost
            fluence += e.fluence
        t = max(t0 - lost, 0.0)
        vm, disp, margin = solve_structure(case, out, step, t, response.pressure, use_ccx)
        for k, v in zip(h, (step * dt, fluence, t, vm, disp, margin)):
            h[k].append(v)
        if t <= 0:
            break

    result = SimulationResult(
        engines={"flow": flow_engine, "erosion": "fluence x yield (ASTM E2089 / MISSE 2)", "structure": struct_engine},
        flow=flow, history=h, dsmc_layout=lay,
    )
    _write_outputs(case, result, out)
    return result


def _clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, np.generic):
        return _clean(x.item())
    return x


def _write_outputs(case: Case, r: SimulationResult, out: Path) -> None:
    report = {
        "orbitwear": __import__("orbitwear").__version__,
        "engines": r.engines,
        "inputs": {
            "altitude_km": case.orbit.altitude_m / 1e3,
            "duration_days": case.duration_s / 86400.0,
            "steps": case.steps,
            "material": case.material,
            "panel_m": [case.geometry.width, case.geometry.height],
            "thickness_um": case.geometry.thickness * 1e6,
            "incidence_deg": case.geometry.incidence_deg,
            "membrane_load_N_per_m": case.membrane_load,
            "composition": case.flow.composition,
            "accommodation": case.flow.accommodation,
        },
        "flow": r.flow,
        "final": r.final(),
        "dsmc_layout": r.dsmc_layout,
    }
    (out / "report.json").write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(r.history))
        for row in zip(*r.history.values()):
            w.writerow(["" if (isinstance(v, float) and math.isnan(v)) else v for v in row])


def write_inputs(case: Case, out_dir: str | Path = "orbitwear-out") -> dict[str, Path]:
    """Write the OpenFOAM case and the as-built CalculiX deck without running anything.

    For running the engines somewhere else (a cluster, a container) and
    reading the results back with :func:`orbitwear.openfoam.read_wall_loads`
    and :func:`orbitwear.calculix.read_dat`.
    """
    out = Path(out_dir)
    write_case(out / "openfoam", case.geometry, case.flow, case.dsmc)
    resp = maxwell_kernel(ram_flux(case), case.flow.wall_temperature, case.flow.accommodation)
    inp = write_inp(out / "calculix" / "as_built" / "panel.inp", case.geometry, case.geometry.thickness,
                    case.shell, ShellLoads(case.membrane_load, resp.pressure, case.delta_t))
    return {"openfoam": out / "openfoam", "calculix": inp}


def summary(r: SimulationResult) -> str:
    f = r.final()
    years = f["time_s"] / (365.25 * 86400.0)
    disp = f["displacement_m"]
    lines = [
        f"engines: flow = {r.engines['flow']}; structure = {r.engines['structure']}",
        f"after {years:.2f} years",
        f"  drag pressure        {r.flow['drag_pressure_Pa']:.3e} Pa   (Cd {r.flow['drag_coefficient']:.2f}, drag {r.flow['drag_force_N']:.3e} N)",
        f"  heat flux absorbed   {r.flow['heat_flux_W_m2']:.3f} W/m^2",
        f"  AO fluence           {f['fluence_atoms_m2'] / 1e4:.3e} atoms/cm^2",
        f"  thickness remaining  {f['thickness_m'] * 1e6:.1f} um",
        f"  max von Mises        {f['von_mises_Pa'] / 1e6:.2f} MPa   (margin to yield {f['margin']:.2f})",
    ]
    if not (isinstance(disp, float) and math.isnan(disp)):
        lines.append(f"  max displacement     {disp * 1e3:.3f} mm")
    return "\n".join(lines)
