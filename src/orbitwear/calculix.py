"""CalculiX adapter: the structural side of the coupling.

Writes a shell model of the panel at its current (eroded) thickness, runs
``ccx``, and reads stresses and displacements from the ``.dat`` file.

The model, in the panel's own frame (panel in the x-y plane, ram normal +z):

* S3 triangle shells from :meth:`orbitwear.geometry.PanelGeometry.shell_mesh`
* simply supported on all four edges (z fixed), the edge at x = -w/2 held in x,
  one corner held in y to remove rigid-body motion
* the membrane load per unit width applied in +x on the edge at x = +w/2
* the flow pressure from the ram face applied as a shell pressure ``P``,
  positive pushing in -z (into the ram face), as Abaqus and CalculiX define it
* optionally a uniform temperature change, with thermal expansion

The ``.dat`` output carries von Mises inputs (six stress components) per
integration point and three displacement components per node.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import PanelGeometry


@dataclass(frozen=True)
class ShellMaterial:
    youngs_modulus: float            # Pa
    poisson_ratio: float
    yield_strength: float            # Pa
    expansion: float | None = None   # 1/K, linear thermal expansion


@dataclass(frozen=True)
class ShellLoads:
    membrane_load: float             # N per metre of panel height, along +x at x = +w/2
    pressure: float = 0.0            # Pa, on the ram face, pushing into it
    delta_t: float = 0.0             # K, uniform temperature change


@dataclass(frozen=True)
class ShellResult:
    max_von_mises: float             # Pa
    max_displacement: float          # m, largest nodal displacement magnitude
    margin: float                    # yield / max_von_mises - 1


def write_inp(path: str | os.PathLike, geom: PanelGeometry, thickness: float,
              material: ShellMaterial, loads: ShellLoads) -> Path:
    """Write the CalculiX input deck. ``thickness`` is the current shell thickness, m."""
    if thickness <= 0:
        raise ValueError("thickness must be positive; the panel has eroded through")
    nx, ny = geom.nx, geom.ny
    us = np.linspace(-geom.width / 2, geom.width / 2, nx + 1)
    vs = np.linspace(-geom.height / 2, geom.height / 2, ny + 1)
    node_id = lambda i, j: j * (nx + 1) + i + 1     # 1-based
    lines = ["*HEADING", "orbitwear panel shell"]

    lines.append("*NODE, NSET=NALL")
    for j, y in enumerate(vs):
        for i, x in enumerate(us):
            lines.append(f"{node_id(i, j)}, {x:.9e}, {y:.9e}, 0.0")

    lines.append("*ELEMENT, TYPE=S3, ELSET=EALL")
    e = 1
    for j in range(ny):
        for i in range(nx):
            p, p1 = node_id(i, j), node_id(i + 1, j)
            q, q1 = node_id(i, j + 1), node_id(i + 1, j + 1)
            lines.append(f"{e}, {p}, {p1}, {q1}")
            lines.append(f"{e + 1}, {p}, {q1}, {q}")
            e += 2

    def nset(name: str, ids: list[int]) -> None:
        lines.append(f"*NSET, NSET={name}")
        for k in range(0, len(ids), 12):
            lines.append(", ".join(str(n) for n in ids[k:k + 12]))

    umin = [node_id(0, j) for j in range(ny + 1)]
    umax = [node_id(nx, j) for j in range(ny + 1)]
    vmin = [node_id(i, 0) for i in range(nx + 1)]
    vmax = [node_id(i, ny) for i in range(nx + 1)]
    edge = sorted(set(umin + umax + vmin + vmax))
    nset("EDGE_UMIN", umin)
    nset("EDGE_UMAX", umax)
    nset("EDGE_ALL", edge)

    lines += [
        "*MATERIAL, NAME=PANEL",
        "*ELASTIC",
        f"{material.youngs_modulus:.9e}, {material.poisson_ratio:.6g}",
    ]
    thermal = material.expansion is not None and loads.delta_t != 0.0
    if thermal:
        lines += ["*EXPANSION, ZERO=0.", f"{material.expansion:.9e}"]
    lines += [
        "*SHELL SECTION, ELSET=EALL, MATERIAL=PANEL",
        f"{thickness:.9e}",
        "*BOUNDARY",
        "EDGE_ALL, 3, 3",
        "EDGE_UMIN, 1, 1",
        f"{node_id(0, 0)}, 2, 2",
    ]
    if thermal:
        lines += ["*INITIAL CONDITIONS, TYPE=TEMPERATURE", "NALL, 0."]

    lines += ["*STEP", "*STATIC"]
    dv = geom.height / ny
    lines.append("*CLOAD")
    for j, n in enumerate(umax):
        trib = dv / 2 if j in (0, ny) else dv
        lines.append(f"{n}, 1, {loads.membrane_load * trib:.9e}")
    if loads.pressure != 0.0:
        lines += ["*DLOAD", f"EALL, P, {loads.pressure:.9e}"]
    if thermal:
        lines += ["*TEMPERATURE", f"NALL, {loads.delta_t:.9e}"]
    lines += [
        "*NODE PRINT, NSET=NALL",
        "U",
        "*EL PRINT, ELSET=EALL",
        "S",
        "*NODE FILE",
        "U",
        "*EL FILE",
        "S",
        "*END STEP",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _blocks(text: str, keyword: str) -> list[list[list[float]]]:
    """Numeric rows of every ``.dat`` block whose header starts with ``keyword``."""
    blocks, cur, inside = [], None, False
    for ln in text.splitlines():
        s = ln.strip()
        if s.lower().startswith(keyword):
            cur, inside = [], True
            blocks.append(cur)
            continue
        if inside:
            if not s:
                if cur:            # blank line after data ends the block
                    inside = False
                continue
            if s[0].isalpha():
                inside = False
                continue
            cur.append([float(x) for x in re.findall(_FLOAT, s)])
    return blocks


def von_mises(sxx, syy, szz, sxy, sxz, syz) -> float:
    return float(np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                         + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)))


def read_dat(path: str | os.PathLike, yield_strength: float) -> ShellResult:
    """Largest von Mises stress and nodal displacement in a CalculiX ``.dat`` file."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    stress_blocks = _blocks(text, "stresses")
    disp_blocks = _blocks(text, "displacements")
    if not stress_blocks or not stress_blocks[-1]:
        raise ValueError(f"{path}: no stress block")
    vm = max(von_mises(*row[2:8]) for row in stress_blocks[-1] if len(row) >= 8)
    disp = 0.0
    if disp_blocks and disp_blocks[-1]:
        disp = max(float(np.linalg.norm(row[1:4])) for row in disp_blocks[-1] if len(row) >= 4)
    margin = yield_strength / vm - 1.0 if vm > 0 else float("inf")
    return ShellResult(max_von_mises=vm, max_displacement=disp, margin=margin)


@dataclass
class CalculiXRunner:
    """How to launch CalculiX. ``ORBITWEAR_CCX`` names the executable (default ``ccx``)."""
    executable: str = "ccx"
    timeout_s: float | None = 600

    @classmethod
    def from_env(cls) -> "CalculiXRunner":
        return cls(executable=os.environ.get("ORBITWEAR_CCX", "ccx"))

    def available(self) -> bool:
        return shutil.which(self.executable) is not None or Path(self.executable).is_file()

    def run(self, inp: str | os.PathLike) -> Path:
        inp = Path(inp).resolve()
        job = inp.with_suffix("")
        with (inp.parent / f"{job.name}.log").open("w", encoding="utf-8") as log:
            done = subprocess.run([self.executable, "-i", job.name], cwd=inp.parent,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=self.timeout_s)
        dat = job.with_suffix(".dat")
        if done.returncode != 0 or not dat.is_file():
            raise RuntimeError(f"CalculiX failed on {inp.name} (see {job.name}.log)")
        return dat
