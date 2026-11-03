"""OpenFOAM adapter: the flow side of the coupling.

Writes a dsmcFoam case for a panel in the free stream, runs it, and reads the
wall force and heat flux back as a :class:`orbitwear.surface.SurfaceResponse`.

Written against ESI OpenFOAM (openfoam.com) v2306 to v2406, whose
``discreteMethods/dsmc`` library and tutorials (``freeSpaceStream``) this
case follows. dsmcFoam is not in every OpenFOAM distribution; check with
``orbitwear check``.

The case:

* ``system/blockMeshDict``  a box around the panel, one ``freestream`` patch
* ``constant/triSurface/panel.stl``  the panel as a thin closed box with three
  named solids, which snappyHexMesh turns into wall patches ``ram``, ``back``
  and ``sides``
* ``constant/dsmcProperties``  species, VHS collisions, Maxwell
  diffuse/specular walls (``MixedDiffuseSpecular``), free-stream inflow
* ``system/dsmcInitialiseDict``  initial fill at free-stream conditions
* ``system/controlDict``  dsmcFoam, time averaging after the flow settles, and
  one ``surfaceFieldValue`` per wall patch integrating ``qMean`` and ``fDMean``

``fD`` is the force the gas exerts on the wall per unit area, ``q`` the heat
flux into the wall. The run sequence is ``blockMesh``, ``snappyHexMesh
-overwrite``, ``dsmcInitialise``, ``dsmcFoam``.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

import numpy as np

from .atmosphere import DEFAULT_COMPOSITION, K_B, SPECIES, Orbit, exponential_density, mean_mass, number_densities
from .geometry import PanelGeometry
from .surface import SurfaceResponse

WALL_PATCHES = ("ram", "back", "sides")
RUN_SEQUENCE = ("blockMesh", "snappyHexMesh -overwrite", "dsmcInitialise", "dsmcFoam")


@dataclass
class DSMCSettings:
    """Numerical settings for the dsmcFoam case."""
    target_particles: float = 2.0e5     # simulated particles in the domain
    flow_throughs: float = 5.0          # run length, in domain crossings at orbital speed
    average_after: float = 2.0          # start time-averaging after this many crossings
    base_cells: int = 24                # blockMesh cells along the longest domain side
    surface_level: int = 4              # snappyHexMesh refinement level at the panel
    step_fraction: float = 1.0          # time step, as a fraction of the finest cell crossing time
    padding: float = 0.5                # domain padding around the panel, in panel sizes
    upstream_padding: float = 1.0       # extra padding on the upstream side


@dataclass
class FlowConditions:
    orbit: Orbit
    temperature: float = 1000.0         # K, free-stream kinetic temperature
    wall_temperature: float = 300.0     # K
    accommodation: float = 0.9          # diffuse fraction of wall interactions
    composition: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COMPOSITION))

    @property
    def velocity(self) -> np.ndarray:
        """Free-stream gas velocity in the panel frame: the wind blows in -x."""
        return np.array([-self.orbit.speed, 0.0, 0.0])


# --------------------------------------------------------------------------
# Dictionary writing
# --------------------------------------------------------------------------

def _header(cls: str, obj: str, location: str | None = None) -> str:
    loc = f'    location    "{location}";\n' if location else ""
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "  Written by orbitwear. Edit the orbitwear inputs, not this file.\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {cls};\n"
        f"{loc}"
        f"    object      {obj};\n"
        "}\n\n"
    )


def _vec(v) -> str:
    return f"({v[0]:.9g} {v[1]:.9g} {v[2]:.9g})"


@dataclass
class CaseLayout:
    """Numbers derived from the geometry and settings, kept for the report."""
    domain_min: np.ndarray
    domain_max: np.ndarray
    base_cell: float
    finest_cell: float
    delta_t: float
    end_time: float
    average_start: float
    n_equivalent: float
    number_densities: dict[str, float]


def layout(geom: PanelGeometry, flow: FlowConditions, s: DSMCSettings) -> CaseLayout:
    lo, hi = geom.bounding_box()
    size = max(geom.width, geom.height)
    pad = s.padding * size
    lo = lo - pad
    hi = hi + pad
    hi[0] += s.upstream_padding * size        # wind comes from +x
    ext = hi - lo
    base = float(ext.max()) / s.base_cells
    finest = base / (2 ** s.surface_level)
    speed = flow.orbit.speed
    dt = s.step_fraction * finest / speed
    crossing = float(ext[0]) / speed
    nd = number_densities(flow.orbit.altitude_m, flow.composition)
    volume = float(np.prod(ext))
    n_eq = sum(nd.values()) * volume / s.target_particles
    return CaseLayout(
        domain_min=lo, domain_max=hi, base_cell=base, finest_cell=finest,
        delta_t=dt, end_time=s.flow_throughs * crossing, average_start=s.average_after * crossing,
        n_equivalent=n_eq, number_densities=nd,
    )


def _block_mesh(L: CaseLayout) -> str:
    lo, hi = L.domain_min, L.domain_max
    n = np.maximum(1, np.round((hi - lo) / L.base_cell)).astype(int)
    pts = [
        (lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
        (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2]),
    ]
    verts = "\n".join(f"    {_vec(p)}" for p in pts)
    return _header("dictionary", "blockMeshDict") + f"""scale 1;

vertices
(
{verts}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1)
);

boundary
(
    freestream
    {{
        type patch;
        faces
        (
            (0 4 7 3)
            (1 2 6 5)
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);
"""


def _snappy(L: CaseLayout, s: DSMCSettings) -> str:
    # A point in the gas: inside the domain, off any cell face, away from the panel.
    inside = L.domain_min + 0.1137 * (L.domain_max - L.domain_min)
    lvl = s.surface_level
    regions = "\n".join(
        f"            {p} {{ level ({lvl} {lvl}); patchInfo {{ type wall; }} }}" for p in WALL_PATCHES
    )
    names = "\n".join(f"            {p} {{ name {p}; }}" for p in WALL_PATCHES)
    return _header("dictionary", "snappyHexMeshDict") + f"""castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    panel.stl
    {{
        type triSurfaceMesh;
        name panel;
        regions
        {{
{names}
        }}
    }}
}}

castellatedMeshControls
{{
    maxLocalCells       2000000;
    maxGlobalCells      8000000;
    minRefinementCells  0;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;
    features            ();
    refinementSurfaces
    {{
        panel
        {{
            level ({lvl} {lvl});
            patchInfo {{ type wall; }}
            regions
            {{
{regions}
            }}
        }}
    }}
    resolveFeatureAngle 30;
    refinementRegions   {{}}
    locationInMesh      {_vec(inside)};
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch        3;
    tolerance           2.0;
    nSolveIter          30;
    nRelaxIter          5;
    nFeatureSnapIter    10;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes       true;
    layers              {{}}
    expansionRatio      1.0;
    finalLayerThickness 0.3;
    minThickness        0.1;
    nGrow               0;
    featureAngle        60;
    nRelaxIter          3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-15;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
}}

mergeTolerance 1e-6;
debug 0;
"""


def _dsmc_properties(L: CaseLayout, flow: FlowConditions) -> str:
    species = list(L.number_densities)
    dens = "\n".join(f"        {sp} {L.number_densities[sp]:.6e};" for sp in species)
    mols = []
    for sp in species:
        m, d, dof, om = SPECIES[sp]
        mols.append(
            f"    {sp}\n    {{\n        mass {m:.6e};\n        diameter {d:.6e};\n"
            f"        internalDegreesOfFreedom {dof};\n        omega {om};\n    }}"
        )
    mol_block = "\n".join(mols)
    return _header("dictionary", "dsmcProperties", "constant") + f"""nEquivalentParticles {L.n_equivalent:.6e};

WallInteractionModel MixedDiffuseSpecular;

MixedDiffuseSpecularCoeffs
{{
    diffuseFraction {flow.accommodation:.6g};
}}

BinaryCollisionModel VariableHardSphere;

VariableHardSphereCoeffs
{{
    Tref 273;
}}

InflowBoundaryModel FreeStream;

FreeStreamCoeffs
{{
    numberDensities
    {{
{dens}
    }}
}}

typeIdList ({" ".join(species)});

moleculeProperties
{{
{mol_block}
}}
"""


def _dsmc_initialise(L: CaseLayout, flow: FlowConditions) -> str:
    dens = "\n".join(f"    {sp} {n:.6e};" for sp, n in L.number_densities.items())
    return _header("dictionary", "dsmcInitialiseDict") + f"""numberDensities
{{
{dens}
}};

massTemperature {flow.temperature:.6g};

velocity {_vec(flow.velocity)};
"""


def _control_dict(L: CaseLayout) -> str:
    write_every = L.end_time / 5.0
    walls = "\n".join(
        f"""    wall_{p}
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        log             false;
        writeFields     false;
        regionType      patch;
        name            {p};
        operation       areaIntegrate;
        fields          (qMean fDMean);
    }}"""
        for p in WALL_PATCHES
    )
    mean = "\n".join(
        f"            {f} {{ mean on; prime2Mean off; base time; }}"
        for f in ("rhoN", "rhoM", "dsmcRhoN", "momentum", "linearKE", "internalE", "iDof", "q", "fD")
    )
    return _header("dictionary", "controlDict") + f"""application     dsmcFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {L.end_time:.9g};
deltaT          {L.delta_t:.9g};
writeControl    runTime;
writeInterval   {write_every:.9g};
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   10;
runTimeModifiable true;
adjustTimeStep  no;

functions
{{
    fieldAverage1
    {{
        type            fieldAverage;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        timeStart       {L.average_start:.9g};
        fields
        (
{mean}
        );
    }}

    dsmcFields1
    {{
        type            dsmcFields;
        libs            (lagrangianFunctionObjects);
        writeControl    writeTime;
        timeStart       {L.average_start:.9g};
    }}

{walls}
}}
"""


_FIELDS = {
    # name: (class, dimensions, internal value)
    "dsmcRhoN":  ("volScalarField", "[0 -3 0 0 0 0 0]", "0"),
    "rhoN":      ("volScalarField", "[0 -3 0 0 0 0 0]", "0"),
    "rhoM":      ("volScalarField", "[1 -3 0 0 0 0 0]", "0"),
    "linearKE":  ("volScalarField", "[1 -1 -2 0 0 0 0]", "0"),
    "internalE": ("volScalarField", "[1 -1 -2 0 0 0 0]", "0"),
    "iDof":      ("volScalarField", "[0 -3 0 0 0 0 0]", "0"),
    "q":         ("volScalarField", "[1 0 -3 0 0 0 0]", "0"),
    "momentum":  ("volVectorField", "[1 -2 -1 0 0 0 0]", "(0 0 0)"),
    "fD":        ("volVectorField", "[1 -1 -2 0 0 0 0]", "(0 0 0)"),
}

_WALL_REGEX = '"(' + "|".join(WALL_PATCHES) + ')"'


def _zero_fields(flow: FlowConditions) -> dict[str, str]:
    out = {}
    for name, (cls, dims, val) in _FIELDS.items():
        out[name] = _header(cls, name, "0") + f"""dimensions      {dims};

internalField   uniform {val};

boundaryField
{{
    ".*"
    {{
        type            calculated;
        value           uniform {val};
    }}
}}
"""
    out["boundaryT"] = _header("volScalarField", "boundaryT", "0") + f"""dimensions      [0 0 0 1 0 0 0];

internalField   uniform 0;

boundaryField
{{
    freestream
    {{
        type            fixedValue;
        value           uniform {flow.temperature:.6g};
    }}
    {_WALL_REGEX}
    {{
        type            fixedValue;
        value           uniform {flow.wall_temperature:.6g};
    }}
}}
"""
    out["boundaryU"] = _header("volVectorField", "boundaryU", "0") + f"""dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    freestream
    {{
        type            fixedValue;
        value           uniform {_vec(flow.velocity)};
    }}
    {_WALL_REGEX}
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
}}
"""
    return out


_FV_SCHEMES = _header("dictionary", "fvSchemes") + """ddtSchemes { default none; }
gradSchemes { default none; }
divSchemes { default none; }
laplacianSchemes { default none; }
interpolationSchemes { default linear; }
snGradSchemes { default none; }
"""

_FV_SOLUTION = _header("dictionary", "fvSolution") + "\n"


def write_case(case_dir: str | os.PathLike, geom: PanelGeometry, flow: FlowConditions,
               settings: DSMCSettings | None = None) -> CaseLayout:
    """Write a complete dsmcFoam case to ``case_dir`` and return its layout."""
    s = settings or DSMCSettings()
    case = Path(case_dir)
    for sub in ("0", "constant/triSurface", "system"):
        (case / sub).mkdir(parents=True, exist_ok=True)
    L = layout(geom, flow, s)
    files = {
        "system/blockMeshDict": _block_mesh(L),
        "system/snappyHexMeshDict": _snappy(L, s),
        "system/controlDict": _control_dict(L),
        "system/dsmcInitialiseDict": _dsmc_initialise(L, flow),
        "system/fvSchemes": _FV_SCHEMES,
        "system/fvSolution": _FV_SOLUTION,
        "constant/dsmcProperties": _dsmc_properties(L, flow),
        "constant/triSurface/panel.stl": geom.box_stl(),
    }
    for name, text in _zero_fields(flow).items():
        files[f"0/{name}"] = text
    for rel, text in files.items():
        (case / rel).write_text(text, encoding="utf-8", newline="\n")
    return L


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def wsl_path(path: str | os.PathLike) -> str:
    """C:\\Users\\x\\case -> /mnt/c/Users/x/case, for running OpenFOAM inside WSL."""
    p = PureWindowsPath(str(path))
    if not p.drive:
        p = PureWindowsPath(os.path.abspath(path))
    if not p.drive:
        return Path(path).as_posix()
    return "/mnt/" + p.drive.rstrip(":").lower() + "/" + "/".join(p.parts[1:])


@dataclass
class OpenFOAMRunner:
    """How to launch OpenFOAM commands.

    ``template`` is a shell command with ``{cmd}`` for the OpenFOAM command and
    ``{case}`` / ``{case_wsl}`` for the case directory. The default runs the
    command directly, which works where OpenFOAM's environment is already
    sourced. Examples:

    * WSL:    ``wsl -- bash -lc "source /usr/lib/openfoam/openfoam2406/etc/bashrc && cd {case_wsl} && {cmd}"``
    * Docker: ``docker run --rm -v {case}:/case -w /case opencfd/openfoam-default:2406 {cmd}``

    The environment variable ``ORBITWEAR_OPENFOAM`` sets the template.
    """
    template: str = "{cmd}"
    timeout_s: float | None = None

    @classmethod
    def from_env(cls) -> "OpenFOAMRunner":
        return cls(template=os.environ.get("ORBITWEAR_OPENFOAM", "{cmd}"))

    def available(self) -> bool:
        if self.template != "{cmd}":
            return True          # the user configured a launcher; trust it
        return shutil.which("dsmcFoam") is not None and shutil.which("blockMesh") is not None

    def run(self, case_dir: str | os.PathLike, commands=RUN_SEQUENCE) -> None:
        case = Path(case_dir).resolve()
        log = (case / "orbitwear-run.log").open("w", encoding="utf-8")
        try:
            for cmd in commands:
                line = self.template.format(cmd=cmd, case=str(case), case_wsl=wsl_path(case))
                log.write(f"$ {line}\n")
                log.flush()
                argv = line if os.name == "nt" else shlex.split(line)
                done = subprocess.run(argv, cwd=case, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=self.timeout_s, shell=(os.name == "nt"))
                if done.returncode != 0:
                    raise RuntimeError(f"OpenFOAM step failed: {cmd} (see {case / 'orbitwear-run.log'})")
        finally:
            log.close()


# --------------------------------------------------------------------------
# Reading results
# --------------------------------------------------------------------------

_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True)
class PatchLoads:
    patch: str
    area: float | None      # m^2, from the file header when OpenFOAM wrote it
    heat: float             # W, area integral of qMean
    force: np.ndarray       # N, area integral of fDMean


def read_surface_field_value(path: str | os.PathLike, patch: str) -> PatchLoads:
    """Last row of a ``surfaceFieldValue.dat`` written with fields (qMean fDMean).

    Handles both ways OpenFOAM writes vectors: ``(x y z)`` in one column, or
    three separate columns.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    area = None
    m = re.search(r"#\s*Area\s*:\s*(" + _FLOAT + ")", text)
    if m:
        area = float(m.group(1))
    rows = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"{path}: no data rows")
    nums = [float(x) for x in re.findall(_FLOAT, rows[-1].replace("(", " ").replace(")", " "))]
    if len(nums) < 5:
        raise ValueError(f"{path}: expected time, q and three force components, got {rows[-1]!r}")
    return PatchLoads(patch=patch, area=area, heat=nums[1], force=np.array(nums[2:5]))


def find_patch_files(case_dir: str | os.PathLike) -> dict[str, Path]:
    """Latest ``surfaceFieldValue.dat`` per wall patch under ``postProcessing``."""
    out = {}
    base = Path(case_dir) / "postProcessing"
    for p in WALL_PATCHES:
        cands = sorted((base / f"wall_{p}").glob("*/surfaceFieldValue*.dat"),
                       key=lambda f: _time_key(f.parent.name))
        if cands:
            out[p] = cands[-1]
    return out


def _time_key(name: str) -> float:
    try:
        return float(name)
    except ValueError:
        return math.inf


def read_wall_loads(case_dir: str | os.PathLike, geom: PanelGeometry, flow: FlowConditions) -> SurfaceResponse:
    """Turn the dsmcFoam wall integrals into a SurfaceResponse for the ram face.

    Pressure and shear are the ram-face force per ram area, resolved on the
    ram normal and in-plane. Heat flux is the ram-face heat per ram area.
    The drag coefficient uses the force on all three wall patches, along the
    wind, over 0.5 rho V^2 times the ram area.
    """
    files = find_patch_files(case_dir)
    if "ram" not in files:
        raise FileNotFoundError(f"no postProcessing/wall_ram/*/surfaceFieldValue.dat under {case_dir}")
    loads = {p: read_surface_field_value(f, p) for p, f in files.items()}
    ram = loads["ram"]
    area = ram.area or geom.width * geom.height
    u, v, n = geom.axes()
    f_ram = ram.force
    pressure = float(-np.dot(f_ram, n)) / area         # gas pushes against the outward ram normal
    in_plane = f_ram - np.dot(f_ram, n) * n
    shear = float(np.linalg.norm(in_plane)) / area
    total = sum((ld.force for ld in loads.values()), np.zeros(3))
    wind = flow.velocity / np.linalg.norm(flow.velocity)
    drag = float(np.dot(total, wind))
    rho = exponential_density(flow.orbit.altitude_m)
    q_dyn = 0.5 * rho * flow.orbit.speed ** 2
    cd = drag / (q_dyn * area) if q_dyn > 0 else 0.0
    return SurfaceResponse(pressure=pressure, shear=shear, heat_flux=ram.heat / area, drag_coefficient=cd)


def free_stream_check(flow: FlowConditions) -> dict[str, float]:
    """Numbers a reader can compare against the case files: speed ratio, Knudsen basis."""
    m = mean_mass(flow.composition)
    v_th = math.sqrt(2.0 * K_B * flow.temperature / m)
    return {"speed_ratio": flow.orbit.speed / v_th, "mean_mass": m}
