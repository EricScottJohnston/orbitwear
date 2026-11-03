"""v1.0: the OpenFOAM and CalculiX adapters and the coupled simulator.

The real engines are not needed. Fixture files stand in for their output,
and two stand-in runners write output the way the engines do, so the whole
coupled path (write case, run, read back, erode, write deck, run, read back)
is exercised.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from orbitwear import calculix, openfoam
from orbitwear.atmosphere import Orbit, mean_mass, number_densities
from orbitwear.calculix import ShellLoads, ShellMaterial, read_dat, write_inp
from orbitwear.cli import main
from orbitwear.config import load
from orbitwear.geometry import PanelGeometry
from orbitwear.openfoam import DSMCSettings, FlowConditions, layout, read_surface_field_value, read_wall_loads, write_case
from orbitwear.simulate import Case, ram_flux, simulate
from orbitwear.surface import maxwell_kernel

HERE = Path(__file__).parent
FIX = HERE / "fixtures"
EXAMPLE = HERE.parent / "examples" / "kapton_400km.toml"
KAPTON = ShellMaterial(youngs_modulus=2.5e9, poisson_ratio=0.34, yield_strength=69e6, expansion=2e-5)


def make_case(**kw) -> Case:
    geom = PanelGeometry(width=1.0, height=1.0, thickness=125e-6, nx=4, ny=4)
    flow = FlowConditions(orbit=Orbit(400e3))
    base = dict(orbit=flow.orbit, duration_s=365.25 * 86400, geometry=geom, material="kapton_H",
                shell=KAPTON, membrane_load=20.0, flow=flow, steps=4,
                openfoam_mode="off", calculix_mode="off")
    base.update(kw)
    return Case(**base)


# ---------------------------------------------------------------- geometry

def test_stl_has_three_named_solids():
    stl = PanelGeometry(1.0, 0.5, 1e-4).box_stl()
    for name, facets in (("ram", 2), ("back", 2), ("sides", 8)):
        block = re.search(rf"solid {name}\n(.*?)endsolid {name}", stl, re.S).group(1)
        assert block.count("facet normal") == facets


def test_ram_normal_faces_the_wind():
    g = PanelGeometry(1.0, 1.0, 1e-4, incidence_deg=30.0)
    u, v, n = g.axes()
    assert math.isclose(np.dot(n, [1, 0, 0]), math.cos(math.radians(30)), rel_tol=1e-12)
    assert math.isclose(np.dot(u, n), 0.0, abs_tol=1e-12)
    assert math.isclose(g.incidence_cos(), math.cos(math.radians(30)))


# ---------------------------------------------------------------- atmosphere

def test_number_densities_recover_mass_density():
    comp = {"O": 0.9, "N2": 0.08, "He": 0.02}
    nd = number_densities(400e3, comp)
    rho = sum(n * {"O": 2.6567e-26, "N2": 4.65e-26, "He": 6.65e-27}[s] for s, n in nd.items())
    assert math.isclose(rho, 3.725e-12, rel_tol=1e-6)
    assert math.isclose(nd["O"] / sum(nd.values()), 0.9)
    assert 2.6e-26 < mean_mass(comp) < 2.9e-26


# ---------------------------------------------------------------- OpenFOAM

def test_write_case_files(tmp_path):
    geom = PanelGeometry(1.0, 1.0, 125e-6)
    flow = FlowConditions(orbit=Orbit(400e3))
    L = write_case(tmp_path, geom, flow)
    for rel in ("system/blockMeshDict", "system/snappyHexMeshDict", "system/controlDict",
                "system/dsmcInitialiseDict", "system/fvSchemes", "system/fvSolution",
                "constant/dsmcProperties", "constant/triSurface/panel.stl",
                "0/boundaryT", "0/boundaryU", "0/q", "0/fD", "0/rhoN", "0/momentum"):
        assert (tmp_path / rel).is_file(), rel
    props = (tmp_path / "constant/dsmcProperties").read_text()
    assert "typeIdList (O N2 He);" in props
    assert "MixedDiffuseSpecular" in props and "diffuseFraction 0.9;" in props
    assert "FreeStream" in props
    control = (tmp_path / "system/controlDict").read_text()
    assert "application     dsmcFoam;" in control
    for p in ("ram", "back", "sides"):
        assert f"wall_{p}" in control
    assert "fields          (qMean fDMean);" in control
    bu = (tmp_path / "0/boundaryU").read_text()
    assert f"uniform (-{flow.orbit.speed:.9g} 0 0)" in bu
    # Layout numbers hang together.
    assert L.end_time > L.average_start > 0
    assert math.isclose(L.delta_t, L.finest_cell / flow.orbit.speed)
    vol = float(np.prod(L.domain_max - L.domain_min))
    assert math.isclose(L.n_equivalent * DSMCSettings().target_particles, sum(L.number_densities.values()) * vol)


def test_domain_has_room_upstream():
    geom = PanelGeometry(1.0, 1.0, 125e-6)
    L = layout(geom, FlowConditions(orbit=Orbit(400e3)), DSMCSettings())
    lo, hi = geom.bounding_box()
    assert L.domain_max[0] - hi[0] > lo[0] - L.domain_min[0]


def test_read_surface_field_value_both_vector_formats():
    a = read_surface_field_value(FIX / "surfaceFieldValue_paren.dat", "ram")
    assert a.area == 1.0 and a.heat == 0.75
    assert np.allclose(a.force, [-2.54e-4, 1e-7, -2e-8])
    b = read_surface_field_value(FIX / "surfaceFieldValue_columns.dat", "sides")
    assert b.area == 0.08 and b.heat == 1.2e-3
    assert np.allclose(b.force, [-4e-7, 0, 0])


def test_wsl_path():
    assert openfoam.wsl_path(r"C:\Users\x\case") == "/mnt/c/Users/x/case"


# ---------------------------------------------------------------- CalculiX

def test_write_inp(tmp_path):
    geom = PanelGeometry(1.0, 0.5, 125e-6, nx=6, ny=3)
    inp = write_inp(tmp_path / "p.inp", geom, 100e-6, KAPTON, ShellLoads(membrane_load=20.0, pressure=2.5e-4, delta_t=10.0))
    text = inp.read_text()
    node_block = text.split("*NODE, NSET=NALL\n")[1].split("*")[0].strip().splitlines()
    elem_block = text.split("*ELEMENT, TYPE=S3, ELSET=EALL\n")[1].split("*")[0].strip().splitlines()
    assert len(node_block) == 7 * 4
    assert len(elem_block) == 2 * 6 * 3
    cload = text.split("*CLOAD\n")[1].split("*")[0].strip().splitlines()
    assert math.isclose(sum(float(l.split(",")[2]) for l in cload), 20.0 * 0.5)
    assert "EALL, P, 2.500000000e-04" in text
    assert "*EXPANSION" in text and "*TEMPERATURE" in text
    assert "*SHELL SECTION, ELSET=EALL, MATERIAL=PANEL\n1.000000000e-04" in text


def test_write_inp_refuses_zero_thickness(tmp_path):
    with pytest.raises(ValueError):
        write_inp(tmp_path / "p.inp", PanelGeometry(1, 1, 1e-4), 0.0, KAPTON, ShellLoads(1.0))


def test_read_dat():
    r = read_dat(FIX / "panel.dat", yield_strength=69e6)
    assert math.isclose(r.max_von_mises, 1.6e5)
    assert math.isclose(r.max_displacement, math.sqrt(4e-10 + 1e-12 + 9e-12))
    assert math.isclose(r.margin, 69e6 / 1.6e5 - 1)


# ---------------------------------------------------------------- stand-in engines

class StandInOpenFOAM:
    """Writes postProcessing output the way dsmcFoam's surfaceFieldValue does,
    with values from the free-molecular model, so the reader can be checked
    against a known answer."""

    def __init__(self, case: Case):
        self.case = case
        self.ran = False

    def available(self) -> bool:
        return True

    def run(self, case_dir):
        case_dir = Path(case_dir)
        assert (case_dir / "system/controlDict").is_file()
        c = self.case
        resp = maxwell_kernel(ram_flux(c), c.flow.wall_temperature, c.flow.accommodation)
        area = c.geometry.width * c.geometry.height
        _, _, n = c.geometry.axes()
        force = -resp.pressure * area * n
        d = case_dir / "postProcessing" / "wall_ram" / "0"
        d.mkdir(parents=True)
        (d / "surfaceFieldValue.dat").write_text(
            f"# Area        : {area:.6e}\n# Time\tareaIntegrate(qMean)\tareaIntegrate(fDMean)\n"
            f"1e-3\t{resp.heat_flux * area:.12e}\t({force[0]:.12e} {force[1]:.12e} {force[2]:.12e})\n")
        self.ran = True


class StandInCalculiX:
    """Writes a .dat with the membrane stress of the thickness in the deck."""

    def __init__(self, case: Case):
        self.case = case
        self.runs = 0

    def available(self) -> bool:
        return True

    def run(self, inp):
        inp = Path(inp)
        text = inp.read_text()
        t = float(text.split("*SHELL SECTION, ELSET=EALL, MATERIAL=PANEL\n")[1].splitlines()[0])
        s = self.case.membrane_load / t
        dat = inp.with_suffix(".dat")
        dat.write_text(
            "\n displacements (vx,vy,vz) for set NALL and time  0.1000000E+01\n\n"
            "         1  1.0000000E-06  0.0000000E+00  0.0000000E+00\n\n"
            " stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz) for set EALL and time  0.1000000E+01\n\n"
            f"         1   1  {s:.7E}  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00  0.0000000E+00\n")
        self.runs += 1
        return dat


def test_simulate_with_stand_in_engines(tmp_path):
    case = make_case(openfoam_mode="on", calculix_mode="on")
    case.openfoam = StandInOpenFOAM(case)
    case.calculix = StandInCalculiX(case)
    r = simulate(case, tmp_path)

    assert r.engines["flow"] == "openfoam dsmcFoam"
    assert r.engines["structure"] == "calculix shell"
    assert case.openfoam.ran and case.calculix.runs == case.steps + 1

    base = maxwell_kernel(ram_flux(case), case.flow.wall_temperature, case.flow.accommodation)
    assert math.isclose(r.flow["drag_pressure_Pa"], base.pressure, rel_tol=1e-9)
    assert math.isclose(r.flow["heat_flux_W_m2"], base.heat_flux, rel_tol=1e-9)
    assert math.isclose(r.flow["drag_coefficient"], base.drag_coefficient, rel_tol=1e-9)

    f = r.final()
    assert f["thickness_m"] < case.geometry.thickness
    assert math.isclose(f["von_mises_Pa"], case.membrane_load / f["thickness_m"], rel_tol=1e-6)

    report = json.loads((tmp_path / "report.json").read_text())
    assert report["engines"]["flow"] == "openfoam dsmcFoam"
    assert report["dsmc_layout"]["n_equivalent"] > 0
    assert (tmp_path / "history.csv").read_text().count("\n") == case.steps + 2
    assert len(list((tmp_path / "calculix").glob("step_*/panel.inp"))) == case.steps + 1


def test_simulate_baselines_match_the_original_pipeline(tmp_path):
    r = simulate(make_case(), tmp_path)
    assert r.engines["flow"].startswith("baseline")
    assert r.engines["structure"].startswith("baseline")
    f = r.final()
    assert 20e-6 <= 125e-6 - f["thickness_m"] <= 500e-6     # MISSE range for ram Kapton
    assert math.isnan(f["displacement_m"])
    assert json.loads((tmp_path / "report.json").read_text())["final"]["displacement_m"] is None


def test_mode_on_without_engine_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITWEAR_CCX", "definitely-not-a-ccx-binary")
    case = make_case(calculix_mode="on")
    case.calculix = calculix.CalculiXRunner.from_env()
    with pytest.raises(RuntimeError, match="CalculiX"):
        simulate(case, tmp_path)


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        make_case(openfoam_mode="maybe")


# ---------------------------------------------------------------- config and CLI

def test_example_config_loads():
    c = load(EXAMPLE)
    assert c.material == "kapton_H"
    assert math.isclose(c.geometry.thickness, 125e-6)
    assert math.isclose(c.shell.youngs_modulus, 2.5e9)
    assert c.flow.composition == {"O": 0.9, "N2": 0.08, "He": 0.02}
    assert c.dsmc.target_particles == 200000 and c.dsmc.surface_level == 4
    assert c.openfoam_mode == "auto" and c.calculix_mode == "auto"


def test_cli_run_and_write(tmp_path, capsys):
    cfg = tmp_path / "case.toml"
    cfg.write_text(EXAMPLE.read_text().replace('enabled = "auto"', 'enabled = "off"'))
    assert main(["run", str(cfg), "-o", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "AO fluence" in out and "baseline" in out
    assert (tmp_path / "out" / "report.json").is_file()
    assert main(["write", str(cfg), "-o", str(tmp_path / "decks")]) == 0
    assert (tmp_path / "decks" / "openfoam" / "system" / "controlDict").is_file()
    assert (tmp_path / "decks" / "calculix" / "as_built" / "panel.inp").is_file()


def test_cli_check(capsys):
    assert main(["check"]) == 0
    assert "OpenFOAM" in capsys.readouterr().out
