"""Runs the real engines where they are installed; skipped elsewhere.

A uniaxial membrane case has a known answer: stress = load per width over
thickness. The flow pressure is set to zero so the shell carries only the
membrane load.
"""

import math

import pytest

from orbitwear.calculix import CalculiXRunner, ShellLoads, ShellMaterial, read_dat, write_inp
from orbitwear.geometry import PanelGeometry
from orbitwear.openfoam import OpenFOAMRunner

ccx = CalculiXRunner.from_env()


@pytest.mark.skipif(not ccx.available(), reason="CalculiX (ccx) not installed")
def test_calculix_uniaxial_membrane(tmp_path):
    geom = PanelGeometry(width=1.0, height=1.0, thickness=125e-6, nx=8, ny=8)
    mat = ShellMaterial(youngs_modulus=2.5e9, poisson_ratio=0.34, yield_strength=69e6)
    t = 100e-6
    inp = write_inp(tmp_path / "panel.inp", geom, t, mat, ShellLoads(membrane_load=20.0, pressure=0.0))
    r = read_dat(ccx.run(inp), mat.yield_strength)
    expected = 20.0 / t
    assert math.isclose(r.max_von_mises, expected, rel_tol=0.25), (r.max_von_mises, expected)
    # Free edge stretches by about strain x length.
    assert math.isclose(r.max_displacement, expected / mat.youngs_modulus * geom.width, rel_tol=0.5)


@pytest.mark.skipif(not OpenFOAMRunner.from_env().available(), reason="OpenFOAM (dsmcFoam) not installed")
def test_openfoam_present():
    # A full dsmcFoam run takes minutes; this only confirms the launcher resolves.
    assert OpenFOAMRunner.from_env().available()
