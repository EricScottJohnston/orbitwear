"""Command line: ``orbitwear run``, ``orbitwear write``, ``orbitwear check``."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .calculix import CalculiXRunner
from .openfoam import OpenFOAMRunner


def _check() -> int:
    of = OpenFOAMRunner.from_env()
    ccx = CalculiXRunner.from_env()
    print(f"orbitwear {__version__}")
    if of.template != "{cmd}":
        print(f"  OpenFOAM: launcher from ORBITWEAR_OPENFOAM: {of.template}")
    else:
        print(f"  OpenFOAM: {'found (dsmcFoam, blockMesh on PATH)' if of.available() else 'not found; flow uses the free-molecular baseline'}")
    print(f"  CalculiX: {'found (' + ccx.executable + ')' if ccx.available() else 'not found; structure uses membrane stress'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="orbitwear", description="Low Earth orbit material durability: OpenFOAM flow, atomic oxygen erosion, CalculiX structure.")
    p.add_argument("--version", action="version", version=f"orbitwear {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a case from a TOML file")
    r.add_argument("config")
    r.add_argument("-o", "--out", default="orbitwear-out", help="output directory (default: orbitwear-out)")

    w = sub.add_parser("write", help="write the OpenFOAM case and CalculiX deck without running them")
    w.add_argument("config")
    w.add_argument("-o", "--out", default="orbitwear-out")

    sub.add_parser("check", help="report which engines are installed")

    a = p.parse_args(argv)
    if a.command == "check":
        return _check()

    from .config import load
    from .simulate import simulate, summary, write_inputs

    case = load(a.config)
    if a.command == "write":
        paths = write_inputs(case, a.out)
        print(f"OpenFOAM case: {paths['openfoam']}")
        print(f"CalculiX deck: {paths['calculix']}")
        return 0
    result = simulate(case, a.out)
    print(summary(result))
    print(f"report: {a.out}/report.json, history: {a.out}/history.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
