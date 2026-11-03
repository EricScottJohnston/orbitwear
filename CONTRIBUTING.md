# Contributing to orbitwear

Issues and pull requests are open to anyone.

## What a contribution needs

- **A source for every physical number.** Erosion yields, accommodation
  coefficients, density models, material properties: cite the paper, handbook
  table or data set in the docstring next to the number.
- **A test.** Each stage has a smoke test in `tests/`. A new model gets a test
  that pins at least one known case from its source.
- **The interface kept intact.** The four stages pass dataclasses defined in
  their modules. Add fields rather than renaming them, so downstream stages
  keep working.

## Where work is most wanted

The engines are OpenFOAM (`orbitwear/openfoam.py`) and CalculiX
(`orbitwear/calculix.py`). Each stage also has a built-in stand-in that runs
with NumPy alone. The most useful contributions now:

| Area | What | Where |
| --- | --- | --- |
| Validation | Run the example against a live dsmcFoam and report the numbers | `openfoam.py`, an issue |
| Geometry | Arbitrary STL parts, mapped from flow faces to shell elements | `geometry.py` |
| Erosion | Species-resolved wall flux from the DSMC solution; angle and temperature yields | `erosion.py`, `openfoam.py` |
| Atmosphere | NRLMSIS 2.x density and composition by date and solar activity | `atmosphere.py` |
| Structure | Nonlinear geometry, thermal fields from the flow heat flux | `calculix.py` |

Keep engines optional. The package must install and run its stand-ins with
NumPy only, and the unit tests must pass without OpenFOAM or CalculiX.
Tests that need a real engine go in `tests/test_engines_live.py` with a skip
when the engine is missing.

## Style

Python 3.10+. Plain functions and dataclasses. No framework. Docstrings say
what the function computes, in what units, and where the numbers come from.

## License

By contributing you agree your contribution is licensed under the Apache
License 2.0, the same as the project.
