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

## Where the real engines go

Each stage has a baseline that runs with NumPy alone, and a hook for a full
engine. Contributions that wire in an engine are welcome:

| Stage | Hook | Candidate engines |
| --- | --- | --- |
| Atmosphere | `atmosphere.FluxModel` | SPARTA (DSMC), NRLMSIS 2.x for density |
| Surface | `surface.ScatteringKernel` | CLL, Maxwell with energy dependence |
| Erosion | `erosion.YieldModel` | ASTM E2089 yields, angle and temperature dependence |
| Structure | `structure.StructuralModel` | CalculiX, FEniCS |

Keep engines optional. The package must still install and run its baselines
with NumPy only.

## Style

Python 3.10+. Plain functions and dataclasses. No framework. Docstrings say
what the function computes, in what units, and where the numbers come from.

## License

By contributing you agree your contribution is licensed under the Apache
License 2.0, the same as the project.
