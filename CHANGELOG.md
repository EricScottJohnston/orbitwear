# Changelog

## 1.0.0

The coupled simulator.

* **OpenFOAM adapter** (`orbitwear.openfoam`): writes a complete dsmcFoam
  case for a panel in the free stream (blockMesh, snappyHexMesh, DSMC
  properties, initialisation, control with time averaging and per-patch
  wall integrals), runs it natively or through a WSL or Docker launcher, and
  reads drag pressure, shear, heat flux and drag coefficient back.
* **CalculiX adapter** (`orbitwear.calculix`): writes an S3 shell deck at the
  eroded thickness with membrane load, flow pressure and optional thermal
  load, runs `ccx`, and reads von Mises stress, displacement and margin.
* **Coupled driver** (`orbitwear.simulate`): flow, erosion and structure
  over a mission, each engine in `auto`, `on` or `off` mode, with built-in
  stand-ins. Writes `report.json` and `history.csv`.
* **Input file and command line**: `orbitwear run`, `orbitwear write`,
  `orbitwear check`, TOML cases.
* **Atmosphere**: species composition, variable hard sphere constants, and
  number densities consistent with the density table. The built-in chain
  now uses the same composition, so both entry points agree.
* **Tests**: 23 unit tests, fixtures for both OpenFOAM vector formats and
  the CalculiX `.dat` layout, stand-in engines for the full coupled path,
  and a live CalculiX test. CI on Python 3.11 to 3.13.
* Python 3.11 or later.

## 0.0.1

The four stages with built-in models and hooks for full engines.
