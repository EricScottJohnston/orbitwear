# orbitwear

A simulator for how materials hold up in low Earth orbit. It couples
OpenFOAM's rarefied-flow solver with CalculiX finite element analysis,
through an atomic oxygen erosion model, in one open pipeline.

Today the pieces exist separately. Rarefied-flow codes give drag and heating.
Finite element codes give structural response. Nobody couples them. orbitwear
chains the probabilities from the molecule up:

1. **Flow (OpenFOAM dsmcFoam).** Direct simulation Monte Carlo of the thin
   upper atmosphere around the part: molecular collisions in the gas, and
   diffuse or specular scattering at each wall. Out come the force and heat
   flux on every wall face.
2. **Erosion.** Atomic oxygen arriving on the ram face, times the material's
   erosion yield, gives thickness lost over each step of the mission.
3. **Structure (CalculiX).** A shell model of the part at its eroded
   thickness, loaded by its service load, the flow pressure and any
   temperature change. Out come stress, deflection and margin to yield.

The result is drag, heating, stress and material loss over a satellite's
life, from one input file.

## Quick start

```
pip install -e .
orbitwear check
orbitwear run examples/kapton_400km.toml -o out/kapton
```

`orbitwear check` reports which engines it found. Each engine runs when it
is installed. When one is missing, a built-in model stands in, and the
report says which produced each number:

| Stage | Engine | Built-in stand-in |
| --- | --- | --- |
| Flow | OpenFOAM `dsmcFoam` | Free-molecular flat plate, Maxwell accommodation |
| Erosion | Fluence × erosion yield | (same) |
| Structure | CalculiX `ccx`, S3 shells | Membrane stress, load over thickness |

The example is a 1 m square, 125 µm Kapton H panel facing the wind at 400 km
for one year. With the built-in models it prints:

```
engines: flow = baseline free-molecular Maxwell; structure = baseline membrane stress
after 1.00 years
  drag pressure        2.539e-04 Pa   (Cd 2.32, drag 2.539e-04 N)
  heat flux absorbed   0.749 W/m^2
  AO fluence           2.924e+21 atoms/cm^2
  thickness remaining  37.3 um
  max von Mises        0.54 MPa   (margin to yield 127.64)
```

The run writes `report.json`, `history.csv`, the OpenFOAM case under
`openfoam/` and one CalculiX deck per step under `calculix/`.

## Installing the engines

**OpenFOAM.** orbitwear targets the ESI release (openfoam.com), v2306 to
v2406, which ships `dsmcFoam`. On Ubuntu or in WSL:

```
curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash
sudo apt-get install openfoam2406-default
source /usr/lib/openfoam/openfoam2406/etc/bashrc
```

From Windows, point orbitwear at WSL or Docker with a launcher. `{cmd}` is
the OpenFOAM command, `{case}` the case folder, `{case_wsl}` the same folder
as WSL sees it:

```toml
[openfoam]
command = 'wsl -- bash -lc "source /usr/lib/openfoam/openfoam2406/etc/bashrc && cd {case_wsl} && {cmd}"'
# command = 'docker run --rm -v {case}:/case -w /case opencfd/openfoam-default:2406 {cmd}'
```

The environment variable `ORBITWEAR_OPENFOAM` does the same.

**CalculiX.** On Ubuntu, `sudo apt-get install calculix-ccx`. On Windows,
any build that provides `ccx.exe`. Set `ORBITWEAR_CCX` or `[calculix]
command` if it is not on the path.

## What the OpenFOAM case contains

`orbitwear write case.toml` writes the inputs without running them, for
running on a cluster or in a container:

* `system/blockMeshDict`: a box around the part, one `freestream` patch,
  with a full part length of room upstream
* `constant/triSurface/panel.stl`: the part as a closed solid with named
  faces, which `snappyHexMesh` turns into wall patches `ram`, `back`, `sides`
* `constant/dsmcProperties`: species and variable hard sphere constants
  (Bird 1994, Table A1), `MixedDiffuseSpecular` walls, `FreeStream` inflow
* `system/dsmcInitialiseDict`: the domain filled at free-stream conditions
* `system/controlDict`: time averaging once the flow has crossed the domain
  twice, and a `surfaceFieldValue` integral of `qMean` and `fDMean` on each
  wall patch

The sequence is `blockMesh`, `snappyHexMesh -overwrite`, `dsmcInitialise`,
`dsmcFoam`. The particle weight, time step and run length are set from the
domain, the orbital speed and a target particle count.

## Status of v1.0

* The four stages, the coupling, the input file, the command line and the
  reports are complete.
* The OpenFOAM and CalculiX adapters are tested against sample output files
  and against stand-in engines that write output the way the real ones do.
  A real `ccx` test runs wherever CalculiX is installed, and in CI. A full
  `dsmcFoam` run has not yet been exercised in CI.
* Geometry is a flat panel at any angle to the wind. Arbitrary STL parts are
  next.
* Atomic oxygen fluence comes from the free-stream flux on the ram face,
  which is exact in the free-molecular limit. Species-resolved wall flux from
  the DSMC solution is next.
* The structure is linear and small-deflection.
* Atmosphere density is the US Standard Atmosphere 1976 mean. For a real
  mission, set the composition and density from NRLMSIS.

## Layout

```
src/orbitwear/
  simulate.py     the coupled chain over a mission, engine or stand-in per stage
  openfoam.py     dsmcFoam case writer, launcher, wall-load reader
  calculix.py     shell deck writer, launcher, .dat reader
  geometry.py     panel as a closed STL for the flow and a shell mesh for the structure
  atmosphere.py   density, composition, species constants, free-stream flux
  surface.py      free-molecular Maxwell kernel: drag and heating stand-in
  erosion.py      atomic oxygen erosion yields and fluence
  structure.py    membrane stress stand-in
  pipeline.py     the built-in chain alone, no engines
  config.py       TOML input
  cli.py          orbitwear run | write | check
tests/            unit tests, fixtures, live-engine tests
examples/         kapton_400km.toml, kapton_400km.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests are open.
Every physical number needs a source: a paper, a handbook table or a
measured data set, cited next to the number.

## License

Apache License 2.0. See [LICENSE](LICENSE).
