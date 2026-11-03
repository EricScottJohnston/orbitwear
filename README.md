# orbitwear

Open-source prediction of how materials hold up in low Earth orbit.

Today the pieces exist separately. Rarefied-flow codes give drag and heating.
Finite element codes give structural response. Nobody couples them. orbitwear
chains the probabilities from the molecule up, in one open pipeline:

1. **Atmosphere.** Monte Carlo collisions in the thin upper atmosphere give the
   flux of each species, with its speed and angle, onto every surface element.
2. **Surface.** Each arriving atom or molecule either reflects, accommodates or
   reacts. Gas-surface interaction kernels turn the flux into momentum and
   energy transfer per element: drag and heating.
3. **Erosion.** Atomic oxygen reacts with the surface. Fluence and erosion yield
   give material loss per element over time, so the geometry thins.
4. **Structure.** The pressure, heat and thinned sections feed a finite element
   model. Stress and deformation come out over the satellite's life.

The result is drag, heating, stress and material loss over a mission, from one
model of the spacecraft and one description of the orbit.

## Status

Pre-alpha. The package defines the four stages, the data that passes between
them, and a driver that runs them in order. Each stage ships with a baseline
that runs today and a documented hook for the full engine:

| Stage | Baseline in the package | Full engine (hook) |
| --- | --- | --- |
| Atmosphere | Free-molecular flux from an exponential density model | DSMC (SPARTA) |
| Surface | Maxwell accommodation kernel | Energy-dependent scattering kernels |
| Erosion | Fluence × erosion yield, per material | Angle- and temperature-dependent yields |
| Structure | Thin-wall section loss, analytic stress | FEA (CalculiX, FEniCS) |

The baselines are there so the pipeline runs end to end from the first commit.
Every number they produce is a first estimate, not a validated result.

## Install

```
pip install -e .
```

Python 3.10 or later. NumPy is the only hard dependency.

## Run

```
python examples/kapton_400km.py
```

That runs a Kapton panel at 400 km for one year and prints drag, heating,
fluence, thickness lost and the resulting stress.

## Layout

```
src/orbitwear/
  atmosphere.py   stage 1: flux onto each surface element
  surface.py      stage 2: momentum and energy transfer, drag and heating
  erosion.py      stage 3: atomic oxygen fluence and material loss
  structure.py    stage 4: section loss and stress
  pipeline.py     chains the four stages over a mission
tests/            smoke tests for each stage and the chain
examples/         worked cases
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests are open.
Physics contributions need a source: a paper, a handbook table or a measured
data set, cited in the docstring.

## License

Apache License 2.0. See [LICENSE](LICENSE).
