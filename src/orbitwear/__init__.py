"""orbitwear: how materials hold up in low Earth orbit, molecule to structure.

A simulator that couples OpenFOAM (dsmcFoam, rarefied flow) with CalculiX
(finite element structure) through an atomic oxygen erosion model:

1. :mod:`orbitwear.openfoam`    dsmcFoam case: drag, shear and heating on the panel
2. :mod:`orbitwear.erosion`     atomic oxygen fluence and thickness lost
3. :mod:`orbitwear.calculix`    shell model at the eroded thickness: stress, deflection
4. :mod:`orbitwear.simulate`    the chain over a mission, with fallbacks

Built-in models stand in for each engine when it is not installed:
:mod:`orbitwear.atmosphere` and :mod:`orbitwear.surface` for the flow,
:mod:`orbitwear.structure` for the structure. :mod:`orbitwear.pipeline` runs
the built-in chain alone.
"""

from .pipeline import Mission, Panel, run
from .simulate import Case, SimulationResult, simulate

__all__ = ["Case", "Mission", "Panel", "SimulationResult", "run", "simulate"]
__version__ = "1.0.0"
