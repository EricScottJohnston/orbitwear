"""orbitwear: how materials hold up in low Earth orbit, molecule to structure.

Four stages, chained by :mod:`orbitwear.pipeline`:

1. :mod:`orbitwear.atmosphere`  flux of each species onto each surface element
2. :mod:`orbitwear.surface`     momentum and energy transfer: drag and heating
3. :mod:`orbitwear.erosion`     atomic oxygen fluence and material loss
4. :mod:`orbitwear.structure`   section loss and stress
"""

from .pipeline import Mission, Panel, run

__all__ = ["Mission", "Panel", "run"]
__version__ = "0.0.1"
