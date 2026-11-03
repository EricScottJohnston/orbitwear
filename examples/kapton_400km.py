"""A ram-facing Kapton panel at 400 km for one year.

Run from the repository root:

    python examples/kapton_400km.py
"""

from orbitwear import Mission, Panel, run
from orbitwear.atmosphere import Orbit
from orbitwear.pipeline import summary
from orbitwear.structure import Section

YEAR = 365.25 * 86400.0

panel = Panel(
    material="kapton_H",
    # 125 um (5 mil) Kapton H film carrying 20 N per metre of width.
    # Yield strength and modulus from the DuPont Kapton HN data sheet.
    section=Section(thickness=125e-6, membrane_load=20.0, yield_strength=69e6, youngs_modulus=2.5e9),
    incidence_cos=1.0,
    wall_temperature=300.0,
)
mission = Mission(orbit=Orbit(altitude_m=400e3), duration_s=YEAR, steps=12)

history = run(panel, mission)
print("Kapton H, 125 um, ram-facing, 400 km circular")
print(summary(history))
