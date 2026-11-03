"""Geometry for v1.0: a flat panel, as a thin closed box for the flow solver
and as a triangle shell mesh for the structural solver.

The panel lies in a local (u, v) plane. Its normal is turned away from the
wind by ``incidence_deg``: 0 means the panel faces straight into the wind.
The wind blows in -x in the panel's frame (the spacecraft moves in +x).

Faces of the box carry names, so the flow case gets one patch per face
(``ram``, ``back``, ``sides``) and results can be read back by patch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PanelGeometry:
    width: float            # m, along the panel's u axis
    height: float           # m, along v
    thickness: float        # m, structural thickness (shell)
    incidence_deg: float = 0.0
    box_depth: float = 0.01  # m, depth of the closed box given to the flow solver
    nx: int = 8              # structural mesh divisions along u
    ny: int = 8              # along v

    def normal(self) -> np.ndarray:
        """Unit normal of the ram face, in the wind frame (wind along -x)."""
        a = np.radians(self.incidence_deg)
        return np.array([np.cos(a), np.sin(a), 0.0])

    def incidence_cos(self) -> float:
        return float(np.cos(np.radians(self.incidence_deg)))

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(u, v, n): in-plane unit vectors and the ram-face normal."""
        n = self.normal()
        v = np.array([0.0, 0.0, 1.0])
        u = np.cross(v, n)
        u /= np.linalg.norm(u)
        return u, v, n

    def shell_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Triangle mesh of the ram face: (nodes [N,3], triangles [M,3] 0-based)."""
        u, v, n = self.axes()
        us = np.linspace(-self.width / 2, self.width / 2, self.nx + 1)
        vs = np.linspace(-self.height / 2, self.height / 2, self.ny + 1)
        nodes = np.array([[a * u + b * v] for b in vs for a in us]).reshape(-1, 3)
        tris = []
        for j in range(self.ny):
            for i in range(self.nx):
                p = j * (self.nx + 1) + i
                q = p + self.nx + 1
                tris.append([p, p + 1, q + 1])
                tris.append([p, q + 1, q])
        return nodes, np.array(tris, dtype=int)

    def box_stl(self) -> str:
        """ASCII STL of the thin box with three named solids: ram, back, sides."""
        u, v, n = self.axes()
        w, h, d = self.width / 2, self.height / 2, self.box_depth
        # corners of the ram face (at the normal side) and the back face
        def corner(su: float, sv: float, front: bool) -> np.ndarray:
            return su * w * u + sv * h * v + (0.0 if front else -d) * n
        f = [corner(-1, -1, True), corner(1, -1, True), corner(1, 1, True), corner(-1, 1, True)]
        b = [corner(-1, -1, False), corner(1, -1, False), corner(1, 1, False), corner(-1, 1, False)]
        solids = {
            "ram": [(f[0], f[1], f[2]), (f[0], f[2], f[3])],
            "back": [(b[0], b[2], b[1]), (b[0], b[3], b[2])],
            "sides": [
                (f[0], b[1], f[1]), (f[0], b[0], b[1]),
                (f[1], b[2], f[2]), (f[1], b[1], b[2]),
                (f[2], b[3], f[3]), (f[2], b[2], b[3]),
                (f[3], b[0], f[0]), (f[3], b[3], b[0]),
            ],
        }
        out = []
        for name, tris in solids.items():
            out.append(f"solid {name}")
            for a, bb, c in tris:
                nn = np.cross(bb - a, c - a)
                nn = nn / (np.linalg.norm(nn) or 1.0)
                out.append(f"  facet normal {nn[0]:.6e} {nn[1]:.6e} {nn[2]:.6e}")
                out.append("    outer loop")
                for p in (a, bb, c):
                    out.append(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}")
                out.append("    endloop")
                out.append("  endfacet")
            out.append(f"endsolid {name}")
        return "\n".join(out) + "\n"

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        u, v, n = self.axes()
        pts = []
        for su in (-1, 1):
            for sv in (-1, 1):
                for depth in (0.0, -self.box_depth):
                    pts.append(su * self.width / 2 * u + sv * self.height / 2 * v + depth * n)
        pts = np.array(pts)
        return pts.min(axis=0), pts.max(axis=0)
