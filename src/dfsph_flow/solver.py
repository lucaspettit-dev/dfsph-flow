"""High-level Python API for the DFSPH fluid solver.

The heavy lifting (neighborhood search, SPH sums, pressure solves) lives in
the Cython extension ``dfsph_flow._core``. This module provides the
user-facing :class:`DFSPHFlow` class: scene setup, stepping, read-only data
access, and a small rasterized renderer.
"""

import numpy as np

from ._core import DFSPHSolver


class DFSPHFlow:
    """2D Divergence-Free SPH (DFSPH) incompressible fluid solver.

    Parameters
    ----------
    dx : float
        Particle spacing used when sampling fluid/boundary boxes.
    h : float, optional
        SPH support radius. Defaults to ``2.5 * dx``.
    rest_density : float
        Rest density ``rho0`` (1000.0 = water-like).
    gravity : tuple
        Constant gravity acceleration ``(gx, gy)``.
    xsph_epsilon : float
        XSPH viscosity coefficient in [0, 1). 0 disables it.
    cfl : float
        CFL safety factor for the adaptive time step (0.4, as in the
        DFSPH paper).
    max_dt : float
        Upper bound for the adaptive time step.
    max_density_error : float
        Relative density-error tolerance for the constant-density solver
        (fraction of ``rest_density``).
    max_divergence_error : float
        Relative tolerance for the divergence-free solver.
    max_iterations : int
        Hard cap on pressure-solver iterations per step.
    leaf_cap : int
        Quadtree leaf capacity for neighborhood search (paper optimum
        is ~1000; this is the 2D analog of the octree paper's setting).
    """

    def __init__(
        self,
        dx=0.05,
        h=None,
        rest_density=1000.0,
        gravity=(0.0, -9.81),
        xsph_epsilon=0.1,
        cfl=0.4,
        max_dt=0.01,
        max_density_error=1e-3,
        max_divergence_error=1e-3,
        max_iterations=100,
        leaf_cap=1000,
    ):
        self._solver = DFSPHSolver(
            dx=dx,
            h=2.5 * dx if h is None else h,
            rest_density=rest_density,
            gx=float(gravity[0]),
            gy=float(gravity[1]),
            xsph_epsilon=xsph_epsilon,
            cfl=cfl,
            max_dt=max_dt,
            max_density_error=max_density_error,
            max_divergence_error=max_divergence_error,
            max_iterations=max_iterations,
            leaf_cap=leaf_cap,
        )
        self._time = 0.0
        self._steps = 0

    # -- scene setup ----------------------------------------------------
    def add_fluid_box(self, xmin, ymin, xmax, ymax, velocity=(0.0, 0.0)):
        """Fill an axis-aligned rectangle with fluid particles."""
        self._solver.add_fluid_box(xmin, ymin, xmax, ymax, velocity[0], velocity[1])

    def add_boundary_box(self, xmin, ymin, xmax, ymax):
        """Add hollow rectangular walls (sampled boundary particles)."""
        self._solver.add_boundary_box(xmin, ymin, xmax, ymax)

    def add_boundary_block(self, xmin, ymin, xmax, ymax):
        """Add a filled rectangular obstacle (sampled boundary particles)."""
        self._solver.add_boundary_block(xmin, ymin, xmax, ymax)

    # -- stepping --------------------------------------------------------
    def step(self):
        """Advance one adaptive time step. Returns the dt used."""
        dt = self._solver.step()
        self._time += dt
        self._steps += 1
        return dt

    def run(self, steps):
        """Advance ``steps`` time steps."""
        for _ in range(steps):
            self.step()

    # -- read-only state --------------------------------------------------
    @property
    def time(self):
        return self._time

    @property
    def steps(self):
        return self._steps

    @property
    def num_particles(self):
        return self._solver.num_particles

    @property
    def num_fluid(self):
        return self._solver.num_fluid

    @property
    def positions(self):
        """(n, 2) float64 read-only array of particle positions."""
        a = np.column_stack([self._solver.x, self._solver.y])
        a.flags.writeable = False
        return a

    @property
    def velocities(self):
        """(n, 2) float64 read-only array of particle velocities."""
        a = np.column_stack([self._solver.vx, self._solver.vy])
        a.flags.writeable = False
        return a

    @property
    def densities(self):
        """(n,) float64 read-only array of SPH densities."""
        a = np.asarray(self._solver.rho)
        a.flags.writeable = False
        return a

    @property
    def is_boundary(self):
        """(n,) bool read-only mask of static boundary particles."""
        a = np.asarray(self._solver.is_boundary, dtype=bool)
        a.flags.writeable = False
        return a

    # -- rendering ---------------------------------------------------------
    def render(self, width=512, max_speed=None):
        """Rasterize particles to an RGB image.

        Color encodes flow direction on a color wheel (red = up,
        cyan = down) with brightness scaled by speed; boundary
        particles are drawn dim gray. Returns a read-only uint8
        ``(H, W, 3)`` array.
        """
        pos = self.positions
        vel = self.velocities
        bnd = self.is_boundary
        n = pos.shape[0]
        if n == 0:
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            img.flags.writeable = False
            return img

        lo = pos.min(axis=0)
        hi = pos.max(axis=0)
        span = max(hi[0] - lo[0], hi[1] - lo[1], 1e-9)
        height = max(1, int(width * (hi[1] - lo[1]) / span))
        width = max(1, int(width * (hi[0] - lo[0]) / span))
        px = ((pos[:, 0] - lo[0]) / span * (width - 1)).astype(int)
        py = ((hi[1] - pos[:, 1]) / span * (height - 1)).astype(int)
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)

        speed = np.hypot(vel[:, 0], vel[:, 1])
        if max_speed is None:
            max_speed = max(speed.max(), 1e-9)
        bright = np.clip(speed / max_speed, 0.0, 1.0) ** 0.6

        # direction -> hue wheel, rotated so up (+y) is red (hue 0)
        ang = np.arctan2(vel[:, 1], vel[:, 0]) / (2.0 * np.pi)  # 0 = +x
        hue = (ang - 0.25) % 1.0
        sat = np.ones(n)
        val = bright
        rgb = _hsv_to_rgb(hue, sat, val)

        img = np.zeros((height, width, 3), dtype=np.float64)
        depth = np.zeros((height, width), dtype=np.float64)
        for i in range(n):
            yy, xx = py[i], px[i]
            if bnd[i]:
                if depth[yy, xx] < 0.25:
                    img[yy, xx] = (0.25, 0.25, 0.25)
                    depth[yy, xx] = 0.25
            elif bright[i] > depth[yy, xx]:
                img[yy, xx] = rgb[i]
                depth[yy, xx] = bright[i]
        out = (255 * np.clip(img, 0, 1)).astype(np.uint8)
        out.flags.writeable = False
        return out


def _hsv_to_rgb(h, s, v):
    h6 = (h * 6.0) % 6.0
    i = h6.astype(int)
    f = h6 - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1)
