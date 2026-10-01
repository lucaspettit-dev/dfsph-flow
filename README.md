# dfsph-flow

Divergence-Free Smoothed Particle Hydrodynamics (DFSPH) in Cython, with a fast tree-based neighborhood search.

## Status

**This is a 2D implementation.** The solver operates in 2D with a quadtree neighborhood search (the 2D analog of an octree). A 3D version with a true octree is planned but not yet implemented.

The solver is stable for dam-break and falling-block scenarios with appropriate timesteps. It is not yet validated against reference benchmarks.

## Algorithm

- **DFSPH** (Bender & Koschier 2015): Divergence-free SPH with two pressure solves per step:
  1. Divergence-free solve (enforces ∇·v = 0 via velocity update)
  2. Constant-density solve (enforces ρ = ρ₀ via position correction)
- **Neighborhood search**: Quadtree over uniform-grid cells with Morton (z-order) sorting, inspired by the "Fast Octree Neighborhood Search" paper (Fernández-Fernández et al. 2022). Leaves target ~1000 particles; per-leaf brute force.

## Installation

Requires Python ≥3.9, NumPy, Cython.

```bash
pip install -e .
```

This builds the Cython extension in-place.

## Usage

```python
from dfsph_flow import DFSPHFlow

sim = DFSPHFlow(dx=0.05, gravity=(0, -9.81), xsph_epsilon=0.1, max_dt=0.002)
sim.add_fluid_box(0.1, 0.4, 0.5, 0.8)      # dam-break column
sim.add_boundary_box(0.0, 0.0, 2.0, 1.0)   # container walls

for _ in range(500):
    dt = sim.step()

positions = sim.positions      # (N, 2) array
velocities = sim.velocities    # (N, 2) array
```

### Rendering

```python
img = sim.render(width=800, height=400)  # RGB array, direction/speed colormap
```

### Inflow / Outflow (open boundaries)

Define regions that continuously spawn fluid particles at a given velocity (inflow) or delete fluid particles that enter them (outflow). Inflow boxes only fill grid cells that are currently empty, so they will not over-pack an already populated region; spawned particles get a small positional jitter to avoid quadtree degeneracy. Outflow boxes remove fluid particles only — never boundary particles.

```python
# wind-tunnel test: uniform flow enters from the left, exits on the right
sim = DFSPHFlow(dx=0.05, gravity=(0, 0), xsph_epsilon=0.1, max_dt=0.002)
sim.add_boundary_block(0.4, 0.35, 0.5, 0.65)   # obstacle in the stream
sim.add_inflow_box(0.0, 0.0, 0.1, 1.0, velocity=(2.0, 0.0))
sim.add_outflow_box(0.9, 0.0, 1.0, 1.0)

for _ in range(500):
    dt = sim.step()
```

## Parameters

- `dx`: particle spacing (m)
- `h`: kernel support radius (default: 2.5·dx)
- `rest_density`: target density (default: 1000 kg/m²)
- `gravity`: (gx, gy) tuple
- `xsph_epsilon`: XSPH viscosity (0 = off, 0.1 = typical)
- `cfl`: CFL number for adaptive timestep
- `max_dt`: maximum timestep (use small values ~0.002 for stability at impact)
- `max_density_error`, `max_divergence_error`: solver tolerances
- `max_iterations`: cap for pressure solves
- `leaf_cap`: quadtree leaf capacity

## Limitations

- **2D only.** Not the 3D octree from the paper.
- **Simple boundary treatment.** Uses fixed wall particles with mirrored pressure (div-free) and density contributions. Minor penetration (~0.01m) can occur at high impact velocities. Not the full Akinci volume-based coupling.
- **Timestep sensitivity.** Requires small `max_dt` (~0.002) for stable wall impact. The CFL condition alone is insufficient.
- **Under-relaxed Jacobi.** The divergence-free solve uses ω=0.5 under-relaxation for stability (the pressure matrix is not diagonally dominant).
- **Not benchmarked.** No validation against reference DFSPH results yet.

## References

- Bender, J. & Koschier, D. (2015). Divergence-Free Smoothed Particle Hydrodynamics. *SCA '15*. DOI: 10.1145/2786784.2786796
- Fernández-Fernández, J.A. et al. (2022). Fast Octree Neighborhood Search for SPH Simulations. *SIGGRAPH Asia 2022*. DOI: 10.1145/3550454.3555523
- SPlisHSPlasH: https://github.com/InteractiveComputerGraphics/SPlisHSPlasH

## License

MIT
