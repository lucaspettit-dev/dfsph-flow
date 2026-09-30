"""dfsph_flow: fast 2D Divergence-Free SPH fluid solver (Cython core)."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "DFSPHFlow":
        from .solver import DFSPHFlow

        return DFSPHFlow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DFSPHFlow"]
