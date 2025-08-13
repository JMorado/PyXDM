"""PyXDM: Python package for XDM multipole moment calculations."""

from .core import XDMCalculator, XDMSession
from .grids import CustomGrid, load_mesh

import importlib.metadata

__version__ = importlib.metadata.version(__name__)

__all__ = [
    "XDMSession",
    "XDMCalculator",
    "CustomGrid",
    "load_mesh",
]
