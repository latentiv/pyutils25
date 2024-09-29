"""
This module contains some random instance generators.
"""

from .orthogonal import SplitBasedOrthogonalGenerator
from .point_set import PointSetGenerator
from .simple_polygons import SimplePolygonGenerator, SimplePolygonWithExterior
from .verify_instance import verify_instance

__all__ = [
    "SimplePolygonGenerator",
    "SimplePolygonWithExterior",
    "PointSetGenerator",
    "SplitBasedOrthogonalGenerator",
    "verify_instance",
]
