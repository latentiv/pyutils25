from cgshop2025_pyutils.data_schemas.instance import Cgshop2025Instance
from cgshop2025_pyutils.data_schemas.solution import Cgshop2025Solution

from ..geometry import ConstrainedTriangulation, Point


class DelaunayBasedSolver:
    def __init__(self, instance: Cgshop2025Instance):
        self.instance = instance

    def solve(self) -> Cgshop2025Solution:
        ct = ConstrainedTriangulation()
        instance = self.instance
        for x, y in zip(instance.points_x, instance.points_y):
            ct.add_point(Point(x, y))
        ct.add_boundary(instance.region_boundary)
        for constraint in instance.additional_constraints:
            ct.add_segment(constraint[0], constraint[1])
        edges = ct.get_triangulation_edges()
        return Cgshop2025Solution(
            instance_uid=instance.instance_uid,
            steiner_points_x=[],
            steiner_points_y=[],
            edges=edges,
        )
