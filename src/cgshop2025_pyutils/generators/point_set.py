from ..data_schemas.instance import Cgshop2025Instance
from ..geometry import compute_convex_hull
from ._random_points import _n_random_points


class PointSetGenerator:
    def __call__(self, num_points: int) -> Cgshop2025Instance:
        points = _n_random_points(num_points, 200)
        ch_indices = list(compute_convex_hull(points))
        return Cgshop2025Instance(
            instance_uid=f"point_set_{num_points}",
            num_points=num_points,
            points_x=[round(float(point.x())) for point in points],
            points_y=[round(float(point.y())) for point in points],
            region_boundary=ch_indices,
        )
