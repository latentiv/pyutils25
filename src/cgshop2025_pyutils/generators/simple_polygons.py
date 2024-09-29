"""
One of the simplest instance generators which generates a polygon with a given number of vertices.
"""

import math
import random

from ..data_schemas.instance import Cgshop2025Instance
from ..geometry import FieldNumber, Point, Polygon, Segment, compute_convex_hull
from ._random_points import _n_random_points


def _need_to_be_swapped(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """
    Check if the segments p1-p2 and p3-p4 intersect.
    """
    s1 = Segment(p1, p2)
    s2 = Segment(p3, p4)
    if p2 == p3:
        return s1.does_intersect(p4) or s2.does_intersect(p1)
    if p1 == p4:
        return s1.does_intersect(p3) or s2.does_intersect(p2)
    if s1.does_intersect(s2):
        return True
    s1_ = Segment(p1, p3)
    s2_ = Segment(p2, p4)
    return math.sqrt(float(s1.squared_length())) + math.sqrt(
        float(s2.squared_length())
    ) > math.sqrt(float(s1_.squared_length())) + math.sqrt(float(s2_.squared_length()))


def _connect_via_tsp(points: list[Point]) -> list[Point]:
    """
    Connect the points using a 2-Opt TSP algorithm.
    """
    swaps_executed = True
    points.append(points[0])
    while swaps_executed:
        swaps_executed = False
        for i in range(len(points) - 1):
            for j in range(i + 2, len(points) - 1):
                if _need_to_be_swapped(
                    points[i], points[i + 1], points[j], points[j + 1]
                ):
                    points[i + 1 : j + 1] = reversed(points[i + 1 : j + 1])
                    swaps_executed = True
    return points[:-1]


class SimplePolygonGenerator:
    def __call__(self, num_points: int) -> Cgshop2025Instance:
        points = _n_random_points(num_points, 200)
        points = _connect_via_tsp(points)
        # check if ccw
        polygon = Polygon(points)
        if polygon.area() <= FieldNumber(0):
            points.reverse()
        region_boundary = list(range(num_points))
        return Cgshop2025Instance(
            instance_uid=f"simple_polygon_{num_points}",
            num_points=num_points,
            points_x=[round(float(point.x())) for point in points],
            points_y=[round(float(point.y())) for point in points],
            region_boundary=region_boundary,
        )


class SimplePolygonWithExterior:
    def __call__(
        self, num_points: int, edge_del_prob: float = 0.0
    ) -> Cgshop2025Instance:
        points = _n_random_points(num_points, 200)
        points = _connect_via_tsp(points)
        ch_indices = compute_convex_hull(points)
        ch_edges = {
            (ch_indices[i], ch_indices[(i + 1) % len(ch_indices)])
            for i in range(len(ch_indices))
        }
        ch_edges.update(
            {
                (ch_indices[(i + 1) % len(ch_indices)], ch_indices[i])
                for i in range(len(ch_indices))
            }
        )
        return Cgshop2025Instance(
            instance_uid=f"simple_polygon_with_exterior_{num_points}_{round(100*(1-edge_del_prob)):03d}",
            num_points=num_points,
            points_x=[round(float(point.x())) for point in points],
            points_y=[round(float(point.y())) for point in points],
            region_boundary=ch_indices,
            additional_constraints=[
                [i, (i + 1) % num_points]
                for i in range(num_points)
                if (i, (i + 1) % num_points) not in ch_edges
                and random.random() > edge_del_prob
            ],
        )
