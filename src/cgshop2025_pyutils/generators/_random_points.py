import random

from ..geometry import FieldNumber, Point, squared_distance


def _n_random_points(num_points: int, min_dist: float = 1.0) -> list[Point]:
    """
    Generate n random points in the unit square (0, 10_000) x (0, 10_000).
    Make sure they are unique.
    """
    points: list[Point] = []
    while len(points) < num_points:
        x = random.randint(0, 10_000)
        y = random.randint(0, 10_000)
        if points and min(
            squared_distance(Point(x, y), p) for p in points
        ) < FieldNumber(min_dist**2):
            continue
        points.append(Point(x, y))
    return points
