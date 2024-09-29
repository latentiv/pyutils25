import itertools

from ..data_schemas.instance import Cgshop2025Instance
from ..geometry import Point, Segment


def do_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """
    Check if the segments p1-p2 and p3-p4 intersect.
    """
    s1 = Segment(p1, p2)
    s2 = Segment(p3, p4)
    if p2 == p3:
        return s1.does_intersect(p4) or s2.does_intersect(p1)
    if p1 == p4:
        return s1.does_intersect(p3) or s2.does_intersect(p2)
    if p1 == p3:
        return s1.does_intersect(p4) or s2.does_intersect(p2)
    if p2 == p4:
        return s1.does_intersect(p3) or s2.does_intersect(p1)
    return s1.does_intersect(s2)


def verify_instance(instance: Cgshop2025Instance) -> bool:
    """
    Verifies the correctness of a given instance.

    Args:
        instance: The instance to verify.

    Returns:
        True if the instance is valid, False otherwise.
    """
    # make sure, all points are unique
    points = list(zip(instance.points_x, instance.points_y))
    assert len(points) == len(set(points)), "Points must be unique."
    # make sure, no two segments intersect, except at their endpoints
    segments = []
    for p_idx, q_idx in zip(
        instance.region_boundary,
        instance.region_boundary[1:] + [instance.region_boundary[0]],
    ):
        p = Point(instance.points_x[p_idx], instance.points_y[p_idx])
        q = Point(instance.points_x[q_idx], instance.points_y[q_idx])
        segments.append((p, q))
    for constraint in instance.additional_constraints:
        p_idx, q_idx = constraint
        p = Point(instance.points_x[p_idx], instance.points_y[p_idx])
        q = Point(instance.points_x[q_idx], instance.points_y[q_idx])
        segments.append((p, q))
    for (p1, p2), (q1, q2) in itertools.combinations(segments, 2):
        if do_intersect(p1, p2, q1, q2):
            return False
    return True
