import copy
import random
import uuid

from cgshop2025_pyutils import VerificationResult, verify
from cgshop2025_pyutils.data_schemas.instance import Cgshop2025Instance
from cgshop2025_pyutils.data_schemas.solution import Cgshop2025Solution
from cgshop2025_pyutils.geometry import Point, compute_convex_hull
from cgshop2025_pyutils.naive_algorithm import DelaunayBasedSolver


def test_verify():
    instance = Cgshop2025Instance(
        instance_uid="example",
        num_points=8,
        points_x=[0, 1, 2, 3, 4, 1, 3, 1],
        points_y=[0, 3, 2, 6, 2, 1, 2, 2],
        region_boundary=[0, 1, 2, 3, 4][::-1],
        num_constraints=1,
        additional_constraints=[[5, 6]],
    )

    solver = DelaunayBasedSolver(instance)
    solution = solver.solve()
    result = verify(instance, solution)
    assert not result.errors
    assert result.num_obtuse_triangles == 5
    assert result.num_steiner_points == 0


SEEDS = [
    3059623771,
    912801942,
    2417586875,
    4024461812,
    3551918909,
    3475446657,
    2075485937,
    3679785335,
]

def test_negative_steiner_points():
    solution = Cgshop2025Solution(
        instance_uid="test_id",
        steiner_points_x=[-1, "-1", "-1/2"],
        steiner_points_y=[-1, "-1", "2/-1"],
        edges=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
    )



def generate_random_point_sets(seed, num_sets=10, num_points=500):
    rng = random.Random(seed)
    for _ in range(num_sets):
        yield {
            (
                int(round(rng.uniform(0, 100000000))),
                int(round(rng.uniform(0, 100000000))),
            )
            for _ in range(num_points)
        }


def instance_from_point_set(plist, instance_uid):
    chull_indices: list[int] = compute_convex_hull(plist)
    return Cgshop2025Instance(
        instance_uid=instance_uid,
        num_points=len(plist),
        num_constraints=0,
        region_boundary=chull_indices,
        additional_constraints=[],
        points_x=[p[0] for p in plist],
        points_y=[p[1] for p in plist],
    )


def generate_random_instances(seed, num_instances=10, num_points=500):
    for ps in generate_random_point_sets(seed, num_instances, num_points):
        plist = [Point(x, y) for x, y in ps]
        yield instance_from_point_set(plist, f"random_{seed}_{uuid.uuid4()}")


def break_solution_delete_edge(
    instance: Cgshop2025Instance, solution: Cgshop2025Solution
):
    k = len(instance.region_boundary)
    boundary = {
        (instance.region_boundary[i % k], instance.region_boundary[(i + 1) % k])
        for i in range(k)
    }
    boundary.update({(b, a) for a, b in boundary})
    while True:
        i = random.randint(0, len(solution.edges) - 1)
        e = tuple(solution.edges[i])
        if e not in boundary:
            del solution.edges[i]
            return


def verifier_on_random_instance(instance: Cgshop2025Instance):
    solution = DelaunayBasedSolver(instance).solve()
    result: VerificationResult = verify(instance, solution)
    assert result.errors == []
    assert result.num_steiner_points == 0

    broken_solution1 = copy.deepcopy(solution)
    break_solution_delete_edge(instance, broken_solution1)
    result: VerificationResult = verify(instance, broken_solution1)
    assert result.errors != []


def test_verify_random_instances():
    for seed in SEEDS:
        for instance in generate_random_instances(seed):
            verifier_on_random_instance(instance)


def test_verify_correct_solution_extra_points():
    points = [(192, 512), (384, 512), (320, 560), (272, 576), (192, 576)]
    instance = instance_from_point_set(
        [Point(x, y) for x, y in points], "two_extra_points"
    )
    solution = Cgshop2025Solution(
        instance_uid=instance.instance_uid,
        steiner_points_x=[272, 320],
        steiner_points_y=[512, 512],
        edges=[
            [0, 5],
            [5, 6],
            [6, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 0],
            [5, 3],
            [5, 2],
            [2, 6],
            [5, 4],
        ],
        meta={"constructed_by": "the hand"},
    )
    result = verify(instance, solution)
    assert not result.errors
    assert result.num_steiner_points == 2
    assert result.num_obtuse_triangles == 0
