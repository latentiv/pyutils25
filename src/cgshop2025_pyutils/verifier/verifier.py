from typing import List

from pydantic import BaseModel

from cgshop2025_pyutils.data_schemas.instance import Cgshop2025Instance
from cgshop2025_pyutils.data_schemas.solution import Cgshop2025Solution
from cgshop2025_pyutils.geometry import (
    FieldNumber,
    Point,
    Segment,
    Polygon,
    VerificationGeometryHelper,
    points_contain_duplicates,
    intersection_point,
)

# ADDED: check for duplicate edges
def edges_contain_duplicates(edges):

    # Check the rest of the list for occurences of the same edge (both directions)
    for e1 in range(len(edges)):

      cur_edge = edges[e1]
      cur_edge_rev = [edges[e1][1], edges[e1][0]]

      # If there is a duplicate
      if cur_edge in edges[(e1+1):] or cur_edge_rev in edges[(e1+1):]:

        dupl_list = [i for i, sublist in enumerate(edges) if sublist == cur_edge or sublist == cur_edge_rev]

        # Return just two indices
        return(dupl_list[:2])

    return(None)


class VerificationResult(BaseModel):
    num_obtuse_triangles: int
    num_steiner_points: int
    errors: List[str]


def verify(
    instance: Cgshop2025Instance, solution: Cgshop2025Solution
) -> VerificationResult:
    geom_helper = VerificationGeometryHelper()

    # Initialize an error list to collect all issues found during verification
    errors = []
    
    # Combine instance and solution points into one loop to simplify the logic
    all_points = [Point(x, y) for x, y in zip(instance.points_x, instance.points_y)]
    all_points.extend(
        Point(FieldNumber(x), FieldNumber(y))
        for x, y in zip(solution.steiner_points_x, solution.steiner_points_y)
    )

    # check for duplicate points; if found, we cannot properly interpret the indices.
    #duplicates = points_contain_duplicates(all_points)
    #if duplicates:
    #    p1, p2 = duplicates
    #    errors.append(f"Duplicate points found: Indices {p1} and {p2} are the same point ({all_points[p1]})")

    # ADDED: return all duplicate points
    all_points_copy = all_points.copy()
    for p1 in range(len(all_points)):
      if all_points[p1] in all_points_copy:
        duplicates_list = [p2 for p2 in range(p1+1, len(all_points)) if all_points[p1].x() == all_points[p2].x() and all_points[p1].y() == all_points[p2].y()]
        if len(duplicates_list) > 1: errors.append(f"Duplicate point found ({all_points[p1]}): {duplicates_list}")
        all_points_copy = list(filter((all_points[p1]).__ne__, all_points_copy))


    # ADDED: Check for duplicate edges (undirected)

    # Check the the list for occurences of the same edge (both directions)
    #duplicates = edges_contain_duplicates(solution.edges)
    #if duplicates:
    #    e1, e2 = duplicates
    #    errors.append(f"Duplicate edges found: Indices {e1} and {e2} are the same edge ({solution.edges[e1]})")        

    # ADDED: return all duplicate edges
    all_edges_copy = solution.edges.copy()
    for ed in solution.edges:
      if ed in all_edges_copy:
        duplicates_list = [ed2 for ed2 in solution.edges if (ed[0] == ed2[0] and ed[1] == ed2[1]) or (ed[1] == ed2[0] and ed[0] == ed2[1])]
        if len(duplicates_list) > 1: errors.append(f"Duplicate edge found: {duplicates_list}")
        all_edges_copy = list(filter((ed).__ne__, all_edges_copy))
        all_edges_copy = list(filter(([ed[1], ed[0]]).__ne__, all_edges_copy))
          

    # ADDED: Check for edges from a vertex to itself
    for edge in solution.edges:
        if edge[0] == edge[1]:
          errors.append(f"Found edge from vertex to itself: {edge}")  

    # check for out-of-bounds point indices in edges
    for index, edge in enumerate(solution.edges):
        if (
            edge[0] < 0
            or edge[0] >= len(all_points)
            or edge[1] < 0
            or edge[1] >= len(all_points)
        ):
          errors.append(f"Edge {index} ({edge}) contains out-of-bounds point indices (total number of points: {len(all_points)})")


    # ADDED: Check that each point has at least two edges
    for p in range(len(all_points)):

      # Count the edges having p as one of their endpoints
      p_edges_count = sum([1 for edge in solution.edges if p == edge[0] or p == edge[1]])

      if p_edges_count < 2:
        errors.append(f"Found point with {p_edges_count} edge{['s', ''][p_edges_count]}: point[{p}]")    

    
    # ADDED: Create the region boundary Polygon
    region_boundary_poly = Polygon([all_points[i] for i in instance.region_boundary])

    # ADDED: Check if any Steiner point is outside the region boundary
    for point in all_points[instance.num_points:]:

      if region_boundary_poly.contains(point) is False:
        errors.append(f"Found Steiner point outside the region boundary: point[{point}]")


    # ADDED: Check if there is any edge outside the region boundary
    edges_segm = [Segment(all_points[edge[0]], all_points[edge[1]]) for edge in solution.edges]
    region_boundary_segm = [Segment(all_points[instance.region_boundary[i]], all_points[instance.region_boundary[i+1]]) for i in range(len(instance.region_boundary) - 1)] + \
                            [Segment(all_points[instance.region_boundary[-1]], all_points[instance.region_boundary[0]])]
    
    for i in range(len(solution.edges)):
      
      # Check if the current edge intersects any part of the region boundary
      for rb_segm in region_boundary_segm:

        # Intersection point
        inter_p = intersection_point(edges_segm[i], rb_segm)

        if inter_p is not None and inter_p not in all_points:
          errors.append(f"Found edge outside the region boundary: {solution.edges[i]}")

        # Sample the edge to check if it lies outside the region boundary
        num_samples = 10
        t = [i / num_samples for i in range(1, num_samples + 1)]

        for cur_t in t:
            
            cur_segm = edges_segm[i]
            inter_point = Point(
                                cur_segm.source().x() + FieldNumber(cur_t) * (cur_segm.target().x() - cur_segm.source().x()),
                                cur_segm.source().y() + FieldNumber(cur_t) * (cur_segm.target().y() - cur_segm.source().y())
                                )

            if region_boundary_poly.contains(inter_point) is False:
                                
              errors.append(f"Found edge outside the region boundary: {solution.edges[i]}")

          

    # ADDED: Check if there are any edges that cross each other
    for e1 in range(len(solution.edges)):
      for e2 in range(e1 + 1, len(solution.edges)):

        # If the edges intersect and they do so not on a common vertex and this point is not in all_points
        inter_point = intersection_point(edges_segm[e1], edges_segm[e2])

        #if inter_point is not None and inter_point != edges_segm[e1].source() and inter_point != edges_segm[e1].target() and inter_point not in all_points:
        if inter_point is not None and inter_point not in all_points:

          errors.append(f"Found edges crossing each other: {solution.edges[e1]}, {solution.edges[e2]}")



    # ADDED: Check that all the region boundary's edges (split or not) are present in the trinagulation
    region_boundary_edges = [(instance.region_boundary[i], instance.region_boundary[i+1]) for i in range(len(instance.region_boundary) - 1)] + [(instance.region_boundary[-1], instance.region_boundary[0])]
    rb_edges_collection = {rb_e: [] for rb_e in region_boundary_edges}

    # Check each part of the region boundary
    for bound_part in region_boundary_edges:

      bound_part_poly = Polygon([all_points[bound_part[0]], all_points[bound_part[1]]])

      # Check all the edges and collect those that are on this part of the region boundary
      for cur_edge in solution.edges:

        if bound_part_poly.on_boundary(all_points[cur_edge[0]]) and bound_part_poly.on_boundary(all_points[cur_edge[1]]):
          rb_edges_collection[bound_part].append(cur_edge)

      # Check that it is entirely present
      part_edges = rb_edges_collection[bound_part].copy()

      # Start from one point of the boundary part and move until you reach the other end
      part_start = bound_part[0]
      part_end = bound_part[1]

      cur_p = part_start
      found = True

      # Check if the entire part in present and the list has more length > 1
      if len(part_edges) > 1 and ([part_end,part_start] in part_edges or [part_start,part_end] in part_edges):
        errors.append(f"Part of the boundary region has overlapping edges: point[{part_start}] to point[{part_end}]")

      while found is True and len(part_edges) > 0:
      
        found = False

        for ed in part_edges:

          if ed[0] == cur_p or ed[1] == cur_p:
            
            found = True

            if cur_p == ed[0]:
              cur_p = ed[1]
            elif cur_p == ed[1]:
              cur_p = ed[0]

            part_edges.remove(ed)

            break

      # There is an error; either the current boundary part is not covered entirely or there both split and not split edges or overlaps
      if cur_p != part_end: 
        errors.append(f"Part of the boundary region is missing: point[{part_start}] to point[{part_end}]")
        
      if len(part_edges) > 0:
        errors.append(f"Part of the boundary region has overlapping edges: point[{part_start}] to point[{part_end}]")
        

    # ADDED: Check that all the additional contraints' edges (split or not) are present in the trinagulation
    add_constraints_edges = [(constr[0], constr[1]) for constr in instance.additional_constraints]
    addconstr_edges_collection = {rb_e: [] for rb_e in add_constraints_edges}

    # Check each additional constraint
    for bound_part in add_constraints_edges:

      bound_part_poly = Polygon([all_points[bound_part[0]], all_points[bound_part[1]]])

      # Check all the edges and collect those that are on this part of the region boundary
      for cur_edge in solution.edges:

        if bound_part_poly.on_boundary(all_points[cur_edge[0]]) and bound_part_poly.on_boundary(all_points[cur_edge[1]]):

          addconstr_edges_collection[bound_part].append(cur_edge)


      # Check that it is entirely present
      part_edges = addconstr_edges_collection[bound_part].copy()

      # Start from one point of the boundary part and move until you reach the other end
      part_start = bound_part[0]
      part_end = bound_part[1]

      cur_p = part_start
      found = True


      # Check if the entire part is present and the list has more length > 1
      if len(part_edges) > 1 and ([part_end,part_start] in part_edges or [part_start,part_end] in part_edges):
        errors.append(f"Part of an additional constraint has overlapping edges: [{part_start}, {part_end}]")

      while found is True and len(part_edges) > 0:
      
        found = False

        for ed in part_edges:

          if ed[0] == cur_p or ed[1] == cur_p:
            
            found = True

            if cur_p == ed[0]:
              cur_p = ed[1]
            elif cur_p == ed[1]:
              cur_p = ed[0]

            part_edges.remove(ed)

            break

      # There is an error; either the current boundary part is not covered entirely or there both split and not split edges or overlaps
      if cur_p != part_end: 
        errors.append(f"Part of an additional constraint is missing: [{part_start}, {part_end}]")
        
      if len(part_edges) > 0:
        errors.append(f"Part of an additional constraint has overlapping edges: [{part_start}, {part_end}]")


    # Add points to the geometry helper
    for point in all_points:
        geom_helper.add_point(point)

    # Add segments to the geometry helper
    for edge in solution.edges:
        geom_helper.add_segment(edge[0], edge[1])

    # Check for non-triangular faces
    non_triang = geom_helper.search_for_non_triangular_faces()
    if non_triang:
        errors.append(f"Non-triangular face found at {non_triang}")

    # Check for bad edges (edges with the same face on both sides)
    bad_edges = geom_helper.search_for_bad_edges()
    if bad_edges:
        errors.append(f"Edges with the same face on both sides found at {bad_edges}")

    # Check for faces with holes
    holes = geom_helper.search_for_faces_with_holes()
    if holes:
        errors.append(f"Faces with holes found at {holes}")

    # Check for isolated points
    isolated_points = geom_helper.search_for_isolated_points()
    if isolated_points:
        errors.append(f"Isolated points found at {[str(p) for p in isolated_points]}")

    # If any errors were detected, return a result with those errors
    if errors:
        return VerificationResult(
            num_obtuse_triangles=-1, num_steiner_points=-1, errors=errors
        )

    # No errors, return the results of obtuse triangles and steiner points
    return VerificationResult(
        num_obtuse_triangles=geom_helper.count_obtuse_triangles(),
        num_steiner_points=geom_helper.get_num_points() - len(instance.points_x),
        errors=[],
    )
