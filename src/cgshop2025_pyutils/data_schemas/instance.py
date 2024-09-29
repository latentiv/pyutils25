from pydantic import BaseModel, Field, model_validator


class Cgshop2025Instance(BaseModel):
    """
    This schema defines the data format for the CG_SHOP_2025_Instance, which involves finding a
    Non-Obtuse Triangulation with the minimum number of Steiner points. Point coordinates are
    integers, while Steiner point coordinates can be rational numbers.

    The instance includes a region boundary, represented as a list of counter-clockwise oriented
    point indices, describing the area to be triangulated. This boundary can be a convex hull or
    any simple polygon.

    The triangulation must include all points and adhere to the specified constraints, meaning
    that there must be a direct line connection between every two points in the constraints.
    However, the triangulation can split segments of both the boundary and the constraints.

    Special cases of instances:
    - Only points with the region boundary as the convex hull: This represents an unconstrained
      triangulation. Although you may need to add Steiner points, there are no constraints, and
      the region boundary is provided for convenience.
    - No constraints with the region boundary containing all points: This scenario is the
      classical problem of triangulating a simple polygon, for which a solution with a linear
      number of Steiner points is known to exist.

    """

    instance_uid: str = Field(..., description="Unique identifier of the instance.")
    num_points: int = Field(
        ...,
        description="Number of points in the instance. All points must be part of the final triangulation.",
    )
    points_x: list[int] = Field(..., description="List of x-coordinates of the points.")
    points_y: list[int] = Field(..., description="List of y-coordinates of the points.")
    region_boundary: list[int] = Field(
        ...,
        description=(
            "Boundary of the region to be triangulated, given as a list of counter-clockwise oriented "
            "point indices. The triangulation may split boundary segments. The first point is not "
            "repeated at the end of the list."
        ),
    )
    num_constraints: int = Field(
        default=0, description="Number of constraints in the instance."
    )
    additional_constraints: list[list[int]] = Field(
        default_factory=list,
        description=(
            "List of constraints additional to the region_boundary, each given as a list of two point indices. The triangulation may split "
            "constraint segments, but must include a straight line between the two points."
        ),
    )

    @model_validator(mode="after")
    def validate_points(self):
        if (
            len(self.points_x) != self.num_points
            or len(self.points_y) != self.num_points
        ):
            msg = "Number of points does not match the length of the x/y lists."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_region_boundary(self):
        if len(self.region_boundary) < 3:
            msg = "The region boundary must have at least 3 points."
            raise ValueError(msg)
        for idx in self.region_boundary:
            if idx < 0 or idx >= self.num_points:
                msg = "Invalid point index in region boundary."
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_constraints(self):
        for constraint in self.additional_constraints:
            if len(constraint) != 2:
                msg = "Constraints must have exactly two points."
                raise ValueError(msg)
            for idx in constraint:
                if idx < 0 or idx >= self.num_points:
                    msg = "Invalid point index in constraint."
                    raise ValueError(msg)
        return self
