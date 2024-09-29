from typing import Literal, Union

from pydantic import BaseModel, Field, model_validator


class Cgshop2025Solution(BaseModel):
    """
    Represents a solution for the CG:SHOP 2025 challenge. The solution includes the Steiner points
    and the edges triangulating these points, in addition to the given boundary and constraints.

    Important notes:
    - Coordinates often require exact representation using fractions, thus they can be strings
      in the form "123/456". They can also be integers, though this may be less common.
    - Steiner points are indexed starting from |points|, where |points| is the number of points
      in the instance. For example, if the instance has 7 points and there are 3 Steiner points,
      the Steiner points will be indexed from 7 to 9.
    """

    content_type: Literal["CG_SHOP_2025_Solution"] = Field(
        default="CG_SHOP_2025_Solution",
        description="Used to identify the content type.",
    )
    instance_uid: str = Field(..., description="Unique identifier of the instance.")
    steiner_points_x: list[Union[int, str]] = Field(
        default_factory=list,
        description='List of x-coordinates of the Steiner points. Coordinates can be fractions in the form "123/456".',
    )
    steiner_points_y: list[Union[int, str]] = Field(
        default_factory=list,
        description='List of y-coordinates of the Steiner points. Coordinates can be fractions in the form "123/456".',
    )
    edges: list[list[int]] = Field(
        ...,
        description=(
            "List of edges, each represented as a list of two point indices. "
            "Indices range from 0 to |points| + |steiner_points| - 1. "
            "The first Steiner point index is |points|."
        ),
    )
    meta: dict = Field(
        default_factory=dict, description="Additional metadata for the solution."
    )

    @model_validator(mode="after")
    def validate_rational_values(self):
        """
        Validates that the Steiner points' coordinates are either integers or valid fractional strings.

        Raises:
            ValueError: If any x-coordinate or y-coordinate of a Steiner point is invalid.
        """

        def is_rational(value):
            if isinstance(value, int):
                return True
            if isinstance(value, str):
                if "/" in value:
                    components = value.split("/")
                    if len(components) != 2:
                        return False
                    a, b = components
                    return a.isdigit() and b.isdigit()
                return value.isdigit()
            return False

        for x in self.steiner_points_x:
            if not is_rational(x):
                msg = f"Invalid x-coordinate '{x}' of Steiner point."
                raise ValueError(msg)
        for y in self.steiner_points_y:
            if not is_rational(y):
                msg = f"Invalid y-coordinate '{y}' of Steiner point."
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_steiner_points(self):
        """
        Ensures that the number of x-coordinates matches the number of y-coordinates for Steiner points.

        Raises:
            ValueError: If the numbers of x-coordinates and y-coordinates do not match.
        """
        if len(self.steiner_points_x) != len(self.steiner_points_y):
            msg = "The number of x-coordinates and y-coordinates of Steiner points must match."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_edges(self):
        """
        Validates the edges to ensure each edge connects two different points and that indices are valid.

        Raises:
            ValueError: If an edge has an invalid number of points or connects a point to itself.
        """
        for edge in self.edges:
            if len(edge) != 2:
                msg = "Each edge must connect exactly two points."
                raise ValueError(msg)
            if edge[0] == edge[1]:
                msg = "Edges must not connect a point to itself."
                raise ValueError(msg)
            for idx in edge:
                if idx < 0:
                    msg = f"Invalid point index {idx} in edge."
                    raise ValueError(msg)
        return self
