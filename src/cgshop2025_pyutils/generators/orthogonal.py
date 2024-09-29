"""
Generator that creates orthogonal polygon instances.
"""

import math
import random
from typing import Optional

from ..data_schemas.instance import Cgshop2025Instance
from ..geometry import FieldNumber, Point, Segment


class SplitBasedOrthogonalGenerator:
    def _initial_segments(self):
        def point(x, y):
            return Point(FieldNumber(x), FieldNumber(y))

        def segment(x1, y1, x2, y2):
            return Segment(point(x1, y1), point(x2, y2))

        self.segments: list[Segment] = [
            segment(0, 0, 1_000_000, 0),
            segment(1_000_000, 0, 1_000_000, 1_000_000),
            segment(1_000_000, 1_000_000, 0, 1_000_000),
            segment(0, 1_000_000, 0, 0),
        ]
        self._recompute_total_length()

    def _pick_random_perturbation(self):
        perturbations = ["dent_in", "dent_out"]
        return random.choice(perturbations)

    def _check_new_segments(self, new_segments, ignored_indices_):
        ignored_indices = set(ignored_indices_)
        for i, s in enumerate(self.segments):
            if i in ignored_indices:
                continue
            for s2 in new_segments:
                if s.does_intersect(s2):
                    return False
        return True

    def _dent_magnitude(self, seg_length: int) -> Optional[int]:
        if random.random() < self.exceeding_dent_prob:
            return random.randint(int(1.01 * seg_length), int(1.5 * seg_length))
        min_mag = int(0.1 * seg_length)
        max_mag = int(0.9 * seg_length)
        r = random.randint(min_mag, max_mag)
        if r == 0:
            return None
        return r

    def _dent_width(self, seg_length: int) -> Optional[int]:
        if seg_length <= 2:
            return None
        min_width = int(0.1 * seg_length)
        max_width = int(0.8 * seg_length)
        r = random.randint(min_width, max_width)
        if r == 0:
            return None
        return r

    def _dent_offset(self, seg_length: int, dent_width: int) -> Optional[int]:
        if seg_length - dent_width - 1 < 1:
            return None
        return random.randint(1, seg_length - dent_width - 1)

    def _recompute_total_length(self):
        self.total_length = sum(
            math.sqrt(float(s.squared_length())) for s in self.segments
        )

    def _dent_segment(self, segment_index, dent_outwards: bool) -> bool:
        s = self.segments[segment_index]
        src, tar = s.source(), s.target()
        horizontal = src.y() == tar.y()
        dent_xdir = 0 if horizontal else 1
        dent_ydir = 1 if horizontal else 0
        seg_length = int(round(math.sqrt(float(s.squared_length()))))
        if src.x() > tar.x():
            dent_ydir *= -1
        if src.y() < tar.y():
            dent_xdir *= -1
        if dent_outwards:
            dent_xdir *= -1
            dent_ydir *= -1
        dent_xdir = FieldNumber(dent_xdir)
        dent_ydir = FieldNumber(dent_ydir)
        seg_xdir = FieldNumber(
            0 if not horizontal else (1 if src.x() < tar.x() else -1)
        )
        seg_ydir = FieldNumber(0 if horizontal else (1 if src.y() < tar.y() else -1))
        dent_mag = self._dent_magnitude(seg_length)
        if dent_mag is None:
            return False
        dent_width = self._dent_width(seg_length)
        if dent_width is None:
            return False
        dent_offset = self._dent_offset(seg_length, dent_width)
        if dent_offset is None:
            return False
        dent_width, dent_mag, dent_offset = (
            FieldNumber(dent_width),
            FieldNumber(dent_mag),
            FieldNumber(dent_offset),
        )
        p1 = Point(src.x() + seg_xdir * dent_offset, src.y() + seg_ydir * dent_offset)
        p2 = Point(p1.x() + dent_xdir * dent_mag, p1.y() + dent_ydir * dent_mag)
        p3 = Point(p2.x() + seg_xdir * dent_width, p2.y() + seg_ydir * dent_width)
        p4 = Point(
            src.x() + seg_xdir * (dent_offset + dent_width),
            src.y() + seg_ydir * (dent_offset + dent_width),
        )
        new_segments = [
            Segment(src, p1),
            Segment(p1, p2),
            Segment(p2, p3),
            Segment(p3, p4),
            Segment(p4, tar),
        ]
        ignored = [
            (segment_index - 1) % len(self.segments),
            segment_index,
            (segment_index + 1) % len(self.segments),
        ]
        if not self._check_new_segments(new_segments, ignored):
            return False
        self.segments = (
            self.segments[:segment_index]
            + new_segments
            + self.segments[segment_index + 1 :]
        )
        self._recompute_total_length()
        return True

    def _pick_random_segment(self) -> int:
        goal_distance = random.random() * self.total_length
        current_distance = 0.0
        replace_index = 0
        for i, s in enumerate(self.segments):
            current_distance += math.sqrt(float(s.squared_length()))
            if current_distance >= goal_distance:
                replace_index = i
                break
        return replace_index

    def _split_segment(self, i):
        segment = self.segments[i]
        if segment.squared_length() <= FieldNumber(1):
            return
        seg_length = int(round(math.sqrt(float(segment.squared_length()))))
        dist = random.randint(1, seg_length - 1)
        src, tar = segment.source(), segment.target()
        xdir = 0 if src.x() == tar.x() else (1 if src.x() < tar.x() else -1)
        ydir = 0 if src.y() == tar.y() else (1 if src.y() < tar.y() else -1)
        new_point = Point(
            src.x() + FieldNumber(xdir * dist), src.y() + FieldNumber(ydir * dist)
        )
        new_segments = [Segment(src, new_point), Segment(new_point, tar)]
        self.segments = self.segments[:i] + new_segments + self.segments[i + 1 :]

    def _split_segments(self):
        while len(self.segments) < self.desired_num_points:
            i = self._pick_random_segment()
            self._split_segment(i)

    def __call__(self, n, exceeding_dent_prob=0.1) -> Cgshop2025Instance:
        self.desired_num_points = n
        self.exceeding_dent_prob = exceeding_dent_prob
        self._initial_segments()
        actions = {
            "dent_in": (lambda i: self._dent_segment(i, False)),
            "dent_out": (lambda i: self._dent_segment(i, True)),
        }
        while len(self.segments) + 4 <= self.desired_num_points:
            i = self._pick_random_segment()
            action = self._pick_random_perturbation()
            actions[action](i)
        self._split_segments()
        points = [s.source() for s in self.segments]
        return Cgshop2025Instance(
            instance_uid=f"orthogonal_split_instance{len(points)}_{exceeding_dent_prob:.3f}",
            num_points=len(points),
            points_x=[round(float(point.x())) for point in points],
            points_y=[round(float(point.y())) for point in points],
            region_boundary=list(range(len(points))),
            additional_constraints=[],
        )
