"""Building blocks shared by planners: A* global path and point tracking (both from PythonRobotics)."""

import numpy as np

from prlib import AStarPlanner, PathFinderController, angle_mod, quiet
from robot import SINGLE_INTEGRATOR


# --------------------------------------------------------------------------- tracking

def track_point(obs, target, slow_down=True, gains=(2.0, 6.0, -1.0)):
    """Control that drives the robot towards `target`.

    single integrator: v_max towards target (proportional slow-down near it if slow_down).
    unicycle: PythonRobotics move_to_pose PathFinderController, v clipped to v_max.
    """
    spec = obs.spec
    d = np.asarray(target, float) - obs.p
    dist = np.linalg.norm(d)
    if dist < 1e-9:
        return np.zeros(2)
    if spec.model == SINGLE_INTEGRATOR:
        speed = min(spec.v_max, 2.0 * dist) if slow_down else spec.v_max
        return d / dist * speed
    ctrl = PathFinderController(*gains)
    bearing = np.arctan2(d[1], d[0])
    _, v, w = ctrl.calc_control_command(d[0], d[1], obs.theta, bearing)
    if not slow_down:
        v = np.sign(v) * spec.v_max
    # turn on the spot first when the target is far off the heading (avoids wide arcs into walls)
    if abs(angle_mod(bearing - obs.theta)) > np.pi / 3:
        v = 0.0
    return np.array([np.clip(v, -spec.v_max, spec.v_max), w])


# --------------------------------------------------------------------------- A* global path

class _VectorizedAStar(AStarPlanner):
    """PythonRobotics AStarPlanner with a numpy obstacle-map build (same result, ~100x faster)."""

    def calc_obstacle_map(self, ox, oy):
        ox, oy = np.asarray(ox, float), np.asarray(oy, float)
        self.min_x, self.min_y = round(min(ox)), round(min(oy))
        self.max_x, self.max_y = round(max(ox)), round(max(oy))
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        xs = self.min_x + np.arange(self.x_width) * self.resolution
        ys = self.min_y + np.arange(self.y_width) * self.resolution
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        occ = np.zeros(gx.shape, dtype=bool)
        for k in range(0, len(ox), 512):  # chunks keep memory small
            d2 = (gx[..., None] - ox[k:k + 512]) ** 2 + (gy[..., None] - oy[k:k + 512]) ** 2
            occ |= (d2 <= self.rr ** 2).any(axis=-1)
        self.obstacle_map = occ.tolist()


_ASTAR_CACHE = {}


def astar_for(static_map, inflation, resolution):
    """One A* grid per (map, inflation, resolution), reused across robots and runs."""
    key = (static_map.segments.tobytes(), round(inflation, 6), round(resolution, 6))
    if key not in _ASTAR_CACHE:
        pts = static_map.sample_points(step=resolution / 2)
        with quiet():
            _ASTAR_CACHE[key] = _VectorizedAStar(pts[:, 0].tolist(), pts[:, 1].tolist(), resolution, inflation)
    return _ASTAR_CACHE[key]


class GlobalPath:
    """A* path (PythonRobotics) from start to goal around static walls, plus a lookahead 'carrot'."""

    def __init__(self, static_map, start, goal, radius, resolution=0.25, margin=0.05):
        planner = astar_for(static_map, radius + margin, resolution)
        with quiet():
            rx, ry = planner.planning(start[0], start[1], goal[0], goal[1])
        pts = np.column_stack([rx[::-1], ry[::-1]])
        if len(pts) < 2:  # A* failed (goal unreachable on the grid): fall back to a straight line
            pts = np.array([start, goal], float)
        pts[0], pts[-1] = start, goal
        self.points = pts
        self._idx = 0

    def lookahead(self, p, distance):
        """First path point at least `distance` ahead of the closest point already reached."""
        pts = self.points
        window = pts[self._idx:self._idx + 20]
        self._idx += int(np.argmin(np.linalg.norm(window - p, axis=1)))
        for k in range(self._idx, len(pts)):
            if np.linalg.norm(pts[k] - p) >= distance:
                return pts[k]
        return pts[-1]
