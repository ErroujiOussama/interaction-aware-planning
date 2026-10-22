"""ORCA: optimal reciprocal collision avoidance (decentralized, reactive).

Each neighbor adds one half-plane of permitted velocities; each nearby wall
adds one more. The new velocity is the permitted velocity closest to the
preferred one (2D linear program). If no velocity satisfies every constraint,
a 3D linear program minimizes the largest violation of the robot half-planes
while keeping the wall half-planes hard.

The geometry and the three linear programs follow the reference
implementation RVO2 (Agent.cpp, van den Berg et al. 2011). Walls are
simplified: one half-plane per segment, built from its closest point.

A half-plane is stored as (point, direction): permitted velocities v satisfy
det(direction, point - v) <= 0, i.e. they lie on the left of the line.
"""

import numpy as np

from planners.base import Planner
from planners.common import GlobalPath, track_point
from robot import SINGLE_INTEGRATOR

EPS = 1e-6


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def normalize(a):
    return a / max(np.linalg.norm(a), EPS)


# --------------------------------------------------------------------------- half-planes

def neighbor_line(p, v, r, q, w, rq, tau, dt):
    """ORCA half-plane of a robot (p, v, r) with respect to a neighbor (q, w, rq)."""
    rel_pos, rel_vel = q - p, v - w
    dist_sq = rel_pos @ rel_pos
    radius = r + rq
    if dist_sq > radius ** 2:
        # No collision yet: closest point of the boundary of VO^tau.
        w_vec = rel_vel - rel_pos / tau            # from the center of the cut-off disc
        w_len_sq = w_vec @ w_vec
        dot = w_vec @ rel_pos
        if dot < 0 and dot ** 2 > radius ** 2 * w_len_sq:
            # projection on the cut-off disc
            w_len = np.sqrt(w_len_sq)
            unit_w = w_vec / w_len
            direction = np.array([unit_w[1], -unit_w[0]])
            u = (radius / tau - w_len) * unit_w
        else:
            # projection on one of the two legs of the cone
            leg = np.sqrt(dist_sq - radius ** 2)
            if det(rel_pos, w_vec) > 0:            # left leg
                direction = np.array([rel_pos[0] * leg - rel_pos[1] * radius,
                                      rel_pos[0] * radius + rel_pos[1] * leg]) / dist_sq
            else:                                  # right leg
                direction = -np.array([rel_pos[0] * leg + rel_pos[1] * radius,
                                       -rel_pos[0] * radius + rel_pos[1] * leg]) / dist_sq
            u = (rel_vel @ direction) * direction - rel_vel
    else:
        # Already in contact: get out within one time step.
        w_vec = rel_vel - rel_pos / dt
        w_len = np.linalg.norm(w_vec)
        unit_w = w_vec / max(w_len, EPS)
        direction = np.array([unit_w[1], -unit_w[0]])
        u = (radius / dt - w_len) * unit_w
    return v + 0.5 * u, direction                  # each robot takes half of the correction


def wall_line(p, r, closest, tau, dt):
    """Half-plane that keeps the robot away from a wall point for tau seconds (full responsibility)."""
    n = normalize(p - closest)                     # from the wall towards the robot
    d = np.linalg.norm(p - closest)
    if d > r:
        point = -n * (d - r) / tau                 # may approach the wall by d - r within tau
    else:
        point = n * (r - d) / dt                   # inside the wall: leave it within one step
    return point, np.array([n[1], -n[0]])


# --------------------------------------------------------------------------- linear programs (RVO2)

def lp1(lines, i, radius, opt, direction_opt, result):
    """Best velocity on line i, inside the speed disc and the half-planes 0..i-1."""
    point, direction = lines[i]
    dot = point @ direction
    disc = dot ** 2 + radius ** 2 - point @ point
    if disc < 0:
        return False, result                       # the line misses the speed disc
    t_left, t_right = -dot - np.sqrt(disc), -dot + np.sqrt(disc)
    for j in range(i):
        pj, dj = lines[j]
        denom = det(direction, dj)
        numer = det(dj, point - pj)
        if abs(denom) <= EPS:
            if numer < 0:
                return False, result               # parallel and outside
            continue
        t = numer / denom
        if denom >= 0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, result
    if direction_opt:
        t = t_right if opt @ direction > 0 else t_left
    else:
        t = np.clip(direction @ (opt - point), t_left, t_right)
    return True, point + t * direction


def lp2(lines, radius, opt, direction_opt=False):
    """Velocity closest to opt inside all half-planes and the speed disc.

    Returns (index of the first line that could not be satisfied, velocity).
    """
    if direction_opt:
        result = opt * radius
    elif opt @ opt > radius ** 2:
        result = normalize(opt) * radius
    else:
        result = opt.copy()
    for i, (point, direction) in enumerate(lines):
        if det(direction, point - result) > 0:     # current result violates line i
            ok, new = lp1(lines, i, radius, opt, direction_opt, result)
            if not ok:
                return i, result
            result = new
    return len(lines), result


def lp3(lines, n_hard, begin, radius, result):
    """Minimize the largest violation of lines[n_hard:], keeping lines[:n_hard] satisfied."""
    distance = 0.0
    for i in range(begin, len(lines)):
        point_i, dir_i = lines[i]
        if det(dir_i, point_i - result) <= distance:
            continue
        projected = list(lines[:n_hard])
        for j in range(n_hard, i):
            point_j, dir_j = lines[j]
            d = det(dir_i, dir_j)
            if abs(d) <= EPS:
                if dir_i @ dir_j > 0:
                    continue                       # same direction: line j is redundant
                point = 0.5 * (point_i + point_j)
            else:
                point = point_i + (det(dir_j, point_i - point_j) / d) * dir_i
            projected.append((point, normalize(dir_j - dir_i)))
        n_ok, new = lp2(projected, radius, np.array([-dir_i[1], dir_i[0]]), direction_opt=True)
        if n_ok == len(projected):
            result = new
        distance = det(dir_i, point_i - result)
    return result


# --------------------------------------------------------------------------- planner

class OrcaPlanner(Planner):
    name = "orca"
    defaults = {
        "tau": 2.0,            # [s] horizon for other robots
        "tau_obst": 1.0,       # [s] horizon for walls
        "neighbor_dist": 5.0,  # [m] robots farther away are ignored
        "max_neighbors": 10,
        "wall_dist": 1.5,      # [m] wall segments farther away are ignored
        "lookahead": 1.0,      # [m] local target on the A* path
        "margin": 0.05,        # [m] added to the robot radius (ORCA drives robots to exact contact)
    }

    def reset(self, ctx):
        super().reset(ctx)
        self.path = GlobalPath(ctx.static_map, ctx.start, ctx.goal, ctx.spec.radius)
        self.debug["path"] = self.path.points

    def preferred_velocity(self, obs):
        target = self.path.lookahead(obs.p, self.params["lookahead"])
        d = target - obs.p
        speed = min(obs.spec.v_max, 2.0 * np.linalg.norm(obs.goal - obs.p))  # slow down at the goal
        return normalize(d) * speed

    def new_velocity(self, obs, v_pref):
        prm, dt = self.params, self.ctx.dt
        r = obs.spec.radius + prm["margin"]
        lines = []

        # walls first: they are the hard constraints of lp3
        for a, b in self.ctx.static_map.segments:
            ab = b - a
            s = np.clip((obs.p - a) @ ab / max(ab @ ab, EPS), 0.0, 1.0)
            closest = a + s * ab
            if np.linalg.norm(obs.p - closest) < prm["wall_dist"]:
                lines.append(wall_line(obs.p, r, closest, prm["tau_obst"], dt))
        n_walls = len(lines)

        neighbors = sorted(obs.neighbors, key=lambda nb: np.linalg.norm(nb.p - obs.p))
        for nb in neighbors[:prm["max_neighbors"]]:
            if np.linalg.norm(nb.p - obs.p) < prm["neighbor_dist"]:
                lines.append(neighbor_line(obs.p, obs.v, r, nb.p, nb.v, nb.radius, prm["tau"], dt))

        n_ok, v = lp2(lines, obs.spec.v_max, v_pref)
        if n_ok < len(lines):
            v = lp3(lines, n_walls, n_ok, obs.spec.v_max, v)
        return v

    def compute_control(self, obs):
        v_new = self.new_velocity(obs, self.preferred_velocity(obs))
        if obs.spec.model == SINGLE_INTEGRATOR:
            return v_new
        return track_point(obs, obs.p + v_new * 10 * self.ctx.dt, slow_down=False)
