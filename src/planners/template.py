"""Template for a new strategy. Copy this file to planners/<name>.py, rename the class
and `name`, fill in the TODOs, then register it in planners/__init__.py.

Not registered itself. See prioritized.py and orca.py for complete examples.

As is, it already runs: it follows the A* path around the walls and ignores
other robots. Replace the TODO parts with the actual avoidance logic.
"""

import numpy as np

from planners.base import Planner
from planners.common import GlobalPath, track_point
from robot import SINGLE_INTEGRATOR


class TemplatePlanner(Planner):
    name = "template"  # TODO: name used on the command line (--planner <name>)

    # Tunable parameters with their defaults; override with --param key=value.
    defaults = {
        "lookahead": 1.0,  # [m] distance of the local target on the A* path
        # TODO: add your parameters (horizon, tau, K, margin, ...)
    }

    def reset(self, ctx):
        """Called once per run, before t = 0.

        ctx.robot_id, ctx.spec (radius, v_max, w_max, model), ctx.start, ctx.goal,
        ctx.static_map, ctx.dt
        ctx.fleet / ctx.shared: centralized strategies only (e.g. prioritized).
        """
        super().reset(ctx)
        self.path = GlobalPath(ctx.static_map, ctx.start, ctx.goal, ctx.spec.radius)
        self.debug["path"] = self.path.points  # drawn as a dotted line by viz
        # TODO: initial state of the strategy (warm start, reservations, ...)

    def compute_control(self, obs):
        """Called every step until the robot arrives.

        obs.t, obs.p, obs.v, obs.theta, obs.goal, obs.spec
        obs.neighbors: tuple of NeighborObs(id, p, v, radius) within sensing range

        Returns u = (vx, vy) for single_integrator, (v, omega) for unicycle.
        """
        target = self.path.lookahead(obs.p, self.params["lookahead"])

        # Preferred velocity, as used by ORCA as a reference.
        d = target - obs.p
        v_pref = obs.spec.v_max * d / max(np.linalg.norm(d), 1e-9)

        # TODO: avoidance using obs.neighbors -> new velocity v_new
        v_new = v_pref

        if obs.spec.model == SINGLE_INTEGRATOR:
            return v_new
        # unicycle: track the point the chosen velocity leads to
        return track_point(obs, obs.p + v_new * self.ctx.dt * 10, slow_down=False)
