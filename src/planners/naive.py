"""Naive strategy: drive straight to the goal, ignore walls and other robots.

Sanity check of the simulator: two robots in corridor_2 must collide once.
"""

from planners.base import Planner
from planners.common import track_point


class NaivePlanner(Planner):
    name = "naive"

    def compute_control(self, obs):
        return track_point(obs, obs.goal)
