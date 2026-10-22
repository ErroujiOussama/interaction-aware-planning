"""Planner registry. To add a strategy: copy planners/template.py, then add the class here."""

from planners.base import Planner, PlannerContext
from planners.naive import NaivePlanner
from planners.orca import OrcaPlanner
from planners.prioritized import PrioritizedPlanner

PLANNERS = {cls.name: cls for cls in (NaivePlanner, PrioritizedPlanner, OrcaPlanner)}


def make_planner(name, **params) -> Planner:
    if name not in PLANNERS:
        raise KeyError(f"unknown planner {name!r}. Available: {', '.join(sorted(PLANNERS))}")
    return PLANNERS[name](**params)


__all__ = ["PLANNERS", "Planner", "PlannerContext", "make_planner"]
