"""Smoke tests. Run with `python -m pytest tests` or plain `python tests/test_sim.py`."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import metrics  # noqa: E402
import scenario  # noqa: E402
from planners import PLANNERS  # noqa: E402
from robot import Robot, RobotSpec  # noqa: E402
from simulation import simulate  # noqa: E402


def test_scenarios_load():
    for name in ("corridor_2", "crossing_4", "circle_swap_8"):
        sc = scenario.load(name)
        assert len(sc.build_world(0).robots) == {"corridor_2": 2, "crossing_4": 4, "circle_swap_8": 8}[name]


def test_corridor_naive_collides_once():
    """Two robots going straight in the corridor must collide exactly once."""
    m = metrics.compute(simulate("corridor_2", "naive", seed=0))
    assert m["collisions_robot"] == 1 and m["collisions_wall"] == 0 and m["arrived_rate"] == 1.0


def test_crossing_naive_collides():
    assert metrics.compute(simulate("crossing_4", "naive", seed=0))["collisions_robot"] > 0


def test_unicycle_clipping_and_motion():
    r = Robot(0, RobotSpec(model="unicycle", v_max=1.0, w_max=1.0), [0, 0], [5, 0])
    u = r.step([5.0, 0.0], dt=0.1)
    assert np.allclose(u, [1.0, 0.0]) and np.allclose(r.p, [0.1, 0.0])


def test_seed_is_reproducible():
    a, b = simulate("circle_swap_8", "naive", seed=3), simulate("circle_swap_8", "naive", seed=3)
    assert np.allclose(a.pos, b.pos)


def test_every_planner_runs_on_every_scenario():
    for name in scenario.available():
        sc = scenario.load(name)
        sc.t_max = 2.0  # short: we only check the plumbing
        for planner in PLANNERS:
            for model in ("single_integrator", "unicycle"):
                log = simulate(sc, planner, seed=0, model=model)
                assert log.pos.shape[1] == len(sc.robots)


def test_template_runs():
    from planners.template import TemplatePlanner
    from planners import PLANNERS
    PLANNERS["template"] = TemplatePlanner
    try:
        sc = scenario.load("crossing_4")
        sc.t_max = 2.0
        for model in ("single_integrator", "unicycle"):
            simulate(sc, "template", seed=0, model=model)
    finally:
        del PLANNERS["template"]


def test_prioritized_is_collision_free():
    for name in scenario.available():
        m = metrics.compute(simulate(name, "prioritized", seed=0))
        assert m["collisions_robot"] == 0 and m["arrived_rate"] == 1.0


def test_orca_is_collision_free():
    for name in ("corridor_2", "crossing_4", "circle_swap_8"):
        m = metrics.compute(simulate(name, "orca", seed=0))
        assert m["collisions_robot"] == 0 and m["arrived_rate"] == 1.0


def test_jain():
    assert metrics.jain([1, 1, 1]) == 1.0
    assert abs(metrics.jain([1, 0, 0]) - 1 / 3) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
