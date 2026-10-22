"""Run one (scenario, planner, seed) episode. Same rules for every planner."""

import time

import numpy as np

import scenario as scenario_mod
from logger import RunLog
from planners import PlannerContext, make_planner


def simulate(scenario, planner, seed=0, model=None, params=None) -> RunLog:
    """scenario: Scenario or name/path; planner: registry name. Returns the finalized RunLog."""
    if not isinstance(scenario, scenario_mod.Scenario):
        scenario = scenario_mod.load(scenario)
    world = scenario.build_world(seed, model)
    robots = world.robots
    fleet = [{"id": r.id, "start": r.start.copy(), "goal": r.goal.copy(), "spec": r.spec} for r in robots]
    shared = {}
    planners = []
    for r in robots:
        pl = make_planner(planner, **(params or {}))
        pl.reset(PlannerContext(r.id, r.spec, r.start.copy(), r.goal.copy(), world.map, world.dt, fleet, shared,
                                scenario.t_max))
        planners.append(pl)

    log = RunLog(
        scenario=scenario.name, planner=planner, seed=seed, model=robots[0].spec.model,
        dt=world.dt, t_max=scenario.t_max, static_map=world.map,
        radii=np.array([r.spec.radius for r in robots]), v_max=np.array([r.spec.v_max for r in robots]),
        starts=np.array([r.start for r in robots]), goals=np.array([r.goal for r in robots]),
    )
    log.record_state(world)

    n_steps = int(round(scenario.t_max / world.dt))
    for _ in range(n_steps):
        if world.all_arrived:
            break
        # synchronous update: every planner sees the same snapshot, then the world steps once
        controls, solve_times = {}, {}
        for r in robots:
            if r.arrived:
                continue
            obs = world.observe(r.id)
            t0 = time.perf_counter()
            controls[r.id] = planners[r.id].compute_control(obs)
            solve_times[r.id] = time.perf_counter() - t0
        applied = world.step(controls)
        log.record_step(applied, solve_times)
        log.record_state(world)

    log.paths = {i: pl.debug["path"] for i, pl in enumerate(planners) if "path" in pl.debug}
    return log.finalize(world)
