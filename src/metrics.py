"""Metrics computed from a RunLog (one run). See 05_campagne_experiences/README.md."""

import numpy as np


def jain(x):
    """Jain fairness index (Σx)² / (n Σx²) in [1/n, 1]; 1 = perfectly even. All-zero -> 1."""
    x = np.asarray(x, float)
    s2 = np.sum(x ** 2)
    return 1.0 if s2 < 1e-12 else float(np.sum(x) ** 2 / (len(x) * s2))


def compute(log, deadlock_window=3.0, deadlock_speed=0.05):
    n = len(log.radii)
    arrived = ~np.isnan(log.arrival_time)
    robot_coll = [c for c in log.collisions if c[2] >= 0]
    wall_coll = [c for c in log.collisions if c[2] < 0]

    # travel time: arrival time, or t_end for robots that never arrived (penalized lower bound)
    t_end = float(log.t[-1])
    travel = np.where(arrived, log.arrival_time, t_end)
    ideal = np.linalg.norm(log.goals - log.starts, axis=1) / log.v_max
    delay = np.maximum(travel - ideal, 0.0)

    steps = np.linalg.norm(np.diff(log.pos, axis=0), axis=-1)  # (T, N)
    path_len = steps.sum(axis=0)

    # deadlock: not arrived and (almost) not moving during the last `deadlock_window` seconds
    k = max(int(round(deadlock_window / log.dt)), 1)
    recent_speed = steps[-k:].sum(axis=0) / (k * log.dt) if len(steps) else np.zeros(n)
    deadlocked = (~arrived) & (recent_speed < deadlock_speed)

    du = np.linalg.norm(np.diff(log.u, axis=0), axis=-1) if len(log.u) > 1 else np.zeros((0, n))
    st = log.solve_time[~np.isnan(log.solve_time)] * 1e3  # ms

    return {
        "success": bool(arrived.all() and not log.collisions),
        "arrived_rate": float(arrived.mean()),
        "collisions_robot": len(robot_coll),
        "collisions_wall": len(wall_coll),
        "min_clearance": float(np.min(log.clearance)) if n > 1 else float("nan"),
        "travel_time_mean": float(travel.mean()),
        "makespan": float(travel.max()),
        "delay_mean": float(delay.mean()),
        "jain_fairness": jain(delay),
        "path_length_mean": float(path_len.mean()),
        "path_ratio_mean": float(np.mean(path_len / np.maximum(ideal * log.v_max, 1e-9))),
        "smoothness_mean_du": float(du.mean()) if du.size else 0.0,
        "deadlock_rate": float(deadlocked.mean()),
        "solve_ms_mean": float(st.mean()) if st.size else 0.0,
        "solve_ms_p95": float(np.percentile(st, 95)) if st.size else 0.0,
        "solve_ms_max": float(st.max()) if st.size else 0.0,
    }


def format_table(rows, keys=("success", "collisions_robot", "collisions_wall", "min_clearance", "makespan",
                             "delay_mean", "jain_fairness", "deadlock_rate", "solve_ms_mean")):
    """rows: list of (label, metrics dict) -> fixed-width text table."""
    head = f"{'run':<34}" + "".join(f"{k[:12]:>13}" for k in keys)
    lines = [head, "-" * len(head)]
    for label, m in rows:
        cells = []
        for k in keys:
            v = m[k]
            cells.append(f"{str(v):>13}" if isinstance(v, (bool, int, np.integer)) else f"{v:>13.3f}")
        lines.append(f"{label:<34}" + "".join(cells))
    return "\n".join(lines)
