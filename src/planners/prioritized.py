
"""Prioritized space-time A* planner for multi-robot navigation."""

import heapq
import logging

import numpy as np

from planners.base import Planner
from planners.common import astar_for, track_point
from robot import SINGLE_INTEGRATOR

logger = logging.getLogger(__name__)


class PrioritizedPlanner(Planner):
    name = "prioritized"

    defaults = {
        "resolution": 0.5,   # Spatial grid resolution [m]
        "margin": 0.05,       # Safety margin [m]
        "max_steps": 400,     # Maximum number of planning time steps
    }

    def reset(self, ctx):
        """Compute all fleet plans once, then select this robot's plan."""
        super().reset(ctx)

        if "plans" not in ctx.shared:
            self._plan_fleet()

        self.plan = ctx.shared["plans"].get(ctx.robot_id)

        if self.plan is None:
            self.plan = np.asarray([ctx.start], dtype=float)

        self.debug["path"] = self.plan

    def _plan_fleet(self):
        """Plan robots from the longest to the shortest start-goal distance."""
        ctx = self.ctx
        fleet = ctx.fleet

        if not fleet:
            fleet = [{
                "id": ctx.robot_id,
                "start": ctx.start,
                "goal": ctx.goal,
                "spec": ctx.spec,
            }]

        robots = sorted(
            fleet,
            key=lambda r: -np.linalg.norm(
                np.asarray(r["goal"], dtype=float)
                - np.asarray(r["start"], dtype=float)
            ),
        )

        resolution = self.params["resolution"]
        margin = self.params["margin"]
        max_steps = self.params["max_steps"]

        # A common time step allows us to compare all robots' reservations.
        slowest_vmax = min(float(r["spec"].v_max) for r in robots)
        time_step = resolution / slowest_vmax

        reserved_positions = {}
        reserved_edges = {}
        plans = {}

        for robot in robots:
            robot_id = robot["id"]
            start = np.asarray(robot["start"], dtype=float)
            goal = np.asarray(robot["goal"], dtype=float)
            spec = robot["spec"]
            radius = float(spec.radius)

            grid = astar_for(
                ctx.static_map,
                radius + margin,
                resolution,
            )

            start_cell = self._to_cell(grid, start)
            goal_cell = self._to_cell(grid, goal)

            path_cells = self._space_time_astar(
                grid=grid,
                start=start_cell,
                goal=goal_cell,
                radius=radius,
                max_steps=max_steps,
                reserved_positions=reserved_positions,
                reserved_edges=reserved_edges,
            )

            if path_cells is None:
                logger.warning(
                    "Prioritized planner: no path found for robot %s; "
                    "it will remain at its start.",
                    robot_id,
                )
                path_cells = [start_cell]

            # Convert grid indices back to world coordinates.
            path = np.asarray(
                [
                    [
                        grid.min_x + ix * grid.resolution,
                        grid.min_y + iy * grid.resolution,
                    ]
                    for ix, iy in path_cells
                ],
                dtype=float,
            )

            # Keep the exact goal when the final grid cell is reached.
            if path_cells[-1] == goal_cell:
                path[-1] = goal

            plans[robot_id] = path

            # Reserve the path at each time step. After arrival, the robot
            # stays at its goal, so that position remains reserved.
            for t in range(max_steps + 1):
                cell = path_cells[min(t, len(path_cells) - 1)]
                pos = np.asarray(
                    [
                        grid.min_x + cell[0] * grid.resolution,
                        grid.min_y + cell[1] * grid.resolution,
                    ],
                    dtype=float,
                )

                reserved_positions.setdefault(t, []).append(
                    (pos, radius, robot_id)
                )

                if t < max_steps:
                    next_cell = path_cells[
                        min(t + 1, len(path_cells) - 1)
                    ]
                    reserved_edges.setdefault(t, []).append(
                        (cell, next_cell, robot_id, radius)
                    )

        ctx.shared["plans"] = plans
        ctx.shared["time_step"] = time_step

    def _space_time_astar(
        self,
        grid,
        start,
        goal,
        radius,
        max_steps,
        reserved_positions,
        reserved_edges,
    ):
        """A* search over states (x, y, t), including a wait action."""
        if not self._is_free(grid, start) or not self._is_free(grid, goal):
            return None

        # State format: (ix, iy, t).
        start_state = (start[0], start[1], 0)
        frontier = [(self._heuristic(start, goal), 0, *start_state)]
        parent = {start_state: None}
        visited = set()

        moves = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (0, 0),  # Wait one time step.
        ]

        while frontier:
            _, cost, ix, iy, t = heapq.heappop(frontier)
            state = (ix, iy, t)

            if state in visited:
                continue
            visited.add(state)

            if (ix, iy) == goal:
                return self._reconstruct_path(parent, state)

            if t >= max_steps:
                continue

            for dx, dy in moves:
                nx, ny = ix + dx, iy + dy
                nt = t + 1
                next_cell = (nx, ny)

                if not self._is_free(grid, next_cell):
                    continue

                # The simulator stops a robot as soon as it reaches its goal,
                # so enter the goal only if no robot planned before passes
                # there later.
                if next_cell == goal and not self._goal_stays_free(
                    grid, goal, radius, nt, max_steps, reserved_positions
                ):
                    continue

                # Check physical separation from already planned robots.
                wx = grid.min_x + nx * grid.resolution
                wy = grid.min_y + ny * grid.resolution
                candidate = np.array([wx, wy])

                conflict = False
                for other_pos, other_radius, *_ in reserved_positions.get(
                    nt, []
                ):
                    minimum_distance = radius + other_radius + self.params[
                        "margin"
                    ]
                    if np.linalg.norm(candidate - other_pos) < minimum_distance:
                        conflict = True
                        break

                if conflict:
                    continue

                # Check the minimum distance during the complete transition.
                current_position = np.array(
                    [
                        grid.min_x + ix * grid.resolution,
                        grid.min_y + iy * grid.resolution,
                    ],
                    dtype=float,
                )

                for other_from, other_to, _, other_radius in reserved_edges.get(t, []):
                    other_p0 = np.array(
                        [
                            grid.min_x + other_from[0] * grid.resolution,
                            grid.min_y + other_from[1] * grid.resolution,
                        ],
                        dtype=float,
                    )
                    other_p1 = np.array(
                        [
                            grid.min_x + other_to[0] * grid.resolution,
                            grid.min_y + other_to[1] * grid.resolution,
                        ],
                        dtype=float,
                    )

                    min_distance = self._minimum_transition_distance(
                        current_position,
                        candidate,
                        other_p0,
                        other_p1,
                    )

                    required_distance = (
                        radius + other_radius + self.params["margin"]
                    )

                    if min_distance < required_distance:
                        conflict = True
                        break

                if conflict:
                    continue

                # Add the valid successor to the A* frontier.
                next_state = (nx, ny, nt)

                if next_state in visited:
                    continue

                parent[next_state] = state
                heuristic = self._heuristic(next_cell, goal)

                heapq.heappush(
                    frontier,
                    (nt + heuristic, nt, nx, ny, nt),
                )

        return None

    @staticmethod
    def _heuristic(cell, goal):
        """Manhattan distance in grid cells."""
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    def _goal_stays_free(self, grid, goal, radius, t, max_steps, reserved_positions):
        """True if no reserved robot comes too close to the goal after time t."""
        goal_position = np.array(
            [
                grid.min_x + goal[0] * grid.resolution,
                grid.min_y + goal[1] * grid.resolution,
            ],
            dtype=float,
        )
        for later in range(t + 1, max_steps + 1):
            for other_pos, other_radius, *_ in reserved_positions.get(later, []):
                minimum_distance = radius + other_radius + self.params["margin"]
                if np.linalg.norm(goal_position - other_pos) < minimum_distance:
                    return False
        return True

    @staticmethod
    def _is_free(grid, cell):
        ix, iy = cell
        if ix < 0 or iy < 0:
            return False
        if ix >= grid.x_width or iy >= grid.y_width:
            return False
        return not grid.obstacle_map[ix][iy]

    @staticmethod
    def _to_cell(grid, point):
        """Convert world coordinates to grid indices."""
        ix = round((point[0] - grid.min_x) / grid.resolution)
        iy = round((point[1] - grid.min_y) / grid.resolution)
        return ix, iy

    @staticmethod
    def _minimum_transition_distance(p0, p1, q0, q1):
        """Minimum distance between two synchronous linear motions."""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)

        relative_start = p0 - q0
        relative_velocity = (p1 - p0) - (q1 - q0)

        denominator = float(np.dot(relative_velocity, relative_velocity))

        if denominator < 1e-12:
            alpha = 0.0
        else:
            alpha = float(
                np.clip(
                    -np.dot(relative_start, relative_velocity)
                    / denominator,
                    0.0,
                    1.0,
                )
            )

        relative_position = relative_start + alpha * relative_velocity
        return float(np.linalg.norm(relative_position))


    @staticmethod
    def _reconstruct_path(parent, state):
        path = []
        while state is not None:
            path.append((state[0], state[1]))
            state = parent[state]
        path.reverse()
        return path


    def compute_control(self, obs):
        """Follow the next scheduled waypoint and wait if reached early."""
        if len(self.plan) == 0:
            return np.zeros(2)

        time_step = self.ctx.shared["time_step"]

        # Current time slot and its next scheduled waypoint.
        slot = min(
            int(obs.t / time_step),
            len(self.plan) - 1,
        )
        target_index = min(slot + 1, len(self.plan) - 1)
        target = self.plan[target_index]

        direction = target - obs.p
        distance = np.linalg.norm(direction)

        # If the waypoint is reached early, wait for the next time slot.
        if distance < 0.03:
            return np.zeros(2)

        if obs.spec.model == SINGLE_INTEGRATOR:
            # Never step past the waypoint (an overshoot can touch a wall).
            speed = min(obs.spec.v_max, distance / self.ctx.dt)
            return speed * direction / distance

        return track_point(obs, target, slow_down=True)

