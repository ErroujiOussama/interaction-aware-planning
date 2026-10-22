"""Bridge to the PythonRobotics library (git submodule in ../lib).

PythonRobotics is a collection of scripts, not an installable package, so we
put its root on sys.path and re-export the pieces the simulator reuses:

- AStarPlanner        PathPlanning/AStar/a_star.py           (global grid path)
- PathFinderController PathTracking/move_to_pose/move_to_pose.py (unicycle tracking)
- dwa                 PathPlanning/DynamicWindowApproach/dynamic_window_approach.py
- angle_mod           utils/angle.py

The scripts plot and print by default; both are switched off here.
"""

import contextlib
import io
import pathlib
import sys

PR_ROOT = pathlib.Path(__file__).resolve().parents[1] / "lib" / "PythonRobotics"

if not (PR_ROOT / "PathPlanning").is_dir():
    raise ImportError(
        f"PythonRobotics not found at {PR_ROOT}.\n"
        "Run:  git submodule update --init 02_simulateur_2d/lib/PythonRobotics"
    )
if str(PR_ROOT) not in sys.path:
    sys.path.insert(0, str(PR_ROOT))

from PathPlanning.AStar import a_star as _a_star  # noqa: E402
from PathPlanning.DynamicWindowApproach import dynamic_window_approach as dwa  # noqa: E402
from PathTracking.move_to_pose.move_to_pose import PathFinderController  # noqa: E402
from utils.angle import angle_mod  # noqa: E402

_a_star.show_animation = False
dwa.show_animation = False
AStarPlanner = _a_star.AStarPlanner


@contextlib.contextmanager
def quiet():
    """Silence the print() calls inside PythonRobotics scripts."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


__all__ = ["AStarPlanner", "PathFinderController", "dwa", "angle_mod", "quiet", "PR_ROOT"]
