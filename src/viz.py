"""Matplotlib animation (GIF/MP4 export) and static trajectory plot of a RunLog."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle, Polygon


def _setup_axes(log, ax):
    m = log.static_map
    xmin, ymin, xmax, ymax = m.bounds
    ax.set_xlim(xmin - 0.2, xmax + 0.2)
    ax.set_ylim(ymin - 0.2, ymax + 0.2)
    ax.set_aspect("equal")
    ax.add_patch(Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], fill=False, lw=2, ec="0.2"))
    for poly in m.polygons:
        ax.add_patch(Polygon(poly, fc="0.75", ec="0.3"))
    colors = plt.cm.tab10(np.arange(len(log.radii)) % 10)
    for i, (g, c) in enumerate(zip(log.goals, colors)):
        ax.plot(*g, "x", color=c, ms=9, mew=2)
        if i in log.paths:
            ax.plot(*log.paths[i].T, ":", color=c, lw=1, alpha=0.6)
    ax.set_title(f"{log.scenario} | {log.planner} | seed {log.seed}")
    return colors


def plot_trajectories(log, path=None, show=False):
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = _setup_axes(log, ax)
    for i, c in enumerate(colors):
        ax.plot(*log.pos[:, i].T, "-", color=c, lw=1.5, label=f"robot {i}")
        ax.add_patch(Circle(log.pos[-1, i], log.radii[i], fc=c, alpha=0.5))
    for t, i, j in log.collisions:
        k = int(round(t / log.dt))
        ax.plot(*log.pos[k, int(i)], "r*", ms=14)
    ax.legend(loc="upper right", fontsize=7)
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _contacts(log):
    """(T+1, N) bool: robot touches another robot or a wall at that frame (drawn with a red edge)."""
    P, R = log.pos, log.radii
    D = np.linalg.norm(P[:, :, None] - P[:, None], axis=-1) + np.eye(len(R)) * 1e9
    contact = (D < R[:, None] + R[None]).any(axis=-1)
    walls = np.array([[log.static_map.distance(p) < r for p, r in zip(frame, R)] for frame in P])
    return contact | walls


def animate(log, save=None, show=True, fps=None, trail=40):
    """save: None, '*.gif' or '*.mp4'. fps defaults to real time (1/dt)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = _setup_axes(log, ax)
    n = len(log.radii)
    discs = [ax.add_patch(Circle(log.pos[0, i], log.radii[i], fc=colors[i], ec="k", alpha=0.85)) for i in range(n)]
    heads = [ax.plot([], [], "k-", lw=1.5)[0] for _ in range(n)]
    trails = [ax.plot([], [], "-", color=colors[i], lw=1)[0] for i in range(n)]
    labels = [ax.text(*log.pos[0, i], str(i), ha="center", va="center", fontsize=7, color="w") for i in range(n)]
    info = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9)
    coll_steps = [int(round(t / log.dt)) for t, _, _ in log.collisions]
    contact = _contacts(log)

    def update(k):
        for i in range(n):
            p, th = log.pos[k, i], log.theta[k, i]
            discs[i].center = p
            discs[i].set_edgecolor("red" if contact[k, i] else "k")
            discs[i].set_linewidth(3 if contact[k, i] else 1)
            heads[i].set_data([p[0], p[0] + log.radii[i] * np.cos(th)], [p[1], p[1] + log.radii[i] * np.sin(th)])
            trails[i].set_data(*log.pos[max(0, k - trail):k + 1, i].T)
            labels[i].set_position(p)
        n_coll = sum(c <= k for c in coll_steps)
        info.set_text(f"t = {log.t[k]:5.1f} s   collisions: {n_coll}   clearance: {log.clearance[k]:+.2f} m")
        return [*discs, *heads, *trails, *labels, info]

    fps = fps or int(round(1 / log.dt))
    anim = animation.FuncAnimation(fig, update, frames=len(log.t), interval=1000 / fps, blit=False, repeat=False)
    if save:
        writer = animation.PillowWriter(fps=fps) if str(save).endswith(".gif") else animation.FFMpegWriter(fps=fps)
        anim.save(save, writer=writer)
    if show:
        plt.show()
    plt.close(fig)
    return anim
