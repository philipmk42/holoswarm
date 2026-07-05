"""
mysterio_tricks.py
==================
Mysterio's playbook vs the trained agent (Spider-Man), Far From Home style.

Five tricks, all implemented as perception attacks on the SAME trained policy
(policy.npz from illusion_rl.py). Nothing about the agent changes — only what
its sensors and goal-signal report:

  THE WALL   phantom barrier across open space (illusion ADDS fake geometry)
  THE TRAIN  a real deadly obstacle CLOAKED from the sensors (illusion HIDES
             real danger — the "step in front of the train" beat)
  THE DECOY  the goal signal is spoofed to a fake beacon (fake MJ), whose
             approach path crosses a cloaked trap
  THE HERD   funnel of phantom walls steers the agent off its natural path
             into a cloaked hazard it would otherwise never touch
  THE SWARM  phantoms orbit the agent continuously to disorient it

Usage:
    python mysterio_tricks.py            # runs the whole playbook, prints table
Deps: numpy + illusion_rl.py + policy.npz in the same folder.
"""

import csv
import math

import numpy as np

from illusion_rl import (N_SENS, R_SENS, GOAL_TOL, AGENT_R,
                         get_obs, act, load_policy)

START_X = -4.0
REAL_GOAL = np.array([4.0, 0.0])
MAX_STEPS = 300
VMAX = 0.15
N_RUNS = 25


# ----------------------------------------------------------------------------
# Generalized rollout: sensors and perceived goal can lie, and can vary in time
# ----------------------------------------------------------------------------
def rollout_t(parts, start, real, sensed_fn, goal_percv_fn,
              goal_real=REAL_GOAL, max_steps=MAX_STEPS, record=None,
              probes=()):
    """real: (m,3) circles with PHYSICS. sensed_fn(t,pos)->circles the sensors
    see. goal_percv_fn(t,pos)->the goal the agent THINKS it has. Success is
    always measured against goal_real. probes: points to track min distance to."""
    pos = np.asarray(start, dtype=float).copy()
    path = 0.0
    minp = [np.inf] * len(probes)
    for t in range(max_steps):
        obs = get_obs(pos, goal_percv_fn(t, pos), sensed_fn(t, pos))
        a = act(parts, obs)
        new = np.clip(pos + a * VMAX, -5.0, 5.0)
        path += np.linalg.norm(new - pos)
        pos = new
        if record is not None:
            record.append([t, pos[0], pos[1]] + list(obs[:N_SENS] * R_SENS))
        for i, pb in enumerate(probes):
            minp[i] = min(minp[i], np.linalg.norm(pos - np.asarray(pb)))
        if len(real) > 0 and np.any(
                np.linalg.norm(real[:, :2] - pos, axis=1) < real[:, 2] + AGENT_R):
            return dict(status="collision", path=path, steps=t + 1, minp=minp, end=pos)
        if np.linalg.norm(goal_real - pos) < GOAL_TOL:
            return dict(status="goal", path=path, steps=t + 1, minp=minp, end=pos)
    return dict(status="timeout", path=path, steps=max_steps, minp=minp, end=pos)


def static(circles):
    c = np.asarray(circles).reshape(-1, 3) if len(circles) else np.zeros((0, 3))
    return lambda t, pos: c


def fixed_goal(g):
    g = np.asarray(g, dtype=float)
    return lambda t, pos: g


NONE = np.zeros((0, 3))


def wall_circles(p0, p1, r=0.42, spacing=0.6):
    """Line of phantom circles from p0 to p1 — a fake wall."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    n = max(2, int(L / spacing) + 1)
    ts = np.linspace(0, 1, n)
    return np.array([[*(p0 + t * (p1 - p0)), r] for t in ts])


def starts(n=N_RUNS):
    for i in range(n):
        rng = np.random.default_rng(20_000 + i)
        yield np.array([START_X, rng.uniform(-0.4, 0.4)])


def crossing_at(traj, x):
    """Trajectory point closest to vertical line x."""
    return traj[int(np.argmin(np.abs(traj[:, 0] - x)))]


def natural_crossing(parts, start, x=0.0, sensed_fn=None, goal_fn=None):
    rec = []
    rollout_t(parts, start, NONE,
              sensed_fn or static([]), goal_fn or fixed_goal(REAL_GOAL),
              record=rec)
    return crossing_at(np.array(rec)[:, 1:3], x)


def pct(xs):
    return 100.0 * float(np.mean(xs))


# ----------------------------------------------------------------------------
# The playbook
# ----------------------------------------------------------------------------
def trick_wall(parts):
    wall = wall_circles((0.0, -2.6), (0.0, 2.6))
    base, tricked = [], []
    for st in starts():
        base.append(rollout_t(parts, st, NONE, static([]), fixed_goal(REAL_GOAL)))
        tricked.append(rollout_t(parts, st, NONE, static(wall), fixed_goal(REAL_GOAL)))
    bs, ts = [r["status"] == "goal" for r in base], [r["status"] == "goal" for r in tricked]
    bp = np.mean([r["path"] for r in base])
    tp = np.mean([r["path"] for r in tricked])
    return dict(name="THE WALL", cols=(
        f"success {pct(ts):5.1f}% (was {pct(bs):5.1f}%)",
        f"path {tp:5.2f} m (+{(tp/bp-1)*100:4.1f}%)",
        "agent detours around a wall that is not there"))


def trick_train(parts):
    honest_hit, cloaked_hit = [], []
    for st in starts():
        c = natural_crossing(parts, st, x=0.0)
        hazard = np.array([[c[0], c[1], 0.7]])
        rh = rollout_t(parts, st, hazard, static(hazard), fixed_goal(REAL_GOAL))
        rc = rollout_t(parts, st, hazard, static([]), fixed_goal(REAL_GOAL))
        honest_hit.append(rh["status"] == "collision")
        cloaked_hit.append(rc["status"] == "collision")
    return dict(name="THE TRAIN", cols=(
        f"hit rate honest sensing {pct(honest_hit):5.1f}%",
        f"hit rate CLOAKED {pct(cloaked_hit):5.1f}%",
        "the illusion hides a real hazard on the agent's own path"))


def trick_decoy(parts, decoy=np.array([2.2, 2.6])):
    lured, trapped, reached, dmins = [], [], [], []
    trap_probe = None
    for st in starts():
        # place the trap on this run's approach path to the decoy, 0.8m short
        rec = []
        rollout_t(parts, st, NONE, static([]), fixed_goal(decoy), record=rec)
        traj = np.array(rec)[:, 1:3]
        d2 = np.linalg.norm(traj - decoy, axis=1)
        k = int(np.argmin(np.abs(d2 - 0.8)))
        trap = np.array([[traj[k, 0], traj[k, 1], 0.55]])
        trap_probe = trap[0, :2]
        r = rollout_t(parts, st, trap, static([]), fixed_goal(decoy),
                      probes=[decoy])
        dmins.append(r["minp"][0])
        lured.append(r["minp"][0] < 1.8)
        trapped.append(r["status"] == "collision")
        reached.append(r["status"] == "goal")
    return dict(name="THE DECOY", cols=(
        f"lured toward fake goal {pct(lured):5.1f}%  (closest approach {np.mean(dmins):.2f} m)",
        f"caught by hidden trap {pct(trapped):5.1f}%",
        f"reached REAL goal {pct(reached):5.1f}% — chasing fake MJ into the trap"))


def trick_herd(parts):
    upper = wall_circles((-3.0, 1.8), (2.0, 0.0))
    lower = wall_circles((-3.0, -2.6), (2.0, -1.2))
    funnel = np.vstack([upper, lower])
    # find the funnel exit crossing with a probe run, put the cloaked hazard there
    probe_start = np.array([START_X, 0.0])
    exit_pt = natural_crossing(parts, probe_start, x=1.2, sensed_fn=static(funnel))
    hazard = np.array([[exit_pt[0], exit_pt[1], 0.55]])
    nat_hit, herd_hit = [], []
    for st in starts():
        rn = rollout_t(parts, st, hazard, static([]), fixed_goal(REAL_GOAL))
        rh = rollout_t(parts, st, hazard, static(funnel), fixed_goal(REAL_GOAL))
        nat_hit.append(rn["status"] == "collision")
        herd_hit.append(rh["status"] == "collision")
    return dict(name="THE HERD", cols=(
        f"hazard hit on natural path {pct(nat_hit):5.1f}%",
        f"hazard hit when HERDED {pct(herd_hit):5.1f}%",
        "phantom funnel steers the agent into a trap it would never touch"))


def trick_swarm(parts):
    def make_leapfrog():
        """Phantoms leapfrog AHEAD of the agent and then stay put (world-anchored,
        like drones landing in front of Spider-Man to cut him off). A phantom that
        tracks the agent is provably ignored by this policy; anchored ones are not."""
        state = {"blockers": [], "last": -999}
        def sensed(t, pos):
            if t - state["last"] >= 22:
                d = REAL_GOAL - pos
                n = np.linalg.norm(d)
                if n > 0.8:
                    u = d / n
                    state["blockers"].append([pos[0] + 1.15 * u[0],
                                              pos[1] + 1.15 * u[1], 0.5])
                    state["blockers"] = state["blockers"][-4:]
                    state["last"] = t
            return np.array(state["blockers"]) if state["blockers"] else NONE
        return sensed
    base, tricked = [], []
    for st in starts():
        base.append(rollout_t(parts, st, NONE, static([]), fixed_goal(REAL_GOAL)))
        tricked.append(rollout_t(parts, st, NONE, make_leapfrog(), fixed_goal(REAL_GOAL)))
    bs = np.mean([r["steps"] for r in base])
    ts = np.mean([r["steps"] for r in tricked])
    succ = [r["status"] == "goal" for r in tricked]
    to = [r["status"] == "timeout" for r in tricked]
    return dict(name="THE SWARM", cols=(
        f"success {pct(succ):5.1f}%   timeouts {pct(to):5.1f}%",
        f"time-to-goal x{ts/bs:4.2f} (steps {bs:.0f} -> {ts:.0f})",
        "world-anchored blockers keep landing in the agent's path"))


def dump_train_readings(parts, out="train_trick_readings.csv"):
    """The chilling one: per-step sensors while walking into the cloaked hazard."""
    st = np.array([START_X, 0.15])
    c = natural_crossing(parts, st, x=0.0)
    hazard = np.array([[c[0], c[1], 0.7]])
    rec = []
    r = rollout_t(parts, st, hazard, static([]), fixed_goal(REAL_GOAL), record=rec)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "x", "y"] + [f"sensor_{k}" for k in range(N_SENS)])
        w.writerows(rec)
    last = rec[-1]
    print(f"\n[readings] THE TRAIN: status={r['status']} after {r['steps']} steps; "
          f"hazard at ({hazard[0,0]:.2f},{hazard[0,1]:.2f})")
    print(f"[readings] final-step sensors (m): "
          + " ".join(f"{v:4.1f}" for v in last[3:]))
    print(f"[readings] every sensor read CLEAR while it walked into a real hazard.")
    print(f"[readings] wrote {out} ({len(rec)} steps)")


def main(policy_path="policy.npz"):
    parts = load_policy(policy_path)
    print("MYSTERIO'S PLAYBOOK vs the trained agent"
          f"  ({N_RUNS} runs per condition)\n" + "=" * 74)
    for trick in (trick_wall, trick_train, trick_decoy, trick_herd, trick_swarm):
        r = trick(parts)
        print(f"{r['name']:10s} | " + "\n           | ".join(r["cols"]))
        print("-" * 74)
    dump_train_readings(parts)


if __name__ == "__main__":
    main()
