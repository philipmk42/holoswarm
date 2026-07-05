"""
illusion_rl.py
==============
Do drone-projected illusions fool a trained RL agent? — training + experiment core.

An agent is trained (from scratch, numpy-only, no torch) to navigate an arena to a
goal while avoiding REAL obstacles, sensing the world through 12 radial range
sensors. Then we place a PHANTOM obstacle in its path — the drone hologram: it
appears in the sensors but has no physics — and measure behavior:

    fooled mode ....... sensors include the phantom (vision-like perception)
    ground-truth mode . sensors ignore it          (lidar/collision-like perception)

Metrics: min clearance from the phantom, path length, success, % of runs where the
agent detoured around empty air. Per-step sensor readings are logged to CSV — the
"reading of it".

Usage:
    python illusion_rl.py train              # trains, saves policy.npz (~1 min CPU)
    python illusion_rl.py eval               # runs the deception experiment, prints table
Deps: numpy only. (The PyBullet/gym-pybullet-drones visual deployment is
holodrone_rl.py, which imports this file.)
"""

import argparse
import csv
import math
import time

import numpy as np

# ----------------------------------------------------------------------------
# Environment definition
# ----------------------------------------------------------------------------
ARENA     = 5.0          # world is [-5,5]^2
N_SENS    = 12           # radial range sensors
R_SENS    = 3.0          # sensor max range (m)
VMAX      = 0.15         # max displacement per policy step (m)
MAX_STEPS = 250
AGENT_R   = 0.15
GOAL_TOL  = 0.35

SENS_DIRS = np.stack([np.array([math.cos(a), math.sin(a)])
                      for a in np.linspace(0, 2 * math.pi, N_SENS, endpoint=False)])

OBS_DIM = N_SENS + 3     # sensors + goal unit vector + goal distance
H       = 24             # hidden units
ACT     = 2
DIM     = OBS_DIM * H + H + H * ACT + ACT


def radial_sensors(pos, circles, dirs=SENS_DIRS, rmax=R_SENS):
    """Range reading along each direction to the nearest circle edge.
    circles: (m,3) rows [cx, cy, r]. Returns (n_sensors,) clipped to rmax."""
    if circles is None or len(circles) == 0:
        return np.full(len(dirs), rmax)
    C = circles[:, :2] - np.asarray(pos)      # (m,2)
    r = circles[:, 2]
    b = dirs @ C.T                            # (S,m) projection of center along ray
    c2 = (C ** 2).sum(1) - r ** 2             # (m,)
    disc = b * b - c2[None, :]
    t = b - np.sqrt(np.maximum(disc, 0.0))    # first intersection along ray
    t[(disc <= 0) | (t <= 1e-9)] = np.inf     # miss / behind / inside
    return np.clip(t.min(axis=1), 0.0, rmax)


def get_obs(pos, goal, sensed_circles):
    s = radial_sensors(pos, sensed_circles) / R_SENS
    d = np.asarray(goal) - np.asarray(pos)
    dist = np.linalg.norm(d)
    u = d / max(dist, 1e-9)
    return np.concatenate([s, u, [min(dist, 5.0) / 5.0]])


# ----------------------------------------------------------------------------
# Policy: tiny MLP, parameters as a flat vector (trained by CEM)
# ----------------------------------------------------------------------------
def unpack(theta):
    i = 0
    W1 = theta[i:i + OBS_DIM * H].reshape(OBS_DIM, H); i += OBS_DIM * H
    b1 = theta[i:i + H]; i += H
    W2 = theta[i:i + H * ACT].reshape(H, ACT); i += H * ACT
    b2 = theta[i:i + ACT]
    return W1, b1, W2, b2


def act(parts, obs):
    W1, b1, W2, b2 = parts
    h = np.tanh(obs @ W1 + b1)
    return np.tanh(h @ W2 + b2)              # action in [-1,1]^2


def load_policy(path="policy.npz"):
    d = np.load(path)
    return unpack(d["theta"])


# ----------------------------------------------------------------------------
# Episodes
# ----------------------------------------------------------------------------
def make_episode(seed):
    """Random training episode: start left, goal right, 3-5 circular obstacles,
    one guaranteed near the direct path so avoidance is actually exercised."""
    rng = np.random.default_rng(seed)
    start = np.array([-4.0, rng.uniform(-1.0, 1.0)])
    goal = np.array([4.0, rng.uniform(-1.5, 1.5)])
    circ = []
    t = rng.uniform(0.35, 0.65)
    c = start + t * (goal - start) + rng.uniform(-0.4, 0.4, size=2)
    circ.append([c[0], c[1], rng.uniform(0.55, 0.85)])
    for _ in range(int(rng.integers(2, 5))):
        c = np.array([rng.uniform(-2.5, 2.5), rng.uniform(-3.0, 3.0)])
        r = rng.uniform(0.5, 0.9)
        if np.linalg.norm(c - start) > 1.3 and np.linalg.norm(c - goal) > 1.3:
            circ.append([c[0], c[1], r])
    return start, goal, np.array(circ)


def rollout(parts, start, goal, real, sensed, probe=None, record=None,
            max_steps=MAX_STEPS):
    """Run one episode.
      real   - circles with PHYSICS (collision ends episode)
      sensed - circles the SENSORS see (this is where the phantom goes)
      probe  - optional (x,y): track min distance of trajectory to this point
    Returns dict(ret, status, path, min_probe, steps)."""
    pos = np.asarray(start, dtype=float).copy()
    ret, path = 0.0, 0.0
    dprev = np.linalg.norm(goal - pos)
    min_probe = np.linalg.norm(pos - probe) if probe is not None else np.inf
    for t in range(max_steps):
        obs = get_obs(pos, goal, sensed)
        a = act(parts, obs)
        new = np.clip(pos + a * VMAX, -ARENA, ARENA)
        path += np.linalg.norm(new - pos)
        pos = new
        if record is not None:
            record.append([t, pos[0], pos[1]] + list(obs[:N_SENS] * R_SENS))
        if probe is not None:
            min_probe = min(min_probe, np.linalg.norm(pos - probe))
        d = np.linalg.norm(goal - pos)
        ret += 10.0 * (dprev - d) - 0.05
        dprev = d
        if len(real) > 0 and np.any(
                np.linalg.norm(real[:, :2] - pos, axis=1) < real[:, 2] + AGENT_R):
            return dict(ret=ret - 30.0, status="collision", path=path,
                        min_probe=min_probe, steps=t + 1)
        if d < GOAL_TOL:
            return dict(ret=ret + 60.0, status="goal", path=path,
                        min_probe=min_probe, steps=t + 1)
    return dict(ret=ret, status="timeout", path=path,
                min_probe=min_probe, steps=max_steps)


# ----------------------------------------------------------------------------
# Training: cross-entropy method
# ----------------------------------------------------------------------------
POP, ELITE = 40, 8
PROBE_SEEDS = list(range(7000, 7024))


def probe_success(parts):
    ok = 0
    for s in PROBE_SEEDS:
        st, gl, ci = make_episode(s)
        if rollout(parts, st, gl, ci, ci)["status"] == "goal":
            ok += 1
    return ok / len(PROBE_SEEDS)


def train(iters=60, budget_s=100.0, out="policy.npz", seed=0, resume=None):
    rng = np.random.default_rng(seed)
    mean = np.zeros(DIM)
    std = np.full(DIM, 0.5)
    if resume:
        mean = np.load(resume)["theta"]
        std = np.full(DIM, 0.15)
        print(f"resuming from {resume}")
    t0 = time.time()
    best_sr = -1.0
    best_theta = mean.copy()
    for it in range(iters):
        n_eval = 4 if resume else 2
        seeds = [10_000 * it + k for k in range(n_eval)]     # common random numbers
        pop = rng.standard_normal((POP, DIM)) * std + mean
        fits = np.zeros(POP)
        for i in range(POP):
            parts = unpack(pop[i])
            f = 0.0
            for s in seeds:
                st, gl, ci = make_episode(s)
                f += rollout(parts, st, gl, ci, ci)["ret"]
            fits[i] = f / len(seeds)
        idx = np.argsort(fits)[::-1][:ELITE]
        elite = pop[idx]
        mean = elite.mean(0)
        floor = 0.01 if resume else 0.02
        std = elite.std(0) + max(floor, (0.1 if resume else 0.3) * (0.93 ** it))
        if it % 3 == 2 or it == iters - 1:
            sr = probe_success(unpack(mean))
            print(f"iter {it:02d}  elite_fit={fits[idx].mean():7.1f}  "
                  f"probe_success={sr:.2f}  t={time.time()-t0:5.1f}s", flush=True)
            if sr > best_sr:
                best_sr, best_theta = sr, mean.copy()
            if sr >= 0.85:
                print("early stop: probe success target reached")
                break
        if time.time() - t0 > budget_s:
            print("wall-clock budget reached")
            break
    np.savez(out, theta=best_theta, obs_dim=OBS_DIM, hidden=H,
             n_sens=N_SENS, r_sens=R_SENS, vmax=VMAX)
    print(f"saved {out}  (best probe success {best_sr:.2f})")
    return best_theta


# ----------------------------------------------------------------------------
# The deception experiment
# ----------------------------------------------------------------------------
PHANTOM = np.array([[0.0, 0.0, 0.8]])       # the hologram's footprint: circle r=0.8


def evaluate(policy_path="policy.npz", n_runs=25, csv_out="phantom_readings.csv"):
    parts = load_policy(policy_path)

    print("== 1) competence on REAL obstacles (100 random episodes) ==")
    stat = {"goal": 0, "collision": 0, "timeout": 0}
    for s in range(9000, 9100):
        st, gl, ci = make_episode(s)
        stat[rollout(parts, st, gl, ci, ci)["status"]] += 1
    print(f"   success {stat['goal']}%   collisions {stat['collision']}%   "
          f"timeouts {stat['timeout']}%")

    print("\n== 2) PHANTOM experiment: hologram placed ON the agent's natural path ==")
    goal = np.array([4.0, 0.0])
    R_PH = 0.8
    rows = {"fooled": [], "ground-truth": []}
    phantoms = []
    for i in range(n_runs):
        rng = np.random.default_rng(20_000 + i)
        start = np.array([-4.0, rng.uniform(-0.5, 0.5)])
        # baseline run (nothing sensed) to find where this agent naturally crosses x=0
        rec = []
        base = rollout(parts, start, goal, np.zeros((0, 3)), np.zeros((0, 3)),
                       record=rec)
        traj = np.array(rec)[:, 1:3]
        cross = traj[int(np.argmin(np.abs(traj[:, 0])))]
        ph = np.array([[cross[0], cross[1], R_PH]])
        phantoms.append(ph[0])
        # ground-truth mode: sensors ignore the phantom (== baseline, measured vs ph)
        rt = rollout(parts, start, goal, np.zeros((0, 3)), np.zeros((0, 3)),
                     probe=ph[0, :2])
        rows["ground-truth"].append(rt)
        # fooled mode: sensors see the phantom
        rf = rollout(parts, start, goal, np.zeros((0, 3)), ph, probe=ph[0, :2])
        rows["fooled"].append(rf)
    summary = {}
    for mode in ("fooled", "ground-truth"):
        res = rows[mode]
        succ = np.mean([r["status"] == "goal" for r in res]) * 100
        clear = np.array([r["min_probe"] for r in res])
        plen = np.mean([r["path"] for r in res])
        avoided = np.mean(clear > R_PH) * 100
        summary[mode] = (succ, clear.mean(), plen, avoided)
        print(f"   {mode:12s}  success {succ:5.1f}%   min-clearance "
              f"{clear.mean():4.2f} m   path {plen:5.2f} m   "
              f"detoured-around-empty-air {avoided:5.1f}%")

    f_cl, t_cl = summary["fooled"][1], summary["ground-truth"][1]
    f_pl, t_pl = summary["fooled"][2], summary["ground-truth"][2]
    print(f"\n   deception effect: clearance +{f_cl - t_cl:.2f} m, "
          f"path +{(f_pl / max(t_pl, 1e-9) - 1) * 100:.1f}% when fooled")

    print("\n== 3) the reading of it: one fooled run, per-step sensors -> CSV ==")
    start = np.array([-4.0, 0.15])
    rec0 = []
    rollout(parts, start, goal, np.zeros((0, 3)), np.zeros((0, 3)), record=rec0)
    traj0 = np.array(rec0)[:, 1:3]
    cross = traj0[int(np.argmin(np.abs(traj0[:, 0])))]
    ph = np.array([[cross[0], cross[1], 0.8]])
    print(f"   phantom placed at ({ph[0,0]:.2f},{ph[0,1]:.2f}) — on the natural path")
    rec = []
    r = rollout(parts, start, goal, np.zeros((0, 3)), ph,
                probe=ph[0, :2], record=rec)
    with open(csv_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "x", "y"] + [f"sensor_{k}" for k in range(N_SENS)])
        w.writerows(rec)
    arr = np.array(rec)
    k = int(np.argmin(np.linalg.norm(arr[:, 1:3] - ph[0, :2], axis=1)))
    pos_k = arr[k, 1:3]
    fooled_s = arr[k, 3:]
    truth_s = radial_sensors(pos_k, np.zeros((0, 3)))
    print(f"   closest approach step {int(arr[k,0])} at ({pos_k[0]:.2f},{pos_k[1]:.2f}), "
          f"clearance {np.linalg.norm(pos_k - ph[0,:2]):.2f} m")
    print(f"   fooled sensors (m):       " + " ".join(f"{v:4.1f}" for v in fooled_s))
    print(f"   ground-truth sensors (m): " + " ".join(f"{v:4.1f}" for v in truth_s))
    print(f"   -> the phantom is IN the readings; physics says nothing is there.")
    print(f"   wrote {csv_out} ({len(rec)} steps), status={r['status']}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--iters", type=int, default=60)
    tr.add_argument("--budget", type=float, default=100.0)
    tr.add_argument("--out", default="policy.npz")
    tr.add_argument("--resume", default=None)
    ev = sub.add_parser("eval")
    ev.add_argument("--policy", default="policy.npz")
    ev.add_argument("--runs", type=int, default=25)
    a = ap.parse_args()
    if a.cmd == "train":
        train(iters=a.iters, budget_s=a.budget, out=a.out, resume=a.resume)
    else:
        evaluate(policy_path=a.policy, n_runs=a.runs)
