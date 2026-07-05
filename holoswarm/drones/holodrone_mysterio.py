"""
holodrone_mysterio.py
=====================
Watch Mysterio's playbook happen: real gym-pybullet-drones project the illusions
while the trained agent (Spider-Man, blue sphere) falls for each trick.

    python holodrone_mysterio.py --trick wall     fire wall across open ground
    python holodrone_mysterio.py --trick train    cloaked real hazard on his path
    python holodrone_mysterio.py --trick decoy    green beacon lure + hidden trap
    python holodrone_mysterio.py --trick herd     fire funnel into a hidden trap
    python holodrone_mysterio.py --trick swarm    blockers keep landing ahead of him

Cloaked REAL hazards are drawn as translucent dark-red cylinders (they exist,
physics-wise, but the agent's sensors do not report them). Sensor rays from the
agent: red = reading a hit, green = clear. Trail shows the path taken. Ends with
a verdict + CSV log. 'q' quits.

Needs: policy.npz, illusion_rl.py, mysterio_tricks.py in this folder, plus
gym-pybullet-drones.
"""

import argparse
import csv
import math
import time

import numpy as np

from illusion_rl import (N_SENS, R_SENS, SENS_DIRS, AGENT_R, GOAL_TOL,
                         get_obs, act, load_policy)
from mysterio_tricks import (REAL_GOAL, NONE, static, fixed_goal,
                             wall_circles, natural_crossing)

SIM_HZ, CTRL_HZ = 240, 48
N_DRONES = 10
AGENT_SPEED = 1.5
START = np.array([-4.0, 0.15])
DURATION = 120.0
C_BEAM = [0.20, 0.85, 1.00]
C_HUD = [0.9, 1.0, 1.0]
FIRE = [(0.0, (1.0, 0.9, 0.35)), (0.5, (1.0, 0.45, 0.1)), (1.0, (0.85, 0.12, 0.05))]
GREEN = [(0.0, (0.6, 1.0, 0.6)), (1.0, (0.1, 0.9, 0.3))]


# ----------------------------------------------------------------------------
# Hologram cloud generators
# ----------------------------------------------------------------------------
def _cols(h, stops):
    hs = np.array([s[0] for s in stops]); cs = np.array([s[1] for s in stops])
    out = np.zeros((h.shape[0], 3))
    for c in range(3):
        out[:, c] = np.interp(h, hs, cs[:, c])
    return out


def _flicker(col, t, phase):
    col = col * (0.6 + 0.4 * np.sin(t * 26.0 + phase))[:, None]
    wink = np.random.default_rng(int(t * 16)).random(len(col)) > 0.05
    col[~wink] *= 0.1
    return np.clip(col, 0, 1)


def curtain_cloud(circles, t, n=520, height=2.3, stops=FIRE):
    """Fire curtain rising above a line of wall circles."""
    rng = np.random.default_rng(11)
    idx = rng.integers(0, len(circles), n)
    h = rng.random(n)
    jx = rng.normal(0, 0.18, n); jy = rng.normal(0, 0.18, n)
    base = circles[idx]
    z = h * height + 0.25 * np.sin(h * 6 + t * 4.0) * h
    x = base[:, 0] + jx + 0.08 * np.sin(z * 3 + t * 2)
    y = base[:, 1] + jy + 0.08 * np.cos(z * 3 + t * 2.3)
    pos = np.stack([x, y, z], 1)
    return pos, _flicker(_cols(h, stops), t, rng.random(n) * 6.28)


def beacon_cloud(center, t, n=300):
    """Green goal beacon (the fake MJ) — a bright swirling column."""
    rng = np.random.default_rng(13)
    h = rng.random(n)
    ang = rng.random(n) * 6.28 + t * 2.2
    r = 0.35 * np.sqrt(rng.random(n)) * (1.0 + 0.3 * np.sin(h * 9 + t * 3))
    pos = np.stack([center[0] + r * np.cos(ang),
                    center[1] + r * np.sin(ang),
                    h * 2.4], 1)
    return pos, _flicker(_cols(h, GREEN), t, rng.random(n) * 6.28)


def shimmer_cloud(center, radius, t, n=160):
    """Faint grey shimmer over a CLOAKED real hazard — the lie itself."""
    rng = np.random.default_rng(17)
    d = rng.standard_normal((n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 2] = np.abs(d[:, 2])
    pos = np.asarray([center[0], center[1], 0.0]) + d * (radius + 0.15)
    pos[:, 2] = np.clip(pos[:, 2], 0.1, None)
    col = np.full((n, 3), 0.45) * (0.5 + 0.5 * np.sin(t * 20 + rng.random(n) * 6.28))[:, None]
    return pos, np.clip(col, 0, 1)


def orbs_cloud(blockers, t, n_per=90):
    """Small fire orbs at each leapfrog blocker."""
    if not len(blockers):
        return np.zeros((0, 3)), np.zeros((0, 3))
    P, C = [], []
    for i, b in enumerate(blockers):
        rng = np.random.default_rng(23 + i)
        d = rng.standard_normal((n_per, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        rr = 0.45 * rng.random(n_per) ** (1 / 3)
        pos = np.array([b[0], b[1], 0.7]) + d * rr[:, None]
        pos[:, 2] = np.clip(pos[:, 2], 0.08, None)
        h = rng.random(n_per)
        P.append(pos); C.append(_flicker(_cols(h, FIRE), t, rng.random(n_per) * 6.28))
    return np.vstack(P), np.vstack(C)


def line_stations(circles, n, z=2.6):
    """Spread n drones evenly above a wall's circles."""
    idx = np.linspace(0, len(circles) - 1, n).astype(int)
    return np.stack([circles[idx, 0], circles[idx, 1], np.full(n, z)], 1)


def dome_stations(center, n, r=1.9, cz=1.0):
    pts = np.zeros((n, 3)); phi = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        z = 0.2 + 0.72 * (i / (n - 1) if n > 1 else 0.5)
        rr = math.sqrt(max(0, 1 - z * z)); th = phi * i
        pts[i] = [center[0] + r * rr * math.cos(th),
                  center[1] + r * rr * math.sin(th), cz + r * z]
    return pts


# ----------------------------------------------------------------------------
# Trick scenes
# ----------------------------------------------------------------------------
def build_scene(trick, parts):
    """Returns dict with: sensed_fn, goal_fn, real (circles), holo_fn(t, apos),
    stations_fn(t, apos), cloaked (real circles drawn translucent), title."""
    if trick == "wall":
        wall = wall_circles((0.0, -2.6), (0.0, 2.6))
        st = line_stations(wall, N_DRONES)
        return dict(sensed_fn=static(wall), goal_fn=fixed_goal(REAL_GOAL),
                    real=NONE, cloaked=NONE,
                    holo_fn=lambda t, ap: curtain_cloud(wall, t),
                    stations_fn=lambda t, ap: st,
                    title="THE WALL: a fire wall that is not there")

    if trick == "train":
        c = natural_crossing(parts, START, x=0.0)
        hazard = np.array([[c[0], c[1], 0.7]])
        st = dome_stations(c, N_DRONES)
        return dict(sensed_fn=static([]), goal_fn=fixed_goal(REAL_GOAL),
                    real=hazard, cloaked=hazard,
                    holo_fn=lambda t, ap: shimmer_cloud(c, 0.7, t),
                    stations_fn=lambda t, ap: st,
                    title="THE TRAIN: a real hazard, cloaked from his sensors")

    if trick == "decoy":
        decoy = np.array([2.2, 2.6])
        rec = []
        from mysterio_tricks import rollout_t
        rollout_t(parts, START, NONE, static([]), fixed_goal(decoy), record=rec)
        traj = np.array(rec)[:, 1:3]
        k = int(np.argmin(np.abs(np.linalg.norm(traj - decoy, axis=1) - 0.8)))
        trap = np.array([[traj[k, 0], traj[k, 1], 0.55]])
        stb = dome_stations(decoy, N_DRONES // 2, r=1.5)
        stt = dome_stations(trap[0, :2], N_DRONES - N_DRONES // 2, r=1.4)
        st = np.vstack([stb, stt])

        def holo(t, ap):
            p1, c1 = beacon_cloud(decoy, t)
            p2, c2 = shimmer_cloud(trap[0, :2], 0.55, t)
            return np.vstack([p1, p2]), np.vstack([c1, c2])
        return dict(sensed_fn=static([]), goal_fn=fixed_goal(decoy),
                    real=trap, cloaked=trap,
                    holo_fn=holo, stations_fn=lambda t, ap: st,
                    title="THE DECOY: fake goal beacon, hidden trap on the way")

    if trick == "herd":
        upper = wall_circles((-3.0, 1.8), (2.0, 0.0))
        lower = wall_circles((-3.0, -2.6), (2.0, -1.2))
        funnel = np.vstack([upper, lower])
        exit_pt = natural_crossing(parts, np.array([START[0], 0.0]), x=1.2,
                                   sensed_fn=static(funnel))
        hazard = np.array([[exit_pt[0], exit_pt[1], 0.55]])
        st = np.vstack([line_stations(upper, N_DRONES // 2),
                        line_stations(lower, N_DRONES - N_DRONES // 2)])

        def holo(t, ap):
            p1, c1 = curtain_cloud(funnel, t, n=620)
            p2, c2 = shimmer_cloud(exit_pt, 0.55, t)
            return np.vstack([p1, p2]), np.vstack([c1, c2])
        return dict(sensed_fn=static(funnel), goal_fn=fixed_goal(REAL_GOAL),
                    real=hazard, cloaked=hazard,
                    holo_fn=holo, stations_fn=lambda t, ap: st,
                    title="THE HERD: fire funnel steering him into a hidden trap")

    if trick == "swarm":
        state = {"blockers": [], "last": -999.0}

        def sensed(t, pos):
            if t - state["last"] >= 22 * (1.0 / 10.0) * 10:  # ~2.2 s in sim time
                d = REAL_GOAL - pos; n = np.linalg.norm(d)
                if n > 0.8:
                    u = d / n
                    state["blockers"].append([pos[0] + 1.15 * u[0],
                                              pos[1] + 1.15 * u[1], 0.5])
                    state["blockers"] = state["blockers"][-4:]
                    state["last"] = t
            return (np.array(state["blockers"]) if state["blockers"] else NONE)

        def stations(t, ap):
            bl = state["blockers"][-4:]
            st = []
            for i in range(N_DRONES):
                if bl:
                    b = bl[i % len(bl)]
                    ring = 0.9 + 0.3 * (i // max(len(bl), 1))
                    a = 2 * math.pi * i / N_DRONES + t * 0.4
                    st.append([b[0] + ring * math.cos(a),
                               b[1] + ring * math.sin(a), 1.9])
                else:
                    a = 2 * math.pi * i / N_DRONES
                    st.append([ap[0] + 2.2 * math.cos(a),
                               ap[1] + 2.2 * math.sin(a), 2.2])
            return np.array(st)
        return dict(sensed_fn=sensed, goal_fn=fixed_goal(REAL_GOAL),
                    real=NONE, cloaked=NONE,
                    holo_fn=lambda t, ap: orbs_cloud(state["blockers"], t),
                    stations_fn=stations,
                    title="THE SWARM: blockers keep landing ahead of him")

    raise ValueError(trick)


# ----------------------------------------------------------------------------
def run(trick="wall", duration=DURATION, gui=True, policy_path="policy.npz"):
    import pybullet as p
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    parts = load_policy(policy_path)
    sc = build_scene(trick, parts)
    # sensed_fn in the pure env is step-indexed; here we pass sim-time * 10 as a
    # pseudo-step so spawn cadences roughly match (policy itself is stateless)
    sensed_fn, goal_fn = sc["sensed_fn"], sc["goal_fn"]

    init = sc["stations_fn"](0.0, START)
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=N_DRONES,
                     initial_xyzs=init, initial_rpys=np.zeros((N_DRONES, 3)),
                     physics=Physics.PYB,
                     pyb_freq=SIM_HZ, ctrl_freq=CTRL_HZ,  # older: freq=/aggregate_phy_steps=
                     gui=gui, user_debug_gui=False)
    cid = env.getPyBulletClient()
    CTRL_DT = getattr(env, "CTRL_TIMESTEP", 1.0 / CTRL_HZ)
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(N_DRONES)]
    ret = env.reset(); obs = ret[0] if isinstance(ret, tuple) else ret

    # scene dressing
    gvs = p.createVisualShape(p.GEOM_CYLINDER, radius=GOAL_TOL, length=0.02,
                              rgbaColor=[0.2, 1.0, 0.3, 0.9], physicsClientId=cid)
    p.createMultiBody(0, -1, gvs, basePosition=[REAL_GOAL[0], REAL_GOAL[1], 0.01],
                      physicsClientId=cid)
    for cx, cy, cr in np.asarray(sc["cloaked"]).reshape(-1, 3):
        vs = p.createVisualShape(p.GEOM_CYLINDER, radius=cr, length=1.1,
                                 rgbaColor=[0.55, 0.08, 0.08, 0.35],
                                 physicsClientId=cid)
        p.createMultiBody(0, -1, vs, basePosition=[cx, cy, 0.55], physicsClientId=cid)
    avs = p.createVisualShape(p.GEOM_SPHERE, radius=AGENT_R,
                              rgbaColor=[0.25, 0.55, 1.0, 1.0], physicsClientId=cid)
    agent = p.createMultiBody(0, -1, avs,
                              basePosition=[START[0], START[1], AGENT_R],
                              physicsClientId=cid)
    if gui:
        p.resetDebugVisualizerCamera(10.5, 55, -40, [0, 0, 1.0], physicsClientId=cid)

    apos = START.astype(float).copy()
    action = np.zeros((N_DRONES, 4))
    beam_ids = [-1] * N_DRONES
    ray_ids = [-1] * N_SENS
    elem_id, hud_id = -1, -1
    trail_prev, tick = apos.copy(), 0
    real = np.asarray(sc["real"]).reshape(-1, 3)

    log, path_len, status = [], 0.0, "timeout"
    t_sim, step_i = 0.0, 0
    wall0 = time.perf_counter()
    print(sc["title"] + "  ('q' to quit)")

    try:
        while t_sim < duration:
            obs = env.step(action)[0]
            stations = sc["stations_fn"](t_sim, apos)
            for j in range(N_DRONES):
                pj = np.asarray(obs[j][0:3])
                action[j, :], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=CTRL_DT, state=obs[j],
                    target_pos=stations[j], target_rpy=np.zeros(3))

            pseudo_step = t_sim * 10.0
            sensed = sensed_fn(pseudo_step, apos)
            o = get_obs(apos, goal_fn(pseudo_step, apos), sensed)
            a = act(parts, o)
            new = np.clip(apos + a * AGENT_SPEED * CTRL_DT, -5.0, 5.0)
            path_len += np.linalg.norm(new - apos)
            apos = new
            p.resetBasePositionAndOrientation(agent, [apos[0], apos[1], AGENT_R],
                                              [0, 0, 0, 1], physicsClientId=cid)
            sens = o[:N_SENS] * R_SENS
            log.append([round(t_sim, 3), apos[0], apos[1]] + list(sens))

            if len(real) > 0 and np.any(
                    np.linalg.norm(real[:, :2] - apos, axis=1) < real[:, 2] + AGENT_R):
                status = "collision"; break
            if np.linalg.norm(REAL_GOAL - apos) < GOAL_TOL:
                status = "goal"; break

            if gui:
                for k in range(N_SENS):
                    hit = sens[k] < R_SENS - 1e-6
                    end = apos + SENS_DIRS[k] * sens[k]
                    ray_ids[k] = p.addUserDebugLine(
                        [apos[0], apos[1], 0.25], [end[0], end[1], 0.25],
                        [1.0, 0.25, 0.2] if hit else [0.3, 0.9, 0.4], 1.2,
                        replaceItemUniqueId=ray_ids[k], physicsClientId=cid)
                tick += 1
                if tick % 10 == 0:
                    p.addUserDebugLine([trail_prev[0], trail_prev[1], 0.05],
                                       [apos[0], apos[1], 0.05],
                                       [0.35, 0.65, 1.0], 2.0, physicsClientId=cid)
                    trail_prev = apos.copy()
                hp, hc = sc["holo_fn"](t_sim, apos)
                if len(hp):
                    elem_id = p.addUserDebugPoints(hp.tolist(), hc.tolist(), 4.0,
                                                   replaceItemUniqueId=elem_id,
                                                   physicsClientId=cid)
                for j in range(N_DRONES):
                    aj = list(map(float, obs[j][0:3]))
                    tgt = stations[j].tolist(); tgt[2] = 0.8
                    beam_ids[j] = p.addUserDebugLine(aj, tgt, C_BEAM, 1.0,
                                                     replaceItemUniqueId=beam_ids[j],
                                                     physicsClientId=cid)
                hud = (f"{trick.upper()}  min-sensor {sens.min():4.2f} m  "
                       f"goal {np.linalg.norm(REAL_GOAL - apos):4.2f} m  t={t_sim:5.1f}s")
                hud_id = p.addUserDebugText(hud, [-4.5, 0, 3.8], C_HUD, 1.3,
                                            replaceItemUniqueId=hud_id,
                                            physicsClientId=cid)
                if ord('q') in p.getKeyboardEvents(physicsClientId=cid):
                    break

            t_sim += CTRL_DT; step_i += 1
            slack = wall0 + step_i * CTRL_DT - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
    finally:
        env.close()

    out = f"run_{trick}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "x", "y"] + [f"sensor_{k}" for k in range(N_SENS)])
        w.writerows(log)
    verdicts = dict(
        goal="reached the goal",
        collision="WALKED INTO THE HIDDEN TRAP" if len(real) else "collision",
        timeout="never reached the goal (pinned down)")
    print(f"\nRESULT [{trick}]: {verdicts[status]}  path={path_len:.2f} m  "
          f"t={t_sim:.1f}s  -> log {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trick", choices=["wall", "train", "decoy", "herd", "swarm"],
                    default="train")
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--policy", default="policy.npz")
    ap.add_argument("--no-gui", action="store_true")
    a = ap.parse_args()
    run(trick=a.trick, duration=a.duration, gui=not a.no_gui,
        policy_path=a.policy)
