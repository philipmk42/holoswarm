"""
holodrone_rl.py
===============
Visual deployment of the illusion-deception experiment on gym-pybullet-drones.

A dome of real cf2x drones (DSLPIDControl) projects a fire elemental onto the
trained agent's path. The agent (blue sphere) navigates from the left spawn to
the green goal disk using the SAME policy and SAME 12-sensor model it was
trained with (imported from illusion_rl.py). Two modes:

    --mode fooled   the phantom appears in the agent's range sensors
                    -> watch it detour around empty air
    --mode truth    sensors ignore the phantom (lidar-like / ground truth)
                    -> watch it walk straight THROUGH the monster

Live sensor rays are drawn from the agent (red = hit, green = clear), the HUD
shows readings + clearance, and a per-step CSV is written at the end.

Run:   python holodrone_rl.py --mode fooled
       python holodrone_rl.py --mode truth
Needs: policy.npz (from `python illusion_rl.py train`) in the same folder,
       plus gym-pybullet-drones. Keep illusion_rl.py next to this file.
"""

import argparse
import csv
import math
import time

import numpy as np

from illusion_rl import (N_SENS, R_SENS, SENS_DIRS, AGENT_R, GOAL_TOL,
                         radial_sensors, get_obs, act, load_policy, rollout)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SIM_HZ, CTRL_HZ = 240, 48
N_DRONES   = 9
AGENT_SPEED = 1.5            # m/s at full action
START      = np.array([-4.0, 0.15])
GOAL       = np.array([4.0, 0.0])
REAL_OBST  = [(-1.6, 1.9, 0.6), (1.7, -2.1, 0.7)]   # physical cylinders, off-path
PH_R       = 0.8             # phantom sensor footprint (m)
ILLU_R     = 0.9             # visual elemental size
ILLU_CZ    = 1.0             # elemental hover height
DOME_STAND = 1.3
N_POINTS   = 420
DURATION   = 90.0

C_BEAM = [0.20, 0.85, 1.00]; C_HUD = [0.85, 1.0, 1.0]
FIRE = [(0.0, (1.00, 0.90, 0.35)), (0.5, (1.00, 0.45, 0.10)), (1.0, (0.85, 0.12, 0.05))]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def fibonacci_dome(n):
    """Upper-hemisphere directions (drones can't go underground for a ground
    illusion, so the projector shell is a dome)."""
    pts = np.zeros((n, 3))
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        z = 0.15 + (0.95 - 0.15) * (i / (n - 1) if n > 1 else 0.5)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        th = phi * i
        pts[i] = [math.cos(th) * r, math.sin(th) * r, z]
    return pts


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a) if b != a else 0.0, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def elemental_cloud(center, radius, t, n=N_POINTS):
    """Compact fire-elemental point cloud (swirl + turbulence + flicker)."""
    rng = np.random.default_rng(7)
    h = rng.random(n)
    ang0 = rng.random(n) * 2 * math.pi
    u = rng.random(n)
    pphase = rng.random(n) * 2 * math.pi
    prof = 0.55 * np.exp(-((h - 0.32) / 0.16) ** 2) + 0.30 * np.exp(-((h - 0.80) / 0.06) ** 2)
    prof *= (1.0 - _ss(0.92, 1.0, h)) * (0.4 + 0.6 * _ss(0.0, 0.08, h))
    H = 2.0 * radius
    rr = prof * radius * np.sqrt(u)
    sw = ang0 + t * (0.5 + 1.3 * h)
    x, y, z = rr * np.cos(sw), rr * np.sin(sw), h * H - 0.5 * H
    ph, amp = t * 1.6, 0.10 * radius
    x += amp * (np.sin(y * 2.2 + ph) + 0.5 * np.sin(z * 3.0 + 1.7 * ph))
    y += amp * (np.cos(x * 2.2 + 1.1 * ph) + 0.5 * np.cos(z * 2.6 + 1.3 * ph))
    z += 0.5 * amp * np.sin(x * 2.5 + 0.9 * ph)
    pos = np.stack([x, y, z], 1) + np.asarray(center)
    hs = np.array([s[0] for s in FIRE]); cs = np.array([s[1] for s in FIRE])
    col = np.zeros((n, 3))
    for c in range(3):
        col[:, c] = np.interp(h, hs, cs[:, c])
    col *= (0.65 + 0.35 * np.sin(t * 28.0 + pphase))[:, None]
    col *= 0.85 + 0.15 * math.sin(t * 7.0)
    wink = np.random.default_rng(int(t * 18)).random(n) > 0.05
    col[~wink] *= 0.12
    return pos, np.clip(col, 0, 1)


# ----------------------------------------------------------------------------
# Deployment
# ----------------------------------------------------------------------------
def run(mode="fooled", num_drones=N_DRONES, duration=DURATION, gui=True,
        policy_path="policy.npz", csv_out=None):
    import pybullet as p
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    parts = load_policy(policy_path)
    fooled = (mode == "fooled")

    # phantom goes ON the agent's natural path: find the baseline crossing
    # in the pure env (same policy, same sensors)
    rec0 = []
    rollout(parts, START, GOAL, np.zeros((0, 3)), np.zeros((0, 3)), record=rec0)
    traj0 = np.array(rec0)[:, 1:3]
    cross = traj0[int(np.argmin(np.abs(traj0[:, 0])))]
    PH_XY = np.array([cross[0], cross[1]])
    real_circ = np.array([[ox, oy, orr] for ox, oy, orr in REAL_OBST])
    phantom_circ = np.array([[PH_XY[0], PH_XY[1], PH_R]])
    sensed_circ = np.vstack([real_circ, phantom_circ]) if fooled else real_circ
    illu_center = np.array([PH_XY[0], PH_XY[1], ILLU_CZ])

    dome = fibonacci_dome(num_drones)
    dome_r = ILLU_R + DOME_STAND
    slots = illu_center + dome_r * dome

    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=num_drones,
                     initial_xyzs=slots, initial_rpys=np.zeros((num_drones, 3)),
                     physics=Physics.PYB,
                     pyb_freq=SIM_HZ, ctrl_freq=CTRL_HZ,  # older: freq=/aggregate_phy_steps=
                     gui=gui, user_debug_gui=False)
    cid = env.getPyBulletClient()
    CTRL_DT = getattr(env, "CTRL_TIMESTEP", 1.0 / CTRL_HZ)
    ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(num_drones)]
    ret = env.reset(); obs = ret[0] if isinstance(ret, tuple) else ret

    # scene: goal disk, real obstacles, agent
    gvs = p.createVisualShape(p.GEOM_CYLINDER, radius=GOAL_TOL, length=0.02,
                              rgbaColor=[0.2, 1.0, 0.3, 0.8], physicsClientId=cid)
    p.createMultiBody(0, -1, gvs, basePosition=[GOAL[0], GOAL[1], 0.01], physicsClientId=cid)
    for ox, oy, orr in REAL_OBST:
        ovs = p.createVisualShape(p.GEOM_CYLINDER, radius=orr, length=1.2,
                                  rgbaColor=[0.45, 0.33, 0.2, 1.0], physicsClientId=cid)
        p.createMultiBody(0, -1, ovs, basePosition=[ox, oy, 0.6], physicsClientId=cid)
    avs = p.createVisualShape(p.GEOM_SPHERE, radius=AGENT_R,
                              rgbaColor=[0.25, 0.55, 1.0, 1.0], physicsClientId=cid)
    agent = p.createMultiBody(0, -1, avs, basePosition=[START[0], START[1], AGENT_R],
                              physicsClientId=cid)
    if gui:
        p.resetDebugVisualizerCamera(9.5, 55, -35, [0, 0, 1.0], physicsClientId=cid)

    apos = START.astype(float).copy()
    action = np.zeros((num_drones, 4))
    beam_ids = [-1] * num_drones
    ray_ids = [-1] * N_SENS
    elem_id, hud_id = -1, -1
    trail_prev, trail_ct = apos.copy(), 0

    log = []
    t_sim, step_i = 0.0, 0
    path_len, min_clear = 0.0, np.inf
    status = "timeout"
    wall0 = time.perf_counter()
    print(f"mode={mode.upper()}  phantom at ({PH_XY[0]:.2f},{PH_XY[1]:.2f})  "
          f"drones={num_drones}  ('q' to quit)")

    try:
        while t_sim < duration:
            obs = env.step(action)[0]

            # drones hold the projector dome
            for j in range(num_drones):
                pj = np.asarray(obs[j][0:3])
                d = illu_center - pj
                yaw = math.atan2(d[1], d[0])
                action[j, :], _, _ = ctrl[j].computeControlFromState(
                    control_timestep=CTRL_DT, state=obs[j],
                    target_pos=slots[j], target_rpy=np.array([0.0, 0.0, yaw]))

            # agent: SAME policy + SAME sensor model as training
            o = get_obs(apos, GOAL, sensed_circ)
            a = act(parts, o)
            new = np.clip(apos + a * AGENT_SPEED * CTRL_DT, -5.0, 5.0)
            path_len += np.linalg.norm(new - apos)
            apos = new
            min_clear = min(min_clear, np.linalg.norm(apos - PH_XY))
            p.resetBasePositionAndOrientation(agent, [apos[0], apos[1], AGENT_R],
                                              [0, 0, 0, 1], physicsClientId=cid)
            sens = o[:N_SENS] * R_SENS
            log.append([round(t_sim, 3), apos[0], apos[1],
                        np.linalg.norm(apos - PH_XY)] + list(sens))

            # termination
            if np.any(np.linalg.norm(real_circ[:, :2] - apos, axis=1)
                      < real_circ[:, 2] + AGENT_R):
                status = "collision"; break
            if np.linalg.norm(GOAL - apos) < GOAL_TOL:
                status = "goal"; break

            if gui:
                # sensor rays: red = hit, green = clear
                for k in range(N_SENS):
                    hit = sens[k] < R_SENS - 1e-6
                    end = apos + SENS_DIRS[k] * sens[k]
                    ray_ids[k] = p.addUserDebugLine(
                        [apos[0], apos[1], 0.25], [end[0], end[1], 0.25],
                        [1.0, 0.25, 0.2] if hit else [0.3, 0.9, 0.4], 1.2,
                        replaceItemUniqueId=ray_ids[k], physicsClientId=cid)
                # trail
                trail_ct += 1
                if trail_ct % 10 == 0:
                    p.addUserDebugLine([trail_prev[0], trail_prev[1], 0.05],
                                       [apos[0], apos[1], 0.05],
                                       [0.35, 0.65, 1.0], 2.0, physicsClientId=cid)
                    trail_prev = apos.copy()
                # illusion + beams
                pos, col = elemental_cloud(illu_center, ILLU_R, t_sim)
                elem_id = p.addUserDebugPoints(pos.tolist(), col.tolist(), 4.0,
                                               replaceItemUniqueId=elem_id,
                                               physicsClientId=cid)
                for j in range(num_drones):
                    aj = list(map(float, obs[j][0:3]))
                    beam_ids[j] = p.addUserDebugLine(aj, illu_center.tolist(),
                                                     C_BEAM, 1.2,
                                                     replaceItemUniqueId=beam_ids[j],
                                                     physicsClientId=cid)
                hud = (f"{mode.upper()}  min-sensor {sens.min():4.2f} m  "
                       f"clearance {np.linalg.norm(apos - PH_XY):4.2f} m  "
                       f"goal {np.linalg.norm(GOAL - apos):4.2f} m")
                hud_id = p.addUserDebugText(hud, [-4.5, 0, 3.6], C_HUD, 1.3,
                                            replaceItemUniqueId=hud_id,
                                            physicsClientId=cid)
                if ord('q') in p.getKeyboardEvents(physicsClientId=cid):
                    break

            t_sim += CTRL_DT
            step_i += 1
            slack = wall0 + step_i * CTRL_DT - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
    finally:
        env.close()

    out = csv_out or f"run_{mode}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "x", "y", "clear_to_phantom"] +
                   [f"sensor_{k}" for k in range(N_SENS)])
        w.writerows(log)
    walked_through = min_clear < PH_R
    print(f"\nRESULT [{mode}]: status={status}  path={path_len:.2f} m  "
          f"min clearance to phantom={min_clear:.2f} m")
    print(("-> agent walked THROUGH the hologram (not fooled)" if walked_through
           else "-> agent detoured around empty air (FOOLED)"))
    print(f"log written to {out} ({len(log)} steps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fooled", "truth"], default="fooled")
    ap.add_argument("--drones", type=int, default=N_DRONES)
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--policy", default="policy.npz")
    ap.add_argument("--no-gui", action="store_true")
    a = ap.parse_args()
    run(mode=a.mode, num_drones=a.drones, duration=a.duration,
        gui=not a.no_gui, policy_path=a.policy)
