#!/usr/bin/env python
"""
demo.py - SO-101 pick-and-place in MuJoCo, rendered headless to demo.mp4 (+ PNG keyframes).

    MUJOCO_GL=egl .venv/bin/python demo.py            # default 960x720 @ 30 fps
    .venv/bin/python demo.py --width 640 --height 480 --camera side

Scripted controller: top-down IK waypoints (damped least squares) -> minimum-jerk
joint-space trajectories -> MuJoCo position actuators (STS3215 servo model).
The grasp is physical (friction contacts between jaws and cube, no "magic" weld).
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np

from so101_sim import CUBE_HALF, GRIP_CLOSED, GRIP_OPEN, HOME_Q, SO101Sim, build_model

OUT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--camera", default="front", choices=["front", "side", "top"])
    ap.add_argument("--cube", type=float, nargs=2, default=(0.22, 0.10), metavar=("X", "Y"))
    ap.add_argument("--target", type=float, nargs=2, default=(0.20, -0.12), metavar=("X", "Y"))
    ap.add_argument("--out", default=str(OUT / "demo.mp4"))
    args = ap.parse_args()

    frames_dir = OUT / "frames"
    frames_dir.mkdir(exist_ok=True)

    model = build_model(cube_pos=args.cube, target_pos=args.target)
    sim = SO101Sim(model, width=args.width, height=args.height, fps=args.fps, camera=args.camera)

    home, _ = sim.ik.solve([0.17, 0.0, 0.10], HOME_Q)
    sim.reset(home, GRIP_OPEN)

    keyframes = {}

    def snap(name):
        keyframes[name] = sim.render()

    cube0 = sim.body_pos("cube")
    target = np.array([*args.target, CUBE_HALF])
    hover = 0.06

    sim.caption = "SO-101 | ready"
    sim.hold(1.0)
    snap("01_ready")

    sim.caption = "1. reach above cube"
    sim.move_to(cube0 + [0, 0, hover], 2.0, grip=GRIP_OPEN)

    sim.caption = "2. descend"
    sim.move_to(cube0 + [0, 0, 0.0], 1.2)
    sim.hold(0.2)
    snap("02_pre_grasp")

    sim.caption = "3. close gripper"
    sim.set_grip(GRIP_CLOSED, 0.8)
    sim.hold(0.3)
    snap("03_grasped")

    sim.caption = "4. lift"
    sim.move_to(cube0 + [0, 0, hover], 1.3)
    lifted_z = sim.body_pos("cube")[2]
    snap("04_lifted")

    sim.caption = "5. carry to target"
    mid = 0.5 * (cube0 + target) + [0, 0, hover]
    sim.move_to(mid, 1.3)
    snap("05_carry")
    sim.move_to(target + [0, 0, hover], 1.3)

    sim.caption = "6. place"
    # lower until the *cube* (not the fingertips) is ~2 mm above the floor
    hang = sim.grasp_point()[2] - sim.body_pos("cube")[2]
    sim.move_to(target + [0, 0, hang + 0.002], 1.2)
    sim.set_grip(GRIP_OPEN, 0.6)
    sim.hold(0.2)
    snap("06_placed")

    sim.caption = "7. retreat + home"
    sim.move_to(target + [0, 0, hover], 1.0)
    sim.move_joints(home, 1.8, grip=GRIP_OPEN)
    final = sim.body_pos("cube")
    err = np.linalg.norm(final[:2] - target[:2])
    sim.caption = f"done: cube placed {err*100:.1f} cm from target"
    sim.hold(1.5)
    snap("07_done")

    sim.save_video(args.out)
    for name, img in keyframes.items():
        imageio.imwrite(frames_dir / f"{name}.png", img)

    print(f"cube start  {np.round(cube0, 3)}")
    print(f"cube lifted to z={lifted_z:.3f} m (rest z={cube0[2]:.3f})")
    print(f"cube final  {np.round(final, 3)}  target {np.round(target, 3)}  xy error {err*100:.1f} cm")
    print(f"video: {args.out}  ({len(sim.frames)} frames, {len(sim.frames)/args.fps:.1f} s)")
    print(f"keyframes: {frames_dir}/")
    success = lifted_z > cube0[2] + 0.03 and err < 0.03
    print("SUCCESS" if success else "FAILED (cube not lifted / not on target)")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
