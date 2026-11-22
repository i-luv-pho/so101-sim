"""
so101_sim.py - small helper library around the official SO-101 MuJoCo model.

Provides:
  * build_model()      - SO-101 (TheRobotStudio/SO-ARM100, new calibration) + cube + target pad + cameras
  * IK                 - damped-least-squares inverse kinematics for a "top-down" grasp pose
  * SO101Sim           - physics stepping, smooth joint-space moves, offscreen video recording

Headless rendering: set MUJOCO_GL=egl (default here) or MUJOCO_GL=osmesa before importing mujoco.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
SO101_DIR = ROOT / "third_party" / "SO-ARM100" / "Simulation" / "SO101"
SCENE_XML = SO101_DIR / "scene.xml"  # includes so101_new_calib.xml + floor + lights

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER = "gripper"
# Gripper joint (radians): larger = more open. ~-0.17 fully closed, 1.75 fully open.
GRIP_OPEN = 0.9
GRIP_CLOSED = -0.15  # commanded past contact -> squeezes with the servo's force limit

CUBE_HALF = 0.0125  # 2.5 cm cube
# Grasp point expressed in the 'gripperframe' site frame (x = approach axis,
# z = jaw-closing direction, origin = tip of the fixed jaw).
GRASP_OFFSET = np.array([-0.016, 0.0, CUBE_HALF + 0.005])  # +5 mm clearance from fixed jaw

HOME_Q = np.array([0.0, -1.57, 1.45, 1.0, 0.0])  # tucked, gripper looking forward/down


def build_model(cube_pos=(0.22, 0.10), target_pos=(0.20, -0.12)) -> mujoco.MjModel:
    """SO-101 scene + a graspable cube + a target pad, compiled via MjSpec."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.quality.shadowsize = 4096
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 3

    # Higher friction on all robot collision geoms helps a stable grasp.
    for g in spec.geoms:
        if g.contype or g.conaffinity:
            g.friction = [1.5, 0.05, 0.001]

    wb = spec.worldbody
    cx, cy = cube_pos
    yaw = np.arctan2(cy, cx)  # face the robot so the jaws meet flat faces
    cube = wb.add_body(name="cube", pos=[cx, cy, CUBE_HALF + 0.001],
                       quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    cube.add_freejoint(name="cube_free")
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                  rgba=[0.85, 0.15, 0.15, 1], mass=0.03, friction=[1.5, 0.05, 0.001],
                  condim=4, solref=[0.01, 1])

    tx, ty = target_pos
    pad = wb.add_body(name="target", pos=[tx, ty, 0.0005])
    pad.add_geom(name="target_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.0005, 0],
                  rgba=[0.1, 0.9, 0.3, 0.6], contype=0, conaffinity=0)

    return spec.compile()


# Free-camera presets (lookat, distance, azimuth, elevation) for offscreen rendering.
CAMERAS = {
    "front": dict(lookat=(0.18, 0.0, 0.05), distance=0.58, azimuth=150, elevation=-26),
    "side": dict(lookat=(0.17, 0.0, 0.08), distance=0.65, azimuth=120, elevation=-20),
    "top": dict(lookat=(0.17, 0.0, 0.0), distance=0.70, azimuth=180, elevation=-80),
}


def make_camera(name_or_dict):
    cfg = CAMERAS[name_or_dict] if isinstance(name_or_dict, str) else name_or_dict
    cam = mujoco.MjvCamera()
    cam.lookat[:] = cfg["lookat"]
    cam.distance, cam.azimuth, cam.elevation = cfg["distance"], cfg["azimuth"], cfg["elevation"]
    return cam


class IK:
    """Damped least squares IK for position + top-down orientation of the grasp point."""

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)
        self.site = model.site("gripperframe").id
        self.dofs = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.lo = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS], 0]
        self.hi = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS], 1]

    def grasp_point(self, d=None):
        d = d or self.d
        R = d.site_xmat[self.site].reshape(3, 3)
        return d.site_xpos[self.site] + R @ GRASP_OFFSET, R

    def solve(self, target, q0, yaw=None, iters=300, tol=1e-4):
        """Return arm joint angles (5,) putting the grasp point at `target` with the
        approach axis pointing straight down. Jaw closing direction is horizontal and
        perpendicular to `yaw` (default: radial direction from the base)."""
        m, d = self.m, self.d
        target = np.asarray(target, float)
        if yaw is None:
            yaw = np.arctan2(target[1], target[0])
        x_des = np.array([0, 0, -1.0])  # approach: down
        z_des = np.array([-np.sin(yaw), np.cos(yaw), 0.0])  # jaws close tangentially
        R_des = np.column_stack([x_des, np.cross(z_des, x_des), z_des])
        q = np.array(q0, float).copy()
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            p, R = self.grasp_point()
            e_pos = target - p
            e_rot = 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))
            err = np.concatenate([e_pos, 0.3 * e_rot])
            if np.linalg.norm(e_pos) < tol and np.linalg.norm(e_rot) < 1e-3:
                break
            mujoco.mj_jac(m, d, jp, jr, p, m.site_bodyid[self.site])
            J = np.vstack([jp[:, self.dofs], 0.3 * jr[:, self.dofs]])
            lam = 1e-4
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return q, float(np.linalg.norm(e_pos))


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class SO101Sim:
    """Physics + position control + optional video recording."""

    def __init__(self, model, width=960, height=720, fps=30, camera="front", record=True):
        self.m = model
        self.d = mujoco.MjData(model)
        self.ik = IK(model)
        self.fps = fps
        self.camera = camera
        self.frames: list[np.ndarray] = []
        self.caption = ""
        self.record = record
        self.renderer = mujoco.Renderer(model, height, width) if record else None
        self.arm_act = np.array([model.actuator(j).id for j in ARM_JOINTS])
        self.grip_act = model.actuator(GRIPPER).id
        self.arm_q = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_dof = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.gravity_comp = True
        self._steps_per_frame = max(1, round(1.0 / (fps * model.opt.timestep)))
        self._step_count = 0
        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self, q=HOME_Q, grip=GRIP_OPEN):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.arm_q] = q
        self.d.qpos[self.m.joint(GRIPPER).qposadr[0]] = grip
        self.d.ctrl[self.arm_act] = q
        self.d.ctrl[self.grip_act] = grip
        mujoco.mj_forward(self.m, self.d)

    @property
    def q(self):
        return self.d.ctrl[self.arm_act].copy()  # commanded arm joints

    @property
    def grip(self):
        return float(self.d.ctrl[self.grip_act])

    def body_pos(self, name):
        return self.d.body(name).xpos.copy()

    def grasp_point(self):
        return self.ik.grasp_point(self.d)[0].copy()

    # -- stepping ---------------------------------------------------------
    def step(self, n=1):
        for _ in range(n):
            if self.gravity_comp:
                # Feed-forward gravity/Coriolis compensation on the arm joints (stands in for
                # the integral action / stiffness a real servo has; avoids sag under gravity).
                self.d.qfrc_applied[self.arm_dof] = self.d.qfrc_bias[self.arm_dof]
            mujoco.mj_step(self.m, self.d)
            self._step_count += 1
            if self.record and self._step_count % self._steps_per_frame == 0:
                self.frames.append(self.render())

    def hold(self, seconds):
        self.step(int(seconds / self.m.opt.timestep))

    def move_joints(self, q_goal, duration, grip=None):
        """Smooth (minimum-jerk) joint-space move of the arm (and optionally gripper)."""
        q0, g0 = self.q, self.grip
        q_goal = np.asarray(q_goal, float)
        n = max(1, int(duration / self.m.opt.timestep))
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            self.d.ctrl[self.arm_act] = q0 + s * (q_goal - q0)
            if grip is not None:
                self.d.ctrl[self.grip_act] = g0 + s * (grip - g0)
            self.step()

    def move_to(self, pos, duration, grip=None, yaw=None):
        """Cartesian goal for the grasp point (top-down), executed as a joint-space move."""
        q, err = self.ik.solve(pos, self.q, yaw=yaw)
        if err > 5e-3:
            print(f"[warn] IK residual {err*1000:.1f} mm for target {np.round(pos, 3)}")
        self.move_joints(q, duration, grip)
        return q

    def set_grip(self, value, duration=0.6):
        self.move_joints(self.q, duration, grip=value)

    # -- rendering --------------------------------------------------------
    def render(self, camera=None):
        self.renderer.update_scene(self.d, camera=make_camera(camera or self.camera))
        img = self.renderer.render().copy()
        if self.caption:
            img = _draw_caption(img, self.caption)
        return img

    def save_video(self, path, fps=None):
        import imageio.v2 as imageio
        imageio.mimwrite(path, self.frames, fps=fps or self.fps, codec="libx264",
                         quality=8, macro_block_size=8)


def _draw_caption(img, text):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(16, img.shape[0] // 30))
    except OSError:
        font = ImageFont.load_default()
    x0, y0, x1, y1 = dr.textbbox((14, 12), text, font=font)
    dr.rectangle((x0 - 8, y0 - 6, x1 + 8, y1 + 6), fill=(0, 0, 0, 150))
    dr.text((14, 12), text, font=font, fill=(255, 255, 255, 255))
    return np.asarray(im)
