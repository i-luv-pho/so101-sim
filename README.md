# SO-101 robot arm simulation (MuJoCo, headless)

A small, showable simulation of the **SO-101** low-cost robot arm (the arm used by Hugging Face
**LeRobot**). It uses the official open-source model from TheRobotStudio. It contains:

| File | What it is |
|---|---|
| `demo.py` | Scripted **pick-and-place**: reach the red cube, grasp it with the gripper (real friction contacts, no "magic" attach), lift it, carry it, and place it on the green target. Renders to `demo.mp4` and `frames/*.png`. |
| `teleop.py` | **Discrete-command control**: `left/right/forward/back/up/down/grip/release/home`. Commands come from a script, the keyboard (stdin), or **UDP**. This is the hook for a brain-controlled (EEG) arm. Renders to `teleop.mp4`. |
| `so101_sim.py` | Small library: scene builder (`MjSpec`), top-down IK (damped least squares), minimum-jerk joint moves, offscreen recorder. |
| `demo.mp4`, `teleop.mp4`, `frames/` | Pre-rendered outputs (960×720 and 800×600, 30 fps, ~15 s each). |
| `third_party/SO-ARM100/` | Shallow clone of the official SO-ARM100 repo (model files are in `Simulation/SO101/`). |

Tested on: Linux, 8 CPU cores, **no GPU, no display** (offscreen rendering through `MUJOCO_GL=egl` with Mesa), Python 3.12, MuJoCo 4.0.1.
On that machine `demo.py` runs in about 1.5 minutes, and nearly all of that time goes to software rendering.

## Sources and licenses

| Asset | Source | License |
|---|---|---|
| SO-101 MJCF/URDF and STL meshes (`so101_new_calib.xml`, `scene.xml`, `assets/`) | <https://github.com/TheRobotStudio/SO-ARM100> (`Simulation/SO101`), commit `141aea7` | **Apache-2.0** (`third_party/SO-ARM100/LICENSE`) |
| MuJoCo | <https://github.com/google-deepmind/mujoco> | Apache-2.0 |
| numpy / imageio / imageio-ffmpeg / Pillow | PyPI | BSD / BSD-2 / BSD-2 (bundled ffmpeg binary is LGPL/GPL) / MIT-CMU |
| This folder's code (`*.py`) | written for this project | use freely (e.g. Apache-2.0 to match the model) |

The model files were not modified. The cube, the target pad and the higher contact friction are added
at load time in `so101_sim.build_model()`.

Related open-source projects that are useful next steps (not cloned here):
* **huggingface/lerobot**: dataset format, policies (ACT, Diffusion, SmolVLA, pi0), and the real-robot drivers for SO-100/SO-101. Apache-2.0.
* **huggingface/gym-hil** and the LeRobot sim envs: MuJoCo Gym environments for human-in-the-loop RL with SO-type arms.
* **google-deepmind/mujoco_menagerie** `trs_so_arm100`: the earlier SO-100 model. Apache-2.0.

## How to run

```bash
cd /workspace/so101-sim
# one-time setup (already done here)
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt     # exact versions: requirements.lock.txt
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100

# 1) pick-and-place demo -> demo.mp4 + frames/
.venv/bin/python demo.py
.venv/bin/python demo.py --camera side --width 640 --height 480 --cube 0.24 0.05 --target 0.18 -0.10

# 2) command / "brain" control -> teleop.mp4
.venv/bin/python teleop.py                                   # built-in command script (pick & place by commands)
.venv/bin/python teleop.py --script "forward,left,left,down,grip,up"
.venv/bin/python teleop.py --stdin                           # type: w a s d (planar), r/f up/down, g grip, o open, h home, q quit
.venv/bin/python teleop.py --udp 5005                        # then send UDP text commands (see below)
```

Headless rendering uses `MUJOCO_GL=egl` (the scripts set it by default). If EGL is unavailable, try
`MUJOCO_GL=osmesa`, which needs `libosmesa6`. On a desktop with a display you can open the model
interactively with `python -m mujoco.viewer --mjcf third_party/SO-ARM100/Simulation/SO101/scene.xml`.

## How the demo works

1. **Model**: `scene.xml` from SO-ARM100 (new calibration: joint zero = middle of the range). It has 6 position
   actuators that model STS3215 servos: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
2. **IK** (`so101_sim.IK`): damped least squares on the `gripperframe` site. It places a grasp point between the
   jaws at a target position, with the approach axis pointing straight down. The jaws close sideways, perpendicular
   to the line from the base to the target.
3. **Trajectory**: Cartesian waypoints (above cube → cube → lift → carry → place) are converted to joint goals and
   executed as minimum-jerk joint-space moves.
4. **Gravity compensation**: a feed-forward term is applied to the arm joints. It stands in for the stiffness and
   integral action of the real servo and stops the arm from sagging about 1 cm. Set `sim.gravity_comp = False` to see the difference.
5. **Grasp**: the gripper is commanded beyond closed, so the force-limited servo squeezes the 2.5 cm, 30 g cube
   through friction contacts. On success the script prints `SUCCESS`. Typical result: cube lifted about 5 cm and placed about 1 cm from the target center.

## The EEG / brain-control hook

`teleop.ArmCommander.apply(cmd: str)` is the **only** function a decoder has to call. Every command moves the
end-effector target by 2 cm (`STEP`), clipped to a safe workspace, or opens/closes the gripper.

Command vocabulary: `left right forward back up down grip release home quit`. These aliases are also accepted:
* Keyboard: `w s a d r f g o h q`
* Example 4-class motor-imagery mapping (`EEG_MAP`): `left_hand→left`, `right_hand→right`, `feet→down`, `tongue→grip`

Sending commands from a separate EEG process (e.g. OpenBCI/BrainFlow or an LSL stream plus a classifier):

```python
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(b"left_hand", ("127.0.0.1", 5005))   # -> arm moves left 2 cm
sock.sendto(b"quit", ("127.0.0.1", 5005))        # -> finish and write teleop.mp4
```

Start `teleop.py --udp` first and wait for the `listening on ...` message. UDP packets sent before then are lost.

Suggested pipeline toward a brain-controlled arm:
1. Acquire EEG (OpenBCI Cyton/Ganglion, Muse, g.tec, ...) with **BrainFlow** or **LSL**.
2. Decode discrete intents: motor imagery (CSP + LDA, or EEGNet), SSVEP (CCA, very reliable for 4–6 targets), or blinks/EMG as a "grip" switch.
3. Add a **confidence threshold and dwell time** (e.g. 2 identical predictions in a row) before sending a command. False positives matter more than speed.
4. Send the label over UDP → `ArmCommander.apply`. Swap the sim for the real arm later (see below) without changing the decoder.
5. Optional shared autonomy: let the decoder only choose a goal ("that cube"), and let the scripted IK in `demo.py` do the fine motion.

## Extending toward LeRobot and a real SO-101

* **Real hardware**: LeRobot ships the SO-101 follower driver (Feetech STS3215 bus). The same joint order is used
  here. Joint angles from this sim are in radians (new calibration, zero = mid-range). LeRobot uses degrees or
  normalized ranges per joint, and its gripper is `0 = closed … 100 = open`, so you need a conversion. Get it working in
  the sim, then send the same targets to `robot.send_action(...)` at low speed. Keep the workspace clamps from `teleop.py`.
* **Datasets**: log `(camera image, joint state, action)` at 30 Hz from `demo.py` with randomized cube and target
  positions, and write it in the LeRobot dataset format. That gives you sim demonstrations to train **ACT** or
  **Diffusion Policy** (CPU training is slow, so use a GPU or Colab for training).
* **Policies in the loop**: wrap `SO101Sim` as a `gymnasium.Env` (obs = joint pos + rendered image; action = 6 joint
  targets) so LeRobot or RL code can control it. `gym-hil` is a good reference.
* **Sim-to-real**: randomize cube size, friction, mass, servo gains and camera pose. Calibrate the camera to match the real setup.

## Limitations / next steps

* The grasp uses convex hulls of the CAD meshes, so the cube sometimes rotates a little in the jaws.
  Small dedicated collision pads on the jaw tips would make grasping more robust.
* Top-down IK limits the reachable height near the base (about 10 cm). Allowing a tilted approach would enlarge the workspace.
* Gravity compensation is a simulation convenience. On the real arm, expect some sag and tune the servo P gain instead.
* Next: a Gymnasium env wrapper, a LeRobot dataset exporter, a live EEG (BrainFlow) source class for `teleop.py`,
  and an interactive viewer mode for machines with a display.
