#!/usr/bin/env python
"""
teleop.py - discrete-command control of the simulated SO-101: the hook for a brain-controlled arm.

Any upstream decoder (keyboard, EEG motor-imagery classifier, SSVEP, eye blink detector, ...)
only has to produce one of a handful of discrete command strings:

    left  right  forward  back  up  down  grip  release  home  quit

Each command nudges a Cartesian end-effector target by STEP metres (or opens/closes the
gripper); the arm follows with top-down IK + a smooth joint move. Everything is recorded to
an MP4 because this box is headless.

Command sources (pick one):
    --script "forward,left,down,grip,up"    fixed sequence (default: a full pick & place)
    --stdin                                 type commands / keys interactively (w a s d r f g o h q)
    --udp 5005                              listen for UTF-8 commands on UDP 127.0.0.1:5005
                                            (e.g. from an EEG classifier process)

Example of an EEG process feeding this sim:
    import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(b"left", ("127.0.0.1", 5005))

For a real-time viewer on a machine with a display, run `python -m mujoco.viewer` or adapt
ArmCommander to `mujoco.viewer.launch_passive`.
"""
import argparse
import os
import socket
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from so101_sim import CUBE_HALF, GRIP_CLOSED, GRIP_OPEN, HOME_Q, SO101Sim, build_model

OUT = Path(__file__).resolve().parent

STEP = 0.02  # metres per discrete command
WORKSPACE_LO = np.array([0.10, -0.20, CUBE_HALF + 0.001])
WORKSPACE_HI = np.array([0.30, 0.20, 0.10])  # top-down IK reach limit
START = np.array([0.17, 0.0, 0.094])

# Keyboard aliases (also handy as the label set for an EEG classifier).
KEYS = {"w": "forward", "s": "back", "a": "left", "d": "right", "r": "up", "f": "down",
        "g": "grip", "o": "release", "h": "home", "q": "quit"}

# Suggested mapping from a 4-class motor-imagery BCI (BCI Competition IV-2a style):
EEG_MAP = {"left_hand": "left", "right_hand": "right", "feet": "down", "tongue": "grip"}


class ArmCommander:
    """Turns discrete commands into arm motion. This is the single API an EEG decoder calls."""

    def __init__(self, sim: SO101Sim, step=STEP, move_time=0.5):
        self.sim, self.step, self.move_time = sim, step, move_time
        self.target = START.copy()
        self.gripping = False
        self.go_home()

    # --- primitive actions (what an EEG class maps onto) ---
    def move_left(self):    self._nudge([0, +1, 0])
    def move_right(self):   self._nudge([0, -1, 0])
    def move_forward(self): self._nudge([+1, 0, 0])
    def move_back(self):    self._nudge([-1, 0, 0])
    def move_up(self):      self._nudge([0, 0, +1])
    def move_down(self):    self._nudge([0, 0, -1])

    def grip(self):
        self.gripping = True
        self.sim.set_grip(GRIP_CLOSED, 0.7)
        self.sim.hold(0.2)

    def release(self):
        self.gripping = False
        self.sim.set_grip(GRIP_OPEN, 0.5)

    def go_home(self):
        self.target = START.copy()
        self.sim.move_to(self.target, 1.0)

    # --- dispatcher ---
    def apply(self, cmd: str) -> bool:
        """Execute one command string. Returns False on 'quit'."""
        cmd = KEYS.get(cmd.strip().lower(), cmd.strip().lower())
        cmd = EEG_MAP.get(cmd, cmd)
        table = {"left": self.move_left, "right": self.move_right, "forward": self.move_forward,
                 "back": self.move_back, "up": self.move_up, "down": self.move_down,
                 "grip": self.grip, "release": self.release, "home": self.go_home}
        if cmd == "quit":
            return False
        if cmd not in table:
            print(f"  ? unknown command {cmd!r}")
            return True
        self.sim.caption = f"cmd: {cmd}"
        table[cmd]()
        p = self.sim.grasp_point()
        print(f"  {cmd:8s} -> target {np.round(self.target, 3)}  ee {np.round(p, 3)}  "
              f"grip={'closed' if self.gripping else 'open'}  cube {np.round(self.sim.body_pos('cube'), 3)}")
        return True

    def _nudge(self, direction):
        self.target = np.clip(self.target + self.step * np.asarray(direction, float),
                              WORKSPACE_LO, WORKSPACE_HI)
        self.sim.move_to(self.target, self.move_time)


# --- command sources -------------------------------------------------------
DEFAULT_SCRIPT = ("forward,forward,left,left,left,left,down,down,down,down,grip,up,up,up,"
                  "right,right,right,right,right,right,right,right,down,down,release,up,home")


def script_source(text):
    yield from (c for c in text.split(",") if c.strip())


def stdin_source():
    print("commands: w/s forward/back, a/d left/right, r/f up/down, g grip, o release, h home, q quit")
    for line in sys.stdin:
        for tok in line.split():
            yield tok


def udp_source(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    print(f"listening for commands on udp://127.0.0.1:{port} (send 'quit' to finish)")
    while True:
        data, _ = sock.recvfrom(1024)
        yield data.decode("utf-8", "ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--script", default=DEFAULT_SCRIPT)
    src.add_argument("--stdin", action="store_true")
    src.add_argument("--udp", type=int, metavar="PORT")
    ap.add_argument("--out", default=str(OUT / "teleop.mp4"))
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    model = build_model(cube_pos=(0.21, 0.08), target_pos=(0.21, -0.08))
    sim = SO101Sim(model, width=args.width, height=args.height)
    sim.reset(sim.ik.solve(START, HOME_Q)[0], GRIP_OPEN)
    arm = ArmCommander(sim)

    if args.stdin:
        source = stdin_source()
    elif args.udp:
        source = udp_source(args.udp)
    else:
        source = script_source(args.script)

    try:
        for cmd in source:
            if not arm.apply(cmd):
                break
    except KeyboardInterrupt:
        pass
    sim.caption = "teleop session end"
    sim.hold(1.0)
    sim.save_video(args.out)
    print(f"video: {args.out} ({len(sim.frames)/sim.fps:.1f} s)")


if __name__ == "__main__":
    main()
