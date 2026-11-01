"""言語指示の解釈、知覚、把持、検証、再試行をつなぐ層。"""
import re
from dataclasses import dataclass, field

import numpy as np

from perception import segment
from robot import HOME, PARKED, IKError

# 左右は動画の視聴者から見た向き(ロボットと向かい合う向き)
BINS = {"left": np.array([0.42, -0.42]), "right": np.array([0.42, 0.42])}
SLOTS = [np.array([0.0, 0.04]), np.array([0.0, -0.04]), np.array([0.04, 0.0]), np.array([-0.04, 0.0])]
HOVER_Z = 0.2
BIN_FLOOR_Z = 0.008
EMPTY_GRIP_WIDTH = 0.01  # 物体の最小幅は44mm。これ未満なら空振り
MAX_ATTEMPTS = 3


def parse_command(text):
    m = re.fullmatch(r"put (.+) in the (left|right) bin", text.strip().lower())
    if not m:
        raise ValueError(f"解釈できない指示です: {text}")
    return m.group(1), m.group(2)


@dataclass
class Result:
    command: str
    success: bool = False
    attempts: int = 0
    chosen_scores: list = field(default_factory=list)
    reason: str = ""


class Sorter:
    def __init__(self, robot, camera, matcher, ui=None, observe_hold=0.0):
        self.r, self.cam, self.matcher = robot, camera, matcher
        self.ui = ui
        self.observe_hold = observe_hold
        self.slots_used = {"left": 0, "right": 0}

    def say(self, msg):
        if self.ui:
            self.ui.caption(msg)

    def observe(self, query=None):
        rgb, depth = self.cam.capture(self.r.data)
        objects = segment(rgb, depth, self.cam, self.r.data)
        scores = self.matcher.rank(query, [o["crop"] for o in objects]) if query and objects else None
        if self.ui:
            self.ui.show_perception(rgb, objects, scores, query)
        self.r.settle(self.observe_hold)
        return rgb, depth, objects, scores

    def bin_fill(self, depth, side):
        """箱の内側で、底より高く写っている画素数を数える。"""
        pts = self.cam.to_world(depth, self.r.data)
        c = BINS[side]
        inside = (np.abs(pts[..., 0] - c[0]) < 0.075) & (np.abs(pts[..., 1] - c[1]) < 0.075)
        return int((inside & (pts[..., 2] > 0.02)).sum())

    def _pick(self, target, disturb=None):
        r = self.r
        x, y = target["xy"]
        yaw = target["yaw"]
        grip_z = max(target["height"] - 0.025, 0.02)
        r.move_joints(HOME, 1.8)
        r.move_tcp(np.r_[x, y, HOVER_Z], yaw, 1.8)
        if disturb:
            disturb()
        r.move_tcp(np.r_[x, y, grip_z], yaw, 1.2)
        r.gripper(True)
        width = r.finger_width()
        r.move_tcp(np.r_[x, y, HOVER_Z], yaw, 1.2)
        return width, grip_z

    def _place(self, side, grip_z):
        r = self.r
        slot = SLOTS[self.slots_used[side] % len(SLOTS)]
        xy = BINS[side] + slot
        r.move_tcp(np.r_[xy, HOVER_Z], 0.0, 1.8)
        r.move_tcp(np.r_[xy, BIN_FLOOR_Z + grip_z + 0.012], 0.0, 0.9)
        r.gripper(False, 0.5)
        r.move_tcp(np.r_[xy, HOVER_Z], 0.0, 0.8)

    def _park(self):
        self.r.gripper(False, 0.2)
        self.r.move_joints(HOME, 1.2)
        self.r.move_joints(PARKED, 1.8)
        self.r.settle(0.3)

    def execute(self, command, disturb=None):
        """指示を実行する。空振りや配置失敗は、撮り直して再試行する。"""
        query, side = parse_command(command)
        res = Result(command)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            res.attempts = attempt
            self.say(f"[{attempt}回目] カメラで観測し、CLIPで「{query}」を探す")
            _, depth, objects, scores = self.observe(query)
            if not objects:
                res.reason = "作業領域に物体がない"
                return res
            best = int(np.argmax(scores))
            res.chosen_scores.append(round(float(scores[best]), 3))
            before_fill = self.bin_fill(depth, side)
            self.say(f"[{attempt}回目] 一致度 {scores[best]:.0%} の物体をつかみに行く")
            try:
                width, grip_z = self._pick(objects[best], disturb if attempt == 1 else None)
            except IKError as e:
                self.say("届く姿勢が見つからない。観測からやり直す")
                res.reason = str(e)
                self._park()
                continue
            if width < EMPTY_GRIP_WIDTH:
                self.say("空振りを検知(指の間隔が閉じきった)。観測からやり直す")
                res.reason = "空振り"
                self._park()
                continue
            self.say(f"把持を確認(指の間隔 {width * 1000:.0f} mm)。{ '左' if side == 'left' else '右'}の箱へ運ぶ")
            try:
                self._place(side, grip_z)
            except IKError as e:
                res.reason = str(e)
            self._park()
            _, depth, _, _ = self.observe()
            if self.bin_fill(depth, side) > before_fill + 30:
                self.slots_used[side] += 1
                self.say("箱に入ったことをカメラで確認")
                self.r.settle(self.observe_hold)
                res.success, res.reason = True, ""
                return res
            res.reason = "箱の中に確認できない"
            self.say("箱の中に確認できない。やり直す")
        return res


def make_slide_disturbance(robot, name, distance=0.09, seconds=1.0):
    """把持の直前に、物体を別の場所へ滑らせる(人が動かす状況の模擬)。"""
    def run():
        m, d = robot.model, robot.data
        adr = m.jnt_qposadr[m.body(name).jntadr[0]]
        vadr = m.jnt_dofadr[m.body(name).jntadr[0]]
        start = d.qpos[adr:adr + 2].copy()
        others = [d.body(n).xpos[:2].copy() for n in ("red_cube", "blue_cube", "green_cylinder", "yellow_cylinder") if n != name]
        def clearance(a):
            p = start + distance * np.array([np.cos(a), np.sin(a)])
            if not (0.36 <= p[0] <= 0.66 and -0.22 <= p[1] <= 0.22):
                return -1.0
            return min(np.linalg.norm(p - o) for o in others)
        a = max(np.linspace(0, 2 * np.pi, 16, endpoint=False), key=clearance)
        goal = start + distance * np.array([np.cos(a), np.sin(a)])
        n = int(seconds * 30)
        for i in range(1, n + 1):
            d.qpos[adr:adr + 2] = start + (goal - start) * i / n
            d.qvel[vadr:vadr + 6] = 0
            robot._step_frame()
    return run
