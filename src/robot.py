"""MuJoCo上のPandaアームを動かす層: 逆運動学、軌道生成、把持。"""
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parent.parent / "assets/franka_emika_panda/sorter_scene.xml"
OBJECTS = ["red_cube", "blue_cube", "green_cylinder", "yellow_cylinder"]
TCP_OFFSET = np.array([0, 0, 0.1034])  # handボディから指先中心まで
HOME = np.array([0, -0.5, 0, -2.2, 0, 1.7, 0.785])
PARKED = np.array([2.6, -0.5, 0, -2.2, 0, 1.7, 0.785])  # カメラの視野から退避する姿勢
SIM_STEPS_PER_FRAME = 17  # 0.002s x 17 = 約30fps


class IKError(RuntimeError):
    pass


def scatter_objects(model, data, seed):
    """作業領域に物体を重ならないようランダム配置する。"""
    rng = np.random.default_rng(seed)
    placed = []
    for name in OBJECTS:
        while True:
            xy = rng.uniform([0.36, -0.2], [0.66, 0.2])
            if all(np.linalg.norm(xy - p) > 0.11 for p in placed):
                break
        placed.append(xy)
        yaw = rng.uniform(-np.pi, np.pi)
        adr = model.jnt_qposadr[model.body(name).jntadr[0]]
        z = model.geom(model.body(name).geomadr[0]).size[2 if "cube" in name else 1]
        data.qpos[adr:adr + 3] = [xy[0], xy[1], z + 0.001]
        data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


class Robot:
    def __init__(self, seed=0, on_frame=None):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.hand = self.model.body("hand").id
        self.on_frame = on_frame
        self.data.qpos[:7] = PARKED
        self.data.qpos[7:9] = 0.04
        self.data.ctrl[:7] = PARKED
        self.data.ctrl[7] = 255
        scatter_objects(self.model, self.data, seed)
        mujoco.mj_forward(self.model, self.data)
        self.settle(0.5)

    @property
    def q(self):
        return self.data.qpos[:7].copy()

    def tcp(self):
        h = self.data.body(self.hand)
        return h.xpos + h.xmat.reshape(3, 3) @ TCP_OFFSET

    def _step_frame(self):
        mujoco.mj_step(self.model, self.data, nstep=SIM_STEPS_PER_FRAME)
        if self.on_frame:
            self.on_frame()

    def settle(self, seconds):
        for _ in range(int(seconds * 30)):
            self._step_frame()

    def ik(self, pos, yaw=0.0, seed_q=None):
        """局所解に捕まったら、目標の方向へ腰を向けた姿勢などから解き直す。"""
        aimed = HOME.copy()
        aimed[0] = np.arctan2(pos[1], pos[0])
        seeds = [] if seed_q is None else [seed_q]
        for q0 in seeds + [aimed, HOME, self.q]:
            q = self._ik_from(pos, yaw, q0)
            if q is not None:
                return q
        raise IKError(f"IKが収束しません: pos={pos}, yaw={yaw:.2f}")

    def _ik_from(self, pos, yaw, seed_q, iters=400):
        """指先を pos へ、指を真下向きにする関節角を減衰最小二乗法で解く。"""
        m, d = self.model, self.ik_data
        d.qpos[:] = self.data.qpos
        q = seed_q.copy()
        y = np.array([np.cos(yaw), np.sin(yaw), 0])
        z = np.array([0, 0, -1.0])
        Rd = np.column_stack([np.cross(y, z), y, z])
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        lo, hi = m.jnt_range[:7, 0], m.jnt_range[:7, 1]
        for _ in range(iters):
            d.qpos[:7] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.xmat[self.hand].reshape(3, 3)
            p = d.xpos[self.hand] + R @ TCP_OFFSET
            e = np.r_[pos - p, 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))]
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 5e-3:
                return q
            mujoco.mj_jac(m, d, jp, jr, p, self.hand)
            J = np.vstack([jp[:, :7], jr[:, :7]])
            JJ = J @ J.T + 1e-3 * np.eye(6)
            Jinv = J.T @ np.linalg.inv(JJ)
            dq = Jinv @ e + (np.eye(7) - Jinv @ J) @ (HOME - q) * 0.02
            q = np.clip(q + np.clip(dq, -0.1, 0.1), lo, hi)
        return None

    def move_joints(self, target, seconds):
        """関節空間の最小躍度軌道で動かす。"""
        start = self.data.ctrl[:7].copy()
        n = max(1, int(seconds * 30))
        for i in range(1, n + 1):
            t = i / n
            s = 10 * t**3 - 15 * t**4 + 6 * t**5
            self.data.ctrl[:7] = start + s * (target - start)
            self._step_frame()
        self.settle(0.15)

    def move_tcp(self, pos, yaw=0.0, seconds=1.5):
        self.move_joints(self.ik(pos, yaw, seed_q=self.data.ctrl[:7].copy()), seconds)

    def gripper(self, closed, seconds=0.7):
        start, end = self.data.ctrl[7], (0.0 if closed else 255.0)
        n = int(seconds * 30)
        for i in range(1, n + 1):
            self.data.ctrl[7] = start + (end - start) * i / n
            self._step_frame()
        self.settle(0.2)

    def finger_width(self):
        return float(self.data.qpos[7] + self.data.qpos[8])
