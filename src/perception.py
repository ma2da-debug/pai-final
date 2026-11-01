"""RGB-Dカメラで物体を切り出し、CLIPで言葉と照合する層。"""
import mujoco
import numpy as np
from scipy import ndimage

W, H = 640, 480
WORK_X, WORK_Y = (0.28, 0.74), (-0.28, 0.28)


class Camera:
    def __init__(self, model, name="top"):
        self.model, self.cam = model, model.camera(name).id
        self.renderer = mujoco.Renderer(model, H, W)
        self.f = 0.5 * H / np.tan(np.deg2rad(model.cam_fovy[self.cam]) / 2)

    def capture(self, data):
        r = self.renderer
        r.disable_depth_rendering()
        r.update_scene(data, self.cam)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, self.cam)
        depth = r.render().copy()
        r.disable_depth_rendering()
        return rgb, depth

    def to_world(self, depth, data):
        """画素ごとの深度をワールド座標の点群に変換する。"""
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        pc = np.stack([(u - W / 2) * depth / self.f, -(v - H / 2) * depth / self.f, -depth], -1)
        return pc @ data.cam_xmat[self.cam].reshape(3, 3).T + data.cam_xpos[self.cam]


def grasp_yaw(xy):
    """上から見た輪郭の外接矩形が最小になる向きを求め、指の方向にする。"""
    xy = xy - xy.mean(0)
    best, best_area = 0.0, np.inf
    for a in np.deg2rad(np.arange(0, 90, 3)):
        c, s = np.cos(a), np.sin(a)
        p, q = xy @ [c, s], xy @ [-s, c]
        area = np.ptp(p) * np.ptp(q)
        if area < best_area:
            best, best_area = a, area
    return float(best - np.pi / 2 if best > np.pi / 4 else best)


def segment(rgb, depth, cam, data, min_pixels=60):
    """机より高い領域を塊に分け、位置・把持向き・切り抜き画像を返す。"""
    pts = cam.to_world(depth, data)
    mask = ((pts[..., 2] > 0.012) & (pts[..., 0] > WORK_X[0]) & (pts[..., 0] < WORK_X[1])
            & (pts[..., 1] > WORK_Y[0]) & (pts[..., 1] < WORK_Y[1]))
    labels, n = ndimage.label(mask)
    found = []
    for i in range(1, n + 1):
        m = labels == i
        if m.sum() < min_pixels:
            continue
        top = pts[m]
        z = top[:, 2].max()
        upper = top[top[:, 2] > z - 0.008]  # 上面だけ使うと斜め視の影響が減る
        vs, us = np.where(m)
        pad = 8
        box = (max(us.min() - pad, 0), max(vs.min() - pad, 0), min(us.max() + pad, W), min(vs.max() + pad, H))
        found.append({
            "xy": upper[:, :2].mean(0), "height": float(z), "yaw": grasp_yaw(upper[:, :2]),
            "box": box, "crop": rgb[box[1]:box[3], box[0]:box[2]],
        })
    return found
