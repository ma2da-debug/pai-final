"""シミュレーション映像に、指示・認識結果・状況説明を重ねて動画にする。"""
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FONT_PATHS = ["/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]


def font(size):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


class VideoUI:
    def __init__(self, path):
        self.writer = imageio.get_writer(path, fps=30, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1)
        self.count = 0
        self.cmd = self.msg = ""
        self.note = ""
        self.inset = None
        self.f_big, self.f_mid, self.f_small = font(30), font(24), font(15)

    def attach(self, robot):
        self.renderer = mujoco.Renderer(robot.model, H, W)

    def caption(self, msg):
        self.msg = msg

    def command(self, text, disturbed):
        self.cmd = text
        self.note = "この指示では把持の直前にスクリプトで物体を動かす(外乱テスト)" if disturbed else ""
        self.inset = None

    def show_perception(self, rgb, objects, scores, query):
        img = Image.fromarray(rgb).copy()
        d = ImageDraw.Draw(img)
        best = int(np.argmax(scores)) if scores is not None else -1
        for i, o in enumerate(objects):
            col = (0, 255, 90) if i == best else (255, 255, 255)
            d.rectangle(o["box"], outline=col, width=4 if i == best else 2)
            if scores is not None:
                d.text((o["box"][0], max(o["box"][1] - 26, 0)), f"{scores[i]:.0%}", fill=col, font=font(24), stroke_width=2, stroke_fill=(0, 0, 0))
        self.inset = img.resize((320, 240), Image.BILINEAR)

    def frame(self, robot):
        self.renderer.update_scene(robot.data, "show")
        self._write(self._compose(Image.fromarray(self.renderer.render())))

    def _compose(self, img):
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle((0, 0, W, 58), fill=(10, 12, 18, 215))
        d.text((20, 12), f"指示: {self.cmd}", fill=(255, 255, 255), font=self.f_big)
        d.rectangle((0, H - 58, W, H), fill=(10, 12, 18, 215))
        d.text((20, H - 45), self.msg, fill=(255, 235, 150), font=self.f_mid)
        if self.note:
            d.rectangle((0, 58, W, 92), fill=(150, 60, 20, 225))
            d.text((20, 63), self.note, fill=(255, 255, 255), font=self.f_small.font_variant(size=20))
        if self.inset:
            x, y = W - 336, 100
            img.paste(self.inset, (x, y))
            d = ImageDraw.Draw(img, "RGBA")
            d.rectangle((x, y, x + 320, y + 240), outline=(255, 255, 255, 255), width=2)
            d.rectangle((x, y + 240, x + 320, y + 266), fill=(10, 12, 18, 215))
            d.text((x + 8, y + 245), "上部カメラの入力 / CLIPの一致度", fill=(255, 255, 255), font=self.f_small)
        return np.asarray(img)

    def _write(self, arr):
        self.writer.append_data(arr)
        self.count += 1

    def title_card(self, seconds=5):
        img = Image.new("RGB", (W, H), (16, 20, 30))
        d = ImageDraw.Draw(img)
        d.text((90, 170), "言葉で指示できる片付けロボット", fill=(255, 255, 255), font=font(56))
        lines = [
            "カメラ画像とCLIPで、指示された物体を探して箱へ運ぶ",
            "つかめたか、箱に入ったかを自分で確かめ、失敗したらやり直す",
            "物理シミュレータ MuJoCo / Franka Panda  ・  等倍速",
        ]
        for i, s in enumerate(lines):
            d.text((90, 290 + i * 54), s, fill=(200, 210, 225), font=font(30))
        arr = np.asarray(img)
        for _ in range(seconds * 30):
            self._write(arr)

    def end_card(self, s, seconds=6):
        img = Image.new("RGB", (W, H), (16, 20, 30))
        d = ImageDraw.Draw(img)
        d.text((90, 150), "配置を変えた20エピソードでの評価", fill=(255, 255, 255), font=font(48))
        lines = [
            f"指示どおり箱に入った: {s['ground_truth_success']} / {s['commands']} 指示",
            f"うち外乱ありの指示: {s['disturbed_success']} / {s['disturbed_commands']} 指示",
            f"成功と誤判定した回数: {s['false_success']} 回",
            "判定はシミュレータの真の物体位置を使用",
        ]
        for i, line in enumerate(lines):
            d.text((90, 260 + i * 62), line, fill=(200, 210, 225), font=font(32))
        arr = np.asarray(img)
        for _ in range(seconds * 30):
            self._write(arr)

    def close(self):
        self.writer.close()
