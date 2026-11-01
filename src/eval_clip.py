"""言い回しごとにCLIPの正答率を測る。  python src/eval_clip.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from language import Matcher  # noqa: E402
from perception import Camera, segment  # noqa: E402
from robot import Robot  # noqa: E402

QUERIES = {
    "the red cube": "red_cube", "the blue cube": "blue_cube",
    "the green cylinder": "green_cylinder", "the yellow cylinder": "yellow_cylinder",
    "a red block": "red_cube", "a blue block": "blue_cube",
    "something green": "green_cylinder", "something yellow": "yellow_cylinder",
    "something the color of a banana": "yellow_cylinder",
    "something the color of a ripe tomato": "red_cube",
    "something the color of the ocean": "blue_cube",
    "something the color of fresh grass": "green_cylinder",
}
LAYOUTS = 20

matcher = Matcher()
hits = dict.fromkeys(QUERIES, 0)
for seed in range(LAYOUTS):
    robot = Robot(seed)
    cam = Camera(robot.model)
    rgb, depth = cam.capture(robot.data)
    objects = segment(rgb, depth, cam, robot.data)
    for query, name in QUERIES.items():
        best = objects[int(matcher.rank(query, [o["crop"] for o in objects]).argmax())]
        hits[query] += int(np.linalg.norm(best["xy"] - robot.data.body(name).xpos[:2]) < 0.02)
print(hits)
Path("results/clip_eval.json").write_text(json.dumps({"layouts": LAYOUTS, "correct": hits}, indent=1))
