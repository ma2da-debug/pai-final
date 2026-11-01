"""デモ動画の生成と、成功率の評価を行うエントリポイント。

  python src/run_demo.py video --seed 3 --out demo.mp4
  python src/run_demo.py eval --episodes 20 --out results/eval.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from language import Matcher  # noqa: E402
from perception import Camera  # noqa: E402
from robot import Robot  # noqa: E402
from sorter import BINS, Sorter, make_slide_disturbance  # noqa: E402

TARGETS = {
    "the red cube": "red_cube", "the blue cube": "blue_cube",
    "the green cylinder": "green_cylinder", "the yellow cylinder": "yellow_cylinder",
}
DEMO_COMMANDS = [
    ("put the red cube in the left bin", None),
    ("put the green cylinder in the right bin", "green_cylinder"),
    ("put the yellow cylinder in the left bin", None),
    ("put the blue cube in the right bin", None),
]


def correct_bin(robot, name, side):
    pos = robot.data.body(name).xpos
    center = BINS[side]
    return abs(pos[0] - center[0]) < 0.09 and abs(pos[1] - center[1]) < 0.09


def run_commands(robot, sorter, commands):
    log = []
    for command, disturbed in commands:
        if sorter.ui:
            sorter.ui.command(command, disturbed is not None)
        disturb = make_slide_disturbance(robot, disturbed) if disturbed else None
        res = sorter.execute(command, disturb)
        query = command[len("put "):].split(" in the ")[0]
        side = "left" if "left" in command else "right"
        truly = bool(correct_bin(robot, TARGETS[query], side))
        log.append({**vars(res), "disturbed": disturbed is not None, "ground_truth_ok": truly})
    return log


def evaluate(episodes, out):
    matcher = Matcher()
    rows = []
    for seed in range(episodes):
        rng = np.random.default_rng(1000 + seed)
        order = rng.permutation(list(TARGETS))
        cmds = []
        for i, q in enumerate(order):
            side = "left" if rng.random() < 0.5 else "right"
            # 各エピソードの1物体目に外乱を与える
            cmds.append((f"put {q} in the {side} bin", TARGETS[q] if i == 0 else None))
        robot = Robot(seed)
        sorter = Sorter(robot, Camera(robot.model), matcher)
        rows.append({"seed": seed, "log": run_commands(robot, sorter, cmds)})
        ok = sum(x["ground_truth_ok"] for x in rows[-1]["log"])
        print(f"seed {seed}: {ok}/4", [x["attempts"] for x in rows[-1]["log"]], flush=True)
    logs = [x for r in rows for x in r["log"]]
    n = len(logs)
    summary = {
        "commands": n,
        "ground_truth_success": sum(x["ground_truth_ok"] for x in logs),
        "self_reported_success": sum(x["success"] for x in logs),
        "false_success": sum(x["success"] and not x["ground_truth_ok"] for x in logs),
        "disturbed_commands": sum(x["disturbed"] for x in logs),
        "disturbed_success": sum(x["disturbed"] and x["ground_truth_ok"] for x in logs),
        "undisturbed_first_try": sum((not x["disturbed"]) and x["ground_truth_ok"] and x["attempts"] == 1 for x in logs),
    }
    print(summary)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["video", "eval"])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.mode == "eval":
        evaluate(a.episodes, a.out or "results/eval.json")
        return
    from video import VideoUI
    ui = VideoUI(a.out or "demo.mp4")
    robot = Robot(a.seed)
    ui.attach(robot)
    robot.on_frame = lambda: ui.frame(robot)
    sorter = Sorter(robot, Camera(robot.model), Matcher(), ui, observe_hold=1.8)
    ui.title_card()
    log = run_commands(robot, sorter, DEMO_COMMANDS)
    ui.caption("全指示を完了")
    robot.settle(2.0)
    if Path("results/eval.json").exists():
        ui.end_card(json.loads(Path("results/eval.json").read_text())["summary"])
    ui.close()
    print(json.dumps(log, indent=1, ensure_ascii=False))
    print("duration_sec", round(ui.count / 30, 1))


if __name__ == "__main__":
    main()
