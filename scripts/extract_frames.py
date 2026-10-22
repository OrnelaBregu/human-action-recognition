"""Extract frames from every video for the frame-level baseline.

Frames are resized to 60x80 (width x height) and stored with their label and the id of the
video they come from, grouped into training / validation / testing by subject or group.
--frame-step N keeps every Nth frame, which cuts memory for longer videos such as UCF101.

Usage:
    python scripts/extract_frames.py --dataset kth
    python scripts/extract_frames.py --dataset ucf101 --frame-step 5
"""

import argparse
import os

import cv2
import joblib
import numpy as np

from common import SUBSETS, add_dataset_args, categories, video_index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(parser)
    parser.add_argument("--out", help="output folder (default: data/<dataset>_frames)")
    parser.add_argument("--frame-step", type=int, default=1, help="keep every Nth frame")
    args = parser.parse_args()
    out_dir = args.out or f"data/{args.dataset}_frames"

    videos = video_index(args.dataset, args.videos)
    labels = categories(videos)
    data = {subset: {"X": [], "y": [], "video": []} for subset in SUBSETS}

    for video in videos:
        subset = data[video["subset"]]
        cap = cv2.VideoCapture(video["path"])
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.frame_step == 0:
                subset["X"].append(cv2.resize(frame, dsize=(60, 80), interpolation=cv2.INTER_CUBIC))
                subset["y"].append(labels.index(video["label"]))
                subset["video"].append(video["id"])
            i += 1
        cap.release()

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(labels, os.path.join(out_dir, "labels.joblib"))
    for name, subset in data.items():
        X = np.array(subset["X"], dtype=np.uint8).reshape(-1, 80, 60, 3)
        out = os.path.join(out_dir, f"{name}.joblib")
        joblib.dump({"X": X, "y": np.array(subset["y"], dtype=int), "video": np.array(subset["video"])}, out)
        print(f"{name}: {len(subset['y'])} frames from {len(set(subset['video']))} videos -> {out}")


if __name__ == "__main__":
    main()
