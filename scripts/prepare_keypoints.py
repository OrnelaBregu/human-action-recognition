"""Turn the Detectron2 keypoint files into fixed-length pose sequences for classical models.

Reconstructed from the project report: the original notebook for this step was not kept.
For each frame we take the most confident detected person and keep 13 of the 17 COCO
keypoints (eyes and ears removed, since faces were often missed and add no limb motion).
Frames with no detection repeat the previous pose. Sequences are padded to the longest
video by repeating the first frame at the start (half the missing frames) and the last
frame at the end (the rest), then split by subject or group.

Usage:
    bash scripts/extract_keypoints.sh data/divide data/kth_poses
    python scripts/prepare_keypoints.py --dataset kth --poses data/kth_poses
"""

import argparse
import glob
import os

import numpy as np

from common import SUBSETS, add_dataset_args, categories, video_index

FACE_KEYPOINTS = [1, 2, 3, 4]  # left/right eye, left/right ear in COCO order
KEEP = [i for i in range(17) if i not in FACE_KEYPOINTS]


def load_poses(path):
    """Return an array of shape (frames, 13, 2) with the x, y position of each kept keypoint."""
    data = np.load(path, allow_pickle=True)
    boxes, keypoints = data["boxes"], data["keypoints"]
    poses, last = [], np.zeros((len(KEEP), 2))
    for frame_boxes, frame_kps in zip(boxes, keypoints):
        people_boxes, people_kps = frame_boxes[1], frame_kps[1]
        if len(people_boxes) > 0:
            best = int(np.argmax(np.asarray(people_boxes)[:, 4]))  # last column is the detection score
            kps = np.asarray(people_kps)[best]  # shape (4, 17): x, y, logit, probability
            last = kps[:2, KEEP].T
        poses.append(last)
    return np.array(poses)


def pad(sequence, length):
    missing = length - len(sequence)
    before = missing // 2
    return np.concatenate([
        np.repeat(sequence[:1], before, axis=0),
        sequence,
        np.repeat(sequence[-1:], missing - before, axis=0),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(parser)
    parser.add_argument("--poses", help="keypoint files (default: data/<dataset>_poses)")
    parser.add_argument("--out", help="output folder (default: data/<dataset>_keypoints)")
    args = parser.parse_args()
    poses_dir = args.poses or f"data/{args.dataset}_poses"
    out_dir = args.out or f"data/{args.dataset}_keypoints"

    videos = video_index(args.dataset, args.videos)
    labels = categories(videos)
    files = {os.path.basename(p).split(".")[0]: p for p in glob.glob(os.path.join(poses_dir, "**", "*.npz"), recursive=True)}

    sequences, missing = [], 0
    for video in videos:
        if video["id"] not in files:
            missing += 1
            continue
        poses = load_poses(files[video["id"]])
        if len(poses) > 0:
            sequences.append((video, poses))
    length = max(len(poses) for _, poses in sequences)
    print(f"{len(sequences)} videos ({missing} without keypoint files), padding to {length} frames")

    os.makedirs(out_dir, exist_ok=True)
    for subset in SUBSETS:
        chosen = [(v, p) for v, p in sequences if v["subset"] == subset]
        X = np.array([pad(p, length) for _, p in chosen]).reshape(-1, length, len(KEEP), 2)
        y = np.array([labels.index(v["label"]) for v, _ in chosen], dtype=int)
        np.savez(os.path.join(out_dir, f"{subset}.npz"), X=X, y=y, labels=np.array(labels))
        print(f"{subset}: {len(y)} videos")


if __name__ == "__main__":
    main()
