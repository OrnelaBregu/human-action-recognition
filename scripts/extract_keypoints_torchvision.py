"""Extract COCO body keypoints with torchvision's Keypoint R-CNN (no Detectron2 needed).

An alternative to extract_keypoints.sh that runs on CPU, CUDA or Apple Silicon (MPS).
It uses torchvision's Keypoint R-CNN (ResNet-50 FPN, COCO) rather than the original
Detectron2 R-101 FPN model, and writes one .npz per video in the same format as
VideoPose3D's infer_video_d2.py, so prepare_keypoints.py works on either output.

Usage:
    python scripts/extract_keypoints_torchvision.py --dataset ucf101 [--frame-step 2]
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn

from common import add_dataset_args, video_index

SCORE_THRESHOLD = 0.7  # same person-detection threshold as VideoPose3D's Detectron2 script


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_frames(path, frame_step):
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % frame_step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


@torch.no_grad()
def detect(model, frames, device, batch_size):
    """Return Detectron-style per-frame boxes [x1, y1, x2, y2, score] and keypoints (4, 17)."""
    boxes, keypoints = [], []
    for start in range(0, len(frames), batch_size):
        batch = [torch.from_numpy(f).permute(2, 0, 1).float().div(255).to(device)
                 for f in frames[start:start + batch_size]]
        for out in model(batch):
            keep = out["scores"] >= SCORE_THRESHOLD
            if keep.any():
                b = torch.cat([out["boxes"][keep], out["scores"][keep, None]], dim=1).cpu().numpy()
                k = out["keypoints"][keep].cpu().numpy()          # (n, 17, 3): x, y, visibility
                s = out["keypoints_scores"][keep].cpu().numpy()   # (n, 17)
                kps = np.stack([k[:, :, 0], k[:, :, 1], np.zeros_like(s), s], axis=1)  # (n, 4, 17)
            else:
                b, kps = [], []
            boxes.append([[], b])
            keypoints.append([[], kps])
    return boxes, keypoints


def to_object_array(items):
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):  # element-wise, so numpy does not try to broadcast nested lists
        arr[i] = item
    return arr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(parser)
    parser.add_argument("--out", help="output folder (default: data/<dataset>_poses)")
    parser.add_argument("--frame-step", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-size", type=int, default=480,
                        help="shorter image side used by the model (torchvision default is 800)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    args = parser.parse_args()
    out_dir = args.out or f"data/{args.dataset}_poses"

    device = pick_device(args.device)
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1, min_size=args.min_size)
    model.eval().to(device)
    print(f"running on {device}")

    videos = video_index(args.dataset, args.videos)
    start = time.time()
    for n, video in enumerate(videos, 1):
        out = os.path.join(out_dir, video["subset"], os.path.basename(video["path"]) + ".npz")
        if os.path.exists(out):
            continue
        frames = read_frames(video["path"], args.frame_step)
        boxes, keypoints = detect(model, frames, device, args.batch_size)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        h, w = frames[0].shape[:2] if frames else (0, 0)
        np.savez_compressed(out, boxes=to_object_array(boxes), segments=to_object_array([None] * len(boxes)),
                            keypoints=to_object_array(keypoints), metadata={"w": w, "h": h})
        if n % 25 == 0:
            rate = n / (time.time() - start)
            print(f"{n}/{len(videos)} videos, {(len(videos) - n) / rate / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
