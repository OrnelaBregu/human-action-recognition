"""Encode every video with a frozen V-JEPA 2 video foundation model.

V-JEPA 2 (Meta, 2025) is a video transformer pretrained without labels on over a million
hours of internet video, so it has never been trained on action labels. Each video is
sampled as --frames frames spread evenly over the whole clip, resized to 256x256 (the
full frame is kept rather than center-cropped, so people at the edge of KTH frames stay
in view) and encoded once. The encoder's tokens are averaged per time step and saved as
one small .npy per video (time steps x 1024, float16) for train_few_shot.py.

Runs on CPU, CUDA or Apple Silicon (MPS). Videos already encoded are skipped, so an
interrupted run can be restarted.

Usage:
    python scripts/extract_video_features.py --dataset ucf101
    python scripts/extract_video_features.py --dataset kth --model facebook/vjepa2-vitl-fpc64-256
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
from transformers import VJEPA2Model

from common import add_dataset_args, video_index
from extract_keypoints_torchvision import pick_device

SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet statistics, as in V-JEPA 2's processor
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_clip(path, n_frames):
    """Return n_frames RGB frames spread evenly over the video, resized to SIZE x SIZE."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"no frames in {path}")
    picks = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
    clip = [cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            for i in picks]
    return (np.stack(clip).astype(np.float32) / 255 - MEAN) / STD  # (T, H, W, C)


@torch.no_grad()
def encode(model, clip, device, tubelet):
    x = torch.from_numpy(clip).permute(0, 3, 1, 2)[None].to(device)  # (1, T, C, H, W)
    tokens = model.get_vision_features(x)[0]                         # (T/tubelet * patches, D)
    steps = clip.shape[0] // tubelet
    return tokens.reshape(steps, -1, tokens.shape[-1]).mean(dim=1).cpu().numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(parser)
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256", help="Hugging Face V-JEPA 2 checkpoint")
    parser.add_argument("--frames", type=int, default=16, help="frames sampled per video")
    parser.add_argument("--out", help="output folder (default: data/<dataset>_vjepa2)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    args = parser.parse_args()
    out_dir = args.out or f"data/{args.dataset}_vjepa2"

    device = pick_device(args.device)
    model = VJEPA2Model.from_pretrained(args.model).eval().to(device)
    tubelet = model.config.tubelet_size
    print(f"{args.model} on {device}, {args.frames} frames per video")

    videos = video_index(args.dataset, args.videos)
    start, done = time.time(), 0
    for n, video in enumerate(videos, 1):
        out = os.path.join(out_dir, video["subset"], video["id"] + ".npy")
        if os.path.exists(out):
            continue
        features = encode(model, read_clip(video["path"], args.frames), device, tubelet)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, features)
        done += 1
        if done % 50 == 0:
            rate = done / (time.time() - start)
            print(f"{n}/{len(videos)} videos, {(len(videos) - n) / rate / 60:.0f} min left", flush=True)
    print(f"encoded {done} videos in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
