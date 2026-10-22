"""Split the raw KTH videos into the individual action sequences listed by the dataset authors.

Reconstructed: the original team used a pre-split copy of the data, so this step was not
part of the course code. It reads 00sequences.txt from the KTH website and writes one clip
per sequence, named <person>_<action>_<scenario>_<start>to<end>.avi, which is the naming
the other scripts expect.

Usage:
    python scripts/divide_sequences.py --raw data/kth --sequences data/kth/00sequences.txt --out data/divide
"""

import argparse
import os
import re

import cv2

from common import KTH_CATEGORIES as CATEGORIES


def parse_sequences(path):
    """Map each video name (e.g. person01_boxing_d1) to its list of (start, end) frame ranges."""
    sequences = {}
    with open(path) as f:
        for line in f:
            match = re.match(r"\s*(person\d+_\w+_d\d)\s+frames\s+(.*)", line)
            if not match:
                continue
            ranges = [tuple(int(x) for x in r.split("-")) for r in match.group(2).split(",") if r.strip()]
            sequences[match.group(1)] = ranges
    return sequences


def split_video(src, ranges, out_dir, name):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        print(f"warning: could not read {src}")
        return
    height, width = frames[0].shape[:2]
    for start, end in ranges:
        out = os.path.join(out_dir, f"{name}_{start}to{end}.avi")
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
        for frame in frames[start - 1:end]:  # KTH frame numbers start at 1
            writer.write(frame)
        writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default="data/kth", help="folder with the original KTH .avi files")
    parser.add_argument("--sequences", default="data/kth/00sequences.txt")
    parser.add_argument("--out", default="data/divide")
    args = parser.parse_args()

    sequences = parse_sequences(args.sequences)
    for category in CATEGORIES:
        os.makedirs(os.path.join(args.out, category), exist_ok=True)

    n_done = 0
    for root, _, files in os.walk(args.raw):
        for filename in sorted(files):
            if not filename.endswith(".avi"):
                continue
            name = filename.replace("_uncomp.avi", "").replace(".avi", "")
            if name not in sequences:
                print(f"warning: no sequence info for {filename}, skipping")
                continue
            category = name.split("_")[1]
            split_video(os.path.join(root, filename), sequences[name], os.path.join(args.out, category), name)
            n_done += 1
            if n_done % 50 == 0:
                print(f"split {n_done} videos")
    print(f"done: split {n_done} videos into {args.out}")


if __name__ == "__main__":
    main()
