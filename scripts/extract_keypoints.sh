#!/bin/bash
# Run Detectron2 Keypoint R-CNN (R-101 FPN) on every video under a folder.
#
# Needs Detectron2 and infer_video_d2.py from VideoPose3D:
#   https://github.com/facebookresearch/VideoPose3D/tree/main/inference
# Copy infer_video_d2.py into scripts/, then run from the repository root:
#   bash scripts/extract_keypoints.sh data/divide data/kth_poses        # KTH: one subfolder per action
#   bash scripts/extract_keypoints.sh data/ucf101 data/ucf101_poses     # UCF101: train/ and test/
#
# Writes one .npz file per video, mirroring the input subfolders.
# KTH took just under 8 hours on an RTX 2060.

set -e

videos=${1:-data/divide}
out=${2:-data/kth_poses}

for dir in "$videos"/*/; do
    name=$(basename "$dir")
    mkdir -p "$out/$name"
    python scripts/infer_video_d2.py \
        --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
        --output-dir "$out/$name/" \
        --image-ext avi \
        "$dir"
done
