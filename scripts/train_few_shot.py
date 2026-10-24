"""Few-label comparison: a frozen video foundation model vs. pose keypoints.

For each budget of k labelled training videos per class, k videos per class are drawn at
random from the training (+ validation) videos, a logistic regression is trained on them,
and it is tested on the whole official test set. Each budget is repeated over several
random draws, and both feature sets use exactly the same draws, so the comparison is
paired. "all" trains once on every training video.

Features:
  vjepa2  frozen V-JEPA 2 features from extract_video_features.py, averaged over time
          and L2-normalized (data/<dataset>_vjepa2)
  pose    pose sequences from prepare_keypoints.py, standardized (data/<dataset>_keypoints)

The classifier is not tuned (C=1): with one or two videos per class there is nothing to
tune it on, and using the same classifier at every budget keeps the curve comparable.

Usage:
    python scripts/train_few_shot.py --dataset ucf101 [--shots 1 2 5 10 20] [--draws 5]
"""

import argparse
import csv
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model

from common import SUBSETS, add_dataset_args, categories, output_path, report, save_confusion_matrix, video_index
from train_keypoints import load as load_pose

FEATURES = {"vjepa2": "V-JEPA 2 (frozen)", "pose": "Pose keypoints"}
COLORS = {"vjepa2": "#2a78d6", "pose": "#eb6834"}  # categorical slots 1-2, validated for colour-blind readers
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def load_vjepa2(videos, feature_dir):
    X = {}
    for subset in SUBSETS:
        chosen = [v for v in videos if v["subset"] == subset]
        if not chosen:  # UCF101 has no validation set
            X[subset] = np.zeros((0, 0), dtype=np.float32)
            continue
        feats = np.array([np.load(os.path.join(feature_dir, subset, v["id"] + ".npy")).astype(np.float32).mean(axis=0)
                          for v in chosen])
        X[subset] = feats / np.linalg.norm(feats, axis=1, keepdims=True).clip(1e-8)
    return X


def classifier():
    return sklearn.linear_model.LogisticRegression(C=1.0, max_iter=10000)


def draw(y, k, rng):
    """Indices of k random examples per class."""
    return np.concatenate([rng.choice(np.flatnonzero(y == c), size=k, replace=False) for c in np.unique(y)])


def plot(rows, shots, labels_title, path):
    budgets = [str(k) for k in shots] + ["all"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for name, title in FEATURES.items():
        mean = [np.mean([r["accuracy"] for r in rows if r["features"] == name and r["shots"] == b]) for b in budgets]
        std = [np.std([r["accuracy"] for r in rows if r["features"] == name and r["shots"] == b]) for b in budgets]
        x = np.arange(len(budgets))
        ax.fill_between(x, np.subtract(mean, std), np.add(mean, std), color=COLORS[name], alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=COLORS[name], linewidth=2, marker="o", markersize=6, label=title,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{mean[-1]:.1f}%", (x[-1], mean[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xticks(np.arange(len(budgets)), budgets)
    ax.set_xlim(-0.3, len(budgets) - 0.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("labelled training videos per class", color=MUTED)
    ax.set_ylabel("test accuracy (%)", color=MUTED)
    ax.set_title(labels_title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(parser)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 5, 10, 20], help="labelled videos per class")
    parser.add_argument("--draws", type=int, default=5, help="random draws per budget")
    args = parser.parse_args()

    videos = video_index(args.dataset, args.videos)
    labels = categories(videos)
    y = {s: np.array([labels.index(v["label"]) for v in videos if v["subset"] == s], dtype=int) for s in SUBSETS}
    X = {"vjepa2": load_vjepa2(videos, f"data/{args.dataset}_vjepa2")}
    X["pose"], y_pose, _ = load_pose(f"data/{args.dataset}_keypoints")
    for s in SUBSETS:  # pose sequences follow the dataset index, so the same rows are the same videos
        assert np.array_equal(y_pose[s], y[s]), f"pose and video features do not line up in {s}"

    parts = [s for s in ["training", "validation"] if len(y[s])]
    pool = {name: np.vstack([X[name][s] for s in parts]) for name in FEATURES}
    y_pool = np.hstack([y[s] for s in parts])
    print(f"{len(y_pool)} labelled videos to draw from, {len(y['testing'])} test videos, {len(labels)} classes")

    rows = []
    for k in args.shots:
        for d in range(args.draws):
            idx = draw(y_pool, k, np.random.default_rng(1000 * k + d))
            for name in FEATURES:
                model = classifier().fit(pool[name][idx], y_pool[idx])
                acc = 100 * (model.predict(X[name]["testing"]) == y["testing"]).mean()
                rows.append(dict(features=name, shots=str(k), draw=d, accuracy=acc))
        for name in FEATURES:
            accs = [r["accuracy"] for r in rows if r["features"] == name and r["shots"] == str(k)]
            print(f"{k:>3} per class  {name:7s} {np.mean(accs):5.1f} ± {np.std(accs):4.1f}")

    for name in FEATURES:
        model = classifier().fit(pool[name], y_pool)
        y_pred = model.predict(X[name]["testing"])
        rows.append(dict(features=name, shots="all", draw=0, accuracy=100 * (y_pred == y["testing"]).mean()))
        report(f"{name} (all labels)", y["testing"], y_pred, model.predict_proba(X[name]["testing"]))
        save_confusion_matrix(y["testing"], y_pred, output_path(args.dataset, "few_shot", f"{name}_all_confusion.png"), labels)
        joblib.dump(model, output_path(args.dataset, "few_shot", f"{name}_all.joblib"))

    with open(output_path(args.dataset, "few_shot", "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["features", "shots", "draw", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    title = f"{'KTH' if args.dataset == 'kth' else 'UCF101'}: accuracy vs. labelled videos"
    plot(rows, args.shots, title, output_path(args.dataset, "few_shot", "accuracy_vs_labels.png"))


if __name__ == "__main__":
    main()
