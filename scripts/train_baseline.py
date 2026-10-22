"""Baseline: classify individual raw frames, then label each video from its frame predictions.

Trains logistic regression, a 2-layer MLP and AdaBoost on standardized pixels (training and
validation merged), and reports frame-level and video-level metrics on the test set. As in
the report, a video's label is the rounded mean of its frame predictions; --video-agg vote
uses a majority vote instead. Averaging class indices was the report's choice, so vote is
the better option for new datasets.

Needs a lot of memory for KTH (about 200,000 frames of 14,400 values each).

Usage:
    python scripts/extract_frames.py --dataset kth
    python scripts/train_baseline.py --dataset kth [--video-agg mean|vote]
"""

import argparse
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing

from common import DATASETS, flatten, output_path, report, save_confusion_matrix


def video_predictions(video_ids, y_true, y_pred, agg, n_classes):
    """Aggregate frame predictions into one prediction per video."""
    true, pred = [], []
    for video in sorted(set(video_ids)):
        mask = video_ids == video
        true.append(int(np.bincount(y_true[mask]).argmax()))
        if agg == "mean":
            pred.append(int(np.round(y_pred[mask].mean())))
        else:
            pred.append(int(np.bincount(y_pred[mask], minlength=n_classes).argmax()))
    return np.array(true), np.array(pred)


def plot_weight_grid(W, path):
    """Plot the first learned weight images (one per class or hidden unit) in a grid."""
    k = min(len(W), 24)
    ncol = 6
    nrow = (k + ncol - 1) // ncol
    vmin, vmax = np.percentile(W[:k], [0.1, 99.9])
    fig, axes = plt.subplots(nrow, ncol, figsize=(2 * ncol, 2 * nrow * 80 / 180), squeeze=False)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for w, ax in zip(W[:k], axes.flat):
        ax.matshow(w.reshape(80, 180), vmin=vmin, vmax=vmax, cmap="gray")
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="kth")
    parser.add_argument("--frames", help="frames folder (default: data/<dataset>_frames)")
    parser.add_argument("--video-agg", choices=["mean", "vote"], default="mean")
    args = parser.parse_args()
    frames_dir = args.frames or f"data/{args.dataset}_frames"

    labels = joblib.load(os.path.join(frames_dir, "labels.joblib"))
    trn, val, tst = (joblib.load(os.path.join(frames_dir, f"{n}.joblib")) for n in ["training", "validation", "testing"])
    X_trn = flatten(np.concatenate([trn["X"], val["X"]]))
    y_trn = np.concatenate([trn["y"], val["y"]])
    X_tst, y_tst, video_tst = flatten(tst["X"]), tst["y"], tst["video"]
    print(f"{len(y_trn)} training frames, {len(y_tst)} test frames, {len(labels)} classes")

    scaler = sklearn.preprocessing.StandardScaler().fit(X_trn)
    X_trn = scaler.transform(X_trn)
    X_tst = scaler.transform(X_tst)

    models = {
        "logistic_regression": sklearn.linear_model.LogisticRegression(random_state=0, C=0.1),
        "mlp_2_layers": sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=(100, 50), activation="relu", solver="sgd", batch_size=100,
            learning_rate_init=0.01, max_iter=10, random_state=0, verbose=True, momentum=0.9,
        ),
        # SAMME, the only algorithm in current scikit-learn
        "adaboost": sklearn.ensemble.AdaBoostClassifier(n_estimators=50, random_state=0),
    }

    for name, model in models.items():
        start = time.time()
        model.fit(X_trn, y_trn)
        print(f"\n{name}: trained in {time.time() - start:.0f} s")
        joblib.dump(model, output_path(args.dataset, "baseline", f"{name}.joblib"))

        y_pred = model.predict(X_tst)
        report(f"{name} (frames)", y_tst, y_pred, model.predict_proba(X_tst))
        v_true, v_pred = video_predictions(video_tst, y_tst, y_pred, args.video_agg, len(labels))
        report(f"{name} (videos, {args.video_agg})", v_true, v_pred)
        save_confusion_matrix(v_true, v_pred, output_path(args.dataset, "baseline", f"{name}_confusion.png"), labels)

        if name == "logistic_regression":
            plot_weight_grid(model.coef_, output_path(args.dataset, "baseline", f"{name}_weights.png"))
        elif name == "mlp_2_layers":
            plot_weight_grid(model.coefs_[0].T, output_path(args.dataset, "baseline", f"{name}_weights.png"))


if __name__ == "__main__":
    main()
