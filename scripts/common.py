"""Shared dataset index, subject/group splits, metrics and plotting helpers.

Every script works on a list of videos ("the index"), so the same pipelines run on:
  kth     KTH human actions, split into sequences by divide_sequences.py (data/divide/<action>/*.avi)
  ucf101  UCF101 (or a subset of it) laid out as train/, test/, train.csv, test.csv (data/ucf101)
"""

import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")  # scripts save figures instead of showing them
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

DATASETS = ["kth", "ucf101"]
DEFAULT_VIDEOS = {"kth": "data/divide", "ucf101": "data/ucf101"}

# KTH subject split recommended by the dataset authors:
# https://www.csc.kth.se/cvap/actions/00sequences.txt
KTH_TRAIN_PEOPLE = [11, 12, 13, 14, 15, 16, 17, 18]
KTH_VALIDATION_PEOPLE = [19, 20, 21, 23, 24, 25, 1, 4]
KTH_TEST_PEOPLE = [22, 2, 3, 5, 6, 7, 8, 9, 10]
KTH_CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]

SUBSETS = ["training", "validation", "testing"]
OUTPUT_DIR = "outputs"


def kth_person(filename):
    """Return the subject number from a KTH filename such as person01_boxing_d1_1to95.avi."""
    return int(os.path.basename(filename).split("_")[0][len("person"):])


def kth_subset(filename):
    person = kth_person(filename)
    if person in KTH_TRAIN_PEOPLE:
        return "training"
    if person in KTH_VALIDATION_PEOPLE:
        return "validation"
    return "testing"


def video_index(dataset, root=None):
    """Return a list of videos as dicts with path, id, label, subset and group.

    The group is the KTH subject or the UCF101 group (videos from the same group share
    actors and scene), and splits always keep a group on one side.
    """
    root = root or DEFAULT_VIDEOS[dataset]
    videos = []
    if dataset == "kth":
        for path in sorted(glob.glob(os.path.join(root, "*", "*.avi"))):
            videos.append(dict(path=path, label=os.path.basename(os.path.dirname(path)),
                               subset=kth_subset(path), group=kth_person(path)))
    elif dataset == "ucf101":
        # UCF101 official split 1: groups 1-7 are the test set. There is no separate
        # validation set; the classical models use cross-validation on the training set.
        for subset, folder in [("training", "train"), ("testing", "test")]:
            with open(os.path.join(root, f"{folder}.csv")) as f:
                for row in csv.DictReader(f):
                    name = row["video_name"]
                    videos.append(dict(path=os.path.join(root, folder, name), label=row["tag"], subset=subset,
                                       group=int(re.search(r"_g(\d+)_", name).group(1))))
    else:
        raise ValueError(f"unknown dataset {dataset}")
    for video in videos:
        video["id"] = os.path.splitext(os.path.basename(video["path"]))[0]
    if not videos:
        raise SystemExit(f"no videos found for {dataset} in {root}")
    return videos


def categories(videos):
    return sorted({v["label"] for v in videos})


def add_dataset_args(parser):
    parser.add_argument("--dataset", choices=DATASETS, default="kth")
    parser.add_argument("--videos", help="dataset folder (default: data/divide for kth, data/ucf101 for ucf101)")


def report(name, y_true, y_pred, y_proba=None):
    """Print the metrics used in the project report (all weighted except accuracy)."""
    print(f"\n##### {name} #####")
    print("accuracy:           {:.2f}".format(100 * sklearn.metrics.accuracy_score(y_true, y_pred)))
    print("weighted f1:        {:.2f}".format(100 * sklearn.metrics.f1_score(y_true, y_pred, average="weighted")))
    print("weighted precision: {:.2f}".format(
        100 * sklearn.metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)))
    print("weighted recall:    {:.2f}".format(100 * sklearn.metrics.recall_score(y_true, y_pred, average="weighted")))
    if y_proba is not None:
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
        auc = sklearn.metrics.roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        print("weighted auc:       {:.2f}".format(100 * auc))


def save_confusion_matrix(y_true, y_pred, path, labels):
    """Save raw and row-normalized confusion matrices side by side."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, normalize in zip(axes, [None, "true"]):
        sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=list(range(len(labels))), display_labels=labels,
            xticks_rotation="vertical", normalize=normalize, ax=ax, colorbar=False,
        )
        ax.set_title("normalized confusion matrix" if normalize else "confusion matrix")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def output_path(dataset, *parts):
    path = os.path.join(OUTPUT_DIR, dataset, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def flatten(X):
    """Flatten every sample of X into a 1D feature vector."""
    X = np.asarray(X)
    return X.reshape(X.shape[0], int(np.prod(X.shape[1:])))  # explicit size, so empty arrays work too
