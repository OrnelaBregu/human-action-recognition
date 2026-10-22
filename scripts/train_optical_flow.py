"""Optical flow + bag of visual words, classified with classical models.

Pipeline:
  1. Dense Farneback optical flow between consecutive grayscale frames (resized to 160x120),
     sampled on a 12x16 grid (every 10th pixel): a 384-value motion descriptor per frame pair.
  2. K-means on the training descriptors builds a visual dictionary (1000 words by default).
  3. Each video becomes a TF-IDF weighted histogram of visual words.
  4. Logistic regression, linear SVM, random forest, KNN, AdaBoost and an MLP are tuned with
     5-fold cross-validation on the training (+ validation) videos and tested on the test set.

Intermediate results are cached in data/<dataset>_optf/; use --force to recompute them.

Usage:
    python scripts/train_optical_flow.py --dataset kth [--clusters 1000] [--force]
"""

import argparse
import os
import random
import time

import cv2
import joblib
import numpy as np
import scipy.stats
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from common import add_dataset_args, categories, output_path, report, save_confusion_matrix, video_index

RANDOM_STATE = 0
FARNEBACK_PARAMS = dict(
    winsize=20, iterations=1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
    pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None,
)


def flow_features(path, frame_step=1):
    """Return one optical-flow descriptor per pair of consecutive (kept) frames."""
    cap = cv2.VideoCapture(path)
    features, prev, i = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % frame_step:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.shape != (120, 160):
            frame = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, frame, **FARNEBACK_PARAMS)
            features.append(flow[0:120:10, 0:160:10, :].reshape(-1))  # (dx, dy) on a 12x16 grid
        prev = frame
    cap.release()
    return np.array(features, dtype=np.float32).reshape(-1, 384)


def extract_all(videos, frame_step):
    for n, video in enumerate(videos, 1):
        video["features"] = flow_features(video["path"], frame_step)
        if n % 50 == 0:
            print(f"optical flow: {n}/{len(videos)} videos")
    return videos


def bag_of_words(videos, centers, tfidf=True):
    bow = np.zeros((len(videos), centers.shape[0]))
    for i, video in enumerate(videos):
        if len(video["features"]):
            for word in vq(video["features"], centers)[0]:
                bow[i, word] += 1
    if tfidf:
        doc_freq = np.sum(bow > 0, axis=0)
        bow = bow * np.log((len(videos) + 1) / (doc_freq + 1))
    return bow


def cached(path, compute, force):
    if os.path.exists(path) and not force:
        return joblib.load(path)
    result = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(result, path)
    return result


def search_spaces(n_iter, n_train):
    mlp_params = dict(
        alpha=scipy.stats.uniform(1e-6, 1e-1).rvs(100, random_state=RANDOM_STATE),
        learning_rate_init=scipy.stats.uniform(1e-6, 1e-1).rvs(100, random_state=RANDOM_STATE + 1),
        hidden_layer_sizes=[(int(n),) for n in scipy.stats.randint(1, 2500).rvs(100, random_state=RANDOM_STATE)],
    )
    max_k = int(n_train * 0.8)  # neighbours must fit in a cross-validation training fold
    cv = dict(cv=5, verbose=1, n_jobs=-1)
    return {
        "svm_linear": sklearn.model_selection.RandomizedSearchCV(
            SVC(kernel="linear", max_iter=int(1e6), random_state=RANDOM_STATE),
            {"C": scipy.stats.reciprocal(1e-3, 1e3)}, n_iter=n_iter, random_state=RANDOM_STATE, **cv),
        # The original grid was ["auto", "sqrt"]; for classifiers "auto" meant "sqrt" and has been removed.
        "random_forest": sklearn.model_selection.GridSearchCV(
            sklearn.ensemble.RandomForestClassifier(random_state=RANDOM_STATE),
            {"max_depth": [2 ** i for i in range(12)], "max_features": ["sqrt"]}, **cv),
        "logistic_regression": sklearn.model_selection.GridSearchCV(
            sklearn.linear_model.LogisticRegression(random_state=RANDOM_STATE, max_iter=10000),
            {"C": [10.0 ** k for k in range(-5, 6)]}, **cv),
        "knn": sklearn.model_selection.GridSearchCV(
            sklearn.neighbors.KNeighborsClassifier(),
            {"n_neighbors": [k for k in [1] + [3 * i for i in range(1, 100)] if k <= max_k]}, **cv),
        "adaboost": sklearn.model_selection.GridSearchCV(
            sklearn.ensemble.AdaBoostClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [10, 50, 100, 500]}, **cv),
        "mlp_1_layer": sklearn.model_selection.RandomizedSearchCV(
            sklearn.neural_network.MLPClassifier(random_state=RANDOM_STATE),
            mlp_params, n_iter=n_iter, random_state=RANDOM_STATE, **cv),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(parser)
    parser.add_argument("--work", help="folder for cached results (default: data/<dataset>_optf)")
    parser.add_argument("--clusters", type=int, default=1000, help="visual dictionary size")
    parser.add_argument("--n-iter", type=int, default=100, help="iterations for randomized searches")
    parser.add_argument("--frame-step", type=int, default=1, help="use every Nth frame")
    parser.add_argument("--force", action="store_true", help="recompute cached features and dictionary")
    args = parser.parse_args()
    work = args.work or f"data/{args.dataset}_optf"

    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    videos = video_index(args.dataset, args.videos)
    labels = categories(videos)
    videos = cached(os.path.join(work, f"optical_flow_step{args.frame_step}.joblib"),
                    lambda: extract_all(videos, args.frame_step), args.force)
    # Training and validation videos are merged and used for cross-validation.
    train = [v for v in videos if v["subset"] != "testing"]
    test = [v for v in videos if v["subset"] == "testing"]
    print(f"{len(train)} training videos, {len(test)} test videos, {len(labels)} classes")

    def fit_dictionary():
        descriptors = np.vstack([v["features"] for v in train])
        print(f"running k-means on {len(descriptors)} descriptors")
        return KMeans(init="k-means++", n_clusters=args.clusters, n_init=10, verbose=1,
                      random_state=RANDOM_STATE).fit(descriptors)

    kmeans = cached(os.path.join(work, f"kmeans_{args.clusters}_step{args.frame_step}.joblib"), fit_dictionary, args.force)
    X_trn = bag_of_words(train, kmeans.cluster_centers_)
    X_tst = bag_of_words(test, kmeans.cluster_centers_)
    y_trn = np.array([labels.index(v["label"]) for v in train])
    y_tst = np.array([labels.index(v["label"]) for v in test])

    scaler = sklearn.preprocessing.StandardScaler().fit(X_trn)
    X_trn, X_tst = scaler.transform(X_trn), scaler.transform(X_tst)

    for name, search in search_spaces(args.n_iter, len(train)).items():
        start = time.time()
        search.fit(X_trn, y_trn)
        print(f"\n{name}: best params {search.best_params_}, trained in {time.time() - start:.0f} s")
        model = search.best_estimator_
        joblib.dump(model, output_path(args.dataset, "optical_flow", f"{name}_c{args.clusters}.joblib"))

        y_pred = model.predict(X_tst)
        report(name, y_tst, y_pred)
        save_confusion_matrix(y_tst, y_pred, output_path(args.dataset, "optical_flow", f"{name}_confusion.png"), labels)


if __name__ == "__main__":
    main()
