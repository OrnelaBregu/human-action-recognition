"""Classify pose-keypoint sequences (from Detectron2 Keypoint R-CNN) with classical models.

Tunes dummy baselines, logistic regression, linear and polynomial SVMs, random forest,
AdaBoost, KNN and 1-, 2- and 3-layer MLPs with randomized search (5-fold CV on the training
+ validation videos), then reports test metrics and saves confusion matrices.

Usage:
    python scripts/prepare_keypoints.py --dataset kth
    python scripts/train_keypoints.py --dataset kth [--only mlp_3_layers ...]
"""

import argparse
import os
import random
import time

import joblib
import numpy as np
import scipy.stats
import sklearn.dummy
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm

from common import DATASETS, SUBSETS, flatten, output_path, report, save_confusion_matrix

RANDOM_STATE = 1


def load(data_dir):
    files = {s: np.load(os.path.join(data_dir, f"{s}.npz"), allow_pickle=True) for s in SUBSETS}
    labels = list(files["training"]["labels"])
    X = {s: flatten(files[s]["X"]) for s in SUBSETS}
    y = {s: files[s]["y"] for s in SUBSETS}
    scaler = sklearn.preprocessing.StandardScaler().fit(X["training"])
    X = {s: scaler.transform(X[s]) if len(X[s]) else X[s] for s in X}
    return X, y, labels


def mlp_space(layers, max_units):
    sizes = [tuple(int(n) for n in scipy.stats.randint(1, max_units).rvs(layers)) for _ in range(100)]
    rates = scipy.stats.uniform(1e-6, 1e-1).rvs(100)
    return dict(alpha=rates, learning_rate_init=rates, hidden_layer_sizes=sizes)


def models(n_train):
    """Return (name, estimator, search space, n_iter); a None space means no tuning."""
    reciprocal = scipy.stats.reciprocal(1e-3, 1e3)
    svm_space = dict(C=reciprocal.rvs(100), gamma=reciprocal.rvs(100), degree=list(range(1, 11)))
    mlp = lambda: sklearn.neural_network.MLPClassifier(max_iter=500, early_stopping=True, random_state=RANDOM_STATE)
    max_k = min(500, int(n_train * 0.8))  # neighbours must fit in a cross-validation training fold
    entries = [
        (f"dummy_{s}", sklearn.dummy.DummyClassifier(strategy=s, random_state=RANDOM_STATE), None, 0)
        for s in ["stratified", "most_frequent", "prior", "uniform"]
    ]
    entries += [
        ("logistic_regression", sklearn.linear_model.LogisticRegression(random_state=RANDOM_STATE, max_iter=int(1e5)),
         dict(C=reciprocal.rvs(1000)), 56),
        ("svm_linear", sklearn.svm.SVC(kernel="linear", probability=True, max_iter=int(1e8), random_state=RANDOM_STATE),
         svm_space, 50),
        ("svm_poly", sklearn.svm.SVC(kernel="poly", probability=True, max_iter=int(1e8), random_state=RANDOM_STATE),
         svm_space, 50),
        ("random_forest", sklearn.ensemble.RandomForestClassifier(random_state=RANDOM_STATE),
         dict(max_depth=scipy.stats.randint(1, 2 ** 12).rvs(1000), n_estimators=scipy.stats.randint(1, 2 ** 12).rvs(1000)), 50),
        ("adaboost", sklearn.ensemble.AdaBoostClassifier(random_state=RANDOM_STATE),
         dict(n_estimators=scipy.stats.randint(1000, 2000).rvs(10)), "grid"),
    ]
    entries += [
        (f"knn_{w}", sklearn.neighbors.KNeighborsClassifier(weights=w, n_jobs=-1),
         dict(n_neighbors=scipy.stats.randint(1, max_k).rvs(1000)), 50)
        for w in ["uniform", "distance"]
    ]
    entries += [
        ("mlp_1_layer", mlp(), mlp_space(1, 2500), 50),
        ("mlp_2_layers", mlp(), mlp_space(2, 2000), 50),
        ("mlp_3_layers", mlp(), mlp_space(3, 2000), 50),
    ]
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="kth")
    parser.add_argument("--data", help="prepared keypoints (default: data/<dataset>_keypoints)")
    parser.add_argument("--only", nargs="*", help="train only these models (names as printed)")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    X, y, labels = load(args.data or f"data/{args.dataset}_keypoints")
    parts = [s for s in ["training", "validation"] if len(y[s])]
    X_trnval, y_trnval = np.vstack([X[s] for s in parts]), np.hstack([y[s] for s in parts])

    for name, estimator, space, n_iter in models(len(y_trnval)):
        if args.only and name not in args.only:
            continue
        start = time.time()
        if space is None:
            model = estimator.fit(X["training"], y["training"])
        elif n_iter == "grid":
            model = sklearn.model_selection.GridSearchCV(estimator, space, cv=2, n_jobs=-1, verbose=1).fit(X_trnval, y_trnval)
        else:
            model = sklearn.model_selection.RandomizedSearchCV(
                estimator, space, n_iter=n_iter, random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
            ).fit(X_trnval, y_trnval)
        if space is not None:
            print(f"{name}: best params {model.best_params_}, CV accuracy {100 * model.best_score_:.2f}")
            model = model.best_estimator_
        print(f"{name}: trained in {time.time() - start:.1f} s")

        y_pred = model.predict(X["testing"])
        report(name, y["testing"], y_pred, model.predict_proba(X["testing"]))
        save_confusion_matrix(y["testing"], y_pred, output_path(args.dataset, "keypoints", f"{name}_confusion.png"), labels)
        joblib.dump(model, output_path(args.dataset, "keypoints", f"{name}.joblib"))


if __name__ == "__main__":
    main()
