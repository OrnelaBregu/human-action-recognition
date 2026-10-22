"""CNN-LSTM: a frozen ImageNet CNN encodes each frame, an LSTM models the sequence.

Each video is read as up to --max-frames frames of 120x160 RGB (resized if needed),
zero-padded at the start if shorter. The pretrained backbone (VGG16 or ResNet50, global
average pooled) is applied to every frame with TimeDistributed; an LSTM(256) and a
Dense(512) layer then classify the actions. Trained for 12 epochs with Adam.

By default the videos are split at random (80% train / 20% test), as in the original KTH
project. That lets the same person or scene appear in both sets, so --split official uses
the dataset's own split instead (KTH subjects, UCF101 groups) and is the better choice.

Memory: KTH with 250 frames per video needs roughly 35 GB of RAM. --frame-step and
--max-frames reduce it (e.g. --frame-step 4 --max-frames 60 for UCF101 uses about 3 GB).

Usage:
    python scripts/train_cnn_lstm.py --dataset kth --backbone vgg16
    python scripts/train_cnn_lstm.py --dataset ucf101 --backbone resnet50 --split official --frame-step 4 --max-frames 60
"""

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import LSTM, Dense, GlobalAveragePooling2D, Input, TimeDistributed
from tensorflow.keras.models import Model
from tqdm import tqdm

from common import add_dataset_args, categories, output_path, report, save_confusion_matrix, video_index

ROWS, COLUMNS, CHANNELS = 120, 160, 3
BACKBONES = {"vgg16": VGG16, "resnet50": ResNet50}


def read_video(path, max_frames, frame_step):
    """Read every frame_step-th frame, keep up to max_frames, zero-pad at the start."""
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % frame_step == 0:
            if frame.shape[:2] != (ROWS, COLUMNS):
                frame = cv2.resize(frame, (COLUMNS, ROWS), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        i += 1
    cap.release()
    buf = np.zeros((max_frames, ROWS, COLUMNS, CHANNELS), dtype=np.uint8)
    if frames:
        buf[max_frames - len(frames):] = frames
    return buf


def build_model(backbone, max_frames, n_classes):
    base = BACKBONES[backbone](input_shape=(ROWS, COLUMNS, CHANNELS), weights="imagenet", include_top=False)
    cnn = Model(inputs=base.input, outputs=GlobalAveragePooling2D()(base.output))
    cnn.trainable = False

    video = Input(shape=(max_frames, ROWS, COLUMNS, CHANNELS))
    encoded = LSTM(256)(TimeDistributed(cnn)(video))
    hidden = Dense(512, activation="relu")(encoded)
    # The original notebooks also defined Dropout(0.3) -> Dense(256) -> Dropout(0.3) after this
    # layer, but connected the output to `hidden` directly, so those layers were never used.
    # The reported results come from this simpler model.
    outputs = Dense(n_classes, activation="softmax")(hidden)
    model = Model(video, outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])
    return model


def save_history(history, epochs, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["acc", "loss"]):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="val")
        ax.set_xticks(np.arange(epochs))
        ax.set_xlabel("epoch")
        ax.set_title(f"model {'accuracy' if metric == 'acc' else 'loss'}")
        ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(parser)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="vgg16")
    parser.add_argument("--split", choices=["random", "official"], default="random")
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--frame-step", type=int, default=1, help="use every Nth frame")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    videos = video_index(args.dataset, args.videos)
    labels = categories(videos)
    X = np.array([read_video(v["path"], args.max_frames, args.frame_step) for v in tqdm(videos, desc="reading videos")])
    y = np.array([labels.index(v["label"]) for v in videos])
    if args.split == "random":
        X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        test = np.array([v["subset"] == "testing" for v in videos])
        X_trn, X_tst, y_trn, y_tst = X[~test], X[test], y[~test], y[test]
        # Keras takes the validation set from the end of the training data, which is
        # ordered by class here, so shuffle first.
        order = np.random.default_rng(42).permutation(len(y_trn))
        X_trn, y_trn = X_trn[order], y_trn[order]
    del X

    model = build_model(args.backbone, args.max_frames, len(labels))
    model.summary()
    history = model.fit(X_trn, y_trn, validation_split=0.2, epochs=args.epochs)

    name = f"{args.backbone}_lstm_{args.split}"
    save_history(history, args.epochs, output_path(args.dataset, "cnn_lstm", f"{name}_history.png"))
    model.save(output_path(args.dataset, "cnn_lstm", f"{name}.keras"))

    y_pred = np.argmax(model.predict(X_tst), axis=1)
    report(name, y_tst, y_pred)
    save_confusion_matrix(y_tst, y_pred, output_path(args.dataset, "cnn_lstm", f"{name}_confusion.png"), labels)


if __name__ == "__main__":
    main()
