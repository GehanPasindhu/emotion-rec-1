import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


SEED = 42
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_runtime_optimizations() -> bool:
    gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
    if gpu_available:
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    return gpu_available


def resolve_dataset_dirs(explicit_root: Optional[str] = None) -> tuple[Path, Path]:
    candidates = []
    if explicit_root:
        root = Path(explicit_root)
        candidates.append((root / "train", root / "test"))
    else:
        for root in [Path("."), Path("archive"), Path("../archive")]:
            candidates.append((root / "train", root / "test"))

    for train_dir, test_dir in candidates:
        if train_dir.is_dir() and test_dir.is_dir():
            return train_dir, test_dir

    searched = [str(train.parent.resolve()) for train, _ in candidates]
    raise FileNotFoundError(
        "Could not find FER2013 train/test folders. Checked: " + ", ".join(searched)
    )


def count_images(split_dir: Path) -> dict[str, int]:
    counts = {}
    for emotion in EMOTIONS:
        emotion_dir = split_dir / emotion
        counts[emotion] = len(
            [
                file
                for file in emotion_dir.iterdir()
                if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        ) if emotion_dir.is_dir() else 0
    return counts


def save_class_distribution(train_counts: dict[str, int], test_counts: dict[str, int], output_dir: Path) -> pd.DataFrame:
    df_counts = pd.DataFrame(
        {
            "Emotion": EMOTIONS,
            "Train": [train_counts[e] for e in EMOTIONS],
            "Test": [test_counts[e] for e in EMOTIONS],
        }
    )
    df_counts["Total"] = df_counts["Train"] + df_counts["Test"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, NUM_CLASSES))
    for ax, split in zip(axes, ["Train", "Test"]):
        vals = df_counts[split].values
        bars = ax.bar(df_counts["Emotion"].values, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{split} Set Class Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Number of Images")
        ax.tick_params(axis="x", rotation=30)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30, str(value), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(fig)
    return df_counts


def save_sample_images(train_dir: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(NUM_CLASSES, 5, figsize=(12, 2.5 * NUM_CLASSES))
    fig.suptitle("Sample Images per Emotion Class", fontsize=14, fontweight="bold", y=1.01)

    for row, emotion in enumerate(EMOTIONS):
        emotion_dir = train_dir / emotion
        image_paths = [path for path in emotion_dir.iterdir() if path.is_file()]
        samples = random.sample(image_paths, min(5, len(image_paths)))
        for col, image_path in enumerate(samples):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            axes[row, col].imshow(image, cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(emotion.capitalize(), fontsize=11, rotation=0, labelpad=55, va="center")

        for col in range(len(samples), 5):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "sample_images.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def dataset_file_labels(dataset) -> np.ndarray:
    class_to_idx = {name: idx for idx, name in enumerate(dataset.class_names)}
    return np.array([class_to_idx[Path(path).parent.name] for path in dataset.file_paths], dtype=np.int32)


def build_image_datasets(
    train_dir: Path,
    test_dir: Path,
    image_size: int,
    batch_size: int,
    color_mode: str,
    validation_split: float,
):
    common_kwargs = dict(
        labels="inferred",
        label_mode="categorical",
        class_names=EMOTIONS,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        seed=SEED,
        interpolation="bilinear",
    )

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="training",
        shuffle=True,
        color_mode=color_mode,
        **common_kwargs,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="validation",
        shuffle=False,
        color_mode=color_mode,
        **common_kwargs,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=False,
        color_mode=color_mode,
        **common_kwargs,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds


def build_class_weights(train_ds) -> dict[int, float]:
    y_train = dataset_file_labels(train_ds)
    weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    return dict(enumerate(weights))


def grayscale_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.05, fill_mode="nearest", seed=SEED),
            layers.RandomTranslation(0.10, 0.10, fill_mode="nearest", seed=SEED),
            layers.RandomZoom(0.10, fill_mode="nearest", seed=SEED),
            layers.RandomContrast(0.10, seed=SEED),
        ],
        name="gray_augmentation",
    )


def rgb_augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.04, fill_mode="nearest", seed=SEED),
            layers.RandomTranslation(0.08, 0.08, fill_mode="nearest", seed=SEED),
            layers.RandomZoom(0.10, fill_mode="nearest", seed=SEED),
            layers.RandomContrast(0.10, seed=SEED),
        ],
        name="rgb_augmentation",
    )


def build_custom_cnn(input_shape=(48, 48, 1), num_classes=NUM_CLASSES, l2_lambda=1e-4) -> keras.Model:
    reg = regularizers.l2(l2_lambda)
    augmentation = grayscale_augmentation()

    inputs = keras.Input(shape=input_shape, name="input")
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    for filters, spatial_dropout in [(32, 0.10), (64, 0.10), (128, 0.20), (256, 0.20)]:
        x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.SpatialDropout2D(spatial_dropout)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(384, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)
    x = layers.Dense(192, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inputs, outputs, name="CustomCNN")


def build_mobilenet_model(image_size: int, num_classes=NUM_CLASSES, l2_lambda=1e-4) -> tuple[keras.Model, keras.Model]:
    reg = regularizers.l2(l2_lambda)
    base = MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3), name="input")
    x = rgb_augmentation()(inputs)
    x = layers.Lambda(lambda t: mobilenet_preprocess(tf.cast(t, tf.float32)), name="mobilenet_preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inputs, outputs, name="MobileNetV2_FER"), base


def build_efficientnet_model(image_size: int, num_classes=NUM_CLASSES, l2_lambda=1e-4) -> tuple[keras.Model, keras.Model]:
    reg = regularizers.l2(l2_lambda)
    base = EfficientNetB0(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3), name="input")
    x = rgb_augmentation()(inputs)
    x = layers.Lambda(lambda t: tf.cast(t, tf.float32), name="efficientnet_input_cast")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(320, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(160, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inputs, outputs, name="EfficientNetB0_FER"), base


def compile_classifier(model: keras.Model, learning_rate: float, label_smoothing: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )


def build_callbacks(output_dir: Path, model_name: str, patience: int) -> tuple[Path, list]:
    checkpoint_path = output_dir / f"best_{model_name}.keras"
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, patience // 2), min_lr=1e-6, verbose=1),
    ]
    return checkpoint_path, callbacks


def merge_histories(first, second) -> dict[str, list[float]]:
    merged = {}
    for key in first.history:
        merged[key] = [float(value) for value in first.history[key] + second.history[key]]
    return merged


def plot_history(history_dict: dict[str, list[float]], model_name: str, output_dir: Path) -> None:
    epochs = range(1, len(history_dict["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} Training Curves", fontsize=13, fontweight="bold")

    ax1.plot(epochs, history_dict["accuracy"], label="Train acc", color="royalblue")
    ax1.plot(epochs, history_dict["val_accuracy"], label="Val acc", color="darkorange")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history_dict["loss"], label="Train loss", color="royalblue")
    ax2.plot(epochs, history_dict["val_loss"], label="Val loss", color="darkorange")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-entropy")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"history_{model_name.lower()}.png", dpi=150)
    plt.close(fig)


def evaluate_model(model: keras.Model, test_ds, model_name: str, output_dir: Path) -> dict:
    y_true = dataset_file_labels(test_ds)
    y_prob = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_pre = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype="float"), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{model_name} Confusion Matrix", fontsize=13, fontweight="bold")
    for ax, data, fmt, title in [
        (axes[0], cm_norm, ".2%", "Normalized"),
        (axes[1], cm, "d", "Raw counts"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS, linewidths=0.5, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{model_name.lower()}.png", dpi=150)
    plt.close(fig)

    return {
        "model_name": model_name,
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "macro_f1": float(macro_f1),
        "macro_pre": float(macro_pre),
        "macro_rec": float(macro_rec),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "report": report,
    }


def train_custom_cnn(train_ds, val_ds, test_ds, class_weights: dict[int, float], output_dir: Path, args) -> tuple[keras.Model, dict, dict]:
    model = build_custom_cnn(input_shape=(args.img_size, args.img_size, 1))
    compile_classifier(model, learning_rate=8e-4, label_smoothing=args.label_smoothing)
    _, callbacks = build_callbacks(output_dir, "custom_cnn", patience=10)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_baseline,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
    plot_history(history_dict, "custom_cnn", output_dir)
    results = evaluate_model(model, test_ds, "custom_cnn", output_dir)
    model.save(output_dir / "final_custom_cnn.keras")
    return model, history_dict, results


def train_mobilenet(train_ds, val_ds, test_ds, class_weights: dict[int, float], output_dir: Path, args) -> tuple[keras.Model, dict, dict]:
    model, base = build_mobilenet_model(args.img_size_tl)
    compile_classifier(model, learning_rate=5e-4, label_smoothing=args.label_smoothing)
    _, callbacks = build_callbacks(output_dir, "mobilenet", patience=7)

    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_tl_head,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    compile_classifier(model, learning_rate=1e-5, label_smoothing=args.label_smoothing)
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_tl_finetune,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    history_dict = merge_histories(head_history, fine_tune_history)
    plot_history(history_dict, "mobilenet", output_dir)
    results = evaluate_model(model, test_ds, "mobilenet", output_dir)
    model.save(output_dir / "final_mobilenet.keras")
    return model, history_dict, results


def train_efficientnet(train_ds, val_ds, test_ds, class_weights: dict[int, float], output_dir: Path, args) -> tuple[keras.Model, dict, dict]:
    model, base = build_efficientnet_model(args.img_size_tl)
    compile_classifier(model, learning_rate=4e-4, label_smoothing=args.label_smoothing)
    _, callbacks = build_callbacks(output_dir, "efficientnet", patience=7)

    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_tl_head,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    compile_classifier(model, learning_rate=5e-6, label_smoothing=args.label_smoothing)
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_tl_finetune,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    history_dict = merge_histories(head_history, fine_tune_history)
    plot_history(history_dict, "efficientnet", output_dir)
    results = evaluate_model(model, test_ds, "efficientnet", output_dir)
    model.save(output_dir / "final_efficientnet.keras")
    return model, history_dict, results


def save_summary(results: list[dict], histories: dict, output_dir: Path) -> None:
    comparison_df = pd.DataFrame(
        [
            {
                "Model": item["model_name"],
                "Test Acc.": item["test_acc"],
                "Test Loss": item["test_loss"],
                "Macro F1": item["macro_f1"],
                "Macro Precision": item["macro_pre"],
                "Macro Recall": item["macro_rec"],
            }
            for item in results
        ]
    ).sort_values("Test Acc.", ascending=False)

    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)

    with open(output_dir / "training_histories.json", "w", encoding="utf-8") as file:
        json.dump(histories, file, indent=2)

    with open(output_dir / "evaluation_results.txt", "w", encoding="utf-8") as file:
        for item in results:
            file.write(f"Model: {item['model_name']}\n")
            file.write(f"  Test Accuracy : {item['test_acc']:.4f}\n")
            file.write(f"  Test Loss     : {item['test_loss']:.4f}\n")
            file.write(f"  Macro F1      : {item['macro_f1']:.4f}\n")
            file.write(f"  Macro Prec.   : {item['macro_pre']:.4f}\n")
            file.write(f"  Macro Recall  : {item['macro_rec']:.4f}\n")
            file.write("\nClassification Report:\n")
            file.write(item["report"])
            file.write("\n" + "-" * 70 + "\n")

    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare FER2013 emotion-recognition models.")
    parser.add_argument("--data-root", type=str, default=None, help="Folder containing train/ and test/ directories.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory where models and reports are saved.")
    parser.add_argument("--models", nargs="+", choices=["cnn", "mobilenet", "efficientnet"], default=["cnn", "mobilenet", "efficientnet"])
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--img-size-tl", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--epochs-baseline", type=int, default=40)
    parser.add_argument("--epochs-tl-head", type=int, default=10)
    parser.add_argument("--epochs-tl-finetune", type=int, default=18)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    gpu_available = enable_runtime_optimizations()

    if args.batch_size is None:
        args.batch_size = 64 if gpu_available else 32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir, test_dir = resolve_dataset_dirs(args.data_root)
    print(f"TensorFlow version : {tf.__version__}")
    print(f"GPU available      : {gpu_available}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Train dir          : {train_dir.resolve()}")
    print(f"Test dir           : {test_dir.resolve()}")

    train_counts = count_images(train_dir)
    test_counts = count_images(test_dir)
    save_class_distribution(train_counts, test_counts, output_dir)
    save_sample_images(train_dir, output_dir)

    gray_train, gray_val, gray_test = build_image_datasets(
        train_dir, test_dir, args.img_size, args.batch_size, "grayscale", args.validation_split
    )
    rgb_train, rgb_val, rgb_test = build_image_datasets(
        train_dir, test_dir, args.img_size_tl, args.batch_size, "rgb", args.validation_split
    )

    class_weights = build_class_weights(gray_train)
    print("Class weights:")
    for idx, emotion in enumerate(EMOTIONS):
        print(f"  {emotion:10s} -> {class_weights[idx]:.4f}")

    histories = {}
    results = []

    if "cnn" in args.models:
        _, history, result = train_custom_cnn(gray_train, gray_val, gray_test, class_weights, output_dir, args)
        histories["custom_cnn"] = history
        results.append(result)

    if "mobilenet" in args.models:
        _, history, result = train_mobilenet(rgb_train, rgb_val, rgb_test, class_weights, output_dir, args)
        histories["mobilenet"] = history
        results.append(result)

    if "efficientnet" in args.models:
        _, history, result = train_efficientnet(rgb_train, rgb_val, rgb_test, class_weights, output_dir, args)
        histories["efficientnet"] = history
        results.append(result)

    save_summary(results, histories, output_dir)


if __name__ == "__main__":
    main()
