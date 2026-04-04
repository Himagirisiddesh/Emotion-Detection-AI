import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

from emotion_utils import BASE_DIR, MODEL_DIR, canonicalize_label, ensure_runtime_directories


IMAGE_SIZE = (48, 48)
DEFAULT_DATASET_DIR = BASE_DIR / "archive (2)" / "Data"
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train an improved CNN for facial emotion recognition.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATASET_DIR), help="Dataset directory with class folders.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial Adam learning rate.")
    return parser.parse_args()


def count_images_by_class(data_dir, raw_class_names):
    counts = {}
    for class_name in raw_class_names:
        folder = data_dir / class_name
        counts[class_name] = sum(1 for item in folder.iterdir() if item.is_file())
    return counts


def compute_class_weights(class_counts, raw_class_names):
    total = sum(class_counts.values())
    num_classes = len(raw_class_names)
    weights = {}

    for index, class_name in enumerate(raw_class_names):
        count = class_counts[class_name]
        if count > 0:
            weights[index] = total / (num_classes * count)

    return weights


def prepare_datasets(data_dir, batch_size, validation_split):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="categorical",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    return train_ds, val_ds


def conv_block(x, filters, dropout_rate):
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_model(num_classes):
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.15, value_range=(0, 255)),
        ],
        name="data_augmentation",
    )

    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    x = conv_block(x, 32, 0.10)
    x = conv_block(x, 64, 0.15)
    x = conv_block(x, 128, 0.20)
    x = conv_block(x, 256, 0.25)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.45)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="emotion_cnn_pro")


def plot_training_history(history, output_path):
    history_data = history.history
    epochs = range(1, len(history_data["accuracy"]) + 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history_data["accuracy"], label="Train Accuracy", linewidth=2.5)
    axes[0].plot(epochs, history_data["val_accuracy"], label="Validation Accuracy", linewidth=2.5)
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history_data["loss"], label="Train Loss", linewidth=2.5)
    axes[1].plot(epochs, history_data["val_loss"], label="Validation Loss", linewidth=2.5)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_metadata(
    model_path,
    raw_class_names,
    class_counts,
    history,
    dataset_path,
    metadata_path,
):
    display_labels = [canonicalize_label(label) for label in raw_class_names]
    metadata = {
        "raw_labels": raw_class_names,
        "labels": display_labels,
        "folder_labels": raw_class_names,
        "input_size": list(IMAGE_SIZE),
        "color_mode": "grayscale",
        "confidence_threshold": 0.50,
        "model_path": str(model_path.relative_to(BASE_DIR)),
        "dataset_path": str(dataset_path),
        "class_counts": {canonicalize_label(key): value for key, value in class_counts.items()},
        "best_val_accuracy": round(float(max(history.history["val_accuracy"])), 4),
        "final_train_accuracy": round(float(history.history["accuracy"][-1]), 4),
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main():
    # IMPORTANT: To include Neutral detection, add a "Neutral" folder to your dataset
    # directory containing neutral face images (e.g. from FER2013 or AffectNet).
    # The model will automatically pick it up as a 6th class during training.
    # Recommended: 3000-5000 neutral images to match other class sizes.
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    ensure_runtime_directories()
    tf.keras.utils.set_random_seed(SEED)

    train_ds, val_ds = prepare_datasets(data_dir, args.batch_size, args.validation_split)
    raw_class_names = list(train_ds.class_names)
    display_labels = [canonicalize_label(label) for label in raw_class_names]
    class_counts = count_images_by_class(data_dir, raw_class_names)
    class_weights = compute_class_weights(class_counts, raw_class_names)

    print("Class order used for training:", raw_class_names)
    print("Display labels:", display_labels)
    print("Class counts:", class_counts)

    model = build_model(num_classes=len(raw_class_names))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.10),
        metrics=["accuracy"],
    )

    model_path = MODEL_DIR / "emotion_model.keras"
    h5_path = MODEL_DIR / "emotion_model.h5"
    metadata_path = MODEL_DIR / "emotion_metadata.json"
    plot_path = MODEL_DIR / "training_history.png"
    log_path = MODEL_DIR / "training_log.csv"

    training_callbacks = [
        callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.CSVLogger(str(log_path)),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=training_callbacks,
    )

    model.save(model_path)
    model.save(h5_path)
    plot_training_history(history, plot_path)
    save_metadata(model_path, raw_class_names, class_counts, history, data_dir, metadata_path)

    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Saved TensorFlow model to: {model_path}")
    print(f"Saved H5 model to: {h5_path}")
    print(f"Saved training plot to: {plot_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
