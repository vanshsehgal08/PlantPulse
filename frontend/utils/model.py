from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def get_paths() -> Tuple[Path, Path]:
    # Prefer models and data at the project root, fall back to frontend-local copies
    frontend_root = Path(__file__).resolve().parents[1]
    project_root = frontend_root.parent

    # Model path preference: project_root/models → frontend/models
    # Try multiple model file names
    candidate_model_paths = [
        project_root / "models" / "best_model.keras",
        frontend_root / "models" / "best_model.keras",
        project_root / "models" / "my_model_24.keras",
        frontend_root / "models" / "my_model_24.keras",
    ]
    model_path = next((p for p in candidate_model_paths if p.exists()), candidate_model_paths[0])

    # Data path preference: project_root/data/train → frontend/data/train
    candidate_train_dirs = [
        project_root / "data" / "train",
        frontend_root / "data" / "train",
    ]
    data_train = next((p for p in candidate_train_dirs if p.exists()), candidate_train_dirs[0])

    return model_path, data_train


def list_class_names(train_dir: Path) -> List[str]:
    if train_dir.exists():
        names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
        if names:
            return names
    return [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
    ]


def load_tf_model():
    import tensorflow as tf

    model_path, _ = get_paths()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train and save a model to this path, "
            f"or place a Keras model named 'best_model.keras' in either 'models/' at the project root "
            f"or 'frontend/models/'."
        )
    model = tf.keras.models.load_model(str(model_path))
    return model


def prepare_image_batch(pil_images, target_size=(128, 128)):
    import numpy as np
    arrs = []
    for img in pil_images:
        img = img.convert("RGB").resize(target_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arrs.append(arr)
    batch = np.stack(arrs, axis=0)
    return batch


