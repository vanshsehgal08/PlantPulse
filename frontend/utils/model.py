from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def get_paths() -> Tuple[Path, Path]:
    # Prefer models and data at the project root, fall back to frontend-local copies
    frontend_root = Path(__file__).resolve().parents[1]
    project_root = frontend_root.parent

    # Model path preference: ALWAYS use best_model.keras first
    candidate_model_paths = [
        project_root / "models" / "best_model.keras",
        frontend_root / "models" / "best_model.keras",

    ]
    # Find first existing model, prioritizing best_model.keras
    model_path = None
    for path in candidate_model_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        model_path = candidate_model_paths[0]  # Will raise error in load_tf_model

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
    
    # Memory optimization for production deployment
    # Limit TensorFlow memory growth to prevent OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setup error: {e}")
    
    # Limit CPU memory usage (important for Render free tier)
    try:
        tf.config.set_soft_device_placement(True)
        # Set memory limit to prevent crashes (adjust based on available RAM)
        # For 512MB instances, use smaller limit
        cpu_devices = tf.config.list_physical_devices('CPU')
        if cpu_devices:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    cpu_devices[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=400)]
                )
            except Exception as mem_error:
                # Memory limit setting may not be supported, continue anyway
                print(f"Memory limit configuration not available: {mem_error}")
    except Exception as e:
        # If memory configuration fails, continue anyway
        print(f"Memory configuration warning: {e}")

    model_path, _ = get_paths()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train and save a model to this path, "
            f"or place a Keras model named 'best_model.keras' in either 'models/' at the project root "
            f"or 'frontend/models/'."
        )
    
    # Load model with optimizations
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        # Compile with optimizations
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        # Fallback to basic loading
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


