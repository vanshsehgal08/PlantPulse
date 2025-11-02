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
    # Fallback: Comprehensive list of all 38 PlantVillage classes
    # This ensures the model can use all its outputs even if train_dir doesn't exist
    return [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ]


def load_tf_model():
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setup error: {e}")
    
    try:
        tf.config.set_soft_device_placement(True)
        cpu_devices = tf.config.list_physical_devices('CPU')
        if cpu_devices:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    cpu_devices[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=400)]
                )
            except Exception as mem_error:
                print(f"Memory limit configuration not available: {mem_error}")
    except Exception as e:
        print(f"Memory configuration warning: {e}")

    model_path, _ = get_paths()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train and save a model to this path, "
            f"or place a Keras model named 'best_model.keras' in either 'models/' at the project root "
            f"or 'frontend/models/'."
        )
    
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
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


