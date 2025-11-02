from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_gradcam(model, image_array: np.ndarray, class_index: int) -> np.ndarray:
    """
    Compute a simple Grad-CAM heatmap for a single image (H,W,3) scaled 0..1.
    Returns a (H,W) heatmap normalized to 0..1.
    """
    import tensorflow as tf

    # Find the last conv layer
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
            break
    if last_conv is None:
        raise ValueError("No Conv2D layer found for Grad-CAM.")

    # Some Sequential models loaded from disk may not have defined inputs/outputs
    # until they are called once. Try to use existing tensors; if that fails,
    # reconstruct a forward graph with a fresh Input tensor.
    try:
        conv_tensor = last_conv.output
        out_tensor = model.output
        in_tensors = model.inputs
        # Accessing model.output may raise if model hasn't been built yet.
        _ = out_tensor  # touch to trigger potential errors early
    except Exception:
        # Rebuild the computation graph with a fresh Input tensor
        input_shape = getattr(model, "input_shape", None)
        if input_shape is None or isinstance(input_shape, list):
            # Fallback: infer from common image size
            input_shape = (128, 128, 3)
        else:
            input_shape = input_shape[1:]

        x_in = tf.keras.Input(shape=input_shape)
        x = x_in
        conv_tensor = None
        for layer in model.layers:
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_tensor = x
        out_tensor = x
        in_tensors = [x_in]

        if conv_tensor is None:
            raise ValueError("No Conv2D layer found while rebuilding model for Grad-CAM.")

    grad_model = tf.keras.models.Model(in_tensors, [conv_tensor, out_tensor])

    with tf.GradientTape() as tape:
        inputs = tf.cast(tf.expand_dims(image_array, axis=0), tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (image_array.shape[0], image_array.shape[1]))
    return heatmap.numpy().squeeze()


