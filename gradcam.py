# gradcam.py – FINAL WORKING VERSION FOR YOUR MODEL
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model


def safe_pred_vector(raw_preds):
    """Converts ANY prediction output to a clean 1D prediction vector."""
    raw = tf.convert_to_tensor(raw_preds)
    raw = tf.squeeze(raw)  # remove extra dims
    if len(raw.shape) > 1:
        raw = tf.reshape(raw, [-1])  # flatten
    return raw


def generate_gradcam(model, img_array, original_size, layer_name="Conv_1"):
    """Generates Grad-CAM heatmap for your MobileNetV2 model."""

    orig_h, orig_w = int(original_size[0]), int(original_size[1])

    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)

        conv_outputs, preds = grad_model(img_tensor)

        pred_vec = safe_pred_vector(preds)
        class_index = int(tf.argmax(pred_vec))

        loss = pred_vec[class_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise RuntimeError("Gradients are None! The target conv layer may not be connected.")

    conv = conv_outputs[0].numpy()
    grads = grads[0].numpy()

    weights = np.mean(grads, axis=(0, 1))

    cam = np.zeros(conv.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv[:, :, i]

    cam = np.maximum(cam, 0)

    if cam.max() != 0:
        cam /= cam.max()

    cam_resized = cv2.resize(cam, (orig_w, orig_h))

    heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap.astype(np.uint8), cam_resized
