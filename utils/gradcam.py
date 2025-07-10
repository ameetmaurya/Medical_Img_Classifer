import tensorflow as tf
import numpy as np

def get_gradcam_heatmap(model, image_array, class_index):
    """
    Generates a Grad-CAM heatmap for a specified class index.
    
    Automatically finds the last Conv2D layer in the model.
    """
    # Automatically find last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in the model. Grad-CAM requires convolutional layers.")

    # Build a model that maps the input to the activations of the last conv layer and predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    # Compute the gradient of the target class prediction with respect to the output feature map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, class_index]

    # Compute gradients wrt last conv layer output
    grads = tape.gradient(loss, conv_outputs)[0]  # shape: (H, W, channels)

    # Compute the channel-wise mean of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0,1))

    # Weight the conv feature maps by the mean gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap /= max_val

    return heatmap.numpy()


# 🔵 Blue areas → The model didn’t care much about these regions when deciding.

# 🟡 Yellow/Green areas → Moderately contributed.

# 🔴 Red areas → The model heavily relied on these areas to make its prediction (e.g. NORMAL).