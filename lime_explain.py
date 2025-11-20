import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2

def explain_with_lime(model, img_rgb):

    # Resize image smaller ONLY for LIME (reduces processing cost)
    img_small = cv2.resize(img_rgb, (128, 128))

    img_float = img_small.astype("float32") / 255.0
    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        images = np.array(images)
        return model.predict(images, verbose=0)

    explanation = explainer.explain_instance(
        img_float,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=120,   # From 1000 → 120 (Cloud safe)
    )

    top_label = explanation.top_labels[0]

    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        hide_rest=False,
        num_features=5,  # From 10 → 5
        min_weight=0.0
    )

    lime_image_rgb = mark_boundaries(temp, mask)
    lime_image_rgb = (lime_image_rgb * 255).astype(np.uint8)

    return lime_image_rgb
