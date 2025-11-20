import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries


def explain_with_lime(model, img_rgb):
    """
    Takes the original RGB image (H,W,3) and returns a LIME visualization.
    """

    img_float = img_rgb.astype("float32") / 255.0

    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):
        images = np.array(images)
        return model.predict(images)

    explanation = explainer.explain_instance(
        img_float,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    top_label = explanation.top_labels[0]

    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=False,
        hide_rest=False,
        num_features=10,
        min_weight=0.0
    )

    lime_image_rgb = mark_boundaries(temp, mask)
    lime_image_rgb = (lime_image_rgb * 255).astype(np.uint8)

    return lime_image_rgb
