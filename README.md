A Streamlit-based web application for automated brain tumor classification and visual explainability using MobileNetV2, Grad-CAM, and LIME.
Upload an MRI image and instantly get the tumor prediction along with visual explanations showing why the model predicted it.

 ## Features
 MRI Image Upload + Preprocessing

Accepts .jpg, .jpeg, .png

Automatically resizes and normalizes MRI scans

## Brain Tumor Classification (MobileNetV2)

Predicts four classes:

* Glioma

* Meningioma

* Pituitary

No Tumor

### Returns:

Predicted class

Confidence score

 Explainability Tools

### Built-in XAI methods:

* Grad-CAM

Highlights important regions influencing the prediction.

* LIME

Local Interpretable Model-Agnostic Explanations for pixel-level interpretability.


