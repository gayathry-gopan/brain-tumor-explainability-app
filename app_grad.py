import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from gradcam import generate_gradcam
from lime_explain import explain_with_lime
from io import BytesIO
from PIL import Image

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Explainability",
    
    layout="wide"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.title-box {
    background: linear-gradient(135deg, #4CAF50, #2E7D32);
    padding: 25px;
    text-align: center;
    color: white;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
}

.card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #2E7D32;
    border-left: 5px solid #4CAF50;
    padding-left: 10px;
    margin-bottom: 15px;
}

.upload-box {
    border: 3px dashed #4CAF50;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------
st.markdown("""
<div class="title-box">
    <h1> Brain Tumor Detection & Explainability Dashboard</h1>
    
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# MODEL
# ----------------------------------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("final_mobilenet_brain_tumor.keras")

model = load_cnn_model()
labels = ["glioma", "meningioma", "notumor", "pituitary"]

# ----------------------------------------------------
# UPLOAD SECTION
# ----------------------------------------------------
st.markdown("<div class='section-title'> Upload MRI Scan</div>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded is not None:

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    # Fix image channels
    if img_bgr is None:
        st.error(" ⚠ Unable to read the uploaded image.")
        st.stop()

    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[-1] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    colA, colB = st.columns([1, 1])

    # ----------------------------------------------------
    # COLUMN A — ORIGINAL IMAGE
    # ----------------------------------------------------
    with colA:
        st.markdown("<div class='section-title'>🖼 Uploaded MRI</div>", unsafe_allow_html=True)
        st.image(img_rgb, width=350)

    orig_h, orig_w = img_rgb.shape[:2]

    # ----------------------------------------------------
    # PREPROCESSING
    # ----------------------------------------------------
    IMG_SIZE = 224
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_index = int(np.argmax(preds[0]))
    confidence = float(preds[0][class_index] * 100)

    # ----------------------------------------------------
    # COLUMN B — PREDICTION CARD
    # ----------------------------------------------------
    with colB:
        st.markdown(f"""
            <div class="card">
                <h3 style="color:#2E7D32;">📌 Prediction</h3>
                <h2 style="margin-top:-5px;">{labels[class_index].upper()}</h2>
                <p style="font-size:20px;"><b>Confidence:</b> {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # EXPLAINABILITY SECTION
    # ----------------------------------------------------
    st.markdown("<div class='section-title'> Explainability Results</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ----------------------------------------------------
    # GRAD-CAM
    # ----------------------------------------------------
    with col1:
        st.markdown("<div class='section-title'> Grad-CAM</div>", unsafe_allow_html=True)

        try:
            heatmap_rgb, _ = generate_gradcam(
                model, img_array, (orig_h, orig_w), "block_16_project"
            )

            overlay = cv2.addWeighted(img_rgb.astype(np.float32), 0.6,
                                      heatmap_rgb.astype(np.float32), 0.4, 0)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            st.image(overlay, caption="Grad-CAM Overlay", width=350)

        except Exception as e:
            st.error(f"Grad-CAM Error: {e}")

    # ----------------------------------------------------
    # LIME
    # ----------------------------------------------------
    with col2:
        st.markdown("<div class='section-title'> LIME</div>", unsafe_allow_html=True)

        try:
            lime_img = explain_with_lime(model, img_rgb)
            st.image(lime_img, caption="LIME Explanation", width=350)
        except Exception as e:
            st.error(f"LIME Error: {e}")
