import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

from xai import load_trained_model, GradCAM, generate_plain_language_report

st.set_page_config(page_title="Accessible Glioma Detection", layout="centered")

st.title("🧠 Accessible Glioma Detection System")
st.markdown("Upload a brain MRI scan to generate a visual anomaly map and plain-language summary.")

@st.cache_resource
def get_model():
    return load_trained_model()

model = get_model()
target_layer = model.features[-1]
cam_engine = GradCAM(model, target_layer)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an MRI Scan image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Scan")
        st.image(raw_image, use_container_width=True)

    with st.spinner("Analyzing scan and computing feature heatmaps..."):
        tensor_img = preprocess(raw_image).unsqueeze(0)
        heatmap, pred_class, probs = cam_engine.generate_heatmap(tensor_img)
        confidence = probs[pred_class]

        img_np = np.array(raw_image.resize((224, 224)))
        heatmap_resized = cv2.resize(heatmap, (224, 224))

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

    with col2:
        st.subheader("XAI Heatmap Overlay")
        st.image(overlay, use_container_width=True)

    st.divider()

    report_text = generate_plain_language_report(pred_class, confidence, heatmap)
    
    if pred_class == 0:  # Class 0 is Glioma
        st.error("⚠️ Glioma Features Detected")
    else:
        st.success("✅ No Significant Tumor Indicators Detected")

    st.text_area("Plain-Language Summary Report", report_text, height=220)