from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference import load_predictor
from src.visualize import make_binary_mask, make_heatmap, overlay_heatmap


st.set_page_config(page_title="DefectVision-AD", layout="wide")
st.title("DefectVision-AD")

checkpoint_path = st.sidebar.text_input("Checkpoint", "outputs/checkpoints/autoencoder.pt")
device = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"])
image_size = st.sidebar.number_input("Image size", min_value=64, max_value=1024, value=256, step=32)
uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded and checkpoint_path:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        st.error(f"Checkpoint not found: {checkpoint}")
    else:
        image = Image.open(uploaded).convert("RGB")
        predictor = load_predictor(checkpoint, device=device, image_size=int(image_size))
        tensor = predictor.preprocessor(image).unsqueeze(0)
        prediction = predictor.predict_batch(tensor)
        score = float(prediction["score"][0])
        anomaly_map = prediction["anomaly_map"][0]
        label = "anomaly" if score >= predictor.threshold else "normal"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prediction", label)
        col2.metric("Score", f"{score:.6f}")
        col3.metric("Threshold", f"{predictor.threshold:.6f}")
        col4.metric("Model", predictor.checkpoint.get("model_type", "unknown"))

        heatmap = make_heatmap(anomaly_map)
        overlay = overlay_heatmap(image, anomaly_map)
        mask = make_binary_mask(anomaly_map, predictor.threshold)

        image_col, heatmap_col, overlay_col, mask_col = st.columns(4)
        image_col.image(image, caption="Image", use_container_width=True)
        heatmap_col.image(heatmap, caption="Heatmap", use_container_width=True)
        overlay_col.image(overlay, caption="Overlay", use_container_width=True)
        mask_col.image(mask, caption="Mask", use_container_width=True)
