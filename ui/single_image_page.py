import streamlit as st
from PIL import Image
import torch

from inference.loader import load_model_from_config
from inference.predictor import predict_image
from utils.config import CLASS_COLOR_MAP
from ui.components import show_model_result

@st.cache_resource
def get_models():
    device = torch.device("cpu")
    return load_model_from_config("models/model_config.yaml", device), device

def render_single_image_page():
    st.title("🖼 Single Image Prediction")
    st.write("Upload an X-ray image to see predictions from all trained models.")

    uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        models_info, device = get_models()

        st.markdown("---")
        st.subheader("🔎 Model Predictions")

        for model_info in models_info:
            pred_class, conf, prob_dict = predict_image(image, model_info, device)
            show_model_result(model_info['name'], pred_class, conf, prob_dict, CLASS_COLOR_MAP)

