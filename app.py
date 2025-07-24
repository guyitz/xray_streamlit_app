import streamlit as st
from ui.single_image_page import render_single_image_page
from ui.batch_page import render_batch_page
from ui.about_page import render_about_page

st.set_page_config(
    page_title="X-ray Image Classification",
    layout="wide"
)

# Tabs
tab = st.sidebar.radio("Select Mode", ["🖼 Single Image", "ℹ️ About"])

if tab == "🖼 Single Image":
    render_single_image_page()

elif tab == "📁 Batch Mode":
    render_batch_page()

elif tab == "ℹ️ About":
    render_about_page()

