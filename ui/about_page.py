import streamlit as st

def render_about_page():
    st.title("📄 About This Project")

    st.markdown("""
    This application is an **X-ray Image Classification Demo** built using Streamlit.
    
    It allows you to:
    - Upload a single chest X-ray image and get predictions from multiple trained models.
    - Run batch predictions on a folder of images.
    - Visualize prediction confidence per model using horizontal bar charts.
    - Export batch results as a CSV file.

    ### Models
    Multiple deep learning models (EfficientNet, VGG16, etc.) have been trained on X-ray data to classify:
    - **COVID-19**
    - **Pneumonia**
    - **Normal**

    Each model may use different architectures, input sizes, and label mappings, all configured in a single YAML file.

    ### Team & Credits
    - Developed by: _Your Name or Team_
    - Models trained on: _Eyal's Dataset, Kaggle Dataset_
    - Powered by: PyTorch, Streamlit, and EfficientNet/VGG

    ### Notes
    - All inference runs on **CPU**.
    - Color scheme:
        - 🟥 **COVID-19** = Red
        - 🟩 **Normal** = Green
                
        - 🟦 **Pneumonia** = Blue

    ---
    _You can update this page with more information as your project evolves._
    """)
