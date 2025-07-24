import streamlit as st
from PIL import Image
import os
import pandas as pd
import torch
import zipfile # Import for handling zip files
import tempfile # Import for creating temporary directories
import shutil # Import for removing directories

from inference.loader import load_model_from_config
from inference.predictor import predict_image
from utils.config import CLASS_COLOR_MAP # Assuming this is defined

@st.cache_resource
def get_models():
    # It's good practice to use GPU if available for performance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Loading models to device: {device}") # Optional: show device info
    return load_model_from_config("models/model_config.yaml", device), device

def render_batch_page():
    st.title("📁 Batch Prediction")
    st.write("Upload a ZIP file containing X-ray images to run batch predictions.")

    # File uploader for a zip file
    uploaded_zip_file = st.file_uploader("Upload ZIP archive of images", type=["zip"])

    if uploaded_zip_file:
        # Create a temporary directory to extract images
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract the zip file contents
                with zipfile.ZipFile(uploaded_zip_file, 'r') as zip_ref:
                    st.info(f"Extracting {uploaded_zip_file.name} to a temporary folder...")
                    zip_ref.extractall(temp_dir)
                    st.success("Extraction complete!")

                # Get list of image files from the extracted directory
                # Walk through the temp_dir to find all images, even in subfolders
                image_files = []
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                            image_files.append(os.path.join(root, file))

                if not image_files:
                    st.warning("No image files found in the uploaded ZIP archive.")
                    return

                models_info, device = get_models()

                # Run predictions and collect results
                results = []
                progress_text = st.empty() # Placeholder for progress text
                progress_bar = st.progress(0)
                total_images = len(image_files)

                for idx, image_path in enumerate(image_files):
                    image_name = os.path.basename(image_path) # Get just the filename
                    progress_text.text(f"Processing image {idx + 1}/{total_images}: {image_name}")
                    
                    try:
                        image = Image.open(image_path).convert("RGB")
                    except Exception as e:
                        st.warning(f"Failed to open image {image_name}: {e}")
                        continue

                    row = {"image_name": image_name}
                    for model_info in models_info:
                        pred_class, conf, _ = predict_image(image, model_info, device)
                        row[f"{model_info['name']}_class"] = pred_class
                        row[f"{model_info['name']}_confidence"] = f"{conf:.4f}" # Format confidence
                    results.append(row)
                    progress_bar.progress((idx + 1) / total_images)
                
                progress_text.text("Batch prediction complete!")
                st.success(f"Successfully processed {total_images} images.")

                df_results = pd.DataFrame(results)

                st.markdown("### Prediction Results")
                st.dataframe(df_results)

                # CSV export button
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )

            except zipfile.BadZipFile:
                st.error("The uploaded file is not a valid ZIP archive. Please upload a `.zip` file.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

        # The temporary directory and its contents are automatically cleaned up when exiting `with tempfile.TemporaryDirectory()`.
        # No need for explicit shutil.rmtree(temp_dir) here.

    else:
        st.info("Please upload a ZIP file containing your X-ray images.")