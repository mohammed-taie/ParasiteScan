import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, UnidentifiedImageError
import pandas as pd
import time
from fpdf import FPDF
from datetime import datetime

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set custom page configuration
st.set_page_config(
    page_title="Parasitic Infection Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI enhancements
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
    }
    .stApp header h1 {
        color: #2c3e50;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 24px;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #2c3e50;
        color: #ecf0f1;
    }
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

img_height, img_width = 128, 128

class_names = [
    "Negative", "Cryptosporidium cyst", "Entamoeba histolytica", "Enterobius vermicularis", "Giardia cyst",
    "Hymenolepis nana", "Isospora", "Leishmania", "Toxoplasma", "Trichomonad"
]

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("saved_model/cnn_with_attention_modules.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

def predict_images(images):
    img_arrays = [img_to_array(img.resize((img_height, img_width))) / 255.0 for img in images]
    return model.predict(np.array(img_arrays))

def generate_pdf(summary_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Parasitic Infection Detection Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, txt="Prediction Summary", ln=True)
    pdf.set_font("Arial", size=12)

    for _, row in summary_df.iterrows():
        pdf.cell(0, 10, txt=f"{row['Filename']}: {row['Predicted Class']} (Confidence: {row['Confidence']:.2f})", ln=True)
    
    output_path = "prediction_summary.pdf"
    pdf.output(output_path)
    return output_path

st.title("🔬 Parasitic Infection Detection System")
st.write("Upload one or more microscopy images to detect parasitic infections using a deep learning model.")

uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    images, filenames = [], []
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            images.append(image)
            filenames.append(uploaded_file.name)
        except UnidentifiedImageError:
            st.error(f"Unable to identify image file {uploaded_file.name}. Please upload a valid image.")
    
    if images:
        st.write("### Uploaded Images")

        # Toggle for captions
        show_captions = st.checkbox("Show image captions", value=True)

        # Adjust columns dynamically
        num_cols = min(len(images), 4)
        cols = st.columns(num_cols)

        for idx, image in enumerate(images):
            with cols[idx % num_cols]:
                st.image(image, caption=filenames[idx] if show_captions else "", use_container_width=True)
        
        st.write("### Classification Results")
        start_time = time.time()
        with st.spinner("Processing images..."):
            predictions = predict_images(images)
        end_time = time.time()
        
        results = [{
            "Filename": filename,
            "Predicted Class": class_names[np.argmax(pred)] if np.max(pred) >= 0.5 else "Uncertain",
            "Confidence": np.max(pred),
        } for pred, filename in zip(predictions, filenames)]
        
        summary_df = pd.DataFrame(results)
        st.dataframe(summary_df)

        st.write("### Probability Bar Charts")
        chart_cols = st.columns(2)
        for i, (pred, filename) in enumerate(zip(predictions, filenames)):
            with chart_cols[i % 2]:
                st.write(f"**Results for {filename}**")
                st.bar_chart(pd.DataFrame({"Probability": pred}, index=class_names))

        pdf_path = generate_pdf(summary_df)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                "📄 **Download Prediction Summary as PDF**",
                data=pdf_file,
                file_name="prediction_summary.pdf",
                mime="application/pdf"
            )

        st.success(f"✅ Successfully processed {len(images)} images in {end_time - start_time:.2f} seconds!")

# Add a sidebar for help and model information
st.sidebar.title("About")
st.sidebar.write("This application detects parasitic infections from microscopy images using a deep learning model.")
st.sidebar.write("Ensure your images are in JPG or PNG format.")
st.sidebar.write("For more details, contact the administrator.")
st.sidebar.title("Model Info")
st.sidebar.write("Model Version: 1.1.0")
