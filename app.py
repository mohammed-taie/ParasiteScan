import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import pandas as pd
import time
from fpdf import FPDF

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses INFO and WARNING messages

# Set image dimensions (must match model input)
img_height = 128
img_width = 128

# Class names (from the training data)
class_names = [
    "Negative",
    "Cryptosporidium cyst",
    "Entamoeba histolytica",
    "Enterobius vermicularis",
    "Giardia cyst",
    "Hymenolepis nana",
    "Isospora",
    "Leishmania",
    "Toxoplasma",
    "Trichomonad",
]

# Load the trained model
@st.cache_resource  # Caches the model for faster reload
def load_model():
    model_path = "saved_model/multiclass_model.h5"  # Path to your trained model
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# Define a function to preprocess and predict images in batch
def predict_images(images):
    try:
        img_arrays = np.array([img_to_array(img.resize((img_height, img_width))) / 255.0 for img in images])
        preds = model.predict(img_arrays)
        return preds
    except Exception as e:
        raise RuntimeError(f"Error during batch prediction: {e}")

# Generate a PDF summary
def generate_pdf(summary_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Prediction Summary", ln=True, align="C")
    pdf.ln(10)

    for index, row in summary_df.iterrows():
        pdf.cell(0, 10, txt=f"{row['Filename']}: {row['Predicted Class']} (Confidence: {row['Confidence']:.2f})", ln=True)

    output_path = "prediction_summary.pdf"
    pdf.output(output_path)
    return output_path

# Streamlit UI
st.title("Parasitic Infection Detection")
st.write("Upload one or more microscopy images to detect parasitic infections.")

# Confidence threshold
CONFIDENCE_THRESHOLDS = {class_name: 0.5 for class_name in class_names}  # Default thresholds

# File uploader (accepts multiple files)
uploaded_files = st.file_uploader(
    "Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    images = []
    filenames = []
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            if image.mode != "RGB":
                image = image.convert("RGB")  # Ensure the image is in RGB format
            images.append(image)
            filenames.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Error processing image {uploaded_file.name}: {e}")

    if images:
        # Display images in a grid layout
        st.write("### Uploaded Images")
        cols = st.columns(4)  # Adjust the number of columns as needed
        for idx, image in enumerate(images):
            with cols[idx % 4]:
                st.image(image, caption=filenames[idx], use_container_width=True)

        st.write("Classifying...")
        try:
            start_time = time.time()
            predictions = predict_images(images)
            end_time = time.time()

            results = []
            for idx, (pred, filename) in enumerate(zip(predictions, filenames)):
                predicted_class = np.argmax(pred)
                confidence = np.max(pred)

                # Apply confidence threshold
                if confidence < CONFIDENCE_THRESHOLDS[class_names[predicted_class]]:
                    predicted_label = "Uncertain"
                    st.warning(f"For {filename}, the model is uncertain: confidence for {class_names[predicted_class]} is {confidence:.2f}.")
                else:
                    predicted_label = class_names[predicted_class]

                # Append results for summary
                results.append({
                    "Filename": filename,
                    "Predicted Class": predicted_label,
                    "Confidence": confidence,
                })

                # Display results
                st.write(f"### Results for {filename}")
                st.write(f"**Predicted Class:** {predicted_label}")
                st.write(f"**Confidence:** {confidence:.2f}")

                # Show probabilities for all classes as a bar chart
                st.bar_chart(pd.DataFrame({"Probability": pred}, index=class_names))

                # Optionally display probabilities as a table
                st.write("Probability Details:")
                st.table(
                    pd.DataFrame.from_dict(
                        {class_names[i]: prob for i, prob in enumerate(pred)},
                        orient="index",
                        columns=["Probability"]
                    )
                )

            # Create a summary DataFrame
            summary_df = pd.DataFrame(results)
            st.write("### Prediction Summary")
            st.table(summary_df)

            # Generate and allow download of PDF summary
            pdf_path = generate_pdf(summary_df)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download Prediction Summary as PDF",
                    data=pdf_file,
                    file_name="prediction_summary.pdf",
                    mime="application/pdf",
                )

            st.success(f"Processed {len(images)} images in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            st.error(f"Error during classification: {e}")

# Add a sidebar for help and model information
st.sidebar.title("About")
st.sidebar.write("This application detects parasitic infections from microscopy images using a deep learning model.")
st.sidebar.write("Ensure your images are in JPG or PNG format.")
st.sidebar.write("For more details, contact the administrator.")
st.sidebar.title("Model Info")
st.sidebar.write("Model Version: 1.0.0")
