Here's a `README.md` file you can include in your project to describe your Streamlit app:

---

### **README.md**

# Parasitic Infection Detection App

This Streamlit application uses a Convolutional Neural Network (CNN) to detect parasitic infections in microscopy images. The app is designed to classify images into ten categories, including both infected and uninfected classes, making it a valuable tool for diagnostics in clinical and research settings.

---

## **Features**
- Upload microscopy images for automated classification.
- Classifies images into the following categories:
  - Negative (Uninfected)
  - Cryptosporidium cyst
  - Entamoeba histolytica
  - Enterobius vermicularis
  - Giardia cyst
  - Hymenolepis nana
  - Isospora
  - Leishmania
  - Toxoplasma
  - Trichomonad
- Displays confidence scores and probabilities for all classes.
- Interactive visualizations of class probabilities.

---

## **Technologies Used**
- **Frontend**: [Streamlit](https://streamlit.io/) for the user interface.
- **Backend**: TensorFlow-based Convolutional Neural Network for image classification.
- **Python Libraries**:
  - TensorFlow
  - Streamlit
  - Pillow (for image preprocessing)
  - Numpy (for numerical operations)

---

## **Getting Started**

### **1. Prerequisites**
Ensure you have the following installed on your system:
- Python 3.7 or higher
- pip (Python package manager)

### **2. Clone the Repository**
Clone this repository to your local machine:
```bash
git clone https://github.com/mohammed-taie/<repository-name>.git
cd <repository-name>
```

### **3. Install Dependencies**
Create a virtual environment and install the required libraries:
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **4. Run the App Locally**
Start the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## **Deployment**
This app is deployed on **Streamlit Cloud**. Access the live version here:  
[Parasitic Infection Detection App](https://<your-app>.streamlit.app)

---

## **Usage**
1. Upload a microscopy image using the file uploader.
2. View the predicted class and confidence score.
3. Explore the class probabilities through interactive visualizations.

---

## **License**
This project is licensed under the MIT License.

---

"# ParasiteScan" 
