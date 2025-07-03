import streamlit as st 
import cv2
import joblib
from PIL import Image
import numpy as np
import pandas as pd

# Load model
model = joblib.load("trained_model_KNN.pkl")

# --- UI Components ---
st.snow()
st.title("Say Hello!")
st.camera_input("label")
st.divider()
st.title("Dell Global Business Sdn Bhd")
st.image("factory.jpg")
st.divider()
st.date_input("Transaction Date")
st.divider()
st.radio("Your department:", ["GPP", "Operations", "Quality", "GOQ", "Logistics", "HR"])
st.divider()
st.slider("Slide me")
st.divider()
st.button("Click me")
st.divider()

# --- CSV Upload ---
uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
    st.write("### CSV File Content")
    st.dataframe(df)

# --- Image Preprocessing ---
def preprocess_image(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def extract_features(img):
    resized = cv2.resize(img, (64, 64))
    return resized.flatten().reshape(1, -1)

# --- Image Upload & Prediction ---
uploaded_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_img is not None:
    image = Image.open(uploaded_img).convert("L")
    img_array = np.array(image)
    
    st.image(image, caption="You have successfully uploaded this image")

    processed_image = preprocess_image(img_array)
    features = extract_features(processed_image)

    prediction = model.predict(features)[0]
    label = "Positive" if prediction == 1 else "Negative"
    st.success(f"**Prediction:** {label}")
