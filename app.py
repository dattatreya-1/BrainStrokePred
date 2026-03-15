import streamlit as st
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# load models
xgb_model = joblib.load("xgboost_model.pkl")
vgg = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))

labels = {
    0: "Hemorrhagic Stroke",
    1: "Ischemic Stroke"
}

st.title("Brain Stroke Type Prediction")

file = st.file_uploader("Upload Brain Scan Image", type=["jpg","png","jpeg"])

if file is not None:

    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224,224))
    img = np.array(img)

    img = preprocess_input(np.expand_dims(img, axis=0))

    features = vgg.predict(img)
    features = features.reshape(1,-1)

    pred = xgb_model.predict(features)

    st.success(f"Prediction: {labels[pred[0]]}")
