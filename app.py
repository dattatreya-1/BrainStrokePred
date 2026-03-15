import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image

st.title("Brain Stroke Prediction")

model = joblib.load("xgboost_model.pkl")

file = st.file_uploader("Upload MRI", type=["jpg","png","jpeg"])

if file is not None:

    img = Image.open(file)
    st.image(img)

    img = np.array(img)
    img = cv2.resize(img, (64,64))
    img = img.flatten().reshape(1,-1)

    pred = model.predict(img)

    if st.button("Predict"):
        if pred[0]==0:
            st.error("Hemorrhagic Stroke")
        else:
            st.success("Ischemic Stroke")
