import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from PIL import Image


def preprocess_image(image: Image.Image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def classify_image(model, image: Image.Image):
    try:
        preprocessed_img = preprocess_image(image)
        predictions = model.predict(preprocessed_img)
        return decode_predictions(predictions, top=3)[0]
    except Exception as e:
        st.error(f"Error in classification: {e}")
        return None


@st.cache_resource
def load_model_cached():
    return MobileNetV2(weights="imagenet")


def main():
    st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="centered")

    st.title("Image Classifier using MobileNetV2")
    st.write("Upload an image, and the model will classify the objects in it.")

    model = load_model_cached()

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Classifying image..."):
                predictions = classify_image(model, image)

            if predictions:
                st.subheader("Top Predictions")
                for i, (_, label, prob) in enumerate(predictions, 1):
                    st.write(f"{i}. **{label}** : {prob * 100:.2f}%")

        except Exception as e:
            st.error(f"Error loading image: {e}")


if __name__ == "__main__":
    main()
