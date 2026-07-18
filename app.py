"""
Streamlit Frontend

Author: Matheen Shaik
"""

import requests
import streamlit as st
from PIL import Image

# -----------------------------
# FastAPI Endpoint
# -----------------------------

API_URL = "https://pneumonia-detection-ai-97kl.onrender.com/predict"

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered"
)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("🫁 Pneumonia Detection AI")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### About

This application uses a Deep Learning CNN model deployed using FastAPI.

### Detects

- ✅ NORMAL
- 🦠 PNEUMONIA

### Features

- Chest X-ray Upload
- AI Prediction
- Confidence Score
- Fast API Inference

### Developer

**Matheen Shaik**
""")

# -----------------------------
# Main Page
# -----------------------------

st.title("🫁 Pneumonia Detection AI")

st.write(
    "Upload a Chest X-ray image to detect whether the patient has **Pneumonia** or is **Normal**."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    if st.button("🔍 Predict", use_container_width=True):

        with st.spinner("Analyzing X-ray..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:

                response = requests.post(API_URL, files=files)

                if response.status_code == 200:

                    result = response.json()

                    prediction = result["prediction"]
                    confidence = float(result["confidence"])

                    st.divider()

                    st.subheader("Prediction Result")

                    if prediction.upper() == "PNEUMONIA":

                        st.error(f"🦠 **Prediction:** {prediction}")

                    else:

                        st.success(f"✅ **Prediction:** {prediction}")

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.progress(confidence / 100)

                else:

                    st.error("Prediction failed.")

            except Exception as e:

                st.error(f"Unable to connect to API.\n\n{e}")

st.divider()

st.caption(
    "Developed by Matheen Shaik | TensorFlow • FastAPI • Streamlit"
)