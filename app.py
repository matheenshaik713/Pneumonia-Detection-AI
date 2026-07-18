"""
Streamlit Frontend

Author: Matheen Shaik
"""

import tempfile

import streamlit as st
from PIL import Image

from src.predict import Predictor


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("🫁 Pneumonia Detection AI")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### About



- ✅ NORMAL
- 🦠 PNEUMONIA

### Features

- Image Upload
- AI Prediction
- Confidence Score
- Fast Inference

### Developer

**Matheen Shaik**
""")

# ---------------------------------------------------------
# Main Title
# ---------------------------------------------------------

st.title("🫁 Pneumonia Detection AI")

st.markdown("""
### AI-powered Chest X-ray Classification

Upload a Chest X-ray image and the trained CNN model will predict whether the patient has **Pneumonia** or is **Normal**.
""")

st.markdown("---")

# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    st.markdown("")

    if st.button("🔍 Predict", use_container_width=True):

        with st.spinner("Analyzing Chest X-ray..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as tmp:

                tmp.write(uploaded_file.getbuffer())

                image_path = tmp.name

            predictor = Predictor()

            prediction, confidence = predictor.predict(image_path)

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == "PNEUMONIA":

            st.error(f"🦠 **Prediction:** {prediction}")

        else:

            st.success(f"✅ **Prediction:** {prediction}")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

# ---------------------------------------------------------
# Model Information
# ---------------------------------------------------------

st.markdown("---")



# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Developed by Matheen Shaik | "
    "Deep Learning • TensorFlow • Streamlit • FastAPI"
)