# streamlit_app_ga_tuning.py

import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from ga_tuner import run_genetic_algorithm

st.set_page_config(layout="wide")
st.title("🧬 Genetic Algorithm OMR Tuning")

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload Answer Sheet", type=["jpg", "jpeg", "png"])
show_gen = st.sidebar.checkbox("🔁 Show GA Progress", value=True)


def show_progress(msg):
    if show_gen:
        st.sidebar.text(msg)


if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original = img.copy()

    st.image(original, caption="📷 Original Image", use_container_width=True)

    # Run GA optimization
    best_result = run_genetic_algorithm(original, show_progress=show_progress)

    st.markdown("---")
    st.subheader("🏆 Best Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(best_result["enhanced"],
                 caption="✨ Enhanced", use_container_width=True)
    with col2:
        st.image(best_result["detection"],
                 caption="📍 Marker Overlay", use_container_width=True)
    with col3:
        st.image(best_result["graded"],
                 caption="📊 Graded Sheet", use_container_width=True)

    st.markdown(f"**Best Parameters:** `{best_result['params']}`")
    st.markdown(f"**Valid Answers Detected:** `{best_result['score']}`")

    if best_result["graded"] is not None:
        buf = BytesIO()
        Image.fromarray(cv2.cvtColor(best_result["graded"], cv2.COLOR_BGR2RGB)).save(
            buf, format="PNG")
        st.download_button("📥 Download Graded Sheet", data=buf.getvalue(
        ), file_name="graded_sheet.png", mime="image/png")
