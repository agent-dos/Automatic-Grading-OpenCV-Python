# streamlit_qr_detector.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pyzbar import pyzbar

st.set_page_config(layout="centered")
st.title("🔍 QR Code Detector")

uploaded_file = st.file_uploader(
    "📤 Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.image(img, caption="Original Image",
             channels="BGR", use_container_width=True)

    # Detect QR codes
    decoded_objects = pyzbar.decode(gray)
    qr_info = []

    for obj in decoded_objects:
        # Extract bounding box and data
        points = obj.polygon
        qr_text = obj.data.decode('utf-8')
        qr_info.append(qr_text)

        # Draw bounding box
        pts = np.array([[p.x, p.y] for p in points], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=2)
        x, y = pts[0]
        cv2.putText(img, qr_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)

    if qr_info:
        st.success("✅ QR Code(s) Detected:")
        for i, code in enumerate(qr_info):
            st.markdown(f"**QR {i+1}:** `{code}`")
    else:
        st.warning("❌ No QR Code Detected.")

    st.image(img, caption="🟩 Annotated Image",
             channels="BGR", use_container_width=True)
