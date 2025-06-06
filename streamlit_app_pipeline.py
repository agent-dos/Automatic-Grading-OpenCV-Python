import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from enhance_image import image_enhancer
from grade_paper import ProcessPage
from transform_image import transform_paper_image

# App title and layout
st.set_page_config(layout="wide")
st.title("📄 Static Marker-Based OMR Pipeline")

st.markdown("""
**Pipeline:**  
`Image Enhancement → Static Marker Detection → Perspective Warp → Grading`
""")

# Sidebar controls
st.sidebar.title("🔧 Image Processing Parameters")
with st.sidebar.expander("Gaussian Blur"):
    blur_ksize = st.slider("Kernel Size", 1, 15, 5, step=2)
with st.sidebar.expander("Adaptive Thresholding"):
    block_size = st.slider("Block Size", 3, 100, 11, step=2)
    C = st.slider("C", -10, 10, 2)
with st.sidebar.expander("Morphological Filtering"):
    morph_kernel_size = st.slider("Kernel Size", 1, 10, 2)

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload Answer Sheet", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and decode image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original = img.copy()

    # Step 1: Enhance image
    enhanced = image_enhancer(
        img, blur_ksize, block_size, C, morph_kernel_size)

    # Step 2: Warp using static marker detection
    detection_img, warped_paper, _, method_used, marker_points = transform_paper_image(
        enhanced.copy())

    # Visual feedback for markers
    overlay_img = detection_img.copy()
    if method_used == "static_marker":
        for (x, y) in marker_points:
            cv2.circle(overlay_img, (int(x), int(y)), 10, (0, 0, 255), -1)
        st.sidebar.success("✅ Static Marker-Based Warp Applied")
    else:
        st.sidebar.error("🛑 Marker detection failed. Blank fallback applied.")

    # Step 3: Process for answers and QR
    extracted_answers, codes = [-1], [-1]
    graded_image = None
    expected_shape = (1202, 850, 3)  # static 103 DPI A4

    if (
        warped_paper is not None and
        isinstance(warped_paper, np.ndarray) and
        warped_paper.size != 0 and
        warped_paper.shape == expected_shape
    ):
        extracted_answers, graded_image, codes = ProcessPage(
            warped_paper.copy())
    else:
        st.warning("🛑 Invalid warped paper. Skipping grading.")

    # Display pipeline
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(original, caption="🖼 Original", use_container_width=True)
    with col2:
        st.image(enhanced, caption="✨ Enhanced", use_container_width=True)
    with col3:
        st.image(overlay_img, caption="📍 Marker Overlay",
                 use_container_width=True)
    with col4:
        if graded_image is not None:
            st.image(graded_image, caption="📊 Graded Sheet",
                     use_container_width=True)
        else:
            st.warning("Grading skipped due to failure.")

    # Results
    st.markdown("---")
    st.markdown("### ✅ Extracted Information")
    if codes != [-1]:
        st.markdown(f"**QR Code:** `{codes[0]}`")
    else:
        st.markdown("**QR Code not detected.**")

    if extracted_answers != [-1]:
        st.markdown("**Answers:**")
        for i, ans in enumerate(extracted_answers):
            st.write(f"Q{i+1}: {'❓' if ans == '?' else ans}")
    else:
        st.markdown("No answers extracted.")

    # Download result
    if graded_image is not None:
        result_pil = Image.fromarray(
            cv2.cvtColor(graded_image, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button("📥 Download Graded Sheet", data=buf.getvalue(),
                           file_name="graded_sheet.png", mime="image/png")
