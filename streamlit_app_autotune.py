import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from itertools import product

from enhance_image import image_enhancer, auto_enhance_image
from transform_image import transform_paper_image
from grade_paper import ProcessPage

# ------------------- Constants -------------------
C_RANGE = list(range(2, 11))  # Default C tuning range
EXPECTED_WIDTH = 850
EXPECTED_HEIGHT = 1202

# ------------------- Helper Functions -------------------


def range_slider_param(name, min_val, max_val, default, step, delta=2):
    center = st.sidebar.slider(name, min_val, max_val, default, step=step)
    lower = max(min_val, center - delta * step)
    upper = min(max_val, center + delta * step)
    values = list(range(lower, upper + 1, step))
    return center, values


def auto_enhance_and_process(img, param_grid, blur_ksize, block_size, morph_kernel_size):
    best_result = None
    best_score = -1

    for C in param_grid["C"]:
        try:
            enhanced = image_enhancer(
                img.copy(), blur_ksize, block_size, C, morph_kernel_size)
            detection_img, warped_paper, _, method_used, marker_points = transform_paper_image(
                enhanced)

            if (
                method_used != "fallback"
                and warped_paper is not None
                and warped_paper.shape == (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3)
            ):
                answers, graded_img, codes = ProcessPage(warped_paper.copy())
                num_valid_answers = sum([1 for a in answers if a != '?'])

                # Alignment check
                left_col_x = 126
                right_col_x = 618
                pixel_dist = right_col_x - left_col_x
                normalized_dist = pixel_dist / EXPECTED_WIDTH
                alignment_score = abs(normalized_dist - 0.578) < 0.01

                if num_valid_answers > best_score and alignment_score:
                    best_score = num_valid_answers
                    best_result = {
                        "answers": answers,
                        "codes": codes,
                        "graded_img": graded_img,
                        "enhanced": enhanced,
                        "detection_img": detection_img,
                        "params": {
                            "blur_ksize": blur_ksize,
                            "block_size": block_size,
                            "C": C,
                            "morph_kernel_size": morph_kernel_size
                        }
                    }

        except Exception as e:
            print(f"[Tuning Error] {e}")
            continue

    return best_result


# ------------------- Streamlit App -------------------
st.set_page_config(layout="wide")
st.title("🧠 Enhanced Auto-Tuning OMR Pipeline")

st.sidebar.title("🔧 Auto-Tuning Parameters")
blur_c, _ = range_slider_param("Gaussian Blur (Kernel Size)", 1, 15, 5, 2)
block_c, _ = range_slider_param("Adaptive Threshold Block Size", 3, 101, 11, 2)
morph_c, _ = range_slider_param("Morphological Kernel Size", 1, 10, 2, 1)

st.sidebar.caption(
    "🔁 C is automatically tuned from 2 to 10 based on contrast.")
auto_mode = st.sidebar.checkbox("🧠 Enable Auto-Tuning", value=True)

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload Answer Sheet", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original = img.copy()

    result = None
    if auto_mode:
        param_grid = {"C": C_RANGE}
        result = auto_enhance_and_process(
            original, param_grid, blur_c, block_c, morph_c)

    if result:
        enhanced = result["enhanced"]
        graded_image = result["graded_img"]
        extracted_answers = result["answers"]
        codes = result["codes"]
        detection_img = result["detection_img"]
        st.sidebar.success(f"✅ Params: {result['params']}")
    else:
        st.warning("⚠️ Auto-tuning failed or disabled. Using defaults.")
        enhanced = image_enhancer(original, blur_c, block_c, 2, morph_c)
        detection_img, warped_paper, _, method_used, marker_points = transform_paper_image(
            enhanced.copy())
        if warped_paper is not None and warped_paper.shape == (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
            extracted_answers, graded_image, codes = ProcessPage(
                warped_paper.copy())
        else:
            st.error("❌ Pipeline failed. Check marker detection.")
            extracted_answers, graded_image, codes = [-1], None, [-1]

    # ------------------- Display Results -------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(original, caption="📷 Original", use_container_width=True)
    with col2:
        st.image(enhanced, caption="✨ Enhanced", use_container_width=True)
    with col3:
        st.image(detection_img if result else enhanced,
                 caption="📍 Marker Overlay", use_container_width=True)
    with col4:
        if graded_image is not None:
            st.image(graded_image, caption="📊 Graded",
                     use_container_width=True)
        else:
            st.warning("⚠️ No grading available.")

    st.markdown("---")
    st.subheader("📤 Extracted Results")
    if codes != [-1]:
        st.markdown(f"**QR Code:** `{codes[0]}`")
    else:
        st.markdown("**QR Code:** Not found")

    if extracted_answers != [-1]:
        for i, ans in enumerate(extracted_answers):
            st.write(f"Q{i+1}: {'❓' if ans == '?' else ans}")
    else:
        st.markdown("No answers extracted.")

    if graded_image is not None:
        buf = BytesIO()
        Image.fromarray(cv2.cvtColor(graded_image, cv2.COLOR_BGR2RGB)).save(
            buf, format="PNG")
        st.download_button("📥 Download Graded Sheet", data=buf.getvalue(
        ), file_name="graded_sheet.png", mime="image/png")
