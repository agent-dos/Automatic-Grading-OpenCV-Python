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
EXPECTED_WIDTH = 850
EXPECTED_HEIGHT = 1202

# ------------------- Helper Functions -------------------


def range_slider_param_range(name, min_val, max_val, default_range, step):
    min_selected, max_selected = st.sidebar.slider(
        f"{name} Range", min_val, max_val, default_range, step=step)
    values = list(range(min_selected, max_selected + 1, step))
    return (min_selected + max_selected) // 2, values


def auto_enhance_and_process(img, param_grid):
    best_result = None
    best_score = -1

    for C, blur_ksize, block_size, morph_kernel_size in product(
        param_grid["C"],
        param_grid["blur_ksize"],
        param_grid["block_size"],
        param_grid["morph_kernel_size"]
    ):
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
blur_c, blur_range = range_slider_param_range(
    "Gaussian Blur (Kernel Size)", 1, 30, (3, 7), 1)
block_c, block_range = range_slider_param_range(
    "Adaptive Threshold Block Size", 3, 101, (9, 15), 1)
morph_c, morph_range = range_slider_param_range(
    "Morphological Kernel Size", 1, 10, (1, 3), 1)
c_min, c_max = st.sidebar.slider(
    "C Tuning Range (Adaptive Threshold)", 0, 20, (2, 10), step=1)
c_range = list(range(c_min, c_max + 1))

total_combinations = len(blur_range) * len(block_range) * \
    len(morph_range) * len(c_range)
st.sidebar.markdown(f"🔢 **Total Combinations:** `{total_combinations}`")
st.sidebar.caption(
    "🔁 Define parameter ranges for efficient adaptive threshold tuning.")

auto_mode = st.sidebar.checkbox("🧠 Enable Auto-Tuning", value=True)

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload Answer Sheet", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original = img.copy()

    result = None
    if auto_mode:
        param_grid = {
            "blur_ksize": blur_range,
            "block_size": block_range,
            "C": c_range,
            "morph_kernel_size": morph_range
        }
        result = auto_enhance_and_process(original, param_grid)

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
