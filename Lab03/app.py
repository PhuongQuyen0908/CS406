import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Enhancement", layout="wide")

st.title("Lab 03: Image Enhancing")
st.markdown("---")

# Upload ảnh
uploaded_file = st.file_uploader(" Tải lên ảnh màu", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    img_array = np.array(image)

    # Chuyển từ RGB sang BGR cho OpenCV
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array

    # Hiển thị ảnh gốc
    st.subheader("Ảnh gốc")
    st.image(image, use_container_width=True)

    st.markdown("---")

    # 1. DENOISING / SMOOTHING
    st.subheader("1️. Denoising / Smoothing")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Gaussian Blur**")
        gaussian = cv2.GaussianBlur(img_bgr, (5, 5), 0)
        gaussian_rgb = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
        st.image(gaussian_rgb, use_container_width=True)

    with col2:
        st.write("**Median Blur**")
        median = cv2.medianBlur(img_bgr, 5)
        median_rgb = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
        st.image(median_rgb, use_container_width=True)

    with col3:
        st.write("**Bilateral Filter**")
        bilateral = cv2.bilateralFilter(img_bgr, 9, 75, 75)
        bilateral_rgb = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)
        st.image(bilateral_rgb, use_container_width=True)

    st.markdown("---")

    # 2. SHARPENING
    st.subheader("2️. Sharpening")
    col1, col2 = st.columns(2)

    # Kernel sharpening
    kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    sharpened = cv2.filter2D(img_bgr, -1, kernel_sharpen)
    sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    with col1:
        st.write("**Ảnh gốc**")
        st.image(image, use_container_width=True)

    with col2:
        st.write("**Ảnh sau khi Sharpening**")
        st.image(sharpened_rgb, use_container_width=True)

    st.markdown("---")

    # 3. EDGE DETECTION
    st.subheader("3️. Edge Detection")

    # Chuyển sang grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Sobel Edge Detection**")
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel * 255 / sobel.max())
        st.image(sobel, use_container_width=True, clamp=True)

    with col2:
        st.write("**Prewitt Edge Detection**")
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewittx = cv2.filter2D(gray, -1, kernelx)
        prewitty = cv2.filter2D(gray, -1, kernely)
        prewitt = np.sqrt(prewittx**2 + prewitty**2)
        prewitt = np.uint8(prewitt * 255 / prewitt.max())
        st.image(prewitt, use_container_width=True, clamp=True)

    with col3:
        st.write("**Canny Edge Detection**")
        canny = cv2.Canny(gray, 100, 200)
        st.image(canny, use_container_width=True, clamp=True)

    st.markdown("---")

    # Hiển thị so sánh tổng hợp
    st.subheader(" So sánh tổng hợp Edge Detection")

    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        st.write("**Ảnh gốc (Grayscale)**")
        st.image(gray, use_container_width=True, clamp=True)

    with fig_col2:
        # Tạo ảnh kết hợp 3 edge detection
        combined = np.hstack((sobel, prewitt, canny))
        st.write("**Sobel | Prewitt | Canny**")
        st.image(combined, use_container_width=True, clamp=True)

    st.success("Xử lý ảnh hoàn tất!")

else:
    st.info(" Vui lòng tải lên một ảnh để bắt đầu xử lý")


st.markdown("---")
st.caption("Lab-03: Image Enhancement Application")
