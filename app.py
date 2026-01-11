import streamlit as st
import cv2
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Shape & Contour Analyzer",
    page_icon="🔷",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fb;
}
h2 {
    font-size: 22px !important;
}
h3 {
    font-size: 18px !important;
}
.small {
    font-size: 13px;
    color: #666;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("## Shape & Contour Analyzer (23MIA1120 - Aditya Rao B)")
st.markdown(
    "<p class='small'>Detect geometric shapes, count objects, and compute area & perimeter using contour-based feature extraction.</p>",
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

# ---------------- UTILS ----------------
def resize_for_display(img, max_height=420):
    h, w = img.shape[:2]
    scale = max_height / h
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), max_height))
    return img

# ---------------- SHAPE DETECTION ----------------
def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        vertices = len(approx)

        shape = "Unknown"

        if vertices == 3:
            shape = "Triangle"

        elif vertices == 4:
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / min(w, h)
            shape = "Square" if aspect < 1.15 else "Rectangle"

        elif vertices == 5:
            shape = "Pentagon"

        elif vertices == 6:
            shape = "Hexagon"

        else:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            shape = "Circle" if circularity > 0.8 else "Irregular"

        cv2.drawContours(image, [approx], -1, (80, 200, 120), 3)
        cv2.putText(
            image,
            shape,
            (approx[0][0][0], approx[0][0][1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,   # 👈 reduced font size
            (60, 120, 220),
            2
        )

        results.append([shape, round(area, 2), round(perimeter, 2)])

    return image, results

# ---------------- MAIN ----------------
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    processed_img, data = detect_shapes(image.copy())

    # Resize images for display
    image_disp = resize_for_display(image)
    processed_disp = resize_for_display(processed_img)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Original Image")
        st.image(image_disp, channels="BGR")

    with col2:
        st.subheader("Detected Shapes")
        st.image(processed_disp, channels="BGR")

    if data:
        df = pd.DataFrame(data, columns=["Shape", "Area", "Perimeter"])

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Objects", len(df))
        m2.metric("Unique Shapes", df["Shape"].nunique())
        m3.metric("Max Area", f"{df['Area'].max():.1f}")

        st.subheader("Detailed Measurements")
        st.dataframe(df, use_container_width=True)

else:
    st.info("Upload an image from the sidebar to begin analysis.")
