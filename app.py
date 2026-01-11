import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="NEON VISION HUD",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------- NEON HUD STYLING --------------------- #
st.markdown("""
<style>
/* GRID BACKGROUND */
.stApp {
    background:
        linear-gradient(90deg, rgba(0,255,255,0.05) 1px, transparent 1px),
        linear-gradient(rgba(0,255,255,0.05) 1px, transparent 1px),
        #020617;
    background-size: 40px 40px;
    color: #e0f2fe;
    font-family: 'Orbitron', monospace;
}

/* HEADER */
.hud-header {
    border: 2px solid #22d3ee;
    padding: 1.8rem;
    border-radius: 20px;
    box-shadow: 0 0 25px #22d3ee;
    background: #020617;
}

/* PANELS */
.hud-panel {
    border: 1px solid rgba(34,211,238,0.4);
    padding: 1.4rem;
    border-radius: 16px;
    background: rgba(2,6,23,0.85);
    margin-bottom: 20px;
}

/* METRICS */
.hud-metric {
    border: 2px solid #a855f7;
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 0 20px #a855f7;
}

/* IMAGE FRAME */
.hud-image {
    border: 2px solid #22d3ee;
    border-radius: 18px;
    padding: 0.6rem;
    box-shadow: 0 0 30px rgba(34,211,238,0.6);
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] section {
    background: rgba(2,6,23,0.9);
    border: 1px dashed #22d3ee;
    border-radius: 14px;
}
[data-testid="stFileUploader"] * {
    color: #e0f2fe !important;
}

/* REMOVE FOOTER */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------- HEADER --------------------- #
st.markdown("""
<div class="hud-header">
    <h1>Contour Object Analyzer</h1>
    <p>Geometric Shape Detection & Measurement</p>
</div>
""", unsafe_allow_html=True)

# --------------------- UPLOADER PANEL --------------------- #
st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------- BACKEND LOGIC --------------------- #
def resize_for_display(img):
    h, w = img.shape[:2]
    scale = min(500 / w, 380 / h)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        v = len(approx)

        if v == 3:
            shape = "Triangle"
        elif v == 4:
            w, h = cv2.minAreaRect(cnt)[1]
            if min(w,h) == 0:
                continue
            shape = "Square" if max(w,h)/min(w,h) < 1.15 else "Rectangle"
        elif v == 5:
            shape = "Pentagon"
        elif v == 6:
            shape = "Hexagon"
        else:
            shape = "Circle" if (4*np.pi*area)/(perimeter**2) > 0.8 else "Irregular"

        cv2.drawContours(image, [approx], -1, (0, 220, 120), 3)
        cv2.putText(image, shape, (approx[0][0][0], approx[0][0][1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 2)

        results.append([shape, round(area,2), round(perimeter,2)])

    return image, results

# --------------------- PROCESSING --------------------- #
if uploaded_file:
    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(img_bytes, 1)

    processed, data = detect_shapes(image.copy())

    # --------------------- IMAGE DISPLAY --------------------- #
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="hud-panel"><h3>Original Image</h3>', unsafe_allow_html=True)
        st.image(resize_for_display(image), channels="BGR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="hud-panel"><h3>Detected Contours</h3>', unsafe_allow_html=True)
        st.image(resize_for_display(processed), channels="BGR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------- METRICS --------------------- #
    if data:
        df = pd.DataFrame(data, columns=["Shape", "Area", "Perimeter"])

        m1, m2, m3 = st.columns(3)
        metrics = [len(df), df["Shape"].nunique(), df["Area"].max()]
        labels = ["OBJECTS DETECTED", "SHAPE TYPES", "MAX AREA"]

        for col, val, label in zip([m1, m2, m3], metrics, labels):
            with col:
                st.markdown(f"""
                <div class="hud-metric">
                    <h2>{int(val)}</h2>
                    <span>{label}</span>
                </div>
                """, unsafe_allow_html=True)

        # --------------------- DATA TABLE --------------------- #
        st.markdown('<div class="hud-panel"><h3>Measurements</h3>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Upload an image to start contour analysis.")
