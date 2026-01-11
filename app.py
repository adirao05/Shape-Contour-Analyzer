import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

MIN_CONTOUR_AREA = 500  

st.set_page_config(
    page_title="NEON VISION HUD",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

st.markdown("""
<div class="hud-header">
    <h1>Shape & Contour Analyzer</h1>
    <p>Real-Time Shape Detection System (23MIA1120 - Aditya Rao B)</p>
</div>
""", unsafe_allow_html=True)

st.write("")

st.markdown('<div class="hud-panel">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "UPLOAD IMAGE SIGNAL",
    type=["png", "jpg", "jpeg"]
)

st.markdown('</div>', unsafe_allow_html=True)
def classify_shape(approx, contour):
    v = len(approx)
    if v == 3:
        return "TRIANGLE"
    elif v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        return "SQUARE" if 0.95 <= w / float(h) <= 1.05 else "RECTANGLE"
    elif v > 4:
        return "CIRCLE"
    return "UNKNOWN"

if uploaded_file:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output_image = image_np.copy()
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        shape = classify_shape(approx, cnt)

        cv2.drawContours(output_image, [cnt], -1, (0, 255, 255), 3)

        x, y, w, h = cv2.boundingRect(approx)
        cv2.putText(
            output_image,
            shape,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2
        )

        results.append([shape, area, peri])

    df = pd.DataFrame(
        results,
        columns=["SHAPE", "AREA (px²)", "PERIMETER (px)"]
    )

    m1, m2, m3 = st.columns(3)

    for col, val, label in zip(
        [m1, m2, m3],
        [len(df), df["SHAPE"].nunique(), df["AREA (px²)"].max()],
        ["OBJECTS DETECTED", "SHAPE TYPES", "MAX AREA"]
    ):
        with col:
            st.markdown(f"""
            <div class="hud-metric">
                <h2>{int(val)}</h2>
                <span>{label}</span>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    i1, i2 = st.columns(2)

    with i1:
        st.markdown('<div class="hud-image">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with i2:
        st.markdown('<div class="hud-image">', unsafe_allow_html=True)
        st.image(output_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
    st.subheader("MEASUREMENT MATRIX")

    st.dataframe(
        df.style
        .format({
            "AREA (px²)": "{:.2f}",
            "PERIMETER (px)": "{:.2f}"
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#020617"),
                ("color", "#22d3ee"),
                ("border", "1px solid #22d3ee")
            ]},
            {"selector": "td", "props": [
                ("background-color", "rgba(2,6,23,0.9)"),
                ("color", "#e0f2fe"),
                ("border", "1px solid rgba(34,211,238,0.3)")
            ]}
        ]),
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("AWAITING IMAGE INPUT SIGNAL...")
