import streamlit as st
from ultralytics import YOLO
from PIL import Image

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Fake Photo Detection",
    page_icon="📸",
    layout="centered"
)

# -------------------------------------------------
# Load YOLO Model
# -------------------------------------------------
model = YOLO("best.pt")

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>
📸 Fake Photo Detection
</h1>

<p style='text-align:center; font-size:18px;'>
Detect whether the captured image is a
<b>Real Photo</b> or a <b>Photo of a Screen</b>.
</p>
""", unsafe_allow_html=True)

st.write("")

# -------------------------------------------------
# Camera
# -------------------------------------------------
st.markdown(
    "<h4 style='text-align:center;'>Capture Image</h4>",
    unsafe_allow_html=True
)

camera_image = st.camera_input(
    "",
    label_visibility="collapsed"
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if camera_image is not None:

    image = Image.open(camera_image)

    st.image(
        image,
        caption="Captured Image",
        use_container_width=True
    )

    with st.spinner("🔍 Analyzing Image..."):
        result = model.predict(
            source=image,
            imgsz=224,
            verbose=False
        )[0]

    score = float(result.probs.data[1])

    st.divider()

    if score > 0.5:
        st.error(
            f"📱 **Photo of a Screen Detected**\n\n"
            f"Confidence: **{score*100:.2f}%**"
        )
    else:
        st.success(
            f"📷 **Real Photo Detected**\n\n"
            f"Confidence: **{(1-score)*100:.2f}%**"
        )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:15px;'>
        🚀 Developed by <b>Akshay Raj</b>
    </div>
    """,
    unsafe_allow_html=True
)