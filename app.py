import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

st.set_page_config(
    page_title="Smart Person Detection",
    page_icon="👤",
    layout="wide"
)

st.title("👤 Smart Person Detection")
st.subheader("AI-Powered Person Detection System")

st.write(
    "Upload one or more images to automatically detect people "
    "using an AI-based object detection model."
)

st.divider()

@st.cache_resource
def load_model():
    return YOLO("yolo11s.pt")

model = load_model()

st.subheader("📤 Upload Images")

uploaded_files = st.file_uploader(
    "Choose one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    st.divider()
    st.subheader("🔍 Detection Results")

    total_persons = 0

    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file).convert("RGB")

        results = model.predict(
            source=image,
            classes=[0],
            conf=0.50,
            verbose=False
        )

        result = results[0]
        person_count = len(result.boxes)

        total_persons += person_count

        st.markdown(f"### 🖼️ {uploaded_file.name}")

        if person_count > 0:

            annotated_image = result.plot(
                labels=True,
                boxes=True
            )

            annotated_image = annotated_image[:, :, ::-1]

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:
                st.image(
                    annotated_image,
                    caption="Detected Persons",
                    use_container_width=True
                )

            st.success(
                f"✅ {person_count} person(s) detected"
            )

            image_bytes = Image.fromarray(
                annotated_image
            )

            buffer = io.BytesIO()
            image_bytes.save(buffer, format="PNG")

            st.download_button(
                label="📥 Download Result",
                data=buffer.getvalue(),
                file_name=f"detected_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                mime="image/png"
            )

        else:

            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

            st.warning(
                "⚠️ No person was detected in this image."
            )

        st.divider()

    st.subheader("📊 Detection Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Images Uploaded",
            len(uploaded_files)
        )

    with col2:
        st.metric(
            "Total Persons Detected",
            total_persons
        )

else:

    st.info(
        "📌 Upload one or more images to start AI person detection."
    )

st.divider()

st.subheader("ℹ️ How It Works")

st.write(
    "1. Upload one or more images."
)

st.write(
    "2. The AI model scans each image for people."
)

st.write(
    "3. Detected people are highlighted with bounding boxes."
)

st.write(
    "4. Download the annotated image for further use."
)

st.divider()

st.caption(
    "🤖 Smart Person Detection • AI-Based Image Analysis"
)