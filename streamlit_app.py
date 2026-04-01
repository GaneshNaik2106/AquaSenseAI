"""Run: streamlit run streamlit_app.py"""

from pathlib import Path
import tempfile
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Marine Detection App", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "Fish": BASE_DIR / "final_models" / "best_fish.pt",
    "Marine Waste": BASE_DIR / "final_models" / "best_waste.pt",
    "Oil Spill": BASE_DIR / "final_models" / "best_oil.pt",
}

MODEL_COLORS = {
    "Fish": (0, 255, 0),
    "Marine Waste": (0, 165, 255),
    "Oil Spill": (255, 0, 255),
}


# ✅ Load model (cached)
@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> YOLO:
    return YOLO(str(MODEL_PATHS[model_name]))


def ensure_models_exist():
    return [name for name, path in MODEL_PATHS.items() if not path.exists()]


# ✅ STREAMLIT SAFE VIDEO WRITER (XVID ONLY)
def open_video_writer(path, width, height, fps):
    path = Path(path).with_suffix(".avi")  # force avi
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    return writer, path


# ✅ MAIN PROCESS FUNCTION
def process_video(input_path, model_names, conf, progress_bar):

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output path
    output_path = "output.avi"
    writer, final_path = open_video_writer(output_path, width, height, fps)

    models = {name: load_model(name) for name in model_names}
    counts = {}

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        for model_name, model in models.items():
            results = model.predict(frame, conf=conf, verbose=False)[0]

            if results.boxes is None:
                continue

            names = results.names
            color = MODEL_COLORS[model_name]

            for box in results.boxes:
                cls = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = f"{model_name}: {names[cls]} {score:.2f}"
                counts[label] = counts.get(label, 0) + 1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        writer.write(annotated)

        frame_idx += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()
    progress_bar.progress(1.0)

    return final_path, counts


# ✅ STREAMLIT APP
def app():
    st.title("🌊 Marine Video Object Detection")

    if "video_bytes" not in st.session_state:
        st.session_state.video_bytes = None
    if "counts" not in st.session_state:
        st.session_state.counts = None

    # Check models
    missing = ensure_models_exist()
    if missing:
        st.error(f"Missing models: {', '.join(missing)}")
        st.stop()

    mode = st.radio("Mode", ["Single Model", "All Models"], horizontal=True)

    if mode == "Single Model":
        selected = st.selectbox("Select Model", list(MODEL_PATHS.keys()))
        active_models = [selected]
    else:
        active_models = list(MODEL_PATHS.keys())
        st.info("Running all models")

    conf = st.slider("Confidence", 0.1, 0.9, 0.25)

    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    if uploaded:
        st.video(uploaded)

        if st.button("Run Detection"):

            with tempfile.TemporaryDirectory() as tmp:
                input_path = Path(tmp) / uploaded.name
                input_path.write_bytes(uploaded.getbuffer())

                progress = st.progress(0.0)

                with st.spinner("Processing..."):
                    output_path, counts = process_video(
                        str(input_path),
                        active_models,
                        conf,
                        progress,
                    )

                # ✅ READ VIDEO BYTES (IMPORTANT FIX)
                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                st.session_state.video_bytes = video_bytes
                st.session_state.counts = counts

                st.success("Done!")

    # OUTPUT
    if st.session_state.video_bytes:
        st.subheader("Output")
        st.video(st.session_state.video_bytes)

        if st.session_state.counts:
            df = pd.DataFrame([
                {"Class": k, "Count": v}
                for k, v in st.session_state.counts.items()
            ]).sort_values("Count", ascending=False)

            st.subheader("Detection Summary")
            st.dataframe(df, width="stretch")


if __name__ == "__main__":
    app()