"""Run: streamlit run streamlit_app.py"""
from pathlib import Path
import shutil
import subprocess
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


@st.cache_resource(show_spinner=False)
def load_model(model_name: str) -> YOLO:
    return YOLO(str(MODEL_PATHS[model_name]))


def ensure_models_exist() -> list[str]:
    missing = [name for name, path in MODEL_PATHS.items() if not path.exists()]
    return missing


def _open_video_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """Prefer codecs that work in browser HTML5 players; fall back to mp4v."""
    candidates = ("avc1", "H264", "mp4v")
    for cc in candidates:
        fourcc = cv2.VideoWriter_fourcc(*cc)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
    raise RuntimeError("Could not open video writer for output (try installing ffmpeg).")


def ensure_browser_playable_mp4(src: Path, dst: Path) -> Path:
    """Re-encode to H.264 + yuv420p so st.video can play in the browser."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dst),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return dst
    except (subprocess.CalledProcessError, OSError):
        return src


def process_video(
    input_path: str,
    model_names: list[str],
    conf: float,
    output_path: str,
    progress_bar,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open uploaded video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0:
        fps = 25.0

    writer = _open_video_writer(Path(output_path), width, height, fps)

    models = {name: load_model(name) for name in model_names}
    class_counter = {}
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated = frame.copy()

        for model_name, model in models.items():
            results = model.predict(source=frame, conf=conf, verbose=False)
            result = results[0]
            names = result.names
            color = MODEL_COLORS[model_name]

            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label_name = names.get(cls_id, str(cls_id))
                full_label = f"{model_name}: {label_name} {score:.2f}"

                class_counter[full_label] = class_counter.get(full_label, 0) + 1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    full_label,
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        writer.write(annotated)
        frame_idx += 1
        if frame_count > 0:
            progress_bar.progress(min(frame_idx / frame_count, 1.0))

    cap.release()
    writer.release()
    progress_bar.progress(1.0)
    return class_counter


def app():
    st.title("Marine Video Object Detection")
    st.write(
        "Upload a video and run detection using fish, marine waste, oil spill, or all models."
    )

    if "detection_video_bytes" not in st.session_state:
        st.session_state.detection_video_bytes = None
    if "detection_counts" not in st.session_state:
        st.session_state.detection_counts = None
    if "detection_filename" not in st.session_state:
        st.session_state.detection_filename = None

    missing_models = ensure_models_exist()
    if missing_models:
        st.error(f"Missing model files: {', '.join(missing_models)}")
        st.stop()

    mode = st.radio(
        "Detection Mode",
        ["Single Model", "All Models"],
        horizontal=True,
    )

    if mode == "Single Model":
        selected = st.selectbox("Choose Model", list(MODEL_PATHS.keys()))
        active_models = [selected]
    else:
        active_models = list(MODEL_PATHS.keys())
        st.info("All three models will run on each frame.")

    conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        sig = f"{uploaded_file.name}_{uploaded_file.size}"
        if (
            st.session_state.get("last_upload_sig") is not None
            and st.session_state.last_upload_sig != sig
        ):
            st.session_state.detection_video_bytes = None
            st.session_state.detection_counts = None
            st.session_state.detection_filename = None
        st.session_state.last_upload_sig = sig

        st.subheader("Input")
        st.video(uploaded_file)

        if st.button("Run Detection", type="primary"):
            with tempfile.TemporaryDirectory() as temp_dir:
                td = Path(temp_dir)
                input_video = td / uploaded_file.name
                raw_out = td / f"detected_{Path(uploaded_file.name).stem}_raw.mp4"
                h264_out = td / f"detected_{Path(uploaded_file.name).stem}.mp4"

                input_video.write_bytes(uploaded_file.getbuffer())
                progress_bar = st.progress(0.0)

                with st.spinner("Processing video..."):
                    counts = process_video(
                        input_path=str(input_video),
                        model_names=active_models,
                        conf=conf,
                        output_path=str(raw_out),
                        progress_bar=progress_bar,
                    )

                final_path = ensure_browser_playable_mp4(raw_out, h264_out)
                output_bytes = final_path.read_bytes()
                out_name = h264_out.name if final_path == h264_out else raw_out.name

                st.session_state.detection_video_bytes = output_bytes
                st.session_state.detection_counts = counts
                st.session_state.detection_filename = out_name

                st.success("Detection complete — output preview is below.")

    if st.session_state.detection_video_bytes is not None:
        st.subheader("Output (detected)")
        st.video(st.session_state.detection_video_bytes)

        st.download_button(
            "Download detected video",
            data=st.session_state.detection_video_bytes,
            file_name=st.session_state.detection_filename or "detected.mp4",
            mime="video/mp4",
        )

        counts = st.session_state.detection_counts
        if counts:
            df = (
                pd.DataFrame(
                    [{"Class": key, "Detections": value} for key, value in counts.items()]
                )
                .sort_values("Detections", ascending=False)
                .reset_index(drop=True)
            )
            st.subheader("Detection Summary")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No detections found in the uploaded video.")


if __name__ == "__main__":
    app()
