"""EmotionVision - Streamlit frontend (talks to the FastAPI backend).

Run:
    streamlit run app/streamlit_app.py
Env:
    API_URL (default http://localhost:8000)
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

API_URL = os.getenv("API_URL", "http://localhost:8000")
REPORTS = Path("reports")
EXAMPLES_DIR = Path("app/examples")

CLASS_COLORS = {
    "neutral": "#9CA3AF", "happiness": "#F59E0B", "surprise": "#8B5CF6", "sadness": "#3B82F6",
    "anger": "#EF4444", "disgust": "#10B981", "fear": "#EC4899",
}
CLASS_EMOJI = {
    "neutral": "😐", "happiness": "😄", "surprise": "😮", "sadness": "😢",
    "anger": "😠", "disgust": "🤢", "fear": "😨",
}

st.set_page_config(page_title="EmotionVision", page_icon="🙂", layout="wide")


# ----------------------------------------------------------------- API calls
@st.cache_data(ttl=30, show_spinner=False)
def api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def api_explain(image_bytes: bytes, filename: str, mime: str) -> dict:
    r = requests.post(f"{API_URL}/explain", files={"file": (filename, image_bytes, mime)}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()


# ------------------------------------------------------------------- helpers
def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def draw_boxes(img: Image.Image, faces: list[dict]) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    lw = max(2, out.width // 250)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, out.width // 40))
    except OSError:
        font = ImageFont.load_default()
    for i, f in enumerate(faces, 1):
        b, color = f["box"], CLASS_COLORS.get(f["label"], "#FFFFFF")
        x0, y0, x1, y1 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
        d.rectangle([x0, y0, x1, y1], outline=color, width=lw)
        text = f"#{i} {f['label']} {f['confidence']:.0%}" + (" ?" if f["uncertain"] else "")
        tb = d.textbbox((x0, y0), text, font=font)
        d.rectangle([x0, max(tb[1] - 4, 0) - (tb[3] - tb[1]) - 4, tb[2] + 8, y0], fill=color)
        d.text((x0 + 4, max(y0 - (tb[3] - tb[1]) - 6, 0)), text, fill="black", font=font)
    return out


def proba_chart(probs: dict[str, float]) -> None:
    df = pd.DataFrame({"expression": list(probs), "probability": list(probs.values())})
    df = df.sort_values("probability", ascending=True)
    st.bar_chart(df, x="expression", y="probability", horizontal=True, height=230,
                 color="#6366F1")


def show_results(img: Image.Image, res: dict) -> None:
    faces = res["faces"]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(draw_boxes(img, faces) if res["face_detected"] else img, width="stretch")
    with c2:
        st.metric("Faces detected", res["n_faces"])
        st.metric("Inference time", f"{res['inference_ms']:.0f} ms")
        if not res["face_detected"]:
            st.warning("No face detected - the whole image was classified. "
                       "Try a closer, frontal face for reliable results.")

    for i, f in enumerate(faces, 1):
        st.divider()
        st.subheader(f"Face #{i}: {CLASS_EMOJI.get(f['label'], '')} {f['label'].capitalize()} "
                     f"({f['confidence']:.1%})")
        if f["uncertain"]:
            st.info("Low confidence: the expression is ambiguous between several classes.")
        a, b = st.columns([1, 2])
        with a:
            if f.get("heatmap_png_base64"):
                st.image(base64.b64decode(f["heatmap_png_base64"]),
                         caption="Grad-CAM: regions that drove the prediction", width="stretch")
        with b:
            proba_chart(f["probabilities"])


def run_prediction(image_bytes: bytes, filename: str, mime: str) -> None:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    # Streamlit reruns the script on every click: cache results per image so the API
    # is called (and the session history updated) only once per image.
    key = hashlib.md5(image_bytes).hexdigest()
    cache = st.session_state.results
    if key not in cache:
        with st.spinner("Analyzing..."):
            try:
                cache[key] = api_explain(to_png_bytes(img), Path(filename).stem + ".png", "image/png")
            except requests.RequestException:
                st.error(f"Cannot reach the API at {API_URL}. Is it running?")
                return
            except RuntimeError as e:
                st.error(f"API error: {e}")
                return
        for f in cache[key]["faces"]:
            st.session_state.history.append({"label": f["label"], "confidence": f["confidence"]})
    show_results(img, cache[key])


# ---------------------------------------------------------------------- page
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.results = {}

with st.sidebar:
    st.title("🙂 EmotionVision")
    st.caption("Facial expression recognition with Grad-CAM explanations")
    health = api_get("/health")
    if health and health.get("model_loaded"):
        st.success("API online")
        info = api_get("/model/info") or {}
        st.markdown(f"**Model:** `{info.get('model', '?')}`  \n"
                    f"**Input:** {info.get('input_size', '?')} px  \n"
                    f"**Runtime:** ONNX Runtime (CPU)  \n"
                    f"**Uncertainty threshold:** {info.get('uncertainty_threshold', '?')}")
        st.markdown("**Classes:** " + " ".join(CLASS_EMOJI.values()))
    else:
        st.error(f"API offline ({API_URL})")
    st.divider()
    st.caption("⚠️ Predicts the visible facial *expression*, not a person's internal emotional state. "
               "Images are processed in memory and not stored.")

tab_upload, tab_webcam, tab_examples, tab_dash = st.tabs(
    ["📤 Upload", "📷 Webcam", "🖼️ Examples", "📊 Dashboard"])

with tab_upload:
    up = st.file_uploader("Upload an image with one or more faces",
                          type=["jpg", "jpeg", "png", "webp", "bmp"])
    if up is not None:
        run_prediction(up.getvalue(), up.name, up.type or "image/png")

with tab_webcam:
    shot = st.camera_input("Take a picture")
    if shot is not None:
        run_prediction(shot.getvalue(), "webcam.png", "image/png")

with tab_examples:
    examples = sorted(p for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not examples:
        st.info(f"Add a few face images to `{EXAMPLES_DIR}/` to show them here.")
    else:
        cols = st.columns(min(len(examples), 6))
        for i, p in enumerate(examples):
            with cols[i % len(cols)]:
                st.image(str(p), width="stretch")
                if st.button("Analyze", key=f"ex_{p.name}", width="stretch"):
                    st.session_state.example = p
        if "example" in st.session_state:
            p = st.session_state.example
            st.divider()
            run_prediction(p.read_bytes(), p.name, "image/png")

with tab_dash:
    st.header("Model & pipeline dashboard")

    bench = REPORTS / "benchmark.json"
    if bench.exists():
        st.subheader("Inference benchmark (CPU)")
        b = json.loads(bench.read_text())
        df = pd.DataFrame(b).T
        keep = ["size_mb", "test_accuracy", "test_macro_f1", "latency_ms_p50",
                "latency_ms_p95", "throughput_img_s", "speedup_vs_pytorch"]
        st.dataframe(df[[c for c in keep if c in df.columns]], width="stretch")
        best = b.get("onnx_fp32", {})
        m1, m2, m3 = st.columns(3)
        m1.metric("Test accuracy (ONNX)", f"{best.get('test_accuracy', 0):.1%}")
        m2.metric("Test macro-F1 (ONNX)", f"{best.get('test_macro_f1', 0):.3f}")
        m3.metric("Speed-up vs PyTorch", f"x{best.get('speedup_vs_pytorch', 0)}")

    abl = REPORTS / "quantization_ablation.json"
    if abl.exists():
        with st.expander("INT8 quantization ablation (validation set)"):
            st.dataframe(pd.DataFrame(json.loads(abl.read_text())).T, width="stretch")

    stats = REPORTS / "data_stats.json"
    if stats.exists():
        st.subheader("Dataset (FER2013 + FER+ labels)")
        s = json.loads(stats.read_text())
        c = st.columns(4)
        c[0].metric("Train", s["train_size"])
        c[1].metric("Validation", s["val_size"])
        c[2].metric("Test", s["test_size"])
        c[3].metric("Dropped (ambiguous)", s["dropped"])
        st.bar_chart(pd.DataFrame(s["per_class"]), height=280)

    figs = [("Confusion matrix (test)", sorted((REPORTS / "figures").glob("confusion_matrix_*.png"))),
            ("Grad-CAM examples", [REPORTS / "figures" / "gradcam_examples.png"])]
    for title, paths in figs:
        paths = [p for p in paths if p.exists()]
        if paths:
            st.subheader(title)
            st.image([str(p) for p in paths], width="stretch")

    st.subheader("This session")
    if st.session_state.history:
        h = pd.DataFrame(st.session_state.history)
        c1, c2 = st.columns(2)
        c1.metric("Faces analyzed", len(h))
        c2.metric("Mean confidence", f"{h['confidence'].mean():.1%}")
        st.bar_chart(h["label"].value_counts(), height=220)
    else:
        st.caption("No prediction yet in this session.")
