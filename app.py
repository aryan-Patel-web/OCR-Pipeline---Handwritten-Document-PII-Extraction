"""
app.py
Streamlit Frontend for Hybrid Medical OCR (Tesseract + TrOCR)
Black / ChatGPT-style UI
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import json

from main import MedicalDocumentOCR

st.set_page_config(
    page_title="Handwritten Medical OCR",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- DARK THEME CSS (BLACK UI) ---------------------- #

def apply_black_theme():
    st.markdown(
        """
        <style>
        /* Global */
        .stApp {
            background-color: #000000;
            color: #f1f1f1;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #050505;
            border-right: 1px solid #222;
        }

        /* Titles */
        h1, h2, h3, h4, h5 {
            color: #ffffff !important;
            font-weight: 700;
        }

        /* Text */
        p, span, label, li {
            color: #e0e0e0 !important;
        }

        /* Cards */
        .card {
            background: #111111;
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid #262626;
            box-shadow: 0 0 0 1px #111;
        }

        .card-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #10b981, #0ea5e9);
            color: white;
            border-radius: 999px;
            border: none;
            padding: 0.45rem 1.4rem;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 25px rgba(16,185,129,0.45);
        }

        /* File uploader: smaller, minimal look */
        [data-testid="stFileUploader"] > div:nth-child(1) {
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            border: 1px solid #333;
            background: #111;
        }
        [data-testid="stFileUploader"] section {
            padding: 0.4rem 0;
        }
        [data-testid="stFileUploader"] label {
            font-size: 0.9rem !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #050505;
            border-radius: 999px;
            padding: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #a3a3a3;
            border-radius: 999px;
            padding-top: 0.25rem;
            padding-bottom: 0.25rem;
        }
        .stTabs [aria-selected="true"] {
            background: #111111;
            color: #f9fafb !important;
            box-shadow: 0 0 0 1px #1f2937;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #22c55e;
            font-weight: 700;
        }
        [data-testid="stMetricLabel"] {
            color: #9ca3af;
        }

        /* Text area */
        textarea {
            background-color: #050505 !important;
            color: #e5e7eb !important;
            border-radius: 10px !important;
            border: 1px solid #27272a !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #050505;
        }
        ::-webkit-scrollbar-thumb {
            background: #444;
            border-radius: 999px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #666;
        }

        /* Small caption text */
        .caption {
            font-size: 0.8rem;
            color: #9ca3af;
        }

        /* Gradient header text */
        .gradient-title {
            background: linear-gradient(120deg, #22c55e, #06b6d4, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.3rem;
            text-align: center;
            margin-bottom: 0.3rem;
        }
        .subtitle {
            text-align: center;
            color: #9ca3af;
            font-size: 0.95rem;
            margin-bottom: 1.8rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


apply_black_theme()

# ---------------------- SESSION STATE ---------------------- #

if "ocr_engine" not in st.session_state:
    st.session_state.ocr_engine = MedicalDocumentOCR()

if "results" not in st.session_state:
    st.session_state.results = None

# ---------------------- HEADER ---------------------- #

st.markdown('<div class="gradient-title">Handwritten Medical OCR</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Hybrid Tesseract + TrOCR engine for tough hospital documents (raw text only)</div>',
    unsafe_allow_html=True,
)

# ---------------------- SIDEBAR (SETTINGS) ---------------------- #

with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")
    st.markdown("---")

    use_tesseract = st.checkbox("Use Tesseract (printed text)", value=True)
    use_trocr = st.checkbox("Use TrOCR (handwriting)", value=True)

    st.markdown("### 🧪 Preprocessing")
    deskew = st.checkbox("Deskew pages", value=True)
    denoise = st.checkbox("Denoise", value=True)
    enhance = st.checkbox("Enhance contrast", value=True)

    st.markdown("### ✂️ TrOCR Grid")
    rows = st.slider("Grid rows", 1, 4, 3)
    cols = st.slider("Grid cols", 1, 4, 2)

    st.markdown("---")
    st.markdown("### ℹ️ Info")
    st.caption(
        "TrOCR is a handwriting transformer model. "
        "We split the page into tiles and OCR each tile to capture even faint pen strokes."
    )

# Update engine config from sidebar
st.session_state.ocr_engine.use_tesseract = use_tesseract
st.session_state.ocr_engine.use_trocr = use_trocr
st.session_state.ocr_engine.deskew = deskew
st.session_state.ocr_engine.denoise = denoise
st.session_state.ocr_engine.enhance_contrast = enhance
st.session_state.ocr_engine.trocr_grid_rows = rows
st.session_state.ocr_engine.trocr_grid_cols = cols

# ---------------------- MAIN LAYOUT ---------------------- #

left_col, right_col = st.columns([1, 1])

# ---------- LEFT: IMAGE + RUN BUTTON ---------- #

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📤 Upload document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "JPEG/PNG medical document",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Original scan", use_container_width=True)

        run = st.button("🔍 Run Hybrid OCR", use_container_width=True)
        if run:
            with st.spinner("Processing with Tesseract + TrOCR..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                try:
                    res = st.session_state.ocr_engine.process_document(tmp_path)
                    st.session_state.results = res
                    st.success("✅ OCR completed")
                except Exception as e:
                    st.error(f"Error while running OCR: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    else:
        st.caption("Upload a scan on the right button to start.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT: RESULTS ---------- #

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📊 OCR Results</div>', unsafe_allow_html=True)

    results = st.session_state.results

    if results is None:
        st.info("No document processed yet. Upload on the left and click **Run Hybrid OCR**.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Metrics row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Pages", results["summary"]["pages_detected"])
        with c2:
            st.metric("PII Items", results["summary"]["total_pii_found"])
        with c3:
            st.metric("Text Length", len(results["combined_text_all"]))

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📝 Combined Text", "📄 Per Page", "🔒 PII Summary", "🖼️ Redacted Image", "📥 Download"]
        )

        # --- Tab 1: Combined text (raw) --- #
        with tab1:
            st.markdown("#### Combined OCR Output (Tesseract + TrOCR)")
            st.text_area(
                label="",
                value=results["combined_text_all"],
                height=420,
            )

        # --- Tab 2: Per-page view --- #
        with tab2:
            for page in results["pages"]:
                st.markdown(f"#### Page: `{page['side']}`")
                colp1, colp2 = st.columns(2)
                with colp1:
                    st.markdown("**Tesseract (printed)**")
                    st.text_area(
                        label=f"Tesseract - {page['side']}",
                        value=page["tesseract_text"],
                        height=200,
                    )
                with colp2:
                    st.markdown("**TrOCR (handwriting)**")
                    st.text_area(
                        label=f"TrOCR - {page['side']}",
                        value=page["trocr_text"],
                        height=200,
                    )
                st.markdown("---")

        # --- Tab 3: PII Summary --- #
        with tab3:
            st.markdown("#### Detected PII (regex-based)")
            pii = results["pii"]
            if not any(pii.values()):
                st.info("No PII-like patterns detected.")
            else:
                for key, vals in pii.items():
                    if not vals:
                        continue
                    st.markdown(f"**{key}**")
                    st.write(", ".join(vals))

        # --- Tab 4: Redacted Image --- #
        with tab4:
            st.markdown("#### Redacted Image (PII hidden where possible)")
            red = results["redacted_image"]
            if red is None:
                st.info("Redacted image unavailable.")
            else:
                red_rgb = cv2.cvtColor(red, cv2.COLOR_BGR2RGB)
                st.image(red_rgb, caption="Redacted Image", use_container_width=True)

        # --- Tab 5: Download --- #
        with tab5:
            st.markdown("#### Download OCR Output")

            txt = results["combined_text_all"]
            st.download_button(
                label="📄 Download Full Text (.txt)",
                data=txt.encode("utf-8"),
                file_name=f"ocr_text_{results['timestamp']}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            json_payload = {
                "timestamp": results["timestamp"],
                "summary": results["summary"],
                "pii": results["pii"],
                "pages": results["pages"],
            }
            st.download_button(
                label="🧾 Download JSON (.json)",
                data=json.dumps(json_payload, indent=2),
                file_name=f"ocr_json_{results['timestamp']}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
