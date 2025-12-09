"""
app.py
Streamlit frontend for MedicalDocumentOCR
Dark theme UI similar to ChatGPT.
"""

import streamlit as st
import cv2
from PIL import Image
import json
import tempfile
import os

from main import MedicalDocumentOCR

# ---------------------- PAGE CONFIG ---------------------- #

st.set_page_config(
    page_title="Medical OCR + PII Extractor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------- DARK THEME CSS ---------------------- #

def apply_dark_theme():
    st.markdown(
        """
    <style>
        .stApp {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }

        [data-testid="stSidebar"] {
            background-color: #2D2D2D;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;
            font-weight: 600;
        }

        p, span, label {
            color: #E0E0E0;
        }

        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: #2D2D2D;
            color: #E0E0E0;
            border: 1px solid #404040;
            border-radius: 8px;
        }

        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }

        [data-testid="stFileUploader"] {
            background-color: #2D2D2D;
            border: 2px dashed #667eea;
            border-radius: 12px;
            padding: 2rem;
        }

        .custom-card {
            background: linear-gradient(135deg, #2d2d2d 0%, #1e1e1e 100%);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            border: 1px solid #404040;
            margin-bottom: 1.5rem;
        }

        .gradient-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 2rem;
        }

        .pii-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            margin: 0.15rem;
            font-weight: 600;
            font-size: 0.85rem;
        }

        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #1E1E1E;
        }
        ::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #764ba2;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


apply_dark_theme()


# ---------------------- SESSION STATE ---------------------- #

if "results" not in st.session_state:
    st.session_state.results = None


# ---------------------- HEADER ---------------------- #

st.markdown(
    '<h1 class="gradient-header">🏥 Medical Document OCR + PII Extractor</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align: center; font-size: 1.1rem; color: #B0B0B0; margin-bottom: 2rem;">Advanced preprocessing + OCR tuned for handwritten hospital records (double-page, progress notes, charts)</p>',
    unsafe_allow_html=True,
)


# ---------------------- SIDEBAR ---------------------- #

with st.sidebar:
    st.markdown("### ⚙️ Preprocessing Settings")
    st.markdown("---")

    deskew = st.checkbox("Auto-deskew tilted pages", value=True)
    denoise = st.checkbox("Denoise background", value=True)
    enhance_contrast = st.checkbox("Enhance contrast (CLAHE)", value=True)

    st.markdown("---")
    st.markdown("#### ℹ️ Assignment Notes")
    st.info(
        """
**Pipeline:**  
Input (handwritten JPEG) → Pre-processing → OCR → Text Cleaning →  
PII + structured extraction → Optional redacted image.

Optimized for:  
• Slight tilt  
• Doctor handwriting  
• Double-page scanned forms
"""
    )

    st.markdown("---")
    st.markdown("#### 🔒 Privacy")
    st.warning("All processing is **local only**. No data leaves your machine.")


# ---------------------- MAIN LAYOUT ---------------------- #

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Handwritten Medical Document")

    uploaded_file = st.file_uploader(
        "Upload JPEG/PNG medical notes or hospital forms",
        type=["jpg", "jpeg", "png"],
        help="Use clear scans or photos; double-page images are supported.",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Document", use_column_width=True)

        if st.button("🚀 Run OCR + PII Extraction", use_container_width=True):
            with st.spinner("Processing document with advanced OCR..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".jpg"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    ocr = MedicalDocumentOCR(
                        deskew=deskew,
                        denoise=denoise,
                        enhance_contrast=enhance_contrast,
                    )
                    results = ocr.process_document(tmp_path)
                    st.session_state.results = results
                    st.success("✅ Processing complete!")
                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    else:
        st.info("⬆️ Upload a handwritten medical JPEG/PNG to begin.")

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Extraction Results")

    results = st.session_state.results
    if results is None:
        st.info("Results will appear here after you process a document.")
    else:
        # Top metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("PII Items Found", results["summary"]["total_pii_found"])
        with c2:
            st.metric("PII Categories", len(results["summary"]["pii_categories"]))
        with c3:
            st.metric("Pages Detected", results["summary"]["pages_detected"])

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🔍 Structured Fields", "📝 Text & PII", "🖼️ Redacted Image", "📥 Download"]
        )

        # -------- Structured Fields per Page -------- #
        with tab1:
            st.markdown("#### Per-page structured extraction")
            for page in results.get("pages", []):
                side = page.get("side", "page")
                fields = page.get("fields", {})
                st.markdown(f"##### 📄 {side.upper()} PAGE")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Header / Patient Info**")
                    st.write("Hospital:", fields.get("hospital_name") or "—")
                    st.write("Location:", fields.get("location") or "—")
                    st.write("Form Code:", fields.get("form_code") or "—")
                    st.write("Patient Name:", fields.get("patient_name") or "—")
                    st.write("Age:", fields.get("age") or "—")
                    st.write("Sex:", fields.get("sex") or "—")
                    st.write("IPD No.:", fields.get("ipd_number") or "—")
                    st.write("UHID No.:", fields.get("uhid_number") or "—")
                    st.write("Bed No.:", fields.get("bed_number") or "—")
                    st.write("Date:", fields.get("date") or "—")
                    st.write("Time:", fields.get("time") or "—")

                with col_b:
                    st.markdown("**Vitals**")
                    vitals = fields.get("vitals", {}) or {}
                    st.write("BP:", vitals.get("bp") or "—")
                    st.write("PR:", vitals.get("pr") or "—")
                    st.write("RR:", vitals.get("rr") or "—")
                    st.write("Temp:", vitals.get("temp") or "—")

                    st.markdown("**Clinical Summary**")
                    cs = fields.get("clinical_summary") or "—"
                    st.write(cs)

                st.markdown("---")

        # -------- PII + Text -------- #
        with tab2:
            sub1, sub2 = st.columns([1, 1])
            with sub1:
                st.markdown("#### Extracted PII tokens (for redaction)")
                pii = results.get("pii_extracted", {})
                for cat, vals in pii.items():
                    if not vals:
                        continue
                    st.markdown(f"**{cat.replace('_', ' ').title()}:**")
                    for v in vals:
                        st.markdown(
                            f'<span class="pii-badge">{v}</span>',
                            unsafe_allow_html=True,
                        )
                    st.markdown("")

            with sub2:
                st.markdown("#### Cleaned OCR Text (all pages)")
                st.text_area(
                    label="",
                    value=results.get("cleaned_text", ""),
                    height=400,
                )

        # -------- Redacted Image -------- #
        with tab3:
            if results.get("redacted_image") is not None:
                st.markdown("#### Redacted document preview")
                bgr = results["redacted_image"]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                st.image(
                    Image.fromarray(rgb),
                    caption="Redacted image (PII blocked)",
                    use_column_width=True,
                )
            else:
                st.info("Redacted image not available for this document.")

        # -------- Downloads -------- #
        with tab4:
            st.markdown("#### Download structured output")

            export = {
                "timestamp": results["timestamp"],
                "image_path": results["image_path"],
                "pages": results["pages"],
                "cleaned_text": results["cleaned_text"],
                "pii_extracted": results["pii_extracted"],
                "summary": results["summary"],
            }

            st.download_button(
                label="📄 Download JSON report",
                data=json.dumps(export, indent=2),
                file_name=f"ocr_results_{results['timestamp']}.json",
                mime="application/json",
                use_container_width=True,
            )

            if results.get("redacted_image") is not None:
                _, buf = cv2.imencode(".jpg", results["redacted_image"])
                st.download_button(
                    label="🖼️ Download redacted image (JPG)",
                    data=buf.tobytes(),
                    file_name=f"redacted_{results['timestamp']}.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- FEATURES ROW ---------------------- #

st.markdown("---")
st.markdown("### 🎯 What this pipeline does for your assignment")

c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    "<div style='text-align:center;'><h3>📄</h3><b>Double-page aware</b><br/>Splits left/right pages and extracts per-page fields.</div>",
    unsafe_allow_html=True,
)
c2.markdown(
    "<div style='text-align:center;'><h3>🩺</h3><b>Vitals & diagnosis</b><br/>BP / PR / RR / Temp + mental health summary.</div>",
    unsafe_allow_html=True,
)
c3.markdown(
    "<div style='text-align:center;'><h3>🔐</h3><b>PII extraction</b><br/>Names, IDs, dates, phones for redaction.</div>",
    unsafe_allow_html=True,
)
c4.markdown(
    "<div style='text-align:center;'><h3>✨</h3><b>Preprocessing</b><br/>Deskew, denoise, contrast enhance for doctor handwriting.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='text-align:center;color:#808080;padding:1.5rem;'>Made for OCR Pipeline Assignment – Handwritten PII Extraction</div>",
    unsafe_allow_html=True,
)
