"""
app.py - Streamlit Frontend for Medical OCR System
Advanced Dark UI/UX Dashboard
Author: Aryan Patel
Institution: IIIT Manipur, B.Tech CSE (Pre-final Year)
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Import backend OCR processor
from main import MedicalDocumentOCR

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Medical OCR Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - DARK THEME
# ============================================================================

def load_custom_css():
    st.markdown("""
    <style>
    /* ========== GLOBAL STYLES ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3e 50%, #0f1629 100%);
        color: #e8eaed;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d3e 0%, #0f1629 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e8eaed;
    }
    
    /* ========== HEADERS ========== */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* ========== CARDS & CONTAINERS ========== */
    .custom-card {
        background: rgba(26, 29, 62, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 12px 48px rgba(99, 102, 241, 0.2);
        transform: translateY(-2px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        transform: scale(1.03);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    
    .metric-label {
        color: #9ca3af;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    /* ========== BUTTONS ========== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* ========== FILE UPLOADER ========== */
    [data-testid="stFileUploader"] {
        background: rgba(26, 29, 62, 0.4);
        border: 2px dashed rgba(99, 102, 241, 0.5);
        border-radius: 16px;
        padding: 32px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.8);
        background: rgba(26, 29, 62, 0.6);
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26, 29, 62, 0.4);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #9ca3af;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.2);
        color: #e8eaed;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {
        background: rgba(26, 29, 62, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        color: #e8eaed !important;
        font-weight: 600;
        padding: 16px;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    .streamlit-expanderContent {
        background: rgba(15, 22, 41, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-top: none;
        border-radius: 0 0 12px 12px;
        padding: 20px;
    }
    
    /* ========== DATA DISPLAY ========== */
    .element-container div[data-testid="stMarkdownContainer"] p {
        color: #e8eaed;
        line-height: 1.7;
    }
    
    code {
        background: rgba(99, 102, 241, 0.2);
        color: #a78bfa;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.9em;
    }
    
    pre {
        background: rgba(15, 22, 41, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 16px;
    }
    
    /* ========== TABLE STYLING ========== */
    .dataframe {
        background: rgba(26, 29, 62, 0.6) !important;
        color: #e8eaed !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
    }
    
    .dataframe th {
        background: rgba(99, 102, 241, 0.3) !important;
        color: white !important;
        font-weight: 600;
        padding: 12px !important;
    }
    
    .dataframe td {
        padding: 12px !important;
        border-bottom: 1px solid rgba(99, 102, 241, 0.1) !important;
    }
    
    /* ========== STATUS BADGES ========== */
    .status-success {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    
    .status-warning {
        background: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
        border: 1px solid rgba(251, 191, 36, 0.4);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    /* ========== PROGRESS BAR ========== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* ========== ALERTS ========== */
    .stAlert {
        background: rgba(26, 29, 62, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 12px;
        color: #e8eaed;
    }
    
    /* ========== SELECTBOX & INPUT ========== */
    .stSelectbox > div > div {
        background: rgba(26, 29, 62, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
        color: #e8eaed;
    }
    
    .stTextInput > div > div > input {
        background: rgba(26, 29, 62, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
        color: #e8eaed;
    }
    
    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 22, 41, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* ========== CUSTOM COMPONENTS ========== */
    .field-row {
        display: flex;
        justify-content: space-between;
        padding: 12px 16px;
        margin: 6px 0;
        background: rgba(15, 22, 41, 0.6);
        border-radius: 8px;
        border-left: 3px solid rgba(99, 102, 241, 0.6);
        transition: all 0.2s ease;
    }
    
    .field-row:hover {
        background: rgba(99, 102, 241, 0.15);
        border-left-color: #667eea;
    }
    
    .field-label {
        color: #9ca3af;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .field-value {
        color: #e8eaed;
        font-weight: 500;
        font-size: 1rem;
    }
    
    .pii-tag {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }
    
    /* ========== ANIMATIONS ========== */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .custom-card, .metric-card {
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    .processing-indicator {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* ========== RESPONSIVE ========== */
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem !important;
        }
        
        .custom-card {
            padding: 16px;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def render_metric_card(label, value, icon="📊"):
    """Render a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def render_field_row(label, value):
    """Render a field row with label and value"""
    if value is None or value == "":
        value = "—"
    st.markdown(f"""
    <div class="field-row">
        <span class="field-label">{label}</span>
        <span class="field-value">{value}</span>
    </div>
    """, unsafe_allow_html=True)

def render_status_badge(status, label):
    """Render a status badge"""
    st.markdown(f'<span class="status-{status}">{label}</span>', unsafe_allow_html=True)

def create_pii_chart(pii_data):
    """Create a visualization of PII categories found"""
    categories = []
    counts = []
    
    for key, values in pii_data.items():
        if values:
            categories.append(key.replace('_', ' ').title())
            counts.append(len(values))
    
    if not categories:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=counts,
            y=categories,
            orientation='h',
            marker=dict(
                color=counts,
                colorscale='Purples',
                line=dict(color='rgba(99, 102, 241, 0.8)', width=2)
            ),
            text=counts,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="PII Categories Distribution",
        xaxis_title="Count",
        yaxis_title="Category",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8eaed'),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_processing_summary_chart(summary):
    """Create a pie chart of processing summary"""
    labels = ['Total PII Found', 'Pages Detected', 'Categories']
    values = [
        summary.get('total_pii_found', 0),
        summary.get('pages_detected', 0),
        len(summary.get('pii_categories', []))
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker=dict(
            colors=['#667eea', '#764ba2', '#a78bfa'],
            line=dict(color='rgba(26, 29, 62, 0.8)', width=2)
        )
    )])
    
    fig.update_layout(
        title="Processing Summary",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8eaed'),
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    load_custom_css()
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🏥 Medical OCR Intelligence</h1>
        <p style="color: #9ca3af; font-size: 1.1rem; margin-top: -10px;">
            Advanced Handwritten Medical Document Analysis with AI-Powered Extraction
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        st.markdown("### Processing Options")
        deskew = st.checkbox("📐 Deskew Images", value=True, 
                            help="Automatically correct skewed/tilted images")
        denoise = st.checkbox("🔊 Denoise", value=True,
                             help="Remove noise from images for better OCR")
        enhance_contrast = st.checkbox("🌟 Enhance Contrast", value=True,
                                      help="Improve text visibility")
        
        st.markdown("### AI Enhancement")
        use_llm = st.checkbox("🤖 Use LLM Post-Processing", value=False,
                             help="Use Mistral AI to refine and correct extracted data")
        
        if use_llm:
            st.info("💡 Ensure MISTRAL_API_KEY is set in your .env file")
        
        st.markdown("---")
        st.markdown("### 📊 System Info")
        st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption("⚡ Powered by Tesseract OCR")
        if use_llm:
            st.caption("🧠 Enhanced by Mistral AI")
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developer")
        st.caption("**Aryan Patel**")
        st.caption("IIIT Manipur")
        st.caption("B.Tech CSE (Pre-final Year)")
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Medical Document")
        
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Quick Guide")
        st.markdown("""
        1. Upload a medical document
        2. Configure processing options
        3. Click 'Process Document'
        4. Review extracted data
        5. Download results
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Uploaded Document")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process button
        if st.button("🚀 Process Document", use_container_width=True):
            with st.spinner("🔄 Processing document... This may take a moment."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Initialize OCR processor
                    ocr = MedicalDocumentOCR(
                        deskew=deskew,
                        denoise=denoise,
                        enhance_contrast=enhance_contrast,
                        use_llm=use_llm
                    )
                    
                    # Process document
                    progress_bar = st.progress(0)
                    progress_bar.progress(25, "📄 Reading document...")
                    
                    results = ocr.process_document(tmp_path)
                    progress_bar.progress(70, "🔍 Extracting data...")
                    
                    if use_llm:
                        results = ocr.enhance_with_llm(results)
                        progress_bar.progress(90, "🤖 AI refinement...")
                    
                    progress_bar.progress(100, "✅ Complete!")
                    progress_bar.empty()
                    
                    # Store results in session state
                    st.session_state['results'] = results
                    st.session_state['tmp_path'] = tmp_path
                    
                    st.success("✅ Document processed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
                    st.exception(e)
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Summary Metrics
        st.markdown('<div class="section-header">📊 Processing Summary</div>', unsafe_allow_html=True)
        
        summary = results.get('summary', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_metric_card(
                "Total PII Found",
                summary.get('total_pii_found', 0),
                "🔒"
            )
        
        with col2:
            render_metric_card(
                "Pages Detected",
                summary.get('pages_detected', 0),
                "📄"
            )
        
        with col3:
            render_metric_card(
                "PII Categories",
                len(summary.get('pii_categories', [])),
                "📂"
            )
        
        with col4:
            render_metric_card(
                "Status",
                "Success" if summary.get('total_pii_found', 0) > 0 else "Complete",
                "✅"
            )
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Structured Data",
            "🔒 PII Detection",
            "📝 Raw Text",
            "🖼️ Redacted Image",
            "📥 Download"
        ])
        
        with tab1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            
            # Choose data source
            data_source = "LLM-Enhanced" if results.get('llm_enhanced') and use_llm else "OCR"
            st.markdown(f"### Data Source: **{data_source}**")
            
            if results.get('llm_enhanced') and use_llm:
                # Display LLM-enhanced data
                llm_data = results['llm_enhanced']
                
                for page_idx, page in enumerate(llm_data.get('pages', [])):
                    with st.expander(f"📄 Page: {page.get('side', 'Unknown').upper()}", expanded=True):
                        st.markdown("#### 🏥 Hospital Information")
                        render_field_row("Hospital Name", page.get('hospital_name'))
                        render_field_row("Location", page.get('location'))
                        render_field_row("Form Code", page.get('form_code'))
                        
                        st.markdown("#### 👤 Patient Information")
                        render_field_row("Patient Name", page.get('patient_name'))
                        render_field_row("Age", page.get('age'))
                        render_field_row("Sex", page.get('sex'))
                        render_field_row("IPD Number", page.get('ipd_number'))
                        render_field_row("UHID Number", page.get('uhid_number'))
                        render_field_row("Bed Number", page.get('bed_number'))
                        
                        st.markdown("#### 📅 Date & Time")
                        render_field_row("Date", page.get('date'))
                        render_field_row("Time", page.get('time'))
                        
                        st.markdown("#### 💉 Vital Signs")
                        vitals = page.get('vitals', {})
                        render_field_row("Blood Pressure", vitals.get('bp'))
                        render_field_row("Pulse Rate", vitals.get('pr'))
                        render_field_row("Respiratory Rate", vitals.get('rr'))
                        render_field_row("Temperature", vitals.get('temp'))
                        
                        st.markdown("#### 📋 Clinical Summary")
                        summary_text = page.get('clinical_summary', 'No summary available')
                        st.text_area("", summary_text, height=150, key=f"clinical_{page_idx}")
            
            else:
                # Display OCR data
                for page_idx, page in enumerate(results.get('pages', [])):
                    with st.expander(f"📄 Page: {page.get('side', 'Unknown').upper()}", expanded=True):
                        fields = page.get('fields', {})
                        
                        st.markdown("#### 🏥 Hospital Information")
                        render_field_row("Hospital Name", fields.get('hospital_name'))
                        render_field_row("Location", fields.get('location'))
                        render_field_row("Form Code", fields.get('form_code'))
                        
                        st.markdown("#### 👤 Patient Information")
                        render_field_row("Patient Name", fields.get('patient_name'))
                        render_field_row("Age", fields.get('age'))
                        render_field_row("Sex", fields.get('sex'))
                        render_field_row("IPD Number", fields.get('ipd_number'))
                        render_field_row("UHID Number", fields.get('uhid_number'))
                        render_field_row("Bed Number", fields.get('bed_number'))
                        
                        st.markdown("#### 📅 Date & Time")
                        render_field_row("Date", fields.get('date'))
                        render_field_row("Time", fields.get('time'))
                        
                        st.markdown("#### 💉 Vital Signs")
                        vitals = fields.get('vitals', {})
                        render_field_row("Blood Pressure", vitals.get('bp'))
                        render_field_row("Pulse Rate", vitals.get('pr'))
                        render_field_row("Respiratory Rate", vitals.get('rr'))
                        render_field_row("Temperature", vitals.get('temp'))
                        
                        st.markdown("#### 📋 Clinical Summary")
                        summary_text = fields.get('clinical_summary', 'No summary available')
                        st.text_area("", summary_text, height=150, key=f"ocr_clinical_{page_idx}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 🔒 Detected PII (Personally Identifiable Information)")
            
            pii_data = results.get('pii_extracted', {})
            
            # Create visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                chart = create_pii_chart(pii_data)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("No PII detected in document")
            
            with col2:
                st.markdown("#### PII Categories")
                for category in summary.get('pii_categories', []):
                    st.markdown(f'<div class="pii-tag">{category.replace("_", " ").title()}</div>', 
                              unsafe_allow_html=True)
            
            # Detailed PII list
            st.markdown("#### Detailed PII Extraction")
            for pii_type, values in pii_data.items():
                if values:
                    with st.expander(f"🔐 {pii_type.replace('_', ' ').title()} ({len(values)})"):
                        for val in values:
                            st.markdown(f"- `{val}`")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 📝 Extracted Text (Cleaned)")
            
            cleaned_text = results.get('cleaned_text', '')
            st.text_area("", cleaned_text, height=500)
            
            st.markdown("#### Text Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                render_metric_card("Total Characters", len(cleaned_text), "📊")
            
            with col2:
                word_count = len(cleaned_text.split())
                render_metric_card("Word Count", word_count, "📝")
            
            with col3:
                line_count = len(cleaned_text.split('\n'))
                render_metric_card("Line Count", line_count, "📄")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 🖼️ Redacted Document (PII Removed)")
            
            redacted_img = results.get('redacted_image')
            
            if redacted_img is not None:
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(redacted_img, channels="BGR", use_container_width=True)
                
                st.info("🔒 All detected PII has been redacted (blacked out) from the image")
            else:
                st.warning("⚠️ Redacted image not available")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 📥 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON download
                json_str = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # CSV download for structured data
                if results.get('pages'):
                    df_data = []
                    
                    if results.get('llm_enhanced') and use_llm:
                        for page in results['llm_enhanced'].get('pages', []):
                            row = {
                                'Side': page.get('side'),
                                'Hospital': page.get('hospital_name'),
                                'Patient Name': page.get('patient_name'),
                                'Age': page.get('age'),
                                'Sex': page.get('sex'),
                                'IPD': page.get('ipd_number'),
                                'UHID': page.get('uhid_number'),
                                'BP': page.get('vitals', {}).get('bp'),
                                'PR': page.get('vitals', {}).get('pr'),
                            }
                            df_data.append(row)
                    else:
                        for page in results.get('pages', []):
                            fields = page.get('fields', {})
                            row = {
                                'Side': page.get('side'),
                                'Hospital': fields.get('hospital_name'),
                                'Patient Name': fields.get('patient_name'),
                                'Age': fields.get('age'),
                                'Sex': fields.get('sex'),
                                'IPD': fields.get('ipd_number'),
                                'UHID': fields.get('uhid_number'),
                                'BP': fields.get('vitals', {}).get('bp'),
                                'PR': fields.get('vitals', {}).get('pr'),
                            }
                            df_data.append(row)
                    
                    df = pd.DataFrame(df_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv,
                        file_name=f"ocr_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Redacted image download
            if results.get('redacted_image') is not None:
                st.markdown("#### Download Redacted Image")
                
                # Convert numpy array to bytes
                redacted_img = results['redacted_image']
                is_success, buffer = cv2.imencode(".jpg", redacted_img)
                
                if is_success:
                    st.download_button(
                        label="🖼️ Download Redacted Image",
                        data=buffer.tobytes(),
                        file_name=f"redacted_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9ca3af; padding: 20px;">
        <p>🏥 Medical OCR Intelligence System</p>
        <p style="font-size: 0.875rem;">
            Developed by <strong>Aryan Patel</strong> | IIIT Manipur | B.Tech CSE (Pre-final Year)
        </p>
        <p style="font-size: 0.875rem;">
            Powered by Tesseract OCR & Mistral AI | Built with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()