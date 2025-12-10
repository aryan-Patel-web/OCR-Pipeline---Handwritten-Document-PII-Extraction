# 🏥 Medical OCR Intelligence System

**Advanced Handwritten Medical Document Analysis with AI-Powered Extraction**

## 👨‍💻 Developer Information

**Name:** Aryan Patel  
**Institution:** IIIT Manipur  
**Program:** B.Tech Computer Science & Engineering  
**Year:** Pre-Final Year (3rd Year)  
**Project Type:** AI/ML Medical Document Processing System

---

## 📌 Project Overview

A sophisticated OCR system designed specifically for handwritten medical documents (doctor notes, progress reports, patient charts) with automatic PII detection, structured field extraction, and optional AI-powered refinement using Mistral LLM.

### Key Features

✅ **Multi-Stage Image Preprocessing** (Deskewing, Denoising, Contrast Enhancement)  
✅ **Ensemble OCR** with multiple Tesseract configurations  
✅ **Structured Field Extraction** (Patient info, Vitals, Clinical summary)  
✅ **PII Detection & Redaction** (Names, IDs, Dates, Phone numbers)  
✅ **Dark Theme UI/UX** with 700+ lines responsive Streamlit dashboard  
✅ **Optional LLM Enhancement** using Mistral AI  
✅ **Data Export** (JSON, CSV, Redacted Images)  
✅ **Interactive Visualizations** (Plotly charts)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install streamlit opencv-python numpy pytesseract Pillow plotly pandas python-dotenv mistralai
```

### 2. Install Tesseract OCR

**Windows:** Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)  
**macOS:** `brew install tesseract`  
**Linux:** `sudo apt-get install tesseract-ocr`

### 3. Configure (Optional)

Create `.env` file for LLM features:
```
MISTRAL_API_KEY=your_api_key_here
```

### 4. Run Application

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## 📁 Project Structure

```
medical-ocr-system/
├── app.py              # Streamlit Frontend (700+ lines)
├── main.py             # Backend OCR Engine
├── requirements.txt    # Dependencies
├── .env               # API Keys (optional)
└── README.md          # Documentation
```

---

## 🎯 Usage Workflow

1. **Upload** medical document (JPG/PNG)
2. **Configure** processing options (sidebar)
3. **Process** document with one click
4. **Review** extracted structured data
5. **Download** results (JSON/CSV/Redacted Image)

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **OCR Engine** | Tesseract 4.x |
| **Image Processing** | OpenCV, NumPy |
| **Frontend** | Streamlit |
| **Visualization** | Plotly |
| **AI Enhancement** | Mistral AI (Optional) |
| **Language** | Python 3.12 |

---

## 📊 System Architecture

```
┌─────────────────┐
│ Upload Document │
└────────┬────────┘
         │
    ┌────▼────────────────┐
    │ Image Preprocessing │
    │ • Deskew           │
    │ • Denoise          │
    │ • Enhance Contrast │
    └────────┬────────────┘
             │
    ┌────────▼────────────┐
    │ Ensemble OCR        │
    │ (5 variants x       │
    │  5 configs)         │
    └────────┬────────────┘
             │
    ┌────────▼────────────┐
    │ Text Cleaning &     │
    │ Field Extraction    │
    └────────┬────────────┘
             │
    ┌────────▼────────────┐
    │ PII Detection &     │
    │ Redaction           │
    └────────┬────────────┘
             │
    ┌────────▼────────────┐
    │ Optional: LLM       │
    │ Refinement          │
    └────────┬────────────┘
             │
    ┌────────▼────────────┐
    │ Streamlit Dashboard │
    │ Visualization       │
    └─────────────────────┘
```

---

## 🔐 PII Detection Categories

- Patient Names
- Age
- Sex/Gender
- IPD Numbers
- UHID Numbers
- Bed Numbers
- Dates
- Times
- Phone Numbers
- Email Addresses

---

## 🎨 Dashboard Features

### 📊 Summary Metrics
- Total PII Found
- Pages Detected
- PII Categories
- Processing Status

### 📋 Tabs
1. **Structured Data** - Organized patient information
2. **PII Detection** - Visual charts and detailed lists
3. **Raw Text** - Cleaned OCR output with statistics
4. **Redacted Image** - Privacy-protected document
5. **Download** - Export in multiple formats

---

## 🔮 Future Enhancements

### Phase 1: Advanced Features
- [ ] **Multi-language Support** (Hindi, Bengali, Telugu)
- [ ] **Handwriting Recognition Improvement** using deep learning models
- [ ] **Real-time Processing** with webcam/scanner integration
- [ ] **Batch Processing** for multiple documents
- [ ] **Cloud Storage Integration** (AWS S3, Google Cloud)

### Phase 2: AI/ML Upgrades
- [ ] **Custom Trained OCR Model** specifically for medical handwriting
- [ ] **Named Entity Recognition (NER)** for medical terms
- [ ] **Automated Report Generation** from extracted data
- [ ] **Predictive Analytics** on patient vital trends
- [ ] **Medical Code Mapping** (ICD-10, CPT codes)

### Phase 3: Production Ready
- [ ] **REST API Development** for system integration
- [ ] **Mobile Application** (React Native/Flutter)
- [ ] **Role-Based Access Control** (RBAC)
- [ ] **Audit Logging** and compliance tracking
- [ ] **HIPAA Compliance** features
- [ ] **Database Integration** (PostgreSQL/MongoDB)
- [ ] **Docker Containerization**
- [ ] **Kubernetes Orchestration** for scalability

### Phase 4: Enterprise Features
- [ ] **Hospital Management System Integration**
- [ ] **Electronic Health Record (EHR) Export**
- [ ] **Real-time Collaboration** tools
- [ ] **Advanced Analytics Dashboard**
- [ ] **Machine Learning Model Training Interface**
- [ ] **Automated Quality Checks** and validation
- [ ] **Multi-tenant Architecture**

---

## 📝 Requirements

```txt
streamlit>=1.31.0
opencv-python>=4.8.0
numpy>=1.24.0
pytesseract>=0.3.10
Pillow>=10.0.0
plotly>=5.18.0
pandas>=2.0.0
python-dotenv>=1.0.0
mistralai>=0.1.0
```

---

## 🐛 Troubleshooting

**Issue:** Tesseract not found  
**Solution:** Ensure Tesseract is installed and in PATH, or set path in `main.py` line 34

**Issue:** Poor OCR accuracy  
**Solution:** Enable all preprocessing options (Deskew, Denoise, Enhance Contrast)

**Issue:** LLM not working  
**Solution:** Verify `MISTRAL_API_KEY` is set in `.env` file

---

## 📄 License

This project is developed for educational and research purposes.

---

## 🙏 Acknowledgments

- **Tesseract OCR** - Google's open-source OCR engine
- **Streamlit** - Fast web app framework
- **Mistral AI** - LLM enhancement capabilities
- **IIIT Manipur** - Academic support and guidance

---

## 📧 Contact

**Aryan Patel**  
B.Tech CSE, Pre-Final Year  
IIIT Manipur  

---

**Built with ❤️ for Healthcare Innovation**