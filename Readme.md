# OCR Pipeline - Handwritten Document PII Extraction

> **AI/ML Internship Assignment for Securelytix**

A complete end-to-end pipeline for extracting text and detecting Personal Identifiable Information (PII) from handwritten documents using OCR technology.

## 🎯 Project Overview

This project implements a robust OCR pipeline that:
- Processes handwritten JPEG documents
- Handles tilted/skewed images automatically
- Extracts text with high accuracy
- Detects multiple types of PII
- Optionally generates redacted images

## 🏗️ Architecture

```
┌─────────────────┐      HTTP/REST      ┌──────────────────┐
│   Streamlit     │ ◄─────────────────► │   FastAPI        │
│   Frontend      │                      │   Backend        │
│   (Port 8501)   │                      │   (Port 8000)    │
└─────────────────┘                      └──────────────────┘
                                                   │
                                                   ▼
                                         ┌──────────────────┐
                                         │  OCR Pipeline    │
                                         │  - Preprocessing │
                                         │  - Tesseract     │
                                         │  - PII Detection │
                                         └──────────────────┘
```

## 🚀 Features

### 1. **Image Preprocessing**
- **Deskewing**: Automatically corrects tilted images
- **Denoising**: Removes image noise using Non-Local Means
- **Contrast Enhancement**: CLAHE for better text visibility
- **Binarization**: Adaptive thresholding for optimal OCR
- **Morphological Operations**: Enhances text structure

### 2. **OCR Engine**
- Powered by **Tesseract OCR v4+**
- Optimized configurations for handwriting
- Confidence scoring for reliability assessment
- Multiple PSM (Page Segmentation Mode) support

### 3. **PII Detection**
Automatically detects:
- 📧 **Email Addresses** (multiple formats)
- 📞 **Phone Numbers** (US & International formats)
- 🔐 **Social Security Numbers** (XXX-XX-XXXX)
- 📅 **Dates** (various formats)
- 👤 **Names** (capitalized patterns)
- 🏠 **Addresses** (street addresses)

### 4. **Image Redaction** (Optional)
- Automatically blacks out detected PII
- Preserves original image structure
- Downloadable redacted output

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- Tesseract OCR installed on system

### Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

Verify installation:
```bash
tesseract --version
```

## 🛠️ Installation & Setup

### Step 1: Clone/Download the Project
```bash
# Create project directory
mkdir ocr-pii-pipeline
cd ocr-pii-pipeline
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
# Test Tesseract
tesseract --version

# Test OpenCV
python -c "import cv2; print(cv2.__version__)"

# Test FastAPI
python -c "import fastapi; print('FastAPI OK')"

# Test Streamlit
python -c "import streamlit; print('Streamlit OK')"
```

## 🎮 Usage

### Starting the Application

**Terminal 1 - Start Backend (FastAPI):**
```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Terminal 2 - Start Frontend (Streamlit):**
```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start Streamlit app
streamlit run app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Using the Application

1. **Open Browser**: Navigate to `http://localhost:8501`
2. **Upload Image**: Click "Browse files" and select a handwritten document (JPEG/PNG)
3. **Configure**: (Optional) Enable "Generate Redacted Image" in sidebar
4. **Process**: Click "Extract & Analyze" button
5. **Review Results**: 
   - View extracted text
   - Check detected PII items
   - Download redacted image (if enabled)
   - Export results as JSON

## 📚 API Documentation

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "tesseract": "OK",
  "api_version": "1.0.0"
}
```

#### 2. Extract PII
```http
POST /ocr/extract?redact=false
Content-Type: multipart/form-data

file: <image_file>
```

**Parameters:**
- `file` (required): Image file (JPEG/PNG)
- `redact` (optional): Boolean, default=false

**Response:**
```json
{
  "success": true,
  "extracted_text": "Patient Name: John Doe...",
  "pii_detected": {
    "emails": ["john@example.com"],
    "phone_numbers": ["555-123-4567"],
    "names": ["John Doe"],
    "dates": ["12/25/2024"]
  },
  "confidence_score": 0.85,
  "redacted_image": "base64_encoded_image..."
}
```

### Interactive API Docs
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Testing

### Test with Sample Images

```bash
# Using curl
curl -X POST "http://localhost:8000/ocr/extract?redact=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_document.jpg"
```

### Test Data Format
Create sample handwritten documents with:
- Patient names
- Contact information
- Dates
- Medical records
- Addresses

## 🎓 Technical Implementation Details

### Image Preprocessing Pipeline
```python
1. Grayscale Conversion → 2. Deskewing → 3. Denoising
     ↓
4. CLAHE Enhancement → 5. Adaptive Binarization → 6. Morphological Ops
```

### PII Detection Strategy
Uses **regex patterns** with multiple variations:
- Email: `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}`
- Phone: Multiple patterns for different formats
- SSN: `\d{3}-\d{2}-\d{4}`
- Names: Capitalized word patterns with filtering
- Dates: Multiple date format patterns
- Addresses: Street address patterns

### Confidence Scoring
- Calculated from Tesseract's per-word confidence
- Averaged across all detected words
- Ranges from 0.0 to 1.0
- Thresholds:
  - High: ≥ 0.7
  - Medium: 0.5 - 0.7
  - Low: < 0.5

## 📊 Project Structure

```
ocr-pii-pipeline/
│
├── backend.py              # FastAPI backend server
├── frontend.py               # Streamlit frontend application
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
│
├── samples/            # Test images (create this folder)
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── sample3.jpg
│
└── venv/               # Virtual environment (created during setup)
```

## 🔧 Configuration

### Environment Variables (Optional)
Create `.env` file:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Tesseract Configuration
TESSERACT_CMD=/usr/bin/tesseract

# Processing Options
MAX_IMAGE_SIZE=5242880  # 5MB
ENABLE_LOGGING=true
```

### Tesseract Configuration
Customize in `main.py`:
```python
custom_config = r'--oem 3 --psm 6'
# OEM 3: Default + LSTM
# PSM 6: Assume uniform block of text
```

## 🐛 Troubleshooting

### Common Issues

**1. Tesseract Not Found**
```bash
# Set Tesseract path explicitly
export TESSERACT_CMD=/usr/local/bin/tesseract
# Or in Python:
pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'
```

**2. Low OCR Accuracy**
- Ensure image is high resolution (≥ 300 DPI)
- Check image is not too tilted
- Verify handwriting is clear
- Try different PSM modes

**3. API Connection Error**
- Verify backend is running: `curl http://localhost:8000/health`
- Check firewall settings
- Ensure correct port numbers

**4. Memory Issues**
- Reduce image size before processing
- Process images sequentially
- Increase system RAM allocation

## 📈 Performance Optimization

### Tips for Better Results
1. **Image Quality**: Use high-resolution scans (300+ DPI)
2. **Lighting**: Ensure uniform lighting, no shadows
3. **Orientation**: Pre-rotate extremely tilted images
4. **Handwriting**: Clear, well-spaced writing works best
5. **Background**: White/light backgrounds recommended

### Processing Speed
- Average: 2-5 seconds per image
- Depends on: image size, complexity, hardware

## 🔒 Security Considerations

- **PII Handling**: All processing done locally
- **No Data Storage**: No images/data stored on server
- **Session Isolated**: Each request processed independently
- **Redaction**: PII permanently removed from images

## 📝 Assignment Submission

### What to Submit
1. ✅ Complete source code (main.py, app.py)
2. ✅ requirements.txt
3. ✅ README.md (this file)
4. ✅ Screenshots/demo video (optional)
5. ✅ Processed sample outputs

### Submission Format
```
ocr-pii-pipeline.zip
├── main.py
├── app.py
├── requirements.txt
├── README.md
└── samples/ (with test results)
```

## 🎯 Key Highlights for Interview

### Technical Skills Demonstrated
- **Backend Development**: FastAPI, REST API design
- **Frontend Development**: Streamlit, responsive UI
- **Computer Vision**: OpenCV, image preprocessing
- **Machine Learning**: OCR, pattern recognition
- **Data Processing**: Regex, text parsing
- **Software Engineering**: Clean code, documentation

### Problem-Solving Approach
1. **Modular Design**: Separate preprocessing, OCR, PII detection
2. **Error Handling**: Comprehensive exception management
3. **User Experience**: Clear feedback, progress indicators
4. **Scalability**: API-based architecture
5. **Documentation**: Detailed README and code comments

## 📧 Contact & Support

**Assignment by:** Aryan Patel
**Email:** aryan914078@gmail.com
**Company:** Securelytix
**Position:** AI/ML Intern

---

## 🙏 Acknowledgments

- **Tesseract OCR**: Google's open-source OCR engine
- **FastAPI**: Modern web framework
- **Streamlit**: Interactive data apps
- **OpenCV**: Computer vision library

---

**Note**: This is an internship assignment submission for Securelytix. The implementation demonstrates practical skills in OCR, PII detection, and full-stack development with modern Python frameworks.

**Good luck with your interview! 🚀**