"""
main.py
Advanced OCR + PII / Structured Field Extraction
Optimized for handwritten medical documents (doctor notes, progress reports, charts)
Works with Python 3.12 – Tesseract only (no EasyOCR).
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
import os


# If Tesseract is not in PATH on Windows, uncomment and set this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class MedicalDocumentOCR:
    def __init__(
        self,
        deskew: bool = True,
        denoise: bool = True,
        enhance_contrast: bool = True,
    ):
        """
        OCR + PII extractor focused on handwritten medical documents.
        """
        self.deskew = deskew
        self.denoise = denoise
        self.enhance_contrast = enhance_contrast

        # Generic PII / identifier patterns
        self.pii_patterns = {
            "patient_name": [
                r"Patient\s*Name\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]+?)(?:\s+Age|\s+Sex|\n|$)",
            ],
            "age": [
                r"Age\s*[:\-]?\s*(\d{1,3})\s*(?:yrs?|years?)?",
            ],
            "sex": [
                r"Sex\s*[:\-]?\s*(M|F|Male|Female)",
            ],
            "ipd_number": [
                r"IPD\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\/]+)",
                r"IP\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\/]+)",
            ],
            "uhid_number": [
                r"UHID\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\/]+)",
            ],
            "bed_number": [
                r"Bed\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\/]+)",
            ],
            "date": [
                r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b",
            ],
            "time": [
                r"\b\d{1,2}[:\.]\d{2}\s*(?:AM|PM|am|pm)?\b",
                r"\b\d{1,2}\s*(?:AM|PM|am|pm)\b",
            ],
            "phone": [
                r"(?:\+91[\-\s]?)?[6-9]\d{9}",
            ],
            "email": [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            ],
        }

    # ---------------------- IMAGE PREPROCESSING ---------------------- #

    def _deskew_image(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_inv = cv2.bitwise_not(gray)
        thresh = cv2.threshold(
            gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]

        coords = np.column_stack(np.where(thresh > 0))
        if coords.size == 0:
            return img

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:
            return img

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def _preprocess_page(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Create multiple processed versions of a single page for OCR ensemble.
        """
        if self.deskew:
            img = self._deskew_image(img)

        if self.denoise:
            img = cv2.fastNlMeansDenoisingColored(
                img, None, 10, 10, 7, 21
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        processed_images: List[np.ndarray] = []

        # Adaptive threshold (handles uneven lighting)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )
        processed_images.append(adaptive)

        # Otsu
        _, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        processed_images.append(otsu)

        # Morph close
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        processed_images.append(morph)

        # Contrast enhancement (CLAHE)
        if self.enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, enhanced_thresh = cv2.threshold(
                enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            processed_images.append(enhanced_thresh)

        # Slight dilation for broken strokes
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        dilated = cv2.dilate(otsu, kernel_dilate, iterations=1)
        processed_images.append(dilated)

        return processed_images

    # ---------------------- OCR CORE ---------------------- #

    def _extract_text_tesseract(self, processed_images: List[np.ndarray]) -> str:
        """
        Extract text from multiple processed variants using several Tesseract configs.
        """
        all_texts: List[str] = []

        configs = [
            "--psm 6 --oem 3 -c preserve_interword_spaces=1",
            "--psm 4 --oem 3",
            "--psm 3 --oem 3",
            "--psm 11 --oem 3",
            "--psm 12 --oem 3",
        ]

        for img in processed_images:
            for cfg in configs:
                try:
                    txt = pytesseract.image_to_string(img, lang="eng", config=cfg)
                    if txt and txt.strip():
                        all_texts.append(txt)
                except Exception:
                    continue

        combined = "\n".join(all_texts)
        return combined

    # ---------------------- TEXT CLEANING ---------------------- #

    def clean_text(self, text: str) -> str:
        """
        Clean raw OCR text:
        - normalize newlines
        - remove noise characters
        - collapse spaces
        """
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = []
        for raw in text.split("\n"):
            # keep only allowed chars
            line = re.sub(r"[^\w\s\.\,\:\-\(\)\/\@\+]", "", raw)
            line = re.sub(r"\s+", " ", line).strip()
            if len(line) > 2:
                lines.append(line)
        return "\n".join(lines)

    # ---------------------- STRUCTURED EXTRACTION ---------------------- #

    def _extract_header_metadata(self, text: str) -> Dict[str, Any]:
        """
        Hospital name, location, form code from header.
        """
        data: Dict[str, Any] = {
            "hospital_name": None,
            "location": None,
            "form_code": None,
        }

        lines = text.split("\n")

        for i, line in enumerate(lines):
            low = line.lower()
            if "institute of medical sciences" in low or "sum hospital" in low:
                data["hospital_name"] = line.strip()
                # next 1–2 lines often contain university + location
                if i + 1 < len(lines):
                    loc_line = lines[i + 1].strip()
                    if any(x in loc_line.lower() for x in ["bhubaneswar", "kalinga"]):
                        data["location"] = loc_line
                if i + 2 < len(lines) and data["location"] is None:
                    loc_line2 = lines[i + 2].strip()
                    if any(x in loc_line2.lower() for x in ["bhubaneswar", "kalinga"]):
                        data["location"] = loc_line2

        # Form code
        m = re.search(
            r"FORM\s*NO\.?\s*([A-Za-z0-9\-\/]+)", text, re.IGNORECASE
        )
        if m:
            data["form_code"] = m.group(1)

        return data

    def _extract_patient_block(self, text: str) -> Dict[str, Any]:
        """
        Patient name, age, sex, IPD, UHID, Bed, date, time.
        """
        block: Dict[str, Any] = {
            "patient_name": None,
            "age": None,
            "sex": None,
            "ipd_number": None,
            "uhid_number": None,
            "bed_number": None,
            "date": None,
            "time": None,
        }

        # Patient name
        for pattern in self.pii_patterns["patient_name"]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                block["patient_name"] = m.group(1).strip()
                break

        # Age
        for pattern in self.pii_patterns["age"]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                block["age"] = m.group(1).strip()
                break

        # Sex
        for pattern in self.pii_patterns["sex"]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                block["sex"] = val.upper()[0]  # M / F
                break

        # IPD, UHID, Bed
        for key in ["ipd_number", "uhid_number", "bed_number"]:
            for pattern in self.pii_patterns[key]:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    block[key] = m.group(1).strip()
                    break

        # Date – first date we see on page
        for pattern in self.pii_patterns["date"]:
            m = re.search(pattern, text)
            if m:
                block["date"] = m.group(0)
                break

        # Time – first time we see on page
        for pattern in self.pii_patterns["time"]:
            m = re.search(pattern, text)
            if m:
                block["time"] = m.group(0)
                break

        return block

    def _extract_vitals(self, text: str) -> Dict[str, Any]:
        """
        BP, PR, RR, Temp from vitals block.
        """
        vitals = {"bp": None, "pr": None, "rr": None, "temp": None}

        m = re.search(
            r"BP\s*[:\-]?\s*([0-9]{2,3}\s*\/\s*[0-9]{2,3})", text, re.IGNORECASE
        )
        if m:
            vitals["bp"] = m.group(1).replace(" ", "")

        m = re.search(
            r"PR\s*[:\-]?\s*([0-9]{2,3})", text, re.IGNORECASE
        )
        if m:
            vitals["pr"] = m.group(1)

        m = re.search(
            r"RR\s*[:\-]?\s*([0-9]{1,3})", text, re.IGNORECASE
        )
        if m:
            vitals["rr"] = m.group(1)

        m = re.search(
            r"(?:T|Temp)\s*[:\-]?\s*([0-9]{2,3}\s*(?:[CF]|°C|°F)?)",
            text,
            re.IGNORECASE,
        )
        if m:
            vitals["temp"] = m.group(1).replace(" ", "")

        return vitals

    def _extract_clinical_summary(self, text: str) -> str:
        """
        Pull out the key diagnosis / narrative (e.g. 'Mental and behavioural disorder...').
        """
        lines = text.split("\n")
        key_lines: List[str] = []

        for line in lines:
            low = line.lower()
            if any(
                kw in low
                for kw in [
                    "mental and behavioural",
                    "behavioural disorder",
                    "dependence syndrome",
                    "diagnosis",
                    "alcohol",
                ]
            ):
                key_lines.append(line.strip())

        # Fallback: first long line from PROGRESS NOTES section
        if not key_lines:
            capture = False
            for line in lines:
                low = line.lower()
                if "progress notes" in low:
                    capture = True
                    continue
                if capture and len(line) > 40:
                    key_lines.append(line.strip())
                    if len(key_lines) >= 3:
                        break

        return " ".join(key_lines) if key_lines else ""

    # ---------------------- GENERIC PII FOR REDACTION ---------------------- #

    def extract_pii_for_redaction(self, text: str) -> Dict[str, List[str]]:
        """
        Generic PII list for redaction (names, IDs, dates, phones, emails).
        """
        pii_data: Dict[str, List[str]] = {
            "patient_name": [],
            "age": [],
            "sex": [],
            "ipd_number": [],
            "uhid_number": [],
            "bed_number": [],
            "dates": [],
            "times": [],
            "phone_numbers": [],
            "emails": [],
        }

        for type_key, patterns in self.pii_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                    if match.groups():
                        value = match.group(1).strip()
                    else:
                        value = match.group(0).strip()

                    if not value:
                        continue

                    if type_key == "date":
                        pii_data["dates"].append(value)
                    elif type_key == "time":
                        pii_data["times"].append(value)
                    elif type_key == "phone":
                        pii_data["phone_numbers"].append(value)
                    elif type_key == "email":
                        pii_data["emails"].append(value)
                    else:
                        if type_key in pii_data:
                            pii_data[type_key].append(value)

        # deduplicate
        for key in pii_data:
            pii_data[key] = sorted(list(set(pii_data[key])))

        return pii_data

    # ---------------------- REDACTED IMAGE ---------------------- #

    def create_redacted_image(
        self, image_path: str, pii_strings: List[str]
    ) -> np.ndarray | None:
        """
        Create a redacted image where any OCR word matching PII strings is blacked out.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Basic preprocessing once for bbox detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        try:
            data = pytesseract.image_to_data(
                thresh, output_type=pytesseract.Output.DICT
            )
        except Exception:
            return img

        pii_lower = [p.lower() for p in pii_strings if p]

        for i, word in enumerate(data["text"]):
            if not word or word.strip() == "":
                continue
            if data["conf"][i] < 40:
                continue

            w_low = word.lower()
            should_redact = any(
                w_low in pii_val or pii_val in w_low for pii_val in pii_lower
            )
            if should_redact:
                x, y, w, h = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

        return img

    # ---------------------- MAIN ENTRY ---------------------- #

    def process_document(self, image_path: str) -> Dict[str, Any]:
        """
        Full pipeline:
        - detect single vs double page
        - OCR per page
        - clean text
        - structured field extraction per page
        - global PII list
        - optional redacted image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        color_img = cv2.imread(image_path)
        if color_img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w = color_img.shape[:2]

        # Heuristic: if width >> height, assume left/right pages.
        pages: List[Tuple[str, np.ndarray]] = []
        if w > 1.1 * h:
            mid = w // 2
            left = color_img[:, :mid]
            right = color_img[:, mid:]
            pages.append(("left", left))
            pages.append(("right", right))
        else:
            pages.append(("single", color_img))

        page_results: List[Dict[str, Any]] = []
        combined_cleaned_text_parts: List[str] = []

        for side, page_img in pages:
            processed_variants = self._preprocess_page(page_img)
            raw_text = self._extract_text_tesseract(processed_variants)
            cleaned_text = self.clean_text(raw_text)

            header_meta = self._extract_header_metadata(cleaned_text)
            patient_block = self._extract_patient_block(cleaned_text)
            vitals = self._extract_vitals(cleaned_text)
            clinical_summary = self._extract_clinical_summary(cleaned_text)

            page_info = {
                "side": side,
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "fields": {
                    **header_meta,
                    **patient_block,
                    "vitals": vitals,
                    "clinical_summary": clinical_summary,
                },
            }
            page_results.append(page_info)
            combined_cleaned_text_parts.append(
                f"[{side.upper()} PAGE]\n{cleaned_text}"
            )

        combined_cleaned_text = "\n\n".join(combined_cleaned_text_parts)

        # Global PII for redaction
        pii_for_redaction = self.extract_pii_for_redaction(combined_cleaned_text)

        # Flatten PII strings for redaction
        pii_strings_for_image: List[str] = []
        for vals in pii_for_redaction.values():
            pii_strings_for_image.extend(vals)

        redacted_image = self.create_redacted_image(
            image_path, pii_strings_for_image
        )

        timestamp = datetime.now().isoformat()

        summary = {
            "total_pii_found": sum(len(v) for v in pii_for_redaction.values()),
            "pii_categories": [
                key for key, vals in pii_for_redaction.items() if vals
            ],
            "pages_detected": len(pages),
        }

        results: Dict[str, Any] = {
            "timestamp": timestamp,
            "image_path": image_path,
            "pages": page_results,
            "cleaned_text": combined_cleaned_text,
            "pii_extracted": pii_for_redaction,
            "redacted_image": redacted_image,
            "summary": summary,
        }

        return results


# Simple CLI test
def test_pipeline(image_path: str):
    ocr = MedicalDocumentOCR()
    results = ocr.process_document(image_path)

    print("\n" + "=" * 80)
    print("CLEANED TEXT (first 800 chars)")
    print("=" * 80)
    print(results["cleaned_text"][:800])
    print("\n" + "=" * 80)
    print("PII / STRUCTURED DATA")
    print("=" * 80)
    print(json.dumps(results["pii_extracted"], indent=2))
    print("\nSUMMARY:")
    print(json.dumps(results["summary"], indent=2))

    for page in results["pages"]:
        print("\n--- PAGE:", page["side"], "---")
        print(json.dumps(page["fields"], indent=2))

    return results


if __name__ == "__main__":
    # Replace with a real JPEG path when testing locally
    sample = "sample_page.jpg"
    if os.path.exists(sample):
        test_pipeline(sample)
    else:
        print("Put a test image as 'sample_page.jpg' to run main.py directly.")
