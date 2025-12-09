"""
main.py
Hybrid OCR Engine for Medical / Handwritten Documents
-----------------------------------------------------
- Uses Tesseract for printed text.
- Uses TrOCR (microsoft/trocr-base-handwritten) for pen / handwriting.
- Works with double-page hospital scans (left + right page).
- Outputs RAW text per page (no hardcoded Name/Age/etc).
- Also provides simple PII detection + redacted image.
"""

import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# If Tesseract is not in PATH on Windows, set this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class MedicalDocumentOCR:
    def __init__(
        self,
        use_tesseract: bool = True,
        use_trocr: bool = True,
        deskew: bool = True,
        denoise: bool = True,
        enhance_contrast: bool = True,
        trocr_grid_rows: int = 3,
        trocr_grid_cols: int = 2,
    ):
        """
        Hybrid OCR:
        - Tesseract for printed text (headers, labels, typed fields).
        - TrOCR (base) for handwriting (blue/black pen, scribbles).
        """
        self.use_tesseract = use_tesseract
        self.use_trocr = use_trocr
        self.deskew = deskew
        self.denoise = denoise
        self.enhance_contrast = enhance_contrast
        self.trocr_grid_rows = trocr_grid_rows
        self.trocr_grid_cols = trocr_grid_cols

        # ---- Load TrOCR model (base) once ----
        self.trocr_processor: TrOCRProcessor | None = None
        self.trocr_model: VisionEncoderDecoderModel | None = None

        if self.use_trocr:
            try:
                self.trocr_processor = TrOCRProcessor.from_pretrained(
                    "microsoft/trocr-base-handwritten"
                )
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-base-handwritten"
                )
                self.trocr_model.eval()
                # Use CPU by default (GPU if available)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.trocr_model.to(self.device)
                print("[TrOCR] Loaded microsoft/trocr-base-handwritten on", self.device)
            except Exception as e:
                print(f"[TrOCR] Failed to load model: {e}")
                self.use_trocr = False

        # ---- Simple PII regex for redaction (numbers, phones, emails, IDs) ----
        self.pii_patterns = {
            "phone": r"(?:\+91[\-\s]?)?[6-9]\d{9}",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "id_like": r"\b[0-9]{4,}\b",  # generic numbers (IPD/UHID etc)
            "name_hint": r"(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+[A-Za-z][A-Za-z\s\.]+",
        }

    # ------------------------------------------------------------------
    # IMAGE PREPROCESSING
    # ------------------------------------------------------------------

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

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def _preprocess_for_tesseract(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Create multiple processed versions of a single page for Tesseract.
        """
        if self.deskew:
            img = self._deskew_image(img)

        if self.denoise:
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed: List[np.ndarray] = []

        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 11
        )
        processed.append(adaptive)

        # Otsu
        _, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        processed.append(otsu)

        # Morph close
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        processed.append(morph)

        # CLAHE contrast
        if self.enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, enhanced_thresh = cv2.threshold(
                enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            processed.append(enhanced_thresh)

        return processed

    # ------------------------------------------------------------------
    # TESSERACT OCR
    # ------------------------------------------------------------------

    def _ocr_tesseract(self, variants: List[np.ndarray]) -> str:
        """
        OCR using Tesseract with multiple configs, merge all text.
        """
        if not self.use_tesseract:
            return ""

        all_texts: List[str] = []
        configs = [
            "--psm 6 --oem 3 -c preserve_interword_spaces=1",
            "--psm 4 --oem 3",
            "--psm 11 --oem 3",
        ]

        for img in variants:
            for cfg in configs:
                try:
                    txt = pytesseract.image_to_string(img, lang="eng", config=cfg)
                    if txt and txt.strip():
                        all_texts.append(txt)
                except Exception:
                    continue

        return "\n".join(all_texts)

    # ------------------------------------------------------------------
    # TrOCR HANDWRITING OCR
    # ------------------------------------------------------------------

    def _trocr_single_image(self, img_bgr: np.ndarray) -> str:
        """
        Run TrOCR on a single RGB patch (handwriting-aware).
        """
        if not self.use_trocr or self.trocr_processor is None or self.trocr_model is None:
            return ""

        # Slight upscale to help thin pen strokes
        h, w = img_bgr.shape[:2]
        scale = 1.3
        img_resized = cv2.resize(
            img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )

        pil_img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            inputs = self.trocr_processor(pil_img, return_tensors="pt").pixel_values
            inputs = inputs.to(self.device)
            generated_ids = self.trocr_model.generate(
                inputs,
                max_length=256,
                num_beams=3,
            )
            text = self.trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
        return text.strip()

    def _ocr_trocr_grid(self, page_bgr: np.ndarray) -> str:
        """
        Split page into a grid and run TrOCR on each cell.
        This dramatically increases recall for handwriting.
        """
        if not self.use_trocr:
            return ""

        h, w = page_bgr.shape[:2]
        rows = self.trocr_grid_rows
        cols = self.trocr_grid_cols
        tile_h = h // rows
        tile_w = w // cols

        texts: List[str] = []

        for r in range(rows):
            for c in range(cols):
                y1 = r * tile_h
                y2 = h if r == rows - 1 else (r + 1) * tile_h
                x1 = c * tile_w
                x2 = w if c == cols - 1 else (c + 1) * tile_w
                tile = page_bgr[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                t = self._trocr_single_image(tile)
                if t:
                    texts.append(t)

        return "\n".join(texts)

    # ------------------------------------------------------------------
    # TEXT CLEANING + PII
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = []
        for raw in text.split("\n"):
            # Keep most characters, just normalize spaces
            line = re.sub(r"\s+", " ", raw).strip()
            if len(line) > 1:
                lines.append(line)
        return "\n".join(lines)

    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Very simple PII detection for redaction.
        """
        pii: Dict[str, List[str]] = {k: [] for k in self.pii_patterns.keys()}

        for key, pattern in self.pii_patterns.items():
            for m in re.finditer(pattern, text, re.IGNORECASE):
                val = m.group(0).strip()
                if val and val not in pii[key]:
                    pii[key].append(val)

        return pii

    # ------------------------------------------------------------------
    # REDACTED IMAGE
    # ------------------------------------------------------------------

    def create_redacted_image(
        self, image_path: str, pii_tokens: List[str]
    ) -> np.ndarray | None:
        """
        Use Tesseract word-level boxes to blackout PII tokens.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        if not pii_tokens:
            return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        try:
            data = pytesseract.image_to_data(
                thr, output_type=pytesseract.Output.DICT
            )
        except Exception:
            return img

        pii_lower = [p.lower() for p in pii_tokens if p]

        for i, word in enumerate(data["text"]):
            if not word or word.strip() == "":
                continue
            if data["conf"][i] < 40:
                continue

            w_low = word.lower()
            if any(w_low in p or p in w_low for p in pii_lower):
                x, y, w, h = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

        return img

    # ------------------------------------------------------------------
    # PAGE SPLITTING + MAIN PIPELINE
    # ------------------------------------------------------------------

    def _split_pages(self, img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        If width >> height, treat as left+right page.
        """
        h, w = img.shape[:2]
        pages: List[Tuple[str, np.ndarray]] = []
        if w > 1.1 * h:
            mid = w // 2
            left = img[:, :mid]
            right = img[:, mid:]
            pages.append(("left", left))
            pages.append(("right", right))
        else:
            pages.append(("single", img))
        return pages

    def process_document(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        full_img = cv2.imread(image_path)
        if full_img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        pages = self._split_pages(full_img)

        page_results: List[Dict[str, Any]] = []
        combined_text_all_parts: List[str] = []

        for side, page_img in pages:
            # Tesseract
            tess_raw = ""
            tess_clean = ""
            if self.use_tesseract:
                tess_variants = self._preprocess_for_tesseract(page_img)
                tess_raw = self._ocr_tesseract(tess_variants)
                tess_clean = self._clean_text(tess_raw)

            # TrOCR
            trocr_text = ""
            if self.use_trocr:
                trocr_text = self._ocr_trocr_grid(page_img)
                trocr_text = self._clean_text(trocr_text)

            combined_page_text = ""
            if tess_clean:
                combined_page_text += "[TESSERACT]\n" + tess_clean + "\n"
            if trocr_text:
                combined_page_text += "[HANDWRITING]\n" + trocr_text

            combined_page_text = combined_page_text.strip()

            page_results.append(
                {
                    "side": side,
                    "tesseract_text": tess_clean,
                    "trocr_text": trocr_text,
                    "combined_text": combined_page_text,
                }
            )
            combined_text_all_parts.append(
                f"[{side.upper()} PAGE]\n{combined_page_text}"
            )

        combined_text_all = "\n\n".join(combined_text_all_parts)

        # PII for redaction (simple)
        pii = self.detect_pii(combined_text_all)
        pii_tokens_flat: List[str] = []
        for vals in pii.values():
            pii_tokens_flat.extend(vals)

        redacted_image = self.create_redacted_image(image_path, pii_tokens_flat)

        summary = {
            "pages_detected": len(pages),
            "total_pii_found": sum(len(v) for v in pii.values()),
            "pii_categories": [k for k, v in pii.items() if v],
        }

        results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "pages": page_results,
            "combined_text_all": combined_text_all,
            "pii": pii,
            "redacted_image": redacted_image,
            "summary": summary,
        }
        return results


# Simple CLI test
def test_pipeline(image_path: str):
    ocr = MedicalDocumentOCR()
    res = ocr.process_document(image_path)

    print("\n===== COMBINED TEXT (first 1000 chars) =====")
    print(res["combined_text_all"][:1000])
    print("\n===== SUMMARY =====")
    print(json.dumps(res["summary"], indent=2))
    print("\n===== PII =====")
    print(json.dumps(res["pii"], indent=2))

    for p in res["pages"]:
        print(f"\n--- PAGE: {p['side']} ---")
        print("Tesseract snippet:", p["tesseract_text"][:200])
        print("TrOCR snippet:", p["trocr_text"][:200])

    return res


if __name__ == "__main__":
    sample = "sample_page.jpg"
    if os.path.exists(sample):
        test_pipeline(sample)
    else:
        print("Put a test image as 'sample_page.jpg' to run main.py directly.")
