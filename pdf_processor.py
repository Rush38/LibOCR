from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger("LibraryOCR")

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


class PDFProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("pdf", {})

    def extract_images_from_pdf(
        self, pdf_path: Path
    ) -> List[np.ndarray]:
        if not HAS_FITZ:
            logger.error("PDF support requires PyMuPDF. Install: pip install PyMuPDF")
            return []

        dpi = self.config.get("dpi", 300)
        first_only = self.config.get("first_page_only", False)
        max_pages = self.config.get("max_pages", 50)
        zoom = dpi / 72

        try:
            doc = fitz.open(str(pdf_path))
            total = min(len(doc), max_pages)
            pages_to_process = [doc[i] for i in range(total)] if not first_only else [doc[0]]

            images = []
            for page in pages_to_process:
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                images.append(img)

            doc.close()
            logger.debug(f"PDF {pdf_path.name}: {len(images)} pages extracted")
            return images
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path.name}: {e}")
            return []

    def get_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "filename": pdf_path.name,
            "pages": 0,
            "author": "",
            "creation_date": "",
        }
        if not HAS_FITZ:
            return meta
        try:
            doc = fitz.open(str(pdf_path))
            meta["pages"] = len(doc)
            info = doc.metadata
            meta["author"] = info.get("author", "")
            meta["creation_date"] = info.get("creationDate", "")
            doc.close()
        except Exception:
            pass
        return meta

    @staticmethod
    def detect_orientation(page_image: np.ndarray) -> int:
        import cv2
        gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(binary)
        if coords is None:
            return 0
        _, _, w, h = cv2.boundingRect(coords)
        return 0 if w >= h else 90
