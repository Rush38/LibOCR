from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger("LibraryOCR")

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


class PDFProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("pdf", {})

    def extract_images_from_pdf(
        self, pdf_path: Path
    ) -> List[np.ndarray]:
        if not HAS_PDF2IMAGE:
            logger.error("PDF support requires pdf2image. Install: pip install pdf2image")
            return []

        dpi = self.config.get("dpi", 300)
        first_only = self.config.get("first_page_only", False)
        max_pages = self.config.get("max_pages", 50)
        color = self.config.get("color_conversion", "RGB")
        poppler = self.config.get("poppler_path", None)

        try:
            pages = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                first_page=1,
                last_page=1 if first_only else max_pages,
                fmt="jpeg",
                use_pdftocairo=True,
                poppler_path=poppler if poppler and Path(poppler).exists() else None,
            )
            images = [np.array(page.convert(color)) for page in pages]
            logger.info(f"PDF {pdf_path.name}: {len(images)} pages extracted")
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
        if not HAS_PDF2IMAGE:
            return meta
        try:
            from pdf2image.pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(str(pdf_path))
            meta["pages"] = info.get("Pages", 0)
            meta["author"] = info.get("Author", "")
            meta["creation_date"] = info.get("CreationDate", "")
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
