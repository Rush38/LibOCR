import pytesseract
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, Any, Optional, List


class HybridOCREngine:
    def __init__(self, config):
        self.config = config
        self.easy_config = config["ocr"]["easyocr"]
        self.tesser_config = config["ocr"]["tesseract"]

        pytesseract.pytesseract.tesseract_cmd = self.tesser_config["path"]

        self._easyocr_reader = None
        self._easyocr_languages = self.easy_config["languages"]
        self._easyocr_gpu = self.easy_config["gpu"]

    @property
    def easyocr_reader(self):
        if self._easyocr_reader is None:
            import easyocr

            self._easyocr_reader = easyocr.Reader(
                self._easyocr_languages,
                gpu=self._easyocr_gpu,
                verbose=False,
            )
        return self._easyocr_reader

    def recognize(
        self, image: np.ndarray
    ) -> Dict[str, Any]:
        timeout = self.config["processing"]["timeout_engine"]

        easy_result: Optional[Dict] = None
        tess_result: Optional[Dict] = None
        warnings: List[str] = []

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_easy = executor.submit(self._run_easyocr, image)
                f_tess = executor.submit(self._run_tesseract, image)

                try:
                    easy_result = f_easy.result(timeout=timeout)
                except FuturesTimeoutError:
                    warnings.append("EasyOCR: превышен таймаут")
                except Exception as e:
                    warnings.append(f"EasyOCR: {str(e)}")

                try:
                    tess_result = f_tess.result(timeout=timeout)
                except FuturesTimeoutError:
                    warnings.append("Tesseract: превышен таймаут")
                except Exception as e:
                    warnings.append(f"Tesseract: {str(e)}")

        except Exception as e:
            warnings.append(f"ThreadPool: {str(e)}")

        easy_text = easy_result["text"] if easy_result else ""
        easy_conf = easy_result["confidence"] if easy_result else 0.0

        tess_text = tess_result["text"] if tess_result else ""
        tess_conf = tess_result["confidence"] if tess_result else 0.0

        if easy_conf > 0 and tess_conf > 0:
            overall = (easy_conf + tess_conf) / 2.0
        else:
            overall = easy_conf + tess_conf

        return {
            "easyocr_text": easy_text,
            "tesseract_text": tess_text,
            "easyocr_conf": easy_conf,
            "tesseract_conf": tess_conf,
            "overall_conf": overall,
            "warnings": warnings,
        }

    def _run_easyocr(self, image: np.ndarray) -> Dict[str, Any]:
        results = self.easyocr_reader.readtext(
            image,
            paragraph=self.easy_config.get("paragraph", True),
        )

        text_parts = []
        confidences = []

        for bbox, text, conf in results:
            if text and text.strip():
                text_parts.append(text.strip())
                confidences.append(float(conf))

        text = "\n".join(text_parts)
        confidence = float(np.mean(confidences)) if confidences else 0.0

        return {"text": text, "confidence": confidence}

    def _run_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        oem = self.tesser_config.get("oem", 3)
        psm = self.tesser_config.get("default_psm", 3)
        lang = self.tesser_config["languages"]

        custom_config = f"--oem {oem} --psm {psm} -c tessedit_do_invert=0"

        text = pytesseract.image_to_string(
            image,
            lang=lang,
            config=custom_config,
        )

        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )

        confs = [int(c) for c in data["conf"] if int(c) > 0]
        confidence = float(np.mean(confs)) if confs else 0.0

        return {"text": text, "confidence": confidence}

    def warmup(self):
        import easyocr

        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            self.easyocr_reader.readtext(dummy)
        except Exception:
            pass
