import pytesseract
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, Any, Optional, List


class HybridOCREngine:
    def __init__(self, config):
        self.config = config
        self.easy_config = config["ocr"]["easyocr"]
        self.tesser_config = config["ocr"]["tesseract"]
        self.trocr_config = config["ocr"].get("trocr", {})

        pytesseract.pytesseract.tesseract_cmd = self.tesser_config["path"]

        self._easyocr_reader = None
        self._easyocr_languages = self.easy_config["languages"]
        self._easyocr_gpu = self.easy_config["gpu"]

        self._trocr_pipeline = None
        self._trocr_model_name = self.trocr_config.get("model_name", "microsoft/troat-base-ru-handwriting")
        self._trocr_device = self.trocr_config.get("device", "auto")
        self._trocr_stride = self.trocr_config.get("stride", 128)

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

    @property
    def trocr_pipeline(self):
        if self._trocr_pipeline is None:
            self._init_trocr()
        return self._trocr_pipeline

    def _init_trocr(self):
        if not self.trocr_config.get("enabled", False):
            return

        try:
            from transformers import pipeline, AutoProcessor, AutoModel
            import torch

            device = self._get_trocr_device()
            model_name = self._trocr_model_name

            self._trocr_pipeline = pipeline(
                "ocr",
                model=model_name,
                device=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        except ImportError as e:
            import logging
            logging.getLogger("LibraryOCR").warning(
                f"TrOCR: transformers not installed ({e}). Install with: pip install transformers torch"
            )
            self._trocr_pipeline = None
        except Exception as e:
            import logging
            logging.getLogger("LibraryOCR").warning(f"TrOCR: failed to load model ({e})")
            self._trocr_pipeline = None

    def _get_trocr_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available() and self._trocr_device == "auto":
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def recognize(
        self, image: np.ndarray
    ) -> Dict[str, Any]:
        timeout = self.config["processing"]["timeout_engine"]

        easy_result: Optional[Dict] = None
        tess_result: Optional[Dict] = None
        trocr_result: Optional[Dict] = None
        warnings: List[str] = []

        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                f_easy = executor.submit(self._run_easyocr, image)
                f_tess = executor.submit(self._run_tesseract, image)
                f_troc = executor.submit(self._run_trocr, image)

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

                try:
                    trocr_result = f_troc.result(timeout=timeout * 2)
                except FuturesTimeoutError:
                    warnings.append("TrOCR: превышен таймаут")
                except Exception as e:
                    warnings.append(f"TrOCR: {str(e)}")

        except Exception as e:
            warnings.append(f"ThreadPool: {str(e)}")

        easy_text = easy_result["text"] if easy_result else ""
        easy_conf = easy_result["confidence"] if easy_result else 0.0

        tess_text = tess_result["text"] if tess_result else ""
        tess_conf = tess_result["confidence"] if tess_result else 0.0

        trocr_text = trocr_result["text"] if trocr_result else ""
        trocr_conf = trocr_result["confidence"] if trocr_result else 0.0

        active_confs = [c for c in [easy_conf, tess_conf, trocr_conf] if c > 0]
        if active_confs:
            overall = sum(active_confs) / len(active_confs)
        else:
            overall = 0.0

        return {
            "easyocr_text": easy_text,
            "tesseract_text": tess_text,
            "trocr_text": trocr_text,
            "easyocr_conf": easy_conf,
            "tesseract_conf": tess_conf,
            "trocr_conf": trocr_conf,
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
        psm = self.tesser_config.get("psm", 6)
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

    def _run_trocr(self, image: np.ndarray) -> Dict[str, Any]:
        if self._trocr_pipeline is None:
            return {"text": "", "confidence": 0.0}

        try:
            import torch

            img_rgb = image
            if len(image.shape) == 2:
                img_rgb = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                img_rgb = image[:, :, :3]

            pil_img = self._numpy_to_pil(img_rgb)

            result = self._trocr_pipeline(pil_img, truncate=True)

            text = result[0].get("generated_text", "") if isinstance(result, list) else result.get("generated_text", "")

            return {"text": text, "confidence": 0.75}

        except Exception as e:
            return {"text": "", "confidence": 0.0}

    def _numpy_to_pil(self, img: np.ndarray):
        from PIL import Image
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)

    def warmup(self):
        import easyocr

        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            self.easyocr_reader.readtext(dummy)
        except Exception:
            pass

        if self.trocr_config.get("enabled", False):
            try:
                _ = self.trocr_pipeline
            except Exception:
                pass