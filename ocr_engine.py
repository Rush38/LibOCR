import pytesseract
import numpy as np
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, Any, Optional, List
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)


class HybridOCREngine:
    def __init__(self, config):
        self.config = config
        self.easy_config = config["ocr"]["easyocr"]
        self.tesser_config = config["ocr"]["tesseract"]
        self.trocr_config = config["ocr"].get("trocr", {})
        self.glm_config = config["ocr"].get("glmocr", {})

        pytesseract.pytesseract.tesseract_cmd = self.tesser_config["path"]

        self._easyocr_reader = None
        self._easyocr_languages = self.easy_config["languages"]
        self._easyocr_gpu = self.easy_config["gpu"]

        self._trocr_model = None
        self._trocr_processor = None
        self._trocr_device_name = "cpu"
        self._trocr_model_name = self.trocr_config.get(
            "model_name", "Kansallisarkisto/cyrillic-large-handwritten"
        )
        self._trocr_local_dir = self.trocr_config.get("local_models_dir", "models")

        self._glm_pipeline = None
        self._glm_model_name = self.glm_config.get(
            "model_name", "zai-org/GLM-OCR"
        )
        self._glm_local_dir = self.glm_config.get("local_models_dir", "models")

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
    def trocr_model(self):
        if self._trocr_model is None and self.trocr_config.get("enabled", False):
            self._init_trocr()
        return self._trocr_model

    def download_models(self) -> Dict[str, str]:
        results = {}
        import logging
        logger = logging.getLogger("LibraryOCR")

        if self.trocr_config.get("enabled", False):
            model_name = self._trocr_model_name
            local_dir = self._trocr_local_dir
            logger.info(f"Downloading TrOCR model: {model_name}")
            path = self._download_model_locally(model_name, local_dir)
            results["trocr"] = str(path) if path else "failed"

        if self.glm_config.get("enabled", False):
            model_name = self._glm_model_name
            local_dir = self._glm_local_dir
            logger.info(f"Downloading GLM-OCR model: {model_name}")
            path = self._download_model_locally(model_name, local_dir)
            results["glmocr"] = str(path) if path else "failed"

        return results

    def _get_model_dir(self, model_name: str, local_dir: str) -> Optional[Path]:
        root = Path(__file__).parent
        models_root = root / local_dir
        models_root.mkdir(parents=True, exist_ok=True)
        model_dir = models_root / model_name.replace("/", "_")
        return model_dir if model_dir.exists() else None

    def _download_model_locally(self, model_name: str, local_dir: str) -> Optional[Path]:
        try:
            from huggingface_hub import snapshot_download

            root = Path(__file__).parent
            models_root = root / local_dir
            models_root.mkdir(parents=True, exist_ok=True)
            model_dir = models_root / model_name.replace("/", "_")

            if not model_dir.exists():
                snapshot_download(
                    repo_id=model_name,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False,
                )
            return model_dir
        except Exception:
            return None

    def _init_trocr(self):
        if not self.trocr_config.get("enabled", False):
            return

        try:
            from transformers import (
                TrOCRProcessor,
                VisionEncoderDecoderModel,
            )
            import torch
            logging.getLogger("transformers").setLevel(logging.CRITICAL)

            device = self._get_trocr_device()
            model_name = self._trocr_model_name
            local_dir = self._trocr_local_dir

            local_path = self._get_model_dir(model_name, local_dir)
            if not local_path:
                logger = logging.getLogger("LibraryOCR")
                logger.info(f"TrOCR: model not found locally, downloading...")
                local_path = self._download_model_locally(model_name, local_dir)

            if local_path:
                load_from = str(local_path)
            else:
                load_from = model_name

            logger = logging.getLogger("LibraryOCR")
            logger.info(f"TrOCR: loading model from {load_from}")

            self._trocr_processor = TrOCRProcessor.from_pretrained(load_from)
            self._trocr_model = VisionEncoderDecoderModel.from_pretrained(load_from)
            self._trocr_model.to(device)
            self._trocr_model.eval()
            self._trocr_model.generation_config.max_new_tokens = 256
            logger.info(f"TrOCR: model loaded successfully, device={device}")
            self._trocr_device_name = device

        except ImportError as e:
            logging.getLogger("LibraryOCR").warning(
                f"TrOCR: transformers not installed ({e}). Install with: pip install transformers torch"
            )
            self._trocr_model = None
            self._trocr_processor = None
        except Exception as e:
            logger = logging.getLogger("LibraryOCR")
            logger.error(f"TrOCR: FAILED to load model ({e})", exc_info=True)
            self._trocr_model = None
            self._trocr_processor = None

    def _init_glmocr(self):
        if not self.glm_config.get("enabled", False):
            return None

        try:
            from transformers import pipeline
            import torch

            device = self._get_glm_device()
            model_name = self._glm_model_name
            local_dir = self._glm_local_dir

            local_path = self._get_model_dir(model_name, local_dir)
            if local_path:
                load_from = str(local_path)
            else:
                load_from = model_name

            self._glm_pipeline = pipeline(
                "ocr",
                model=load_from,
                device=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            return self._glm_pipeline

        except ImportError as e:
            import logging
            logging.getLogger("LibraryOCR").warning(
                f"GLM-OCR: transformers not installed ({e})"
            )
            return None
        except Exception as e:
            import logging
            logging.getLogger("LibraryOCR").warning(f"GLM-OCR: failed to load model ({e})")
            return None

    def _get_trocr_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available() and self.trocr_config.get("device", "auto") == "auto":
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _get_glm_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available() and self.glm_config.get("device", "auto") == "auto":
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def recognize(self, image: np.ndarray) -> Dict[str, Any]:
        timeout = self.config["processing"]["timeout_engine"]
        warnings: List[str] = []
        engine_times: Dict[str, float] = {}
        results: Dict[str, Dict] = {}

        _ = self.easyocr_reader
        _ = self.trocr_model
        cascade = self.config.get("processing", {}).get("cascade", False)

        try:
            import time as tmod

            if cascade:
                t0 = tmod.time()
                easy_result = self._run_easyocr(image)
                engine_times["easyocr"] = tmod.time() - t0
                results["easyocr"] = easy_result
                if easy_result.get("error"):
                    warnings.append(f"EasyOCR: {easy_result['error']}")

                if not easy_result.get("text", "").strip() or easy_result.get("confidence", 0) < 0.3:
                    easy_result2 = self._run_easyocr(image, paragraph=False)
                    if easy_result2.get("text", "").strip():
                        results["easyocr"] = easy_result2
                        engine_times["easyocr"] += engine_times.get("easyocr", 0)

                easy_conf = results["easyocr"].get("confidence", 0.0)
                if easy_conf < 0.7:
                    t1 = tmod.time()
                    tess_result = self._run_tesseract(image)
                    engine_times["tesseract"] = tmod.time() - t1
                    results["tesseract"] = tess_result
                    if tess_result.get("error"):
                        warnings.append(f"Tesseract: {tess_result['error']}")

                tess_conf = results.get("tesseract", {}).get("confidence", 0.0) if "tesseract" in results else 0.0
                if max(easy_conf, tess_conf) < 0.6:
                    t2 = tmod.time()
                    trocr_result = self._run_trocr(image)
                    engine_times["trocr"] = tmod.time() - t2
                    results["trocr"] = trocr_result
                    if trocr_result.get("error"):
                        warnings.append(f"TrOCR: {trocr_result['error']}")
            else:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    f_easy = executor.submit(self._run_easyocr, image)
                    f_tess = executor.submit(self._run_tesseract, image)
                    f_troc = executor.submit(self._run_trocr, image)
                    f_glm = executor.submit(self._run_glmocr, image)

                    for name, future, t_out in [
                        ("easyocr", f_easy, timeout),
                        ("tesseract", f_tess, timeout),
                        ("trocr", f_troc, timeout * 2),
                        ("glmocr", f_glm, timeout * 2),
                    ]:
                        t0 = tmod.time()
                        try:
                            res = future.result(timeout=t_out)
                            engine_times[name] = tmod.time() - t0
                            results[name] = res
                            if res.get("error"):
                                warnings.append(f"{name.capitalize()}: {res['error']}")
                        except FuturesTimeoutError:
                            warnings.append(f"{name.capitalize()}: timeout")
                            engine_times[name] = t_out
                        except Exception as e:
                            warnings.append(f"{name.capitalize()}: {str(e)}")
                            engine_times[name] = tmod.time() - t0

        except Exception as e:
            warnings.append(f"OCR: {str(e)}")

        easy_text = results.get("easyocr", {}).get("text", "")
        easy_conf = results.get("easyocr", {}).get("confidence", 0.0)
        tess_text = results.get("tesseract", {}).get("text", "")
        tess_conf = results.get("tesseract", {}).get("confidence", 0.0)
        trocr_text = results.get("trocr", {}).get("text", "")
        trocr_conf = results.get("trocr", {}).get("confidence", 0.0)
        glm_text = results.get("glmocr", {}).get("text", "")
        glm_conf = results.get("glmocr", {}).get("confidence", 0.0)

        conf_map = {"easyocr": easy_conf, "tesseract": tess_conf, "trocr": trocr_conf}
        active = {k: v for k, v in conf_map.items() if v > 0}
        overall = sum(active.values()) / len(active) if active else 0.0

        best_name = max(active, key=active.get) if active else "tesseract"
        best_reason = f"наивысшая уверенность ({active.get(best_name, 0):.0%})"
        if cascade:
            best_reason += " (каскадный режим)"

        return {
            "easyocr_text": easy_text,
            "tesseract_text": tess_text,
            "trocr_text": trocr_text,
            "glmocr_text": glm_text,
            "easyocr_conf": easy_conf,
            "tesseract_conf": tess_conf,
            "trocr_conf": trocr_conf,
            "glmocr_conf": glm_conf,
            "overall_conf": overall,
            "best_engine": {
                "name": best_name,
                "confidence": active.get(best_name, 0.0),
                "reason": best_reason,
            },
            "engine_times": engine_times,
            "warnings": warnings,
        }

    def _run_easyocr(self, image: np.ndarray, paragraph: Optional[bool] = None) -> Dict[str, Any]:
        try:
            use_paragraph = paragraph if paragraph is not None else self.easy_config.get("paragraph", True)
            results = self.easyocr_reader.readtext(
                image,
                paragraph=use_paragraph,
            )

            text_parts = []
            confidences = []

            for item in results:
                text = None
                conf = None

                if len(item) >= 3:
                    if isinstance(item[1], str) and isinstance(item[2], (int, float)):
                        text, conf = item[1], item[2]
                    elif isinstance(item[2], str) and isinstance(item[1], (int, float)):
                        text, conf = item[2], item[1]
                elif len(item) == 2:
                    if isinstance(item[0], str) and isinstance(item[1], (int, float, str)):
                        text, conf = item[0], item[1]
                    elif isinstance(item[1], str) and isinstance(item[0], (int, float, str)):
                        text, conf = item[1], item[0]

                if text is None or conf is None:
                    continue

                if isinstance(text, list):
                    text = " ".join(str(t) for t in text)
                elif not isinstance(text, str):
                    text = str(text)

                try:
                    conf_float = float(conf)
                    if 0 <= conf_float <= 1:
                        conf_value = conf_float
                    else:
                        conf_value = 0.5
                except (ValueError, TypeError):
                    conf_value = 0.0

                if text and text.strip():
                    text_parts.append(text.strip())
                    confidences.append(conf_value)

            text = "\n".join(text_parts)
            confidence = float(np.mean(confidences)) if confidences else 0.0

            return {"text": text, "confidence": confidence}
        except Exception as e:
            return {"text": "", "confidence": 0.0, "error": str(e)}

    def auto_select_psm(self, image: np.ndarray) -> int:
        h, w = image.shape[:2]
        ratio = w / h
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_pixels = cv2.countNonZero(255 - binary)
        total_pixels = h * w
        density = text_pixels / total_pixels if total_pixels else 0

        if ratio > 2.0:
            return 6
        if density < 0.02:
            return 3
        if 0.02 <= density <= 0.08:
            return 7
        return 11

    def _run_tesseract(self, image: np.ndarray) -> Dict[str, Any]:
        oem = self.tesser_config.get("oem", 3)
        psm = self.tesser_config.get("psm", 6)
        lang = self.tesser_config["languages"]

        adaptive_psm = self.tesser_config.get("adaptive_psm", False)
        if adaptive_psm:
            psm = self.auto_select_psm(image)

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
        confidence = (float(np.mean(confs)) / 100.0) if confs else 0.0

        return {"text": text, "confidence": confidence, "psm_used": psm}

    def _run_trocr(self, image: np.ndarray) -> Dict[str, Any]:
        if self._trocr_model is None or self._trocr_processor is None:
            return {"text": "", "confidence": 0.0, "error": "model not loaded"}

        import logging
        logger = logging.getLogger("LibraryOCR")

        try:
            import torch
            from PIL import Image
            pil_img = Image.fromarray(image)

            inputs = self._trocr_processor(
                images=pil_img, return_tensors="pt"
            )

            device = getattr(self, '_trocr_device_name', 'cpu')
            for key in inputs:
                if hasattr(inputs[key], 'to'):
                    inputs[key] = inputs[key].to(device)

            with torch.no_grad():
                generated_ids = self._trocr_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=4,
                    early_stopping=True,
                )

            text = self._trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            if not text or not text.strip():
                return {"text": "", "confidence": 0.0, "error": "empty result"}

            logger.info(f"TrOCR result: '{text[:50]}...'")
            return {"text": text, "confidence": 0.92}

        except Exception as e:
            logger.error(f"TrOCR error: {e}")
            return {"text": "", "confidence": 0.0, "error": str(e)}

    def _run_glmocr(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.glm_config.get("enabled", False):
            return {"text": "", "confidence": 0.0}

        if self._glm_pipeline is None:
            pipeline = self._init_glmocr()
            if pipeline is None:
                return {"text": "", "confidence": 0.0}
        else:
            pipeline = self._glm_pipeline

        try:
            pil_img = self._numpy_to_pil(image)

            result = pipeline(pil_img, truncate=True)

            text = ""
            if isinstance(result, list):
                text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                text = result.get("generated_text", "")

            return {"text": text, "confidence": 0.88}

        except Exception:
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
                _ = self.trocr_model
            except Exception:
                pass

        if self.glm_config.get("enabled", False):
            try:
                _ = self._init_glmocr()
            except Exception:
                pass

    def download_models(self):
        for model_name, local_dir in [
            (self._trocr_model_name, self._trocr_local_dir),
            (self._glm_model_name, self._glm_local_dir),
        ]:
            path = self._download_model_locally(model_name, local_dir)
            if path:
                print(f"Model downloaded: {model_name} -> {path}")
            else:
                print(f"Model already exists or download failed: {model_name}")
