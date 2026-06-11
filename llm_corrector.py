from typing import Optional
from pathlib import Path


class LLMCorrector:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda"):
        self._model = None
        self._tokenizer = None
        self._device = device
        self._model_name = model_name
        self._loaded = False

    def load(self):
        if self._loaded:
            return True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model.eval()
            self._loaded = True
            return True
        except Exception as e:
            import logging
            logging.getLogger("LibraryOCR").warning(f"LLM: failed to load ({e})")
            return False

    def correct_text(self, raw_text: str, engine_name: str = "", confidence: float = 0.0) -> str:
        if not self._loaded and not self.load():
            return raw_text

        if confidence >= 0.8 and len(raw_text) > 50:
            return raw_text

        try:
            prompt = (
                f"Исправь OCR-ошибки в тексте. Сохрани структуру, заголовки, абзацы. "
                f"Исправь только очевидные ошибки, не меняй смысл.\n\n"
                f"Текст (распознан движком {engine_name}):\n{raw_text[:1000]}"
            )
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            import torch
            inputs = {k: v.to(self._device if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                )
            corrected = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            if corrected.startswith(prompt):
                corrected = corrected[len(prompt):].strip()
            import logging
            logging.getLogger("LibraryOCR").info(f"LLM corrected text: '{raw_text[:30]}...' -> '{corrected[:30]}...'")
            return corrected
        except Exception as e:
            import logging
            logging.getLogger("LibraryOCR").warning(f"LLM correction failed: {e}")
            return raw_text
