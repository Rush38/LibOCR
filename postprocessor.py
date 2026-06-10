import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional


class PostProcessor:
    def __init__(self, dict_loader=None):
        self._reader_pattern = re.compile(
            r"[А-Я][а-я]{2,}(?:[-\s][А-Я][а-я]+)?\s+[А-Я]\.[А-Я]\.?"
        )
        self._reader_alt_pattern = re.compile(
            r"[А-Я][а-я]{2,}(?:[-\s][А-Я][а-я]+)?"
        )
        self._date_pattern = re.compile(
            r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b"
        )
        self._inv_pattern = re.compile(
            r"№\s*([А-Яа-яA-Za-z0-9]+[-./]?[А-Яа-яA-Za-z0-9]*)"
        )
        self._inv_pattern2 = re.compile(
            r"\b([A-Za-z]{2,3}[-./]?[0-9]{4,8})\b"
        )
        self._inv_pattern3 = re.compile(r"\b([0-9]{4,8})\b")

        if dict_loader is None:
            from dictionaries.dictionary_loader import DictionaryLoader
            try:
                self._dict = DictionaryLoader()
            except Exception:
                self._dict = None
        else:
            self._dict = dict_loader

    def process(
        self,
        ocr_result: Dict[str, Any],
        filename: str,
        metadata: Dict[str, Any],
        elapsed: float,
    ) -> Dict[str, Any]:
        easy_text = ocr_result.get("easyocr_text", "")
        tess_text = ocr_result.get("tesseract_text", "")
        trocr_text = ocr_result.get("trocr_text", "")
        glm_text = ocr_result.get("glmocr_text", "")
        paddle_text = ocr_result.get("paddleocr_text", "")
        surya_text = ocr_result.get("suryaocr_text", "")
        doctr_text = ocr_result.get("doctr_text", "")
        gotocr_text = ocr_result.get("gotocr_text", "")

        raw_text = f"{tess_text}\n---\n{easy_text}\n---\n{trocr_text}\n---\n{glm_text}\n---\n{paddle_text}\n---\n{surya_text}\n---\n{doctr_text}\n---\n{gotocr_text}"

        combined = f"{easy_text} {tess_text} {trocr_text} {glm_text} {paddle_text} {surya_text} {doctr_text} {gotocr_text}"

        reader_names, possible_readers = self._extract_readers(combined)
        dates, possible_dates = self._extract_dates(raw_text)
        inventory_numbers = self._extract_inventory_numbers(raw_text)
        possible_stamps = self._extract_stamps(raw_text)
        possible_noise = self._extract_noise(raw_text)
        library_terms = self._dictionary_lookup(raw_text)
        fuzzy_corrections = self._fuzzy_corrections(raw_text)

        language = self._detect_language(raw_text)

        text_type = self._detect_text_type(combined)
        primary_engine = self._select_primary_engine(
            text_type,
            ocr_result.get("easyocr_conf", 0.0),
            ocr_result.get("tesseract_conf", 0.0),
            ocr_result.get("trocr_conf", 0.0),
        )

        possible_errors = self.cross_validate_engines(
            easy_text, tess_text, trocr_text
        )

        lang_corrector = getattr(self, "_lang_corrector", None)
        if lang_corrector:
            words = raw_text.split()
            for i, w in enumerate(words):
                prev = words[i - 1] if i > 0 else None
                nxt = words[i + 1] if i < len(words) - 1 else None
                corrected, _ = lang_corrector.correct_context(w, prev, nxt)
                if corrected != w:
                    fuzzy_corrections.append({
                        "original": w,
                        "corrected": corrected,
                        "distance": 1,
                        "engine": "bigram",
                    })

        raw_parts = [
            f"[ОСНОВНОЙ: {primary_engine.upper()}] ({text_type})",
            raw_text,
        ]
        if reader_names:
            raw_parts.append("[ИМЕНА] " + "; ".join(reader_names))
        if possible_readers:
            raw_parts.append("[ИМЕНА?] " + "; ".join(possible_readers) + " — требует проверки")
        if dates:
            raw_parts.append("[ДАТЫ] " + "; ".join(dates))
        if possible_dates:
            raw_parts.append("[ДАТЫ?] " + "; ".join(possible_dates))
        if inventory_numbers:
            raw_parts.append("[ИНВЕНТАРЬ] " + "; ".join(inventory_numbers))
        if possible_stamps:
            raw_parts.append("[ШТАМПЫ] " + "; ".join(possible_stamps))

        full_raw_text = "\n\n".join(raw_parts)
        formatted_text = self._format_text_structure(full_raw_text)
        hf = self.extract_headers_footers(full_raw_text)

        return {
            "filename": filename,
            "detected_angle": metadata.get("detected_angle", 0.0),
            "corrected": metadata.get("corrected", False),
            "processing_time_sec": round(elapsed, 2),
            "engines_used": ["easyocr", "tesseract", "trocr"],
            "text_type": text_type,
            "primary_engine": primary_engine,
            "possible_noise": possible_noise,
            "library_terms": library_terms,
            "fuzzy_corrections": fuzzy_corrections,
            "raw_text": full_raw_text,
            "formatted_text": formatted_text,
            "possible_errors": possible_errors,
            "alternatives": {
                "easyocr_text": easy_text,
                "tesseract_text": tess_text,
                "trocr_text": trocr_text,
                "glmocr_text": glm_text,
                "paddleocr_text": paddle_text,
                "suryaocr_text": surya_text,
                "doctr_text": doctr_text,
                "gotocr_text": gotocr_text,
            },
            "confidence": {
                "overall": float(ocr_result.get("overall_conf", 0.0)),
                "easyocr": float(ocr_result.get("easyocr_conf", 0.0)),
                "tesseract": float(ocr_result.get("tesseract_conf", 0.0)),
                "trocr": float(ocr_result.get("trocr_conf", 0.0)),
                "glmocr": float(ocr_result.get("glmocr_conf", 0.0)),
                "paddleocr": float(ocr_result.get("paddleocr_conf", 0.0)),
                "suryaocr": float(ocr_result.get("suryaocr_conf", 0.0)),
                "doctr": float(ocr_result.get("doctr_conf", 0.0)),
                "gotocr": float(ocr_result.get("gotocr_conf", 0.0)),
            },
            "warnings": ocr_result.get("warnings", []),
            "language": language,
            "document_type": metadata.get("document_type", "default"),
            "headers": hf.get("headers", []),
            "footers": hf.get("footers", []),
            "page_numbers": hf.get("footers", []),
            "best_engine": ocr_result.get("best_engine", {}),
            "engine_times": ocr_result.get("engine_times", {}),
            "consensus_validated": len(possible_errors) == 0,
        }

    def _extract_readers(self, text: str) -> Tuple[List[str], List[str]]:
        matches = self._reader_pattern.findall(text)
        readers = []
        for m in matches:
            name = self._fix_latin_in_cyrillic(m.strip())
            if len(name) >= 5:
                readers.append(name)

        possible = []
        words = text.split()
        for i, word in enumerate(words):
            if (
                len(word) > 3
                and word[0].isupper()
                and word.isalpha()
                and not re.match(r"^[А-Я]\.$", word)
            ):
                if i + 1 < len(words) and re.match(
                    r"^[А-Я]\.$", words[i + 1]
                ):
                    possible.append(word)

        return list(set(readers)), list(set(possible))

    def _fix_latin_in_cyrillic(self, text: str) -> str:
        replacements = [
            (r"B\b", "в"),
            (r"b\b", "ь"),
            (r"H\b", "н"),
            (r"CT\b", "СТ"),
            (r"CTO\b", "СТО"),
            (r"K\b", "К"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        return text

    def _extract_dates(self, text: str) -> Tuple[List[str], List[str]]:
        matches = self._date_pattern.findall(text)
        dates = []
        possible_dates = []

        for m in matches:
            day, month, year = m[0], m[1], m[2]
            fixed = self._fix_date_symbols(f"{day}.{month}.{year}")

            for fmt in [
                "%d.%m.%Y",
                "%d.%m.%y",
                "%d/%m/%Y",
                "%d/%m/%y",
                "%d-%m-%Y",
                "%d-%m-%y",
            ]:
                try:
                    dt = datetime.strptime(fixed, fmt)
                    if fmt.endswith("%y"):
                        fixed = (
                            fixed[:-2] + "20" + fixed[-2:]
                        )
                    dates.append(fixed)
                    break
                except ValueError:
                    continue
            else:
                possible_dates.append(f"{fixed} — требует проверки")

        return list(set(dates)), possible_dates

    def _fix_date_symbols(self, text: str) -> str:
        text = text.replace("О", "0").replace("о", "0")
        text = text.replace("O", "0")
        text = text.replace("l", "1").replace("I", "1")
        text = text.replace("S", "5")
        return text

    def _extract_inventory_numbers(self, text: str) -> List[str]:
        numbers = []

        for m in self._inv_pattern.findall(text):
            if len(m.strip()) >= 3:
                numbers.append(f"№ {m.strip()}")

        for m in self._inv_pattern2.findall(text):
            numbers.append(m.strip())

        for m in self._inv_pattern3.findall(text):
            numbers.append(m)

        return list(set(numbers))

    def _extract_stamps(self, text: str) -> List[str]:
        stamps = []
        for line in text.split("\n"):
            line = line.strip()
            if 5 <= len(line) <= 40 and line.isupper() and " " not in line:
                if not any(c.isdigit() for c in line):
                    stamps.append(line)
        return list(set(stamps))

    def _extract_noise(self, text: str) -> List[str]:
        noise = []
        for line in text.split("\n"):
            line = line.strip()
            if 0 < len(line) < 2:
                noise.append(line)
        return list(set(noise))

    def _detect_language(self, text: str) -> str:
        cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
        old_ru = sum(1 for c in text if c in "ѣѢѲѳѴѵіІѣѢ")
        latin = sum(
            1
            for c in text
            if c.isalpha() and not ("\u0400" <= c <= "\u04FF")
        )

        if old_ru > 3:
            return "ru_old"
        if cyrillic > latin * 2:
            return "ru"
        elif latin > cyrillic * 2:
            return "en"
        elif cyrillic > 0 or latin > 0:
            return "mixed"
        return "unknown"

    def _dictionary_lookup(self, text: str) -> List[str]:
        if not self._dict:
            return []
        results = self._dict.match_patterns(text, max_edit_distance=1)
        return list({r["corrected"] for r in results})

    def _fuzzy_corrections(self, text: str) -> List[Dict]:
        if not self._dict:
            return []
        return self._dict.match_patterns(text, max_edit_distance=1)

    def _detect_text_type(self, text: str) -> str:
        if not text.strip():
            return "unknown"
        handwritten_indicators = 0
        printed_indicators = 0
        irregular_chars = 0
        for char in text:
            if char in "ёЁіІєЄїЇґҐ":
                handwritten_indicators += 1
            if char.isdigit():
                printed_indicators += 1
            if 0x2500 <= ord(char) <= 0x257F:
                irregular_chars += 1
        words = text.split()
        if not words:
            return "printed"
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3:
            handwritten_indicators += 2
        if handwritten_indicators > printed_indicators:
            return "handwritten"
        return "printed"

    def _select_primary_engine(
        self, text_type: str, easy_conf: float, tess_conf: float, trocr_conf: float
    ) -> str:
        if text_type == "handwritten":
            if trocr_conf > 0.1:
                return "trocr"
        if tess_conf >= easy_conf:
            return "tesseract"
        return "easyocr"

    def register_language_corrector(self, corrector):
        self._lang_corrector = corrector

    def cross_validate_engines(
        self, easy_text: str, tess_text: str, trocr_text: str
    ) -> List[str]:
        errors = []
        easy_lines = [l.strip() for l in easy_text.split("\n") if l.strip()]
        tess_lines = [l.strip() for l in tess_text.split("\n") if l.strip()]
        trocr_lines = [l.strip() for l in trocr_text.split("\n") if l.strip()]

        max_lines = max(len(easy_lines), len(tess_lines), len(trocr_lines))
        for i in range(max_lines):
            line1 = easy_lines[i] if i < len(easy_lines) else ""
            line2 = tess_lines[i] if i < len(tess_lines) else ""
            line3 = trocr_lines[i] if i < len(trocr_lines) else ""

            agreement = sum(1 for t in [line1, line2, line3] if t)
            if agreement < 2 and agreement > 0:
                errors.append(
                    f"строка {i+1}: расхождение между движками"
                )
        return errors

    def _format_text_structure(self, raw_text: str) -> str:
        lines = raw_text.split("\n")
        formatted = []
        buffer = []

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                if buffer:
                    formatted.append(" ".join(buffer))
                    buffer = []
                continue

            if self._is_heading_candidate(stripped, lines, idx):
                if buffer:
                    formatted.append(" ".join(buffer))
                    buffer = []
                formatted.append(f"\n## {stripped}\n")
                continue

            if self._is_list_item(stripped):
                if buffer:
                    formatted.append(" ".join(buffer))
                    buffer = []
                formatted.append(f"  - {stripped}")
                continue

            buffer.append(stripped)

        if buffer:
            formatted.append(" ".join(buffer))

        return "\n\n".join(formatted)

    @staticmethod
    def _is_heading_candidate(line: str, all_lines: list, idx: int) -> bool:
        if len(line) > 60:
            return False
        if line.isupper() and len(line) > 3:
            return True
        if all(c.isupper() or c in " .-–—:!?" for c in line.strip()) and len(line.strip()) > 2:
            return True
        if len(line) < 40 and idx + 1 < len(all_lines):
            next_line = all_lines[idx + 1].strip()
            if len(next_line) > len(line) * 2 and not next_line.isupper():
                return True
        return False

    @staticmethod
    def _is_list_item(line: str) -> bool:
        return any(line.startswith(p) for p in ["- ", "* ", "• ", "1.", "2.", "3.", "4.", "5.", "– ", "— "])

    def extract_headers_footers(self, text: str) -> Dict[str, List[str]]:
        lines = text.split("\n")
        headers = []
        footers = []
        main_lines = []
        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                main_lines.append(line)
                continue
            if len(s) < 30 and any(c.isdigit() for c in s) and not any(c.isalpha() for c in s):
                footers.append(s)
                continue
            if len(s) < 20 and s.isupper():
                if i < len(lines) // 4:
                    headers.append(s)
                elif i > len(lines) * 3 // 4:
                    footers.append(s)
                else:
                    main_lines.append(line)
            else:
                main_lines.append(line)
        return {"headers": headers, "footers": footers, "main_text": "\n".join(main_lines)}
