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

        raw_text = f"{tess_text}\n---\n{easy_text}"

        combined = easy_text + " " + tess_text

        reader_names, possible_readers = self._extract_readers(combined)
        dates, possible_dates = self._extract_dates(raw_text)
        inventory_numbers = self._extract_inventory_numbers(raw_text)
        possible_stamps = self._extract_stamps(raw_text)
        possible_noise = self._extract_noise(raw_text)
        library_terms = self._dictionary_lookup(raw_text)

        language = self._detect_language(raw_text)

        return {
            "filename": filename,
            "detected_angle": metadata.get("detected_angle", 0.0),
            "corrected": metadata.get("corrected", False),
            "processing_time_sec": round(elapsed, 2),
            "engines_used": ["easyocr", "tesseract"],
            "reader_names": reader_names,
            "possible_readers": possible_readers,
            "dates": dates,
            "possible_dates": possible_dates,
            "inventory_numbers": inventory_numbers,
            "possible_stamps": possible_stamps,
            "possible_noise": possible_noise,
            "library_terms": library_terms,
            "raw_text": raw_text,
            "alternatives": {
                "easyocr_text": easy_text,
                "tesseract_text": tess_text,
            },
            "confidence": {
                "overall": float(ocr_result.get("overall_conf", 0.0)),
                "easyocr": float(ocr_result.get("easyocr_conf", 0.0)),
                "tesseract": float(ocr_result.get("tesseract_conf", 0.0)),
            },
            "warnings": ocr_result.get("warnings", []),
            "language": language,
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
        latin = sum(
            1
            for c in text
            if c.isalpha() and not ("\u0400" <= c <= "\u04FF")
        )

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
        return [r["dictionary"] for r in results]
