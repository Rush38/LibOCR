import json
import re
from pathlib import Path
from typing import Dict, List, Set


class DictionaryLoader:
    _instance = None
    _data: Dict[str, List[str]] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, dict_path: Path | None = None):
        if not self._data:
            self.load(dict_path)

    def load(self, dict_path: Path | None = None):
        if dict_path is None:
            from utils import get_project_root
            dict_path = get_project_root() / "dictionaries" / "library_dictionary.json"

        if not dict_path.exists():
            self._data = {}
            return

        with open(dict_path, "r", encoding="utf-8-sig") as f:
            self._data = json.load(f)

        self._compiled: Dict[str, Set[str]] = {}
        for key, words in self._data.items():
            self._compiled[key] = set(w.lower() for w in words)

    def get_category(self, key: str) -> List[str]:
        return self._data.get(key, [])

    def all_words(self) -> List[str]:
        result = []
        for words in self._data.values():
            result.extend(words)
        return result

    def search(self, text: str, categories: List[str] | None = None) -> Dict[str, List[str]]:
        text_lower = text.lower()
        found = {}
        keys = categories if categories else list(self._data.keys())
        for key in keys:
            if key not in self._compiled:
                continue
            matches = [w for w in self._compiled[key] if w in text_lower]
            if matches:
                found[key] = matches
        return found

    def highlight_in_text(self, text: str) -> str:
        all_words = []
        for words in self._compiled.values():
            all_words.extend(words)

        pattern = "|".join(re.escape(w) for w in all_words)
        if not pattern:
            return text
        return re.sub(pattern, lambda m: f"**{m.group()}**", text, flags=re.IGNORECASE)

    def match_patterns(self, text: str, max_edit_distance: int = 1) -> List[Dict]:
        results = []
        words = re.findall(r"[А-Яа-яЁёA-Za-z\-]{3,}", text)
        all_dict_words = self.all_words()

        for word in words:
            word_lower = word.lower()
            for dict_word in all_dict_words:
                if dict_word in word_lower or word_lower in dict_word:
                    dist = self._levenshtein(word_lower, dict_word)
                    if dist <= max_edit_distance:
                        results.append({
                            "found": word,
                            "dictionary": dict_word,
                            "distance": dist
                        })
                        break
        return results

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return DictionaryLoader._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev[j + 1] + 1
                deletions = curr[j] + 1
                substitutions = prev[j] + (c1 != c2)
                curr.append(min(insertions, deletions, substitutions))
            prev = curr
        return prev[-1]
