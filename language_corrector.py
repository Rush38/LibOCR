import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LanguageModelCorrector:
    def __init__(self, bigrams_path: Optional[Path] = None):
        self._bigrams: Dict[str, int] = {}
        if bigrams_path and bigrams_path.exists():
            try:
                with open(bigrams_path, "r", encoding="utf-8") as f:
                    self._bigrams = json.load(f)
            except Exception:
                pass

    def correct_context(
        self, word: str, prev_word: Optional[str] = None, next_word: Optional[str] = None
    ) -> Tuple[str, int]:
        if not word or len(word) < 3:
            return word, 0

        candidates = self._find_candidates(word, prev_word, next_word)
        if candidates:
            best = candidates[0]
            if best[1] > 0:
                return best[0], best[1]
        return word, 0

    def _find_candidates(
        self, word: str, prev_word: Optional[str], next_word: Optional[str]
    ) -> List[Tuple[str, int]]:
        if not self._bigrams:
            return []

        best: List[Tuple[str, int]] = []
        lower = word.lower()

        for bigram, freq in self._bigrams.items():
            parts = bigram.split("_")
            if len(parts) < 2:
                continue

            w1, w2 = parts[0], parts[1]

            if prev_word is not None and prev_word.lower() == w1:
                dist = self._levenshtein(lower, w2)
                if dist <= 2:
                    best.append((w2 if word.islower() else w2.capitalize(), freq))

            if next_word is not None and next_word.lower() == w2:
                dist = self._levenshtein(lower, w1)
                if dist <= 2:
                    best.append((w1 if word.islower() else w1.capitalize(), freq))

            if prev_word is None and next_word is None:
                dist = self._levenshtein(lower, w1)
                if dist <= 1:
                    best.append((w1 if word.islower() else w1.capitalize(), freq))
                dist2 = self._levenshtein(lower, w2)
                if dist2 <= 1:
                    best.append((w2 if word.islower() else w2.capitalize(), freq))

        best.sort(key=lambda x: -x[1])
        return best

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[m][n]
