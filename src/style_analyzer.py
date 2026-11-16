# src/style_analyzer.py
# Utilities to load sample texts & analyze writing style

from pathlib import Path
import json
import re
from collections import Counter
from typing import List, Dict, Any

DATA_DIR = Path("data/samples")


def load_samples() -> List[Dict[str, Any]]:
    """
    Load text samples from data/samples/*.txt
    Returns list of dicts: {"id": ..., "text": ...}
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for p in sorted(DATA_DIR.glob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        if text:
            cleaned_text = " ".join(
                line.strip() for line in text.splitlines() if line.strip()
            )
            samples.append({"id": p.stem, "text": cleaned_text})
    return samples


def save_samples_jsonl(samples, out_path="data/samples.jsonl"):
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return out_path


class StyleAnalyzer:
    """
    Simple analyzer that computes basic writing style metrics like:
    - avg_sentence_length
    - vocab_richness
    - readability_score
    - formality_score
    - common_phrases

    It also compares two sets of metrics and returns a similarity score.
    """

    def _sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    def _words(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z']+", text.lower())

    def _bigrams(self, words: List[str]) -> List[str]:
        return [f"{w1} {w2}" for w1, w2 in zip(words, words[1:])]

    def analyze_text(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {}

        sentences = self._sentences(text)
        words = self._words(text)
        num_sentences = max(len(sentences), 1)
        num_words = max(len(words), 1)

        avg_sentence_length = num_words / num_sentences
        vocab_richness = len(set(words)) / num_words

        # Dumb readability and formality metrics
        readability_score = max(0.0, min(100.0, 100.0 - avg_sentence_length * 2))
        long_words = [w for w in words if len(w) >= 7]
        formality_score = max(0.0, min(100.0, 40.0 + (len(long_words) / num_words) * 100))

        bigrams = self._bigrams(words)
        phrase_counts = Counter(bigrams).most_common(20)

        return {
            "avg_sentence_length": round(avg_sentence_length, 2),
            "vocab_richness": round(vocab_richness, 3),
            "readability_score": round(readability_score, 1),
            "formality_score": round(formality_score, 1),
            "common_phrases": phrase_counts,
        }

    def compare_styles(self, base: Dict[str, Any], target: Dict[str, Any]) -> float:
        """
        Compare two sets of metrics and return similarity (0–100)
        """
        if not base or not target:
            return 0.0

        score = 0
        keys = ["avg_sentence_length", "vocab_richness", "readability_score", "formality_score"]

        for key in keys:
            b = float(base.get(key, 0.0))
            t = float(target.get(key, 0.0))
            diff = abs(b - t)
            if key == "vocab_richness":
                score += max(0.0, 1.0 - diff * 4.0)
            else:
                score += max(0.0, 1.0 - diff / 50.0)

        return round((score / len(keys)) * 100, 1)
