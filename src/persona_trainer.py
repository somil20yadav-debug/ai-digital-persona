# src/persona_trainer.py
# Trainer for collecting samples, analyzing style, and building embeddings/index

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import hnswlib

from src.style_analyzer import StyleAnalyzer, load_samples
from src.database import get_all_samples  # <- load existing samples from DB

EMBED_MODEL = "all-MiniLM-L6-v2"
OUT_DIR = Path("models/embeddings")


class PersonaTrainer:
    def __init__(self, embed_model: str = EMBED_MODEL, out_dir: Path | str = OUT_DIR):
        self.embed_model = embed_model
        self.out_dir = Path(out_dir)

        # In-memory writing samples captured from UI / DB
        self.samples: List[str] = []
        self.sources: List[str] = []

        # Cached style metrics from user samples
        self.style_metrics: Dict = {}

        # Analyzer for writing style
        self._analyzer = StyleAnalyzer()

        # 🔹 Auto-load samples from the database on startup
        try:
            db_samples = get_all_samples()
            if db_samples:
                texts = [s.content for s in db_samples]
                sources = [s.source or "db" for s in db_samples]
                self.add_samples(texts, sources)
                print(f"Loaded {len(self.samples)} samples from DB into trainer.")
        except Exception as e:
            print("Warning: could not load DB samples into trainer:", e)

    # ----------- UI Utility Methods -----------

    def add_samples(self, texts: List[str], sources: List[str]) -> None:
        """Add new samples (from UI or file) and recompute style metrics."""
        for text, src in zip(texts, sources):
            t = (text or "").strip()
            if not t:
                continue
            self.samples.append(t)
            self.sources.append(src or "unknown")

        combined = "\n\n".join(self.samples)
        if combined:
            self.style_metrics = self._analyzer.analyze_text(combined)

    def clear_samples(self) -> None:
        """Clear in-memory samples and style metrics (does NOT touch DB)."""
        self.samples.clear()
        self.sources.clear()
        self.style_metrics = {}

    def get_sample_count(self) -> int:
        return len(self.samples)

    # ------------ Build Embedding Index ------------

    def build(self):
        """
        Build embeddings and HNSW index based on sample .txt files in data/samples/.
        Used when running offline training.
        """
        samples = load_samples()
        if not samples:
            raise RuntimeError("No samples found in data/samples. Add .txt files and try again.")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        texts = [s["text"] for s in samples]

        # Create embeddings
        model = SentenceTransformer(self.embed_model)
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Initialize index
        dim = embeddings.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        index.add_items(embeddings, np.arange(len(embeddings)))
        index.set_ef(50)

        # Save
        index.save_index(str(self.out_dir / "hnsw.index"))
        np.save(self.out_dir / "embeddings.npy", embeddings)
        with open(self.out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print("Index built successfully")
        return {
            "num_samples": len(samples),
            "index_path": str(self.out_dir / "hnsw.index"),
            "embeddings_path": str(self.out_dir / "embeddings.npy"),
            "meta_path": str(self.out_dir / "meta.json"),
        }


# Keep compatibility with old CLI:
def build_index():
    trainer = PersonaTrainer()
    return trainer.build()
