"""
Unit tests for persona training and style analyzer modules.
Can be executed with: python -m unittest tests/test_persona.py
"""
import unittest
import tempfile
import shutil
import os
from pathlib import Path

from src.style_analyzer import StyleAnalyzer, load_samples
from src.persona_trainer import PersonaTrainer


class TestStyleAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = StyleAnalyzer()

    def test_analyze_empty_text(self):
        result = self.analyzer.analyze_text("")
        self.assertEqual(result, {})

    def test_analyze_sample_text(self):
        text = "Machine learning models require robust evaluation. Clean data is critical for accurate inference."
        metrics = self.analyzer.analyze_text(text)
        self.assertIn("avg_sentence_length", metrics)
        self.assertIn("vocab_richness", metrics)
        self.assertIn("readability_score", metrics)
        self.assertIn("formality_score", metrics)
        self.assertGreater(metrics["avg_sentence_length"], 0)

    def test_compare_styles_similarity(self):
        metrics_a = {
            "avg_sentence_length": 12.0,
            "vocab_richness": 0.8,
            "readability_score": 75.0,
            "formality_score": 50.0,
        }
        # Identical metrics should have 100% similarity
        score = self.analyzer.compare_styles(metrics_a, metrics_a)
        self.assertEqual(score, 100.0)


class TestPersonaTrainer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_trainer_empty_initialization(self):
        trainer = PersonaTrainer(out_dir=self.test_dir)
        trainer.clear_samples()
        self.assertEqual(trainer.get_sample_count(), 0)

    def test_trainer_add_samples(self):
        trainer = PersonaTrainer(out_dir=self.test_dir)
        trainer.clear_samples()
        trainer.add_samples(["This is a test writing sample for persona training."], ["test"])
        self.assertEqual(trainer.get_sample_count(), 1)
        self.assertTrue(bool(trainer.style_metrics))


if __name__ == "__main__":
    unittest.main()
