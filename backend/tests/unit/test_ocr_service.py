"""
test_ocr_service.py — Unit tests for ocr_service.py

Strategy:
  - Uses FAKE images synthesised with PIL (no Tesseract binary required for
    most tests) and monkey-patching pytesseract for CI environments.
  - When Tesseract IS available, integration-level smoke tests are run.
  - All tests use deterministic, synthetic data.

Run with:
    pytest backend/tests/unit/test_ocr_service.py -v
"""

from __future__ import annotations

import sys
import os

# Ensure the backend package root is on the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import unittest
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from app.models.findings import BoundingBox, OCRWord
from app.services.ocr_service import (
    MIN_WORD_CONFIDENCE,
    _load_image,
    _preprocess_image,
    _scale_bbox,
    get_word_at_char_offset,
    get_words_in_char_span,
    run_ocr,
    words_to_plain_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_white_image(width: int = 400, height: int = 100) -> Image.Image:
    """Create a plain white RGB image for testing."""
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def _fake_tesseract_data(texts, lefts, tops, widths, heights, confs):
    """Build a dict matching pytesseract.image_to_data output format."""
    return {
        "text": texts,
        "conf": confs,
        "left": lefts,
        "top": tops,
        "width": widths,
        "height": heights,
    }


def _make_sample_words() -> list[OCRWord]:
    """Return a small deterministic list of OCRWords for testing helpers."""
    return [
        OCRWord("Hello", BoundingBox(10, 10, 50, 20), confidence=0.92, page=0),
        OCRWord("Phone:", BoundingBox(70, 10, 60, 20), confidence=0.90, page=0),
        OCRWord("9876543210", BoundingBox(140, 10, 80, 20), confidence=0.88, page=0),
        OCRWord("Email:", BoundingBox(10, 40, 55, 20), confidence=0.91, page=0),
        OCRWord("user@example.com", BoundingBox(75, 40, 120, 20), confidence=0.95, page=0),
    ]


# ===========================================================================
# BoundingBox tests
# ===========================================================================

class TestBoundingBox(unittest.TestCase):

    def test_basic_creation(self):
        bb = BoundingBox(x=10, y=20, width=100, height=50)
        self.assertEqual(bb.x, 10)
        self.assertEqual(bb.y, 20)
        self.assertEqual(bb.width, 100)
        self.assertEqual(bb.height, 50)

    def test_as_dict(self):
        bb = BoundingBox(5, 5, 50, 25)
        d = bb.as_dict()
        self.assertEqual(d, {"x": 5, "y": 5, "width": 50, "height": 25})

    def test_from_tesseract(self):
        bb = BoundingBox.from_tesseract(left=10, top=20, width=100, height=30)
        self.assertEqual(bb.x, 10)
        self.assertEqual(bb.y, 20)

    def test_union_single(self):
        bb = BoundingBox(10, 10, 50, 20)
        result = BoundingBox.union([bb])
        self.assertEqual(result, bb)

    def test_union_multiple(self):
        boxes = [
            BoundingBox(0, 0, 50, 20),
            BoundingBox(60, 0, 50, 20),
            BoundingBox(0, 25, 120, 15),
        ]
        result = BoundingBox.union(boxes)
        self.assertEqual(result.x, 0)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.width, 110)   # 0 to 60+50=110
        self.assertEqual(result.height, 40)   # 0 to 25+15=40

    def test_union_empty_raises(self):
        with self.assertRaises(ValueError):
            BoundingBox.union([])

    def test_negative_dimension_raises(self):
        with self.assertRaises(ValueError):
            BoundingBox(0, 0, -1, 10)


# ===========================================================================
# OCRWord tests
# ===========================================================================

class TestOCRWord(unittest.TestCase):

    def test_valid_word(self):
        word = OCRWord("test", BoundingBox(0, 0, 40, 15), confidence=0.85, page=0)
        self.assertEqual(word.text, "test")
        self.assertAlmostEqual(word.confidence, 0.85)

    def test_confidence_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            OCRWord("x", BoundingBox(0, 0, 10, 10), confidence=1.5)

    def test_confidence_zero_valid(self):
        word = OCRWord("x", BoundingBox(0, 0, 10, 10), confidence=0.0)
        self.assertEqual(word.confidence, 0.0)

    def test_page_default_zero(self):
        word = OCRWord("hello", BoundingBox(0, 0, 50, 20), confidence=0.9)
        self.assertEqual(word.page, 0)


# ===========================================================================
# Image loading tests
# ===========================================================================

class TestLoadImage(unittest.TestCase):

    def test_load_pil_image(self):
        img = _make_white_image()
        result = _load_image(img)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (400, 100))

    def test_load_bytes(self):
        import io
        img = _make_white_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        result = _load_image(buf.read())
        self.assertIsInstance(result, Image.Image)

    def test_load_numpy_rgb(self):
        arr = np.zeros((100, 200, 3), dtype=np.uint8)
        result = _load_image(arr)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (200, 100))

    def test_load_numpy_grayscale(self):
        arr = np.zeros((100, 200), dtype=np.uint8)
        result = _load_image(arr)
        self.assertIsInstance(result, Image.Image)

    def test_load_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            _load_image(12345)  # type: ignore


# ===========================================================================
# Image pre-processing tests
# ===========================================================================

class TestPreprocessImage(unittest.TestCase):

    def test_returns_numpy_array(self):
        img = _make_white_image()
        result = _preprocess_image(img)
        self.assertIsInstance(result, np.ndarray)

    def test_output_is_2d_grayscale(self):
        img = _make_white_image()
        result = _preprocess_image(img)
        self.assertEqual(result.ndim, 2)

    def test_small_image_upscaled(self):
        """Images smaller than UPSCALE_MIN_DIMENSION should be upscaled."""
        img = Image.new("RGB", (100, 50), color=(200, 200, 200))  # very small
        result = _preprocess_image(img)
        # After upscaling the short dimension (50) should have been enlarged
        self.assertGreater(result.shape[1], 100)

    def test_large_image_not_upscaled(self):
        """Images already large enough should NOT be upscaled."""
        img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        result = _preprocess_image(img)
        # Width should stay close to original (no upscaling)
        self.assertAlmostEqual(result.shape[1], 800, delta=50)


# ===========================================================================
# Scale bbox tests
# ===========================================================================

class TestScaleBbox(unittest.TestCase):

    def test_scale_1_unchanged(self):
        bb = BoundingBox(10, 20, 100, 40)
        result = _scale_bbox(bb, 1.0)
        self.assertEqual(result, bb)

    def test_scale_2_halved(self):
        bb = BoundingBox(20, 40, 200, 80)
        result = _scale_bbox(bb, 2.0)
        self.assertEqual(result.x, 10)
        self.assertEqual(result.y, 20)
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 40)


# ===========================================================================
# Corpus and offset helpers
# ===========================================================================

class TestWordsToPlainText(unittest.TestCase):

    def test_basic(self):
        words = _make_sample_words()
        corpus = words_to_plain_text(words)
        self.assertIn("Hello", corpus)
        self.assertIn("Phone:", corpus)
        self.assertIn("9876543210", corpus)

    def test_empty(self):
        self.assertEqual(words_to_plain_text([]), "")

    def test_spacing(self):
        words = [
            OCRWord("A", BoundingBox(0, 0, 10, 10), confidence=0.9),
            OCRWord("B", BoundingBox(15, 0, 10, 10), confidence=0.9),
        ]
        corpus = words_to_plain_text(words)
        self.assertEqual(corpus, "A B")


class TestGetWordAtCharOffset(unittest.TestCase):

    def _simple_words(self):
        # corpus: "cat dog"
        return [
            OCRWord("cat", BoundingBox(0, 0, 30, 15), confidence=0.9),
            OCRWord("dog", BoundingBox(40, 0, 30, 15), confidence=0.9),
        ]

    def test_first_word(self):
        words = self._simple_words()
        self.assertEqual(get_word_at_char_offset(words, 0), 0)  # 'c'
        self.assertEqual(get_word_at_char_offset(words, 2), 0)  # 't'

    def test_second_word(self):
        words = self._simple_words()
        self.assertEqual(get_word_at_char_offset(words, 4), 1)  # 'd'
        self.assertEqual(get_word_at_char_offset(words, 6), 1)  # 'g'

    def test_out_of_range_returns_none(self):
        words = self._simple_words()
        self.assertIsNone(get_word_at_char_offset(words, 999))


class TestGetWordsInCharSpan(unittest.TestCase):

    def _words(self):
        # corpus: "Hello Phone: 9876543210"
        # positions: 0-4 5-10 11-20
        return [
            OCRWord("Hello", BoundingBox(0, 0, 50, 20), confidence=0.9),
            OCRWord("Phone:", BoundingBox(60, 0, 60, 20), confidence=0.9),
            OCRWord("9876543210", BoundingBox(130, 0, 80, 20), confidence=0.88),
        ]

    def test_exact_word(self):
        words = self._words()
        # "Hello" = chars 0-4
        indices = get_words_in_char_span(words, 0, 5)
        self.assertIn(0, indices)
        self.assertNotIn(1, indices)
        self.assertNotIn(2, indices)

    def test_multi_word_span(self):
        words = self._words()
        # Span covering "Phone:" and "9876543210"
        corpus = words_to_plain_text(words)
        start = corpus.index("Phone:")
        end = corpus.index("9876543210") + len("9876543210")
        indices = get_words_in_char_span(words, start, end)
        self.assertIn(1, indices)
        self.assertIn(2, indices)
        self.assertNotIn(0, indices)


# ===========================================================================
# run_ocr — with mocked pytesseract
# ===========================================================================

class TestRunOCRMocked(unittest.TestCase):
    """Test run_ocr() with pytesseract replaced by a mock."""

    def _make_mock_data(self):
        return _fake_tesseract_data(
            texts=["", "Room:", "1208", "Guest:", "John"],
            lefts=[0, 10, 80, 10, 80],
            tops=[0, 10, 10, 40, 40],
            widths=[0, 55, 45, 55, 40],
            heights=[0, 20, 20, 20, 20],
            confs=[-1, 92, 88, 91, 85],  # -1 = paragraph marker
        )

    @patch("app.services.ocr_service._TESSERACT_AVAILABLE", True)
    @patch("app.services.ocr_service.pytesseract")
    def test_returns_list_of_ocr_words(self, mock_tess):
        mock_tess.image_to_data.return_value = self._make_mock_data()
        mock_tess.Output.DICT = "dict"

        img = _make_white_image(400, 100)
        words = run_ocr(img)

        self.assertIsInstance(words, list)
        self.assertTrue(all(isinstance(w, OCRWord) for w in words))

    @patch("app.services.ocr_service._TESSERACT_AVAILABLE", True)
    @patch("app.services.ocr_service.pytesseract")
    def test_filters_low_confidence(self, mock_tess):
        data = _fake_tesseract_data(
            texts=["Good", "Bad"],
            lefts=[0, 100],
            tops=[0, 0],
            widths=[40, 30],
            heights=[20, 20],
            confs=[90, 10],  # 10 < MIN_WORD_CONFIDENCE (30)
        )
        mock_tess.image_to_data.return_value = data
        mock_tess.Output.DICT = "dict"

        words = run_ocr(_make_white_image())
        texts = [w.text for w in words]
        self.assertIn("Good", texts)
        self.assertNotIn("Bad", texts)

    @patch("app.services.ocr_service._TESSERACT_AVAILABLE", True)
    @patch("app.services.ocr_service.pytesseract")
    def test_filters_whitespace_tokens(self, mock_tess):
        data = _fake_tesseract_data(
            texts=["  ", "Hello"],
            lefts=[0, 50],
            tops=[0, 0],
            widths=[20, 40],
            heights=[20, 20],
            confs=[85, 90],
        )
        mock_tess.image_to_data.return_value = data
        mock_tess.Output.DICT = "dict"

        words = run_ocr(_make_white_image())
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].text, "Hello")

    @patch("app.services.ocr_service._TESSERACT_AVAILABLE", True)
    @patch("app.services.ocr_service.pytesseract")
    def test_confidence_normalised_to_0_1(self, mock_tess):
        data = _fake_tesseract_data(
            texts=["Word"],
            lefts=[0],
            tops=[0],
            widths=[40],
            heights=[20],
            confs=[75],
        )
        mock_tess.image_to_data.return_value = data
        mock_tess.Output.DICT = "dict"

        words = run_ocr(_make_white_image())
        self.assertAlmostEqual(words[0].confidence, 0.75)

    @patch("app.services.ocr_service._TESSERACT_AVAILABLE", False)
    def test_raises_when_tesseract_unavailable(self):
        with self.assertRaises(RuntimeError):
            run_ocr(_make_white_image())

    @patch("app.services.ocr_service._TESSERACT_AVAILABLE", True)
    @patch("app.services.ocr_service.pytesseract")
    def test_page_number_propagated(self, mock_tess):
        data = _fake_tesseract_data(
            texts=["Hello"],
            lefts=[0], tops=[0], widths=[40], heights=[20], confs=[80]
        )
        mock_tess.image_to_data.return_value = data
        mock_tess.Output.DICT = "dict"

        words = run_ocr(_make_white_image(), page=3)
        self.assertEqual(words[0].page, 3)


if __name__ == "__main__":
    unittest.main()
