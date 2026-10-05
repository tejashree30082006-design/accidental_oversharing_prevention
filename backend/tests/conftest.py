"""
conftest.py — Shared pytest fixtures for backend test suite.

Provides:
  - fake_words_factory: factory to create OCRWord lists from arbitrary text
  - white_pil_image: a blank PIL Image for OCR mocking
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PIL import Image

from app.models.findings import BoundingBox, OCRWord


@pytest.fixture
def fake_words_factory():
    """Return a factory function that builds OCRWord lists from text strings."""
    def _factory(sentence: str, conf: float = 0.90, page: int = 0):
        words = []
        x = 10
        y = 10
        for token in sentence.split():
            word = OCRWord(
                text=token,
                bbox=BoundingBox(x=x, y=y, width=len(token) * 8, height=20),
                confidence=conf,
                page=page,
            )
            words.append(word)
            x += len(token) * 8 + 10
        return words
    return _factory


@pytest.fixture
def white_pil_image():
    """Return a plain 400x100 white PIL Image."""
    return Image.new("RGB", (400, 100), color=(255, 255, 255))
