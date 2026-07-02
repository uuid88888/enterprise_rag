"""OCR provider base classes."""
from __future__ import annotations

from abc import ABC, abstractmethod


class OCRProvider(ABC):
    """Abstract OCR provider."""

    @abstractmethod
    def extract_image_bytes(self, image: bytes, mime_type: str = "image/png") -> str:
        """Extract text from image bytes."""

