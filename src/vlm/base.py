from abc import ABC, abstractmethod
from pathlib import Path

from src.schema import ReceiptExtraction

class ExtractionBackend(ABC):
    name: str

    @abstractmethod
    def extract(self, image_path: Path, ocr_text: str | None = None) -> ReceiptExtraction:
        raise NotImplementedError