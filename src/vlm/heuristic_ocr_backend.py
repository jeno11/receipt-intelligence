import re
from pathlib import Path

from src.schema import ReceiptExtraction
from src.vlm.base import ExtractionBackend

_DATE_PATTERN = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
_MONEY_PATTERN = re.compile(r"\d+\.\d{2}")
_EXCLUDE_KEYWORDS = (
    "subtotal", 
    "sub total", 
    "change", 
    "cash", 
    "tender", 
    "total qty", 
    "total quantity",
)

def find_date(ocr_text: str) -> str | None:
    match = _DATE_PATTERN.search(ocr_text)
    return match.group(1) if match else None

def _amount_on_or_after(lines: list[str], index: int) -> float | None:
    """
    Looks for a money amount on this line, or the single line right after
    it (label and amount are sometimes split across two OCR lines).
    """
    amounts = _MONEY_PATTERN.findall(lines[index])
    if amounts:
        return float(amounts[-1])
    if index + 1 < len(lines):
        amounts = _MONEY_PATTERN.findall(lines[index + 1])
        if amounts:
            return float(amounts[-1])
    return None

def _first_matching_amount(lines: list[str], matches) -> float | None:
    """
    Scans lines for the first one where `matches(lower_line)` is True,
    and returns the amount found on or just after it, if any.
    """
    for i, line in enumerate(lines):
        if matches(line.lower()):
            amount = _amount_on_or_after(lines, i)
            if amount is not None:
                return amount
    return None

def find_total(ocr_text: str) -> float | None:
    lines = ocr_text.splitlines()

    def is_rounded_total(lower:str) -> bool:
        return "round" in lower and "total" in lower

    def is_plain_total(lower: str) -> bool:
        return "total" in lower and not any(bad in lower for bad in _EXCLUDE_KEYWORDS)

    amount = _first_matching_amount(lines, is_rounded_total)
    if amount is not None:
        return amount

    return _first_matching_amount(lines, is_plain_total)

class HeuristicOcrBackend(ExtractionBackend):
    name = "heuristic_ocr"

    def extract(self, image_path: Path, ocr_text = str | None) -> ReceiptExtraction:
        if not ocr_text:
            raise ValueError("HeuristicOcrBackend requires ocr_text")

        return ReceiptExtraction(
            transaction_date=find_date(ocr_text),
            total=find_total(ocr_text),
        )

if __name__ == "__main__":
    from src.data_loader import load_example

    backend = HeuristicOcrBackend()
    example = load_example("000")
    result = backend.extract(example.image_path, ocr_text=example.ocr_text)
    print("Extracted: ", result)