import re
from datetime import datetime

from src.data_loader import load_split
from src.schema import ReceiptExtraction
from dateutil import parser as date_parser
from src.vlm.heuristic_ocr_backend import HeuristicOcrBackend

_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")
TOTAL_TOLERANCE = 0.01

def normalize_text(s: str | None) -> str | None:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s) # collapse multiple spaces / newlines into one
    s = re.sub(r"[^\w\s]", "", s) # strip punctuation
    return s

def normalize_date(s: str | None) -> tuple[int, int, int] | None:
    """
    Parse a data string into a (year, month, day) tuple, trying 
    each known format in turn. Returns None if it doesn't match any of them
    """
    if not s:
        return None
    s = s.strip()
    try:
        if _ISO_DATE_PATTERN.match(s):
            dt = datetime.fromisoformat(s[:10])
        else:
            dt = date_parser.parse(s.strip(), dayfirst=True)
        return (dt.year, dt.month, dt.day)
    except (ValueError, OverflowError):
       return None

def evaluate(backend, examples):
    date_correct = 0
    total_correct = 0
    vendor_correct = 0
    address_correct = 0
    results = []

    for example in examples:
        try:
            prediction = backend.extract(example.image_path, ocr_text=example.ocr_text)
            results.append((example.receipt_id, prediction, example.ground_truth))
        except Exception as e:
            print(f" [{example.receipt_id}] extraction failed: {e}")
            prediction = ReceiptExtraction()

        if normalize_text(prediction.vendor_name) == normalize_text(example.ground_truth.get("company")):
            if normalize_text(example.ground_truth.get("company")): # don't count if ground truth itself is empty
                vendor_correct += 1

        if normalize_text(prediction.vendor_address) == normalize_text(example.ground_truth.get("address")):
            if normalize_text(example.ground_truth.get("address")):
                address_correct += 1

        gt_date = normalize_date(example.ground_truth.get("date"))
        pred_date = normalize_date(prediction.transaction_date)
        if gt_date is not None and gt_date == pred_date:
            date_correct += 1

        gt_total_raw = example.ground_truth.get("total")
        try:
            gt_total = float(gt_total_raw) if gt_total_raw else None
        except ValueError:
            gt_total = None
        if (
            gt_total is not None 
            and prediction.total is not None
            and abs(gt_total - prediction.total) <= TOTAL_TOLERANCE
        ):
            total_correct += 1

    n = len(examples)
    print(f"Evaluated on {n} receipts")
    print(f"Date accuracy: {date_correct}/{n} = {date_correct / n:.1%}")
    print(f"Total accuracy: {total_correct}/{n} = {total_correct / n:.1%}")
    print(f"Vendor accuracy: {vendor_correct} / {n} = {vendor_correct / n:.1%}")
    print(f"Vendor address accuracy: {address_correct} / {n} = {address_correct / n:.1%}")
    return results

if __name__ == "__main__":
    _, eval_examples = load_split()
    backend = HeuristicOcrBackend()
    evaluate(backend, eval_examples)

    print("\n--- Vendor address mismatches (first 10) ---")
    shown = 0
    for example in eval_examples:
        prediction = backend.extract(example.image_path, ocr_text=example.ocr_text)
        gt_address = example.ground_truth.get("address")
        if normalize_text(prediction.vendor_address) != normalize_text(gt_address):
            print(f"{example.receipt_id}: gt={gt_address!r} pred={prediction.vendor_address!r}")
            shown += 1
            if shown >= 10:
                break