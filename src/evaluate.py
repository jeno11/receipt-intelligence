from datetime import datetime

from src.data_loader import load_split
from src.vlm.heuristic_ocr_backend import HeuristicOcrBackend

_DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
]
TOTAL_TOLERANCE = 0.01

def normalize_date(s: str | None) -> tuple[int, int, int] | None:
    """
    Parse a data string into a (year, month, day) tuple, trying 
    each known format in turn. Returns None if it doesn't match any of them
    """
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s.strip(), fmt)
            return (dt.year, dt.month, dt.day)
        except ValueError:
            continue
    return None

def evaluate(backend, examples):
    date_correct = 0
    total_correct = 0

    for example in examples:
        prediction = backend.extract(example.image_path, ocr_text=example.ocr_text)

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

if __name__ == "__main__":
    _, eval_examples = load_split()
    backend = HeuristicOcrBackend()
    evaluate(backend, eval_examples)