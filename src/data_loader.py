import json
import hashlib
from pathlib import Path
from dataclasses import dataclass

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "sroie"
EVAL_FRACTION = 0.15

@dataclass
class SroieExample:
    receipt_id: str
    image_path: Path
    ground_truth: dict
    ocr_text: str

def parse_box_csv_line(line:str) -> tuple[str, tuple[int, ...]]:
    """
    Parses one line of a box CSV: 8 coordinate numbers, then
    free text (which may itself contain commas). 
    """
    parts = line.strip().split(",", maxsplit=8)
    coords = tuple(int(p) for p in parts[:8])
    text = parts[8]
    return text, coords

def load_example(receipt_id: str) -> SroieExample:
    image_path = DATA_DIR / "images" / f"{receipt_id}.jpg"
    entity_path = DATA_DIR / "entities" / f"{receipt_id}.json"
    box_path = DATA_DIR / "ocr_boxes" / f"{receipt_id}.csv"

    with open(entity_path) as f:
        ground_truth = json.load(f)

    lines = []
    with open(box_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                text, _coords = parse_box_csv_line(line)
                lines.append(text)
    ocr_text = "\n".join(lines)

    return SroieExample(
        receipt_id=receipt_id,
        image_path=image_path,
        ground_truth=ground_truth,
        ocr_text=ocr_text
    )

def load_all_exaples() -> list[SroieExample]:
    images_dir = DATA_DIR / "images"
    receipt_ids = sorted(p.stem for p in images_dir.glob("*.jpg"))
    return [load_example(rid) for rid in receipt_ids]

def _is_eval(receipt_id: str) -> bool:
    digest = hashlib.sha256(receipt_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < EVAL_FRACTION

def load_split() -> tuple[list[SroieExample], list[SroieExample]]:
    all_examples = load_all_exaples()
    train = [e for e in all_examples if not _is_eval(e.receipt_id)]
    eval_ = [e for e in all_examples if _is_eval(e.receipt_id)]
    return train, eval_

if __name__ == "__main__":
    train, eval_ = load_split()
    print(f"Loaded {len(train) + len(eval_)} eamples: {len(train)} train / {len(eval_)} eval")
    print("Sample:", train[0].receipt_id, train[0].ground_truth)
