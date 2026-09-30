from dotenv import load_dotenv
load_dotenv()

from src.data_loader import load_split
from src.evaluate import evaluate, normalize_date
from src.vlm.hosted_api_backend import HostedApiVlmBackend

_, eval_examples = load_split()
backend = HostedApiVlmBackend()

results = evaluate(backend, eval_examples)

print("\n--- Date mismatches (first 10) ---")
shown = 0
for receipt_id, prediction, ground_truth in results:
    gt_date = normalize_date(ground_truth.get("date"))
    pred_date = normalize_date(prediction.transaction_date)
    if gt_date != pred_date:
        print(f"{receipt_id}: gt={ground_truth.get('date')!r} pred={prediction.transaction_date!r}")
        shown += 1
        if shown >=10:
            break