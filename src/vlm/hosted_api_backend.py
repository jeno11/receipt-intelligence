import base64
import json
import os
import time
from pathlib import Path
from openai import OpenAI, RateLimitError

from src.schema import ReceiptExtraction
from src.vlm.base import ExtractionBackend

_PROMPT = """You are extracting structured data from a receipt image. \
    Respond with ONLY a JSON object (no markdown, no commentary) with exactly these keys:
    
    {
        "vendor_name": string or null,
        "vendor_address": string or null,
        "transaction_date": string or null,
        "total": number or null
    }
    
    use the exact text as printed on the receipt for vendor_name and vendor_address \
    - do not correct spelling or spacing, copy it literally. For total, use the final \
    amount actually paid (after any rounding adjustment, not the pre-rounding total). \
    If a field is not visible on the receiot, use null."""
_MIN_SECONDS_BETWEEN_CALLS = 4.5 

def _encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class HostedApiVlmBackend(ExtractionBackend):
    name = "hosted_api_vlm"

    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["VLM_API_KEY"],
            base_url=os.environ["VLM_API_BASE_URL"],
        )
        self.model = os.environ["VLM_MODEL"]
        self._last_call_time = 0.0

    def _wait_for_rate_limit(self):
        elapsed = time.monotonic() - self._last_call_time
        remaining = _MIN_SECONDS_BETWEEN_CALLS - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call_time = time.monotonic()

    def extract(self, image_path: Path, ocr_text: str | None = None, max_retries: int = 3) -> ReceiptExtraction:
        b64_image = _encode_image(image_path)
        for attempt in range(max_retries):
            self._wait_for_rate_limit()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": _PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                            ]
                        }
                    ],
                    response_format={"type": "json_object"}
                )
                raw = response.choices[0].message.content
                data = json.loads(raw)
                return ReceiptExtraction(**data)
            except RateLimitError:
                wait = 15 * (attempt + 1)
                print(f" rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
        raise RuntimeError("Rate limited after multiple retries")

if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.data_loader import load_example

    load_dotenv()
    backend = HostedApiVlmBackend()
    example = load_example("000")
    result = backend.extract(example.image_path)
    print("Ground truth: ", example.ground_truth)
    print("Extracted: ", result)