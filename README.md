# Receipt Intelligence

An end-to-end document-intelligence pipeline that extracts structured data
(vendor name, address, date, total) from receipt images — comparing a
rule-based OCR baseline against a hosted vision-language model, served
through a FastAPI backend with a web UI, containerized, tested in CI, and
deployed live.

**Live demo:** https://receipt-intelligence-86w3.onrender.com
*(free-tier hosting — the first request after a few minutes of inactivity
can take 30–60s to wake the server up; it's not broken, just asleep.)*

## Results

Both backends evaluated on the same held-out set of 107 real receipts from
the ICDAR 2019 SROIE dataset:

| field            | heuristic (rule-based) | VLM (Gemini 3.5 Flash-Lite) | change |
|------------------|------------------------:|-----------------------------:|-------:|
| transaction_date | 83.2%                   | 98.1%                        | +14.9pp |
| total            | 32.7%                   | 80.4%                        | **+47.7pp** |
| vendor_name      | 68.2%                   | 72.9%                        | +4.7pp |
| vendor_address   | 15.9%                   | ~63%                         | **+47pp** |

The rule-based backend hits a genuine, documented ceiling on `total` and
`vendor_address` — fields that depend on OCR-line layout variation the
regexes can't generalize to. The VLM comparison was built specifically to
test that hypothesis, and the two weakest heuristic fields are exactly
where it improves the most.

## How it works

```
Receipt image
      |
      v
ExtractionBackend (interface, strategy pattern)
      |
      +---> HeuristicOcrBackend    (regex/rules over pre-extracted OCR text)
      |
      +---> HostedApiVlmBackend    (Gemini, via an OpenAI-compatible endpoint)
      |
      v
ReceiptExtraction (Pydantic schema)
      |
      v
FastAPI (/health, /extract) + static web UI
      |
      v
Docker image --> GitHub Actions CI --> Render (live deploy)
```

Both backends implement the same `ExtractionBackend` interface, so the API
and web UI don't need to know which one is running underneath. In practice
only the VLM backend is wired into `/extract`, since the heuristic backend
depends on SROIE's pre-extracted OCR annotations rather than doing its own
image-to-text step — a deliberate scope boundary, not an oversight.

## Tech stack

Python, FastAPI, Pydantic, Google Gemini (via an OpenAI-compatible API),
pytesseract/OpenCV (heuristic backend), Docker, GitHub Actions, Render.

## Project structure

```
src/
  schema.py              ReceiptExtraction Pydantic model
  data_loader.py          SROIE dataset loader + deterministic train/eval split
  evaluate.py              field-accuracy evaluation harness
  api.py                    FastAPI app (/health, /extract)
  vlm/
    base.py                   ExtractionBackend interface
    heuristic_ocr_backend.py   rule-based baseline
    hosted_api_backend.py       Gemini VLM backend
scripts/
  run_vlm_eval.py          runs the VLM backend over the full eval set
static/
  index.html                 drag-and-drop web UI
Dockerfile
.github/workflows/         CI: builds the Docker image on every push
```

## Running it locally

```bash
git clone git@github.com:jeno11/receipt-intelligence.git
cd receipt-intelligence
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your own Gemini API key
uvicorn src.api:app --reload
```

Open `http://localhost:8000` and upload a receipt image.

## Running it with Docker

```bash
docker build -t receipt-intelligence .
docker run -p 8000:8000 --env-file .env receipt-intelligence
```

## API

- `GET /health` — liveness check.
- `POST /extract` — multipart image upload, returns a `ReceiptExtraction`
  JSON object (`vendor_name`, `vendor_address`, `transaction_date`, `total`).

## Reproducing the evaluation

The SROIE dataset isn't committed to the repo (345MB, its own license
terms). Download the ICDAR 2019 SROIE Task 3 data — e.g. from
[this GitHub mirror](https://github.com/zzzDavid/ICDAR-2019-SROIE) — and
place it at `data/raw/sroie/` with `images/`, `entities/`, and `ocr_boxes/`
subfolders.

```bash
python -m src.evaluate            # heuristic baseline
python scripts/run_vlm_eval.py    # VLM backend (needs VLM_API_KEY set)
```

## Notes & limitations

- The Gemini free tier is rate-limited to 15 requests/minute; a full
  107-receipt eval run takes 8–12 minutes because of that pacing, not
  because anything's stuck.
- The VLM occasionally "corrects" an oddly-formatted real vendor name into
  a more plausible-sounding wrong one, despite an explicit
  copy-literally instruction in the prompt — a distinct failure mode the
  heuristic backend doesn't have (it can only be too literal, never too
  clever).
- No automated test suite yet — a natural next addition.

## What I'd do with more time

Prompt-engineering iteration on the VLM backend (deliberately deferred in
favor of shipping the full pipeline first), a proper test suite, and
wiring OCR into the heuristic backend so it can also handle arbitrary
uploaded photos instead of only SROIE's pre-annotated text.
