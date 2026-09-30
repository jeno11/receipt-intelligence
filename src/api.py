import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile

load_dotenv()

from src.schema import ReceiptExtraction
from src.vlm.hosted_api_backend import HostedApiVlmBackend

app = FastAPI(title="Receipt Intelligence API")
backend = HostedApiVlmBackend()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/extract", response_model=ReceiptExtraction)
async def extract(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        return backend.extract(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)