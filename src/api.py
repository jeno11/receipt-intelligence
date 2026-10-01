import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

load_dotenv()

from src.schema import ReceiptExtraction
from src.vlm.hosted_api_backend import HostedApiVlmBackend

app = FastAPI(title="Receipt Intelligence API")
backend = HostedApiVlmBackend()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/extract", response_model=ReceiptExtraction)
def extract(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got content-type '{file.content_type}'",
        )
    
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        return backend.extract(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"The model returned data we couldn't parse: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Extraction failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

app.mount("/", StaticFiles(directory="static", html=True), name="static")