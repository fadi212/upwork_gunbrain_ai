from fastapi import FastAPI, Form, UploadFile, HTTPException
from typing import Optional
from enum import Enum
from src.pipeline.index import upload_and_index

class SourceType(str, Enum):
    book = "books"
    magazine = "magazines"

app = FastAPI()

@app.post("/upload-and-index/")
async def upload_and_index_endpoint(
        source_type: SourceType,
        file: Optional[UploadFile] = None,
        s3_url: Optional[str] = Form(None),
        title: Optional[str] = Form(None),
        manufacturer: Optional[str] = Form(None),
        website_url: Optional[str] = Form(None)
):
    if not file and not s3_url:
        raise HTTPException(
            status_code=400, detail="Either 'file' or 's3_url' must be provided.")
    if file and s3_url:
        raise HTTPException(
            status_code=400, detail="Provide only one of 'file' or 's3_url', not both.")

    if file:
        status = upload_and_index(
            title, source_type, manufacturer, website_url, file=file)
    else:
        status = upload_and_index(
            title, source_type, manufacturer, website_url, s3_url=s3_url)

    if status.get("already_indexed", False):
        return {"status": "Document already indexed", "details": status}
    elif status.get("success", False):
        return {"status": "Document indexed successfully", "details": status}
    elif status.get("error"):
        return {"status": "Indexing failed", "error": status["error"]}
    else:
        return {"status": "Indexing failed", "details": status}
