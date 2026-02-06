import json
import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile

from ..config import get_app_config
from ..service import get_index_service
from ..utils import get_logger, ensure_dir, save_upload_file

router = APIRouter(prefix="/api/v1/index")
logger = get_logger(__name__)


def _parse_json_list(raw: Optional[str]) -> List:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


@router.post("")
def index_items(
    texts: Optional[str] = Form(default=None),
    metadatas: Optional[str] = Form(default=None),
    collection: Optional[str] = Form(default=None),
    pdf_name: Optional[str] = Form(default=None),
    page: Optional[int] = Form(default=None),
    images: Optional[List[UploadFile]] = File(default=None),
):
    cfg = get_app_config()
    upload_dir = ensure_dir(os.path.join(cfg.indexing.assets_dir, "uploads"))

    text_list = _parse_json_list(texts)
    meta_list = _parse_json_list(metadatas)

    image_paths: List[str] = []
    if images:
        for img in images:
            filename = f"{uuid.uuid4().hex}_{img.filename}"
            path = save_upload_file(img, upload_dir, filename)
            image_paths.append(path)

    logger.info(f"Index request: texts={len(text_list)}, images={len(image_paths)}")
    service = get_index_service()
    return service.index_items(
        texts=text_list,
        image_paths=image_paths,
        collection=collection,
        metadatas=meta_list,
        pdf_name=pdf_name,
        page=page,
    )


@router.post("/pdf")
def index_pdf(
    pdf: UploadFile = File(...),
    pdf_name: Optional[str] = Form(default=None),
):
    cfg = get_app_config()
    pdf_dir = ensure_dir(os.path.join(cfg.indexing.assets_dir, "uploads", "pdfs"))
    filename = f"{uuid.uuid4().hex}_{pdf.filename}"
    pdf_path = save_upload_file(pdf, pdf_dir, filename)
    service = get_index_service()
    return service.index_pdf(pdf_path, pdf_name=pdf_name)


@router.post("/pdf_text_global")
def index_pdf_text_global(
    pdf: UploadFile = File(...),
    pdf_name: Optional[str] = Form(default=None),
):
    """
    仅构建全量文本 chunk collection（A: text-only baseline）。
    """
    cfg = get_app_config()
    pdf_dir = ensure_dir(os.path.join(cfg.indexing.assets_dir, "uploads", "pdfs"))
    filename = f"{uuid.uuid4().hex}_{pdf.filename}"
    pdf_path = save_upload_file(pdf, pdf_dir, filename)
    service = get_index_service()
    return service.index_pdf_text_global(pdf_path, pdf_name=pdf_name)
