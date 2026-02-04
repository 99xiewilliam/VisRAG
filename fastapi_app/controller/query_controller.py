import os
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import get_app_config
from ..schemas import TextQueryRequest, QueryResponse
from ..service import get_query_service
from ..utils import get_logger, ensure_dir, save_upload_file, extract_frame_from_video

router = APIRouter(prefix="/api/v1/query")
logger = get_logger(__name__)


@router.post("/text", response_model=QueryResponse)
def query_text(payload: TextQueryRequest):
    service = get_query_service()
    return service.query(
        query_text=payload.query,
        image_collection=payload.image_collection,
        image_top_k=payload.image_top_k,
        text_top_k=payload.text_top_k,
    )


@router.post("/vision", response_model=QueryResponse)
def query_vision(
    query: Optional[str] = Form(default=None),
    image_collection: Optional[str] = Form(default=None),
    image_top_k: Optional[int] = Form(default=None),
    text_top_k: Optional[int] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
):
    """
    多模态检索：用查询向量在 image_collection 检索最相关图像，再在对应页的 text collection 检索并做 OCR。

    与 Qwen3-VL-Embedding 一致，支持多种输入模态及组合（至少提供 query / image / video 其一）：
    - 仅文本：query
    - 仅图像：image
    - 仅视频：video（内部抽一帧作图像）
    - 文本 + 图像：query + image（多模态向量）
    - 文本 + 视频：query + video（多模态向量，视频抽一帧）
    """
    cfg = get_app_config()
    upload_dir = ensure_dir(os.path.join(cfg.indexing.assets_dir, "uploads", "queries"))
    query_image_path = None

    if image and (image.filename or "").strip():
        filename = f"{uuid.uuid4().hex}_{image.filename}"
        path = save_upload_file(image, upload_dir, filename)
        if os.path.getsize(path) > 0:
            query_image_path = path
    elif video and video.filename:
        video_path = save_upload_file(video, upload_dir, f"{uuid.uuid4().hex}_{video.filename}")
        try:
            query_image_path = extract_frame_from_video(video_path, upload_dir)
        except Exception as e:
            logger.exception("Video frame extraction failed")
            raise HTTPException(status_code=400, detail=f"Video frame extraction failed: {e}")

    if not query and not query_image_path:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'query' (text), 'image', or 'video' is required",
        )

    service = get_query_service()
    return service.query(
        query_text=query or None,
        query_image_path=query_image_path,
        image_collection=image_collection,
        image_top_k=image_top_k,
        text_top_k=text_top_k,
    )
