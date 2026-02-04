import argparse
import os
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi_app.utils import init_logging, get_logger
from fastapi_app.service import get_index_service

logger = get_logger(__name__)


def list_pdfs(dataset_dir: str) -> List[str]:
    files: List[str] = []
    for name in os.listdir(dataset_dir):
        if name.lower().endswith(".pdf"):
            files.append(os.path.join(dataset_dir, name))
    files.sort()
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="dataset")
    p.add_argument("--limit", type=int, default=0, help="0 means no limit")
    args = p.parse_args()

    init_logging()
    service = get_index_service()

    dataset_dir = os.path.abspath(args.dataset_dir)
    pdfs = list_pdfs(dataset_dir)
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]

    logger.info(f"Found {len(pdfs)} PDFs under {dataset_dir}")
    for i, pdf_path in enumerate(pdfs, start=1):
        logger.info(f"[{i}/{len(pdfs)}] Indexing {pdf_path}")
        try:
            service.index_pdf(pdf_path)
        except Exception:
            logger.exception(f"Failed indexing: {pdf_path}")


if __name__ == "__main__":
    main()

