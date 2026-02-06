#!/usr/bin/env python3
"""
Batch index PDFs into global text chunk collection (text-only baseline).

Usage:
  python scripts/index_global_texts.py --dataset-dir dataset
  python scripts/index_global_texts.py --dataset-dir dataset --base-url http://127.0.0.1:8000
"""
import argparse
import os
import sys
from typing import List


def _list_pdfs(dataset_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(dataset_dir)):
        if name.lower().endswith(".pdf"):
            files.append(os.path.join(dataset_dir, name))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Index PDFs into global text collection")
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing PDFs")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="VisRAG API base URL")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        print(f"Dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 1

    try:
        import requests
    except ImportError:
        print("需要安装 requests: pip install requests", file=sys.stderr)
        return 1

    url = f"{args.base_url.rstrip('/')}/api/v1/index/pdf_text_global"
    pdfs = _list_pdfs(dataset_dir)
    if not pdfs:
        print(f"No PDFs found in {dataset_dir}")
        return 0

    print(f"Found {len(pdfs)} PDFs in {dataset_dir}")
    for i, pdf_path in enumerate(pdfs, 1):
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"[{i}/{len(pdfs)}] Indexing text-only: {pdf_name}")
        with open(pdf_path, "rb") as f:
            files = {"pdf": (os.path.basename(pdf_path), f, "application/pdf")}
            data = {"pdf_name": pdf_name}
            try:
                resp = requests.post(url, data=data, files=files, timeout=300)
            except Exception as e:
                print(f"  !! Request failed: {e}")
                continue
        if resp.status_code != 200:
            print(f"  !! HTTP {resp.status_code}: {resp.text[:200]}")
            continue
        try:
            out = resp.json()
        except Exception:
            print(f"  !! Bad JSON: {resp.text[:200]}")
            continue
        print(
            f"  -> pages={out.get('pages')} chunks={out.get('global_text_chunks_indexed')} "
            f"collection={out.get('global_text_collection')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
