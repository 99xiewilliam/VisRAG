#!/usr/bin/env python3
"""
Batch index all PDFs under a dataset directory via POST /api/v1/index/pdf.
Builds full collections: image collection, per-page text collections, and optionally global text collection.
"""
import argparse
import os
import sys

import requests

BASE_URL = os.environ.get("VISRAG_BASE_URL", "http://127.0.0.1:8000")


def main():
    parser = argparse.ArgumentParser(description="Batch index PDFs via /api/v1/index/pdf")
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing PDF files")
    parser.add_argument("--base-url", default=BASE_URL, help="VisRAG API base URL")
    parser.add_argument("--dry-run", action="store_true", help="List PDFs only, do not call API")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        print(f"Error: not a directory: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    pdfs = sorted(
        [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.lower().endswith(".pdf")
        ]
    )
    if not pdfs:
        print(f"No PDFs found under {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) under {dataset_dir}")
    for p in pdfs:
        print(f"  - {os.path.basename(p)}")
    if args.dry_run:
        return

    base_url = args.base_url.rstrip("/")
    endpoint = f"{base_url}/api/v1/index/pdf"

    for pdf_path in pdfs:
        filename = os.path.basename(pdf_path)
        pdf_name = os.path.splitext(filename)[0]
        print(f"\nIndexing: {filename} (pdf_name={pdf_name}) ...")
        try:
            with open(pdf_path, "rb") as f:
                files = {"pdf": (filename, f, "application/pdf")}
                data = {"pdf_name": pdf_name}
                r = requests.post(endpoint, files=files, data=data, timeout=600)
            r.raise_for_status()
            res = r.json()
            print(
                f"  pdf_name={res.get('pdf_name')} | "
                f"image_collection={res.get('image_collection')} | "
                f"images_indexed={res.get('images_indexed')} | "
                f"text_chunks_indexed={res.get('text_chunks_indexed')} | "
                f"text_collections={len(res.get('text_collections') or [])} | "
                f"global={res.get('global_text_collection')} | "
                f"global_chunks={res.get('global_text_chunks_indexed')} | "
                f"pages={res.get('pages')}"
            )
        except requests.RequestException as e:
            print(f"  Error: {e}", file=sys.stderr)
            if hasattr(e, "response") and e.response is not None:
                try:
                    print(f"  Body: {e.response.text[:500]}", file=sys.stderr)
                except Exception:
                    pass

    print("\nDone.")


if __name__ == "__main__":
    main()
