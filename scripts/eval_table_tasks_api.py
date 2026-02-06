#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi_app.service.metrics_service import compute_metrics, average_metrics


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_form(url: str, data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    r = requests.post(url, data=data, timeout=timeout)
    r.raise_for_status()
    return r.json()


def query_text_only(
    base_url: str,
    question: str,
    text_collection: Optional[str],
    text_top_k: Optional[int],
    timeout: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"query": question}
    if text_collection:
        payload["text_collection"] = text_collection
    if text_top_k is not None:
        payload["text_top_k"] = text_top_k
    return post_json(f"{base_url}/api/v1/query/text_only", payload, timeout)


def query_vision_only(
    base_url: str,
    question: str,
    image_collection: Optional[str],
    image_top_k: Optional[int],
    timeout: int,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {"query": question}
    if image_collection:
        data["image_collection"] = image_collection
    if image_top_k is not None:
        data["image_top_k"] = image_top_k
    return post_form(f"{base_url}/api/v1/query/vision_only", data, timeout)


def query_vision(
    base_url: str,
    question: str,
    image_collection: Optional[str],
    image_top_k: Optional[int],
    text_top_k: Optional[int],
    timeout: int,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {"query": question}
    if image_collection:
        data["image_collection"] = image_collection
    if image_top_k is not None:
        data["image_top_k"] = image_top_k
    if text_top_k is not None:
        data["text_top_k"] = text_top_k
    return post_form(f"{base_url}/api/v1/query/vision", data, timeout)

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate table QA tasks via query API")
    parser.add_argument("--qa", default="qa/table_understanding_tasks.jsonl", help="QA jsonl path")
    parser.add_argument("--base-url", default=os.environ.get("VISRAG_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--image-collection", default=None)
    parser.add_argument("--text-collection", default=None)
    parser.add_argument("--image-top-k", type=int, default=None)
    parser.add_argument("--text-top-k", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="qa/eval_table_understanding_results.jsonl")
    parser.add_argument("--summary-out", default="qa/eval_table_understanding_summary.json")
    args = parser.parse_args()

    qa_path = os.path.abspath(args.qa)
    if not os.path.exists(qa_path):
        print(f"QA file not found: {qa_path}", file=sys.stderr)
        sys.exit(1)

    items = load_jsonl(qa_path)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    base_url = args.base_url.rstrip("/")
    results: List[Dict[str, Any]] = []
    metrics_text_only: List[Dict[str, float]] = []
    metrics_vision: List[Dict[str, float]] = []
    metrics_vision_only: List[Dict[str, float]] = []

    for idx, item in enumerate(items, start=1):
        qid = item.get("id", idx)
        question = (item.get("question") or "").strip()
        reference = (item.get("answer") or "").strip()
        if not question:
            continue

        print(f"[{idx}/{len(items)}] {qid}: {question}")
        res_text_only = query_text_only(
            base_url,
            question,
            args.text_collection,
            args.text_top_k,
            args.timeout,
        )
        res_vision = query_vision(
            base_url,
            question,
            args.image_collection,
            args.image_top_k,
            args.text_top_k,
            args.timeout,
        )
        res_vision_only = query_vision_only(
            base_url,
            question,
            args.image_collection,
            args.image_top_k,
            args.timeout,
        )

        pred_text_only = (res_text_only.get("answer") or "").strip()
        pred_vision = (res_vision.get("answer") or "").strip()
        pred_vision_only = (res_vision_only.get("answer") or "").strip()

        m_text_only = compute_metrics(pred_text_only, reference)
        m_vision = compute_metrics(pred_vision, reference)
        m_vision_only = compute_metrics(pred_vision_only, reference)

        metrics_text_only.append(m_text_only)
        metrics_vision.append(m_vision)
        metrics_vision_only.append(m_vision_only)

        results.append(
            {
                "id": qid,
                "question": question,
                "reference": reference,
                "prediction_text_only": pred_text_only,
                "prediction_vision": pred_vision,
                "prediction_vision_only": pred_vision_only,
                "retrieval_text_only": {
                    "text_results": res_text_only.get("text_results", []),
                },
                "retrieval_vision": {
                    "image_results": res_vision.get("image_results", []),
                    "text_results": res_vision.get("text_results", []),
                },
                "retrieval_vision_only": {
                    "image_results": res_vision_only.get("image_results", []),
                },
                "metrics_text_only": m_text_only,
                "metrics_vision": m_vision,
                "metrics_vision_only": m_vision_only,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "count": len(results),
        "text_only": average_metrics(metrics_text_only),
        "vision": average_metrics(metrics_vision),
        "vision_only": average_metrics(metrics_vision_only),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary_out)), exist_ok=True)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
