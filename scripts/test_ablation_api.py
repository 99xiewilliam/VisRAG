#!/usr/bin/env python3
"""
Test A/B/C ablation endpoints:
  - A: /api/v1/query/text_only
  - B: /api/v1/query/vision_only
  - C: /api/v1/query/vision

Usage:
  python scripts/test_ablation_api.py --mode text_only --query "..."
  python scripts/test_ablation_api.py --mode vision_only --query "..."
  python scripts/test_ablation_api.py --mode vision --query "..."
"""
import argparse
import json
import os
import sys


def _print_results(out: dict):
    print()
    print("=" * 60)
    image_results = out.get("image_results", []) or []
    print("image_results (命中页面数):", len(image_results))
    has_rerank = any((im.get("metadata") or {}).get("rerank_score") is not None for im in image_results)
    if has_rerank:
        print("  (已按 rerank_score 重排，分数越大越相关)")
    for i, im in enumerate(image_results):
        md = im.get("metadata", {}) or {}
        rid = im.get("id")
        pdf = md.get("pdf_name")
        page = md.get("page")
        score = md.get("rerank_score")
        img_path = md.get("image_path")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        print(f"  [{i}] score={score_str} pdf={pdf} page={page} id={rid}")
        if img_path:
            print(f"       image_path={img_path}")
    print()
    print("text_results (命中文本块数):", len(out.get("text_results", [])))
    for i, tx in enumerate(out.get("text_results", [])[:5]):
        meta = tx.get("metadata", {})
        text_preview = (meta.get("text") or "")[:120]
        print(f"  [{i}] {text_preview}...")
    print()
    ans = out.get("answer", "") or ""
    print("answer:")
    print("-" * 40)
    print(ans or "(empty)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test text-only / vision-only / vision endpoints")
    parser.add_argument("--mode", choices=["text_only", "vision_only", "vision"], default="vision", help="Endpoint mode")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--query", "-q", default=None, help="Text query")
    parser.add_argument("--image", "-i", default=None, help="Path to query image file")
    parser.add_argument("--video", "-v", default=None, help="Path to query video file")
    parser.add_argument("--image-top-k", type=int, default=None, help="Override image_top_k")
    parser.add_argument("--text-top-k", type=int, default=None, help="Override text_top_k")
    parser.add_argument("--image-collection", default=None, help="Override image collection name")
    parser.add_argument("--text-collection", default=None, help="Override text collection name (text_only)")
    args = parser.parse_args()

    if not args.query and not args.image and not args.video:
        print("请至少提供 --query、--image 或 --video 之一。", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)

    try:
        import requests
    except ImportError:
        print("需要安装 requests: pip install requests", file=sys.stderr)
        sys.exit(1)

    base = args.base_url.rstrip("/")
    if args.mode == "text_only":
        url = f"{base}/api/v1/query/text_only"
        payload = {"query": args.query}
        if args.text_collection:
            payload["text_collection"] = args.text_collection
        if args.text_top_k is not None:
            payload["text_top_k"] = args.text_top_k
        print("Request:", "POST", url)
        print("  mode: text_only")
        print("  query:", args.query or "(none)")
        print("  text_collection:", args.text_collection or "(default)")
        print("  text_top_k:", args.text_top_k or "(default)")
        print()
        resp = requests.post(url, json=payload, timeout=300)
    else:
        # vision_only / vision
        if args.mode == "vision_only":
            url = f"{base}/api/v1/query/vision_only"
        else:
            url = f"{base}/api/v1/query/vision"

        data = {}
        files = {}
        if args.query:
            data["query"] = args.query
        if args.image_collection:
            data["image_collection"] = args.image_collection
        if args.image_top_k is not None:
            data["image_top_k"] = args.image_top_k
        if args.text_top_k is not None:
            data["text_top_k"] = args.text_top_k

        if args.image:
            if not os.path.isfile(args.image):
                print(f"图像文件不存在: {args.image}", file=sys.stderr)
                sys.exit(1)
            files["image"] = (os.path.basename(args.image), open(args.image, "rb"), "image/png")
        if args.video:
            if not os.path.isfile(args.video):
                print(f"视频文件不存在: {args.video}", file=sys.stderr)
                sys.exit(1)
            files["video"] = (os.path.basename(args.video), open(args.video, "rb"), "video/mp4")

        print("Request:", "POST", url)
        print(f"  mode: {args.mode}")
        print("  query:", args.query or "(none)")
        print("  image:", args.image or "(none)")
        print("  video:", args.video or "(none)")
        print("  image_collection:", args.image_collection or "(default)")
        print("  image_top_k:", args.image_top_k or "(default)")
        print("  text_top_k:", args.text_top_k or "(default)")
        print()

        if not files:
            import io
            files = {"image": ("", io.BytesIO(b""), "application/octet-stream")}

        try:
            resp = requests.post(url, data=data, files=files, timeout=300)
        finally:
            for f in files.values():
                if hasattr(f[1], "close"):
                    f[1].close()

    print("Status:", resp.status_code)
    if resp.status_code != 200:
        print("Body:", resp.text[:500])
        sys.exit(1)

    out = resp.json()
    _print_results(out)


if __name__ == "__main__":
    raise SystemExit(main())
