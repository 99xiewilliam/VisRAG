#!/usr/bin/env python3
"""
测试 /api/v1/query/vision 接口。

用法:
  # 仅文本
  python scripts/test_vision_api.py --query "machine translation"
  python scripts/test_vision_api.py --query "文档里关于 BERT 的内容"

  # 文本 + 图像（传本地图片路径）
  python scripts/test_vision_api.py --query "这段在讲什么" --image output/api_assets/pdf_images/xxx/page_1.png

  # 仅图像
  python scripts/test_vision_api.py --image output/api_assets/pdf_images/xxx/page_1.png

  # 指定 base_url（默认 http://127.0.0.1:8000）
  python scripts/test_vision_api.py --base-url http://localhost:8000 --query "summary"
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Test /api/v1/query/vision API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--query", "-q", default=None, help="Text query")
    parser.add_argument("--image", "-i", default=None, help="Path to query image file")
    parser.add_argument("--video", "-v", default=None, help="Path to query video file")
    parser.add_argument("--image-top-k", type=int, default=None, help="Override image_top_k")
    parser.add_argument("--text-top-k", type=int, default=None, help="Override text_top_k")
    parser.add_argument("--image-collection", default=None, help="Override image collection name")
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

    url = f"{args.base_url.rstrip('/')}/api/v1/query/vision"
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
    print("  query:", args.query or "(none)")
    print("  image:", args.image or "(none)")
    print("  video:", args.video or "(none)")
    print()

    # 无文件时也要用 multipart，否则 FastAPI 会因缺少 image 报 422
    if not files:
        import io
        files = {"image": ("", io.BytesIO(b""), "application/octet-stream")}

    try:
        resp = requests.post(url, data=data, files=files, timeout=120)
        for f in files.values():
            if hasattr(f[1], "close"):
                f[1].close()
    except requests.exceptions.ConnectionError as e:
        print("连接失败，请确认 VisRAG 服务已启动（如 python -m fastapi_app.app 或 uvicorn）。", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("请求失败:", e, file=sys.stderr)
        sys.exit(1)

    print("Status:", resp.status_code)
    if resp.status_code != 200:
        print("Body:", resp.text[:500])
        sys.exit(1)

    out = resp.json()
    print()
    print("=" * 60)
    print("image_results (命中页面数):", len(out.get("image_results", [])))
    image_results = out.get("image_results", []) or []
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
    for i, tx in enumerate(out.get("text_results", [])[:3]):
        meta = tx.get("metadata", {})
        text_preview = (meta.get("text") or "")[:100]
        print(f"  [{i}] {text_preview}...")
    print()
    ans = out.get("answer", "") or ""
    print("answer:")
    print("-" * 40)
    print(ans or "(empty)")
    print("=" * 60)


if __name__ == "__main__":
    main()
