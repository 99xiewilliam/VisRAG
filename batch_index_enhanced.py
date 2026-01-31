#!/usr/bin/env python3
"""
VisRAG 增强版批量索引脚本

批量处理 dataset 目录中的所有 PDF 文件，生成：
1. 文本 embedding -> 存入 ChromaDB (text_pages)
2. Vision tokens -> 保存为 .pt 文件，存入 ChromaDB (vision_pages)

使用方法:
    # 完整索引
    python batch_index_enhanced.py --dataset-dir ./dataset --output-dir ./output
    
    # 只处理前 N 个 PDF（测试用）
    python batch_index_enhanced.py --dataset-dir ./dataset --max-pdfs 2
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# 解析 --config 参数（需要在导入其他模块之前）
config_path = None
for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        config_path = sys.argv[i + 1]
        break

# 加载配置并初始化日志
from src.config import reload_config
from src.utils import init_logging_from_config, get_logger

if config_path:
    reload_config(config_path)
init_logging_from_config(config_path)

logger = get_logger(__name__)

# 导入其他模块
from src.embedder import create_embedder
from src.text import extract_pdf_text_by_page
from src.vision import extract_pdf_vision_tokens
from src.store import ChromaStore


def find_pdf_files(dataset_dir: str) -> List[str]:
    """查找目录中的所有 PDF 文件"""
    pdf_files = []
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset 目录不存在: {dataset_dir}")
    
    for ext in ['.pdf', '.PDF']:
        pdf_files.extend(dataset_path.glob(f'*{ext}'))
    
    pdf_files = sorted(pdf_files)
    logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")
    
    return [str(f) for f in pdf_files]


def generate_doc_id(pdf_path: str) -> str:
    """从 PDF 路径生成文档 ID"""
    name = Path(pdf_path).stem
    doc_id = "".join(c if c.isalnum() or c in '_-' else '_' for c in name)
    if len(doc_id) > 100:
        doc_id = doc_id[:100]
    return doc_id


def index_pdf(
    pdf_path: str,
    doc_id: str,
    store: ChromaStore,
    embedder,
    tokens_dir: str,
) -> Dict[str, Any]:
    """
    增强版 PDF 索引：同时生成文本、视觉

    Returns:
        {
            "doc_id": str,
            "text_pages": int,
            "vision_pages": int
        }
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"处理 PDF: {pdf_path}")
    logger.info(f"文档 ID: {doc_id}")
    logger.info(f"{'='*60}")
    
    result = {
        "doc_id": doc_id,
        "text_pages": 0,
        "vision_pages": 0
    }
    
    doc_tokens_dir = os.path.join(tokens_dir, doc_id)
    os.makedirs(doc_tokens_dir, exist_ok=True)
    
    text_pages = extract_pdf_text_by_page(pdf_path)
    logger.info(f"[{doc_id}] 提取了 {len(text_pages)} 页文本")
    
    vision_pages = extract_pdf_vision_tokens(pdf_path, doc_tokens_dir)
    logger.info(f"[{doc_id}] 提取了 {len(vision_pages)} 页视觉 tokens")

    text_ids, text_vecs, text_meta = [], [], []
    vision_ids, vision_vecs, vision_meta = [], [], []

    for text_page in text_pages:
        page_num = text_page["page"]
        page_id = f"{doc_id}_p{page_num}"
        text_ids.append(page_id)
        text_vecs.append(embedder.embed_single(text_page["text"]))
        text_meta.append({
            "doc_id": doc_id,
            "page": page_num,
            "text": text_page["text"][:2000],
            "source_pdf": str(pdf_path),
        })

    for page in vision_pages:
        page_id = f"{doc_id}_p{page['page']}"
        vision_ids.append(page_id)
        vision_vecs.append(page["vector"])
        vision_meta.append({
            "doc_id": doc_id,
            "page": page["page"],
            "tokens_path": page["tokens_path"],
            "source_pdf": str(pdf_path),
        })
    
    # 5. 批量存入 ChromaDB
    if text_ids:
        store.add("text_pages", text_ids, text_vecs, text_meta)
        result["text_pages"] = len(text_ids)
        logger.info(f"[{doc_id}] 文本索引完成: {len(text_ids)} 页")
    
    if vision_ids:
        store.add("vision_pages", vision_ids, vision_vecs, vision_meta)
        result["vision_pages"] = len(vision_ids)
        logger.info(f"[{doc_id}] 视觉索引完成: {len(vision_ids)} 页")
    
    logger.info(f"[{doc_id}] 处理完成 ✓")
    return result


def batch_index_enhanced(
    dataset_dir: str,
    output_dir: str,
    max_pdfs: Optional[int] = None
) -> List[Dict[str, Any]]:
    """增强版批量索引"""
    
    persist_dir = os.path.join(output_dir, "chroma_db")
    tokens_dir = os.path.join(output_dir, "vision_tokens")
    
    os.makedirs(persist_dir, exist_ok=True)
    os.makedirs(tokens_dir, exist_ok=True)
    
    logger.info(f"\n{'#'*60}")
    logger.info(f"开始增强版批量索引")
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'#'*60}\n")
    
    store = ChromaStore(persist_dir)
    embedder = create_embedder()
    
    pdf_files = find_pdf_files(dataset_dir)
    if max_pdfs:
        pdf_files = pdf_files[:max_pdfs]
    
    results = []
    for pdf_path in pdf_files:
        doc_id = generate_doc_id(pdf_path)
        try:
            result = index_pdf(
                pdf_path=pdf_path,
                doc_id=doc_id,
                store=store,
                embedder=embedder,
                tokens_dir=tokens_dir
            )
            results.append(result)
        except Exception as e:
            logger.error(f"处理失败 {pdf_path}: {e}")
            results.append({"doc_id": doc_id, "error": str(e)})
    
    # 汇总
    total_text = sum(r.get("text_pages", 0) for r in results)
    total_vision = sum(r.get("vision_pages", 0) for r in results)
    
    logger.info(f"\n{'#'*60}")
    logger.info(f"批量索引完成")
    logger.info(f"总计: {len(results)} 个 PDF")
    logger.info(f"文本页: {total_text}")
    logger.info(f"视觉页: {total_vision}")
    logger.info(f"{'#'*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="VisRAG 增强版批量索引脚本"
    )
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--dataset-dir", default="/data/xwh/VisRAG/dataset", help="PDF 数据集目录")
    parser.add_argument("--output-dir", default="/data/xwh/VisRAG/output", help="输出目录")
    parser.add_argument("--max-pdfs", type=int, default=None, help="最多处理 N 个 PDF")
    parser.add_argument("--report", default=None, help="保存报告到 JSON")
    
    args = parser.parse_args()
    
    results = batch_index_enhanced(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_pdfs=args.max_pdfs
    )
    
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"报告已保存: {args.report}")


if __name__ == "__main__":
    main()
