#!/usr/bin/env python3
"""
VisRAG 批量索引脚本

批量处理 dataset 目录中的所有 PDF 文件，分别生成：
1. 文本 embedding -> 存入 ChromaDB
2. Vision tokens -> 保存为 .pt 文件，同时 pooling 后存入 ChromaDB

使用方法:
    # 使用默认配置 (config.yaml)
    python batch_index.py --dataset-dir ./dataset --output-dir ./output
    
    # 指定配置文件
    python batch_index.py --config config_openai_example.yaml --dataset-dir ./dataset
    
    # 只处理前 N 个 PDF（测试用）
    python batch_index.py --dataset-dir ./dataset --max-pdfs 5
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
from src.config import reload_config, get_config
from src.utils import init_logging_from_config, get_logger

if config_path:
    reload_config(config_path)
init_logging_from_config(config_path)

logger = get_logger(__name__)

# 导入其他模块
from src.embedder import create_embedder, get_text_dim
from src.text import extract_pdf_text_by_page
from src.vision import extract_pdf_vision_tokens, pdf_to_images, DeepseekEncoder, mean_pool
from src.store import ChromaStore
from PIL import Image
import torch


def find_pdf_files(dataset_dir: str) -> List[str]:
    """查找目录中的所有 PDF 文件"""
    pdf_files = []
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset 目录不存在: {dataset_dir}")
    
    for ext in ['.pdf', '.PDF']:
        pdf_files.extend(dataset_path.glob(f'*{ext}'))
    
    # 按文件名排序，确保处理顺序一致
    pdf_files = sorted(pdf_files)
    logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")
    
    return [str(f) for f in pdf_files]


def generate_doc_id(pdf_path: str) -> str:
    """从 PDF 路径生成文档 ID"""
    # 使用文件名（不含扩展名）作为 doc_id
    # 移除特殊字符，确保 ID 合法
    name = Path(pdf_path).stem
    # 替换特殊字符
    doc_id = "".join(c if c.isalnum() or c in '_-' else '_' for c in name)
    # 限制长度
    if len(doc_id) > 100:
        doc_id = doc_id[:100]
    return doc_id


def index_pdf_text(
    pdf_path: str,
    doc_id: str,
    store: ChromaStore,
    embedder,
    collection_name: str = "text_pages"
) -> Dict[str, Any]:
    """
    索引 PDF 的文本内容
    
    Returns:
        {"pages": 页数, "doc_id": 文档ID}
    """
    logger.info(f"[{doc_id}] 开始索引文本...")
    
    # 提取文本
    pages = extract_pdf_text_by_page(pdf_path)
    logger.info(f"[{doc_id}] 提取了 {len(pages)} 页文本")
    
    if not pages:
        logger.warning(f"[{doc_id}] 未提取到文本内容")
        return {"pages": 0, "doc_id": doc_id}
    
    # 准备数据
    text_ids: List[str] = []
    text_contents: List[str] = []
    text_meta: List[Dict[str, Any]] = []
    
    for page in pages:
        page_id = f"{doc_id}_p{page['page']}"
        text_ids.append(page_id)
        text_contents.append(page["text"])
        text_meta.append({
            "doc_id": doc_id,
            "page": page["page"],
            "text": page["text"][:2000],  # 限制元数据大小
            "source_pdf": str(pdf_path)
        })
    
    # 批量编码
    logger.info(f"[{doc_id}] 编码 {len(text_contents)} 页文本...")
    text_vecs = embedder.embed(text_contents)
    
    # 存入 ChromaDB
    store.add(collection_name, text_ids, text_vecs, text_meta)
    logger.info(f"[{doc_id}] 文本索引完成: {len(pages)} 页 -> {collection_name}")
    
    return {"pages": len(pages), "doc_id": doc_id}


def index_pdf_vision(
    pdf_path: str,
    doc_id: str,
    store: ChromaStore,
    tokens_dir: str,
    collection_name: str = "vision_pages"
) -> Dict[str, Any]:
    """
    索引 PDF 的视觉内容
    
    Returns:
        {"pages": 页数, "tokens_dir": token保存目录}
    """
    logger.info(f"[{doc_id}] 开始索引视觉...")
    
    # 创建该文档的 vision tokens 目录
    doc_tokens_dir = os.path.join(tokens_dir, doc_id)
    os.makedirs(doc_tokens_dir, exist_ok=True)
    
    # 提取 vision tokens（这会保存 .pt 文件并返回 pooled vectors）
    vision_pages = extract_pdf_vision_tokens(pdf_path, doc_tokens_dir)
    logger.info(f"[{doc_id}] 提取了 {len(vision_pages)} 页视觉 tokens")
    
    if not vision_pages:
        logger.warning(f"[{doc_id}] 未提取到视觉内容")
        return {"pages": 0, "tokens_dir": doc_tokens_dir}
    
    # 准备数据存入 ChromaDB
    vision_ids: List[str] = []
    vision_vecs: List[List[float]] = []
    vision_meta: List[Dict[str, Any]] = []
    
    for page in vision_pages:
        page_id = f"{doc_id}_p{page['page']}"
        vision_ids.append(page_id)
        vision_vecs.append(page["vector"])
        vision_meta.append({
            "doc_id": doc_id,
            "page": page["page"],
            "tokens_path": page["tokens_path"],
            "source_pdf": str(pdf_path)
        })
    
    # 存入 ChromaDB
    store.add(collection_name, vision_ids, vision_vecs, vision_meta)
    logger.info(f"[{doc_id}] 视觉索引完成: {len(vision_pages)} 页 -> {collection_name}")
    
    return {
        "pages": len(vision_pages),
        "tokens_dir": doc_tokens_dir,
        "avg_dims": len(vision_vecs[0]) if vision_vecs else 0
    }


def process_single_pdf(
    pdf_path: str,
    store: ChromaStore,
    embedder,
    tokens_dir: str,
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    处理单个 PDF 文件
    
    Returns:
        {
            "doc_id": str,
            "pdf_path": str,
            "text": {"pages": int, "success": bool},
            "vision": {"pages": int, "tokens_dir": str, "success": bool},
            "error": str or None
        }
    """
    doc_id = generate_doc_id(pdf_path)
    result = {
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "text": {"pages": 0, "success": False},
        "vision": {"pages": 0, "tokens_dir": "", "success": False},
        "error": None
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"处理 PDF: {pdf_path}")
    logger.info(f"文档 ID: {doc_id}")
    logger.info(f"{'='*60}")
    
    try:
        # 1. 处理文本
        text_result = index_pdf_text(pdf_path, doc_id, store, embedder)
        result["text"]["pages"] = text_result["pages"]
        result["text"]["success"] = True
        
        # 2. 处理视觉
        vision_result = index_pdf_vision(pdf_path, doc_id, store, tokens_dir)
        result["vision"]["pages"] = vision_result["pages"]
        result["vision"]["tokens_dir"] = vision_result["tokens_dir"]
        result["vision"]["success"] = True
        
        logger.info(f"[{doc_id}] 处理完成 ✓")
        
    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        logger.error(f"[{doc_id}] {error_msg}")
        result["error"] = error_msg
    
    return result


def batch_index(
    dataset_dir: str,
    persist_dir: str,
    tokens_dir: str,
    max_pdfs: Optional[int] = None,
    skip_existing: bool = True
) -> List[Dict[str, Any]]:
    """
    批量索引 dataset 中的所有 PDF
    
    Returns:
        每个 PDF 的处理结果列表
    """
    cfg = get_config()
    
    logger.info(f"\n{'#'*60}")
    logger.info(f"开始批量索引")
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"ChromaDB: {persist_dir}")
    logger.info(f"Vision Tokens: {tokens_dir}")
    logger.info(f"Text Embedding: {cfg.embedding.backend}")
    logger.info(f"{'#'*60}\n")
    
    # 初始化组件
    store = ChromaStore(persist_dir)
    embedder = create_embedder()
    
    logger.info(f"文本 embedding 维度: {embedder.dim}")
    
    # 查找 PDF 文件
    pdf_files = find_pdf_files(dataset_dir)
    
    if max_pdfs:
        pdf_files = pdf_files[:max_pdfs]
        logger.info(f"限制处理前 {max_pdfs} 个 PDF")
    
    # 批量处理
    results = []
    success_count = 0
    fail_count = 0
    
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"\n进度: {i}/{len(pdf_files)}")
        
        result = process_single_pdf(
            pdf_path=pdf_path,
            store=store,
            embedder=embedder,
            tokens_dir=tokens_dir,
            skip_existing=skip_existing
        )
        
        results.append(result)
        
        if result["error"]:
            fail_count += 1
        else:
            success_count += 1
    
    # 汇总报告
    logger.info(f"\n{'#'*60}")
    logger.info(f"批量索引完成")
    logger.info(f"总计: {len(pdf_files)} 个 PDF")
    logger.info(f"成功: {success_count}")
    logger.info(f"失败: {fail_count}")
    logger.info(f"{'#'*60}\n")
    
    return results


def save_report(results: List[Dict[str, Any]], output_path: str):
    """保存处理报告为 JSON"""
    report = {
        "total": len(results),
        "success": sum(1 for r in results if not r["error"]),
        "failed": sum(1 for r in results if r["error"]),
        "details": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="VisRAG 批量索引脚本 - 处理 PDF 为文本 embedding 和 vision tokens"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径 (默认: config.yaml)"
    )
    parser.add_argument(
        "--dataset-dir",
        default="/data/xwh/VisRAG/dataset",
        help="PDF 数据集目录"
    )
    parser.add_argument(
        "--output-dir",
        default="/data/xwh/VisRAG/output",
        help="输出目录（包含 ChromaDB 和 vision tokens）"
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="最多处理 N 个 PDF（测试用）"
    )
    parser.add_argument(
        "--report",
        default=None,
        help="保存处理报告到 JSON 文件"
    )
    
    args = parser.parse_args()
    
    # 设置输出目录
    persist_dir = os.path.join(args.output_dir, "chroma_db")
    tokens_dir = os.path.join(args.output_dir, "vision_tokens")
    
    os.makedirs(persist_dir, exist_ok=True)
    os.makedirs(tokens_dir, exist_ok=True)
    
    # 执行批量索引
    results = batch_index(
        dataset_dir=args.dataset_dir,
        persist_dir=persist_dir,
        tokens_dir=tokens_dir,
        max_pdfs=args.max_pdfs
    )
    
    # 保存报告
    if args.report:
        save_report(results, args.report)
    else:
        # 默认报告路径
        default_report = os.path.join(args.output_dir, "index_report.json")
        save_report(results, default_report)
    
    # 打印失败项
    failed = [r for r in results if r["error"]]
    if failed:
        logger.warning(f"\n有 {len(failed)} 个 PDF 处理失败:")
        for r in failed:
            logger.warning(f"  - {r['doc_id']}: {r['error']}")


if __name__ == "__main__":
    main()
