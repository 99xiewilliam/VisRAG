#!/usr/bin/env python3
"""
对比实验：纯文本检索 vs 纯文本+视觉文本融合

测试两种检索策略对最终生成质量的影响：
- 方案 A (Text): 纯文本检索
- 方案 B (Combined): 纯文本检索 + 视觉 token 还原文本

使用方法:
    python compare_retrieval_methods.py \
        --config config.yaml \
        --query "Transformer架构的优势是什么" \
        --output-dir ./experiments/results
"""

import argparse
import json
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 解析 --config 参数
config_path = None
for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        config_path = sys.argv[i + 1]
        break

from src.config import reload_config
from src.utils import init_logging_from_config, get_logger

if config_path:
    reload_config(config_path)
init_logging_from_config(config_path)
logger = get_logger(__name__)

from src.pipeline import VisRAGPipeline
from src.generator import create_generator


@dataclass
class RetrievalResult:
    """检索结果"""
    method: str           # "text", "combined"
    doc_id: str
    page: int
    score: float
    content: str          # 文本内容或描述
    source: str           # 内容来源说明


@dataclass
class GenerationResult:
    """生成结果"""
    method: str
    query: str
    context: str
    response: str
    retrieval_results: List[RetrievalResult]


class RetrievalExperimenter:
    """检索对比实验器"""
    
    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        self.pipe = VisRAGPipeline(persist_dir)
        self.generator = create_generator()
        
        logger.info(f"ChromaDB 目录: {persist_dir}")
    
    def retrieve_text_only(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        方案 A: 纯文本检索
        
        流程: 问题 → 文本Embed → text_pages集合 → 返回文本内容
        """
        logger.info(f"[方案 A] 纯文本检索: '{query[:50]}...'")
        
        results = self.pipe.query_text(query, top_k)
        
        retrieval_results = []
        for i, (doc_id, metadata, distance) in enumerate(zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            # distance 在 cosine 相似度下越小越相似
            score = 1 - distance  # 转换为相似度分数
            
            retrieval_results.append(RetrievalResult(
                method="text",
                doc_id=metadata.get("doc_id", ""),
                page=metadata.get("page", 0),
                score=score,
                content=metadata.get("text", ""),
                source=f"文本检索 (相似度: {score:.4f})"
            ))
        
        return retrieval_results
    
    def retrieve_vision_text(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        视觉 token 检索并解码为文本
        """
        logger.info(f"视觉检索并解码: '{query[:50]}...'")

        results = self.pipe.query_vision(query, top_k)
        metas = results.get("metadatas") or [[]]
        metas = metas[0] if metas else []
        tokens_paths = []
        for md in metas:
            if isinstance(md, dict) and md.get("tokens_path"):
                tokens_paths.append(md["tokens_path"])

        texts = self.pipe.decode_vision_tokens(tokens_paths) if tokens_paths else []
        
        retrieval_results = []
        for metadata, distance, text in zip(
            metas,
            results["distances"][0],
            texts
        ):
            score = 1 - distance
            
            retrieval_results.append(RetrievalResult(
                method="combined",
                doc_id=metadata.get("doc_id", ""),
                page=metadata.get("page", 0),
                score=score,
                content=text or "",
                source=f"视觉检索+token解码 (相似度: {score:.4f})"
            ))
        
        return retrieval_results
    
    def generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        system_prompt: Optional[str] = None
    ) -> str:
        """基于检索结果生成答案"""
        
        if system_prompt is None:
            system_prompt = "你是一个专业的文档问答助手。请基于提供的上下文回答问题。"
        
        # 构建上下文
        contexts = []
        for i, result in enumerate(retrieval_results, 1):
            contexts.append(f"[文档{i}] {result.doc_id} 第{result.page}页\n{result.content[:1500]}")
        
        context_str = "\n\n".join(contexts)
        
        user_prompt = f"""基于以下文档内容回答问题：

{context_str}

问题: {query}

请给出详细回答："""
        
        response = self.generator.generate(system_prompt, user_prompt)
        return response
    
    def run_comparison(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        运行两种方案的对比实验
        
        Returns:
            {
                "query": str,
                "methods": {
                    "text": GenerationResult,
                    "combined": GenerationResult
                }
            }
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"开始对比实验")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*70}\n")
        
        results = {"query": query, "methods": {}}
        
        # 方案 A: 纯文本检索
        text_retrieval = self.retrieve_text_only(query, top_k)
        text_response = self.generate_answer(query, text_retrieval)
        results["methods"]["text"] = GenerationResult(
            method="text",
            query=query,
            context="\n".join([r.content[:500] for r in text_retrieval]),
            response=text_response,
            retrieval_results=text_retrieval
        )
        
        vision_retrieval = self.retrieve_vision_text(query, top_k)
        combined_retrieval = text_retrieval + vision_retrieval
        combined_response = self.generate_answer(query, combined_retrieval)
        results["methods"]["combined"] = GenerationResult(
            method="combined",
            query=query,
            context="\n".join([r.content[:500] for r in combined_retrieval]),
            response=combined_response,
            retrieval_results=combined_retrieval
        )
        
        return results


def format_results(results: Dict[str, Any]) -> str:
    """格式化输出结果"""
    output = []
    output.append(f"\n{'='*70}")
    output.append(f"查询: {results['query']}")
    output.append(f"{'='*70}\n")
    
    for method, result in results["methods"].items():
        output.append(f"\n{'-'*70}")
        output.append(f"方案: {method.upper()}")
        output.append(f"{'-'*70}")
        
        output.append(f"\n[检索结果]")
        for i, r in enumerate(result.retrieval_results[:3], 1):
            output.append(f"  {i}. {r.doc_id} p{r.page} ({r.source})")
            output.append(f"     内容: {r.content[:200]}...")
        
        output.append(f"\n[生成回答]")
        output.append(result.response)
        output.append("")
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="对比三种检索策略的效果"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径"
    )
    parser.add_argument(
        "--persist-dir",
        default="/data/xwh/VisRAG/output/chroma_db",
        help="ChromaDB 持久化目录"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="测试查询"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="检索结果数"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="结果保存路径（JSON）"
    )
    
    args = parser.parse_args()
    
    # 运行实验
    experimenter = RetrievalExperimenter(args.persist_dir)
    results = experimenter.run_comparison(
        query=args.query,
        top_k=args.top_k
    )
    
    # 打印结果
    formatted = format_results(results)
    print(formatted)
    
    # 保存结果
    if args.output:
        # 转换 dataclass 为 dict
        serializable_results = {
            "query": results["query"],
            "methods": {}
        }
        for method, gr in results["methods"].items():
            serializable_results["methods"][method] = {
                "method": gr.method,
                "query": gr.query,
                "context": gr.context,
                "response": gr.response,
                "retrieval_results": [
                    {
                        "method": r.method,
                        "doc_id": r.doc_id,
                        "page": r.page,
                        "score": r.score,
                        "content": r.content[:1000],  # 限制长度
                        "source": r.source
                    }
                    for r in gr.retrieval_results
                ]
            }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()
