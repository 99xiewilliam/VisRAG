import argparse
import os
import re
import sys

# 加载配置
config_path = None
for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        config_path = sys.argv[i + 1]
        break

from src.config import reload_config, get_config
if config_path:
    reload_config(config_path)
else:
    pass  # 使用默认配置

from src.utils import get_logger, init_logging_from_config

# 初始化日志系统
init_logging_from_config(config_path)
logger = get_logger(__name__)

if config_path:
    logger.info(f"使用配置文件: {config_path}")
else:
    logger.info("使用默认配置")

from src.pipeline import VisRAGPipeline
from qa_pipeline import build_qa, eval_metrics, llm_judge, rag_predict, rag_predict_3way
from src.generator import create_generator


def _default_under_project(*parts: str) -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(project_root, *parts)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="配置文件路径 (默认: config.yaml)")
    # 默认改为项目目录下，避免 /data/... 的权限问题
    p.add_argument("--persist", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("index_pdf")
    sp.add_argument("--id", required=True)
    sp.add_argument("--pdf", required=True)
    sp.add_argument("--tokens-dir", default=None)

    qp = sub.add_parser("query_text")
    qp.add_argument("--text", required=True)
    qp.add_argument("--top-k", type=int, default=5)

    vp = sub.add_parser("query_vision_text")
    vp.add_argument("--text", required=True)
    vp.add_argument("--top-k", type=int, default=5)

    g = sub.add_parser("qa_build")
    g.add_argument("--dataset-dir", default="/data/xwh/VisRAG/dataset")
    g.add_argument("--out", default="/data/xwh/VisRAG/qa/qa.jsonl")
    g.add_argument("--qas-per-page", type=int, default=2)
    g.add_argument("--max-pages", type=int, default=2)

    e = sub.add_parser("qa_eval")
    e.add_argument("--qa", required=True)
    e.add_argument("--pred", required=True)
    e.add_argument("--out", default="/data/xwh/VisRAG/qa/metrics.json")
    e.add_argument("--log", default=None)

    j = sub.add_parser("qa_judge")
    j.add_argument("--qa", required=True)
    j.add_argument("--pred", required=True)
    j.add_argument("--out", default="/data/xwh/VisRAG/qa/judge.jsonl")
    j.add_argument("--log", default=None)

    # 新增：给 QA jsonl 补齐 id 字段（用于评测对齐）
    qa_fix = sub.add_parser("qa_add_ids")
    qa_fix.add_argument("--in", dest="in_path", required=True, help="输入 QA jsonl（可无 id）")
    qa_fix.add_argument("--out", dest="out_path", required=True, help="输出 QA jsonl（补齐 id）")

    # 新增：RAG 推理（同时输出 text-only 与 vision+text）
    rp = sub.add_parser("rag_predict")
    rp.add_argument("--qa", required=True, help="QA jsonl 路径（需包含 id/question/answer 等字段）")
    rp.add_argument("--out", required=True, help="输出预测 jsonl 路径")
    rp.add_argument("--top-k", type=int, default=3)
    rp.add_argument("--max-context-chars", type=int, default=6000)

    # 新增：RAG 三路对比（vision-only / text-only / text+vision）
    rp3 = sub.add_parser("rag_predict_3way")
    rp3.add_argument("--qa", required=True, help="QA jsonl 路径（需包含 id/question/answer 等字段）")
    rp3.add_argument("--out", required=True, help="输出预测 jsonl 路径")
    rp3.add_argument("--top-k", type=int, default=3)
    rp3.add_argument("--max-context-chars", type=int, default=6000)
    
    # 新增：直接测试 Generator
    tg = sub.add_parser("test_generator")
    tg.add_argument("--system", default="你是一个有帮助的助手。")
    tg.add_argument("--user", required=True)
    tg.add_argument("--max-tokens", type=int, default=512)
    
    return p


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if s else "item"


def _qa_add_ids(in_path: str, out_path: str):
    """
    为没有 id 的 QA jsonl 补齐 id：
    - 优先使用 title/doc 字段做前缀
    - 追加行号，保证唯一性
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            item = None
            try:
                import json
                item = json.loads(line)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            if "id" not in item or not str(item.get("id") or "").strip():
                prefix = _slug(item.get("title") or item.get("doc") or "qa")
                item["id"] = f"{prefix}_{idx}"
            fout.write(__import__("json").dumps(item, ensure_ascii=False) + "\n")
            n += 1
    logger.info(f"qa_add_ids 完成: {n} 条 -> {out_path}")


def main():
    args = build_parser().parse_args()
    
    cfg = get_config()
    logger.info(f"Generator backend: {cfg.generator.backend}")
    
    if not args.persist:
        args.persist = _default_under_project("output", "chroma_db")

    # 兼容两种目录结构：
    # - 直接把 ChromaDB 持久化目录传进来（包含 chroma.sqlite3）
    # - 传入 output 根目录（其下有 chroma_db/ 与 vision_tokens/），如 batch_index_enhanced.py 的 output_dir
    persist_dir = args.persist
    nested = os.path.join(persist_dir, "chroma_db")
    if os.path.isdir(nested) and os.path.exists(os.path.join(nested, "chroma.sqlite3")):
        logger.info(f"检测到嵌套 chroma_db，使用: {nested}")
        persist_dir = nested

    os.makedirs(persist_dir, exist_ok=True)
    pipe = VisRAGPipeline(persist_dir)
    
    if args.cmd == "index_pdf":
        if not args.tokens_dir:
            args.tokens_dir = _default_under_project("output", "vision_tokens")
        os.makedirs(args.tokens_dir, exist_ok=True)
        pipe.index_pdf(args.id, args.pdf, args.tokens_dir)
        logger.info(f"PDF 索引完成: {args.id}")
    
    elif args.cmd == "query_text":
        res = pipe.query_text(args.text, top_k=args.top_k)
        logger.info(f"文本查询: '{args.text[:50]}...', 返回 {len(res['ids'][0])} 条结果")
        for i, (id_, md, dist) in enumerate(zip(res["ids"][0], res["metadatas"][0], res["distances"][0])):
            print(i, id_, dist, md)
    
    elif args.cmd == "query_vision_text":
        res = pipe.query_vision(args.text, top_k=args.top_k)
        logger.info(f"视觉查询(文本): '{args.text[:50]}...', 返回 {len(res['ids'][0])} 条结果")
        for i, (id_, md, dist) in enumerate(zip(res["ids"][0], res["metadatas"][0], res["distances"][0])):
            print(i, id_, dist, md)
    
    elif args.cmd == "qa_build":
        build_qa(args.dataset_dir, args.out, args.qas_per_page, args.max_pages)
        logger.info(f"QA 构建完成: {args.out}")
    
    elif args.cmd == "qa_eval":
        eval_metrics(args.qa, args.pred, args.out, args.log)
        print("ok")
    elif args.cmd == "qa_judge":
        llm_judge(args.qa, args.pred, args.out, None, args.log)
        llm_judge(args.qa, args.pred, args.out)

    elif args.cmd == "qa_add_ids":
        _qa_add_ids(args.in_path, args.out_path)

    elif args.cmd == "rag_predict":
        rag_predict(
            qa_path=args.qa,
            out_path=args.out,
            persist_dir=persist_dir,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
        )

    elif args.cmd == "rag_predict_3way":
        rag_predict_3way(
            qa_path=args.qa,
            out_path=args.out,
            persist_dir=persist_dir,
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
        )
    
    elif args.cmd == "test_generator":
        logger.info(f"测试 Generator: backend={cfg.generator.backend}")
        
        generator = create_generator()
        response = generator.generate(args.system, args.user)
        
        print("=" * 50)
        print("Response:")
        print(response)
        print("=" * 50)


if __name__ == "__main__":
    main()
