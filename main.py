import argparse
import os
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
from qa_pipeline import build_qa, eval_metrics, llm_judge
from src.generator import create_generator


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="配置文件路径 (默认: config.yaml)")
    p.add_argument("--persist", default="/data/xwh/VisRAG/chroma_db")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("index_pdf")
    sp.add_argument("--id", required=True)
    sp.add_argument("--pdf", required=True)
    sp.add_argument("--tokens-dir", default="/data/xwh/VisRAG/vision_tokens")

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
    
    # 新增：直接测试 Generator
    tg = sub.add_parser("test_generator")
    tg.add_argument("--system", default="你是一个有帮助的助手。")
    tg.add_argument("--user", required=True)
    tg.add_argument("--max-tokens", type=int, default=512)
    
    return p


def main():
    args = build_parser().parse_args()
    
    cfg = get_config()
    logger.info(f"Generator backend: {cfg.generator.backend}")
    
    os.makedirs(args.persist, exist_ok=True)
    pipe = VisRAGPipeline(args.persist)
    
    if args.cmd == "index_pdf":
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
