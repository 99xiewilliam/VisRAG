import argparse
import json
import os
import re
import fitz
from typing import List, Dict, Any

from src.generator import create_generator, BaseGenerator
from src.config import get_config, reload_config
from src.utils import get_logger

logger = get_logger(__name__)


def ensure_nltk():
    import nltk

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(pred: str, ref: str) -> float:
    pred_toks = normalize_text(pred).split()
    ref_toks = normalize_text(ref).split()
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    common = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in ref_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    precision = overlap / len(pred_toks)
    recall = overlap / len(ref_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(ref) else 0.0


def lcs_len(a: List[str], b: List[str]) -> int:
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def rouge_l(pred: str, ref: str) -> float:
    pred_toks = normalize_text(pred).split()
    ref_toks = normalize_text(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = lcs_len(pred_toks, ref_toks)
    precision = lcs / len(pred_toks)
    recall = lcs / len(ref_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def meteor_score(pred: str, ref: str) -> float:
    ensure_nltk()
    from nltk.translate.meteor_score import single_meteor_score
    ref_tokens = normalize_text(ref).split()
    pred_tokens = normalize_text(pred).split()
    return float(single_meteor_score(ref_tokens, pred_tokens))


def load_pdf_pages(pdf_path: str, max_pages: int | None = None) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(doc.page_count):
        if max_pages is not None and i >= max_pages:
            break
        text = doc[i].get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def extract_json_array(text: str):
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def extract_json_object(text: str):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


class QAGenerator:
    """QA 生成器，使用配置中的 Generator"""
    
    def __init__(self, generator: BaseGenerator = None):
        self.generator = generator or create_generator()
    
    def generate_qa(self, context: str, qas_per_page: int = 2) -> List[Dict[str, str]]:
        """根据上下文生成 QA 对"""
        system = "你是一个严谨的文档问答数据生成器。"
        user = (
            "根据给定文档内容生成问题与答案，"
            f"生成 {qas_per_page} 组，必须只基于上下文。"
            "输出严格 JSON 数组，每个元素含 question 和 answer。\n\n"
            f"上下文:\n{context}"
        )
        raw = self.generator.generate(system, user)
        items = extract_json_array(raw)
        if not items:
            return []
        
        qa_pairs = []
        for qa in items:
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if q and a:
                qa_pairs.append({"question": q, "answer": a})
        return qa_pairs


class LLMJudge:
    """LLM 裁判，使用配置中的 Generator"""
    
    def __init__(self, generator: BaseGenerator = None):
        self.generator = generator or create_generator()
    
    def judge(self, question: str, reference: str, prediction: str) -> Dict[str, Any]:
        """评估答案质量"""
        system = "你是严格的答案评估员。"
        user = (
            "给定问题、参考答案、模型答案，输出 JSON："
            "{\"score\":0-1,\"reason\":\"\"}。"
            "只返回 JSON。\n\n"
            f"问题: {question}\n"
            f"参考答案: {reference}\n"
            f"模型答案: {prediction}"
        )
        raw = self.generator.generate(system, user)
        
        # 尝试提取 JSON
        obj = extract_json_object(raw)
        if obj and "score" in obj:
            return obj
        
        # 回退：尝试提取 JSON 数组
        arr = extract_json_array(raw)
        if isinstance(arr, list) and arr and "score" in arr[0]:
            return arr[0]
        
        return {"score": 0.0, "reason": "parse_error"}


def build_qa(
    dataset_dir: str,
    out_path: str,
    qas_per_page: int,
    max_pages: int | None,
    generator: BaseGenerator = None,
    max_pdfs: int | None = None,
):
    """构建 QA 评测集"""
    logger.info(f"开始构建 QA 数据集: dataset={dataset_dir}, out={out_path}")
    gen = QAGenerator(generator)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        processed = 0
        for filename in sorted(os.listdir(dataset_dir)):
            if not filename.lower().endswith(".pdf"):
                continue
            if max_pdfs is not None and processed >= max_pdfs:
                break
            pdf_path = os.path.join(dataset_dir, filename)
            pages = load_pdf_pages(pdf_path, max_pages=max_pages)
            
            for page in pages:
                context = page["text"].strip()
                if len(context) < 200:
                    continue
                context = context[:4000]
                
                qa_pairs = gen.generate_qa(context, qas_per_page)
                for idx, qa in enumerate(qa_pairs):
                    record = {
                        "id": f"{filename}_p{page['page']}_{idx}",
                        "doc": filename,
                        "page": page["page"],
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "context": context,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1
    
    logger.info(f"QA 数据集构建完成: {out_path}")


def eval_metrics(
    qa_path: str,
    pred_path: str,
    out_path: str,
    log_path: str | None = None,
    prediction_field: str = "prediction_text",
):
    """评估 QA 指标"""
    ensure_nltk()
    logger.info(f"开始评测: qa={qa_path}, pred={pred_path}")
    
    qa_map = {}
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qa_map[item["id"]] = item
    
    preds: Dict[str, str] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred = item.get(prediction_field)
            if pred is None:
                pred = item.get("prediction")
            if pred is None:
                pred = item.get("prediction_with_vision", "")
            preds[item["id"]] = pred or ""

    scores = {"f1": [], "em": [], "meteor": [], "rouge_l": []}
    log_file = None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")

    for qid, qa in qa_map.items():
        pred = preds.get(qid, "")
        ref = qa["answer"]
        f1 = token_f1(pred, ref)
        em = exact_match(pred, ref)
        meteor = meteor_score(pred, ref)
        rouge = rouge_l(pred, ref)
        scores["f1"].append(f1)
        scores["em"].append(em)
        scores["meteor"].append(meteor)
        scores["rouge_l"].append(rouge)
        if log_file:
            record = {
                "id": qid,
                "question": qa.get("question", ""),
                "reference": ref,
                "prediction": pred,
                "f1": f1,
                "em": em,
                "meteor": meteor,
                "rouge_l": rouge,
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {k: sum(v) / max(len(v), 1) for k, v in scores.items()}
    logger.info(f"评测样本数: {len(qa_map)}, 平均 F1: {summary['f1']:.4f}")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    
    # 同时打印结果
    print("\n========== 评测结果 ==========")
    for k, v in summary.items():
        print(f"{k.upper()}: {v:.4f}")
    print("==============================\n")
    if log_file:
        log_file.close()


def llm_judge(
    qa_path: str,
    pred_path: str,
    out_path: str,
    generator: BaseGenerator = None,
    log_path: str | None = None,
    prediction_field: str = "prediction_text",
):
    """LLM 裁判评分"""
    judge = LLMJudge(generator)
    
    qa_map = {}
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qa_map[item["id"]] = item
    
    preds: Dict[str, str] = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred = item.get(prediction_field)
            if pred is None:
                pred = item.get("prediction")
            if pred is None:
                pred = item.get("prediction_with_vision", "")
            preds[item["id"]] = pred or ""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    scores = []
    log_file = None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")

    with open(out_path, "w", encoding="utf-8") as f:
        for qid, qa in qa_map.items():
            pred = preds.get(qid, "")
            result = judge.judge(qa["question"], qa["answer"], pred)
            result["id"] = qid
            scores.append(result["score"])
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if log_file:
                detail = {
                    "id": qid,
                    "question": qa.get("question", ""),
                    "reference": qa.get("answer", ""),
                    "prediction": pred,
                    "score": result.get("score", 0.0),
                    "reason": result.get("reason", ""),
                }
                log_file.write(json.dumps(detail, ensure_ascii=False) + "\n")
    
    avg_score = sum(scores) / max(len(scores), 1)
    print(f"\n========== LLM Judge 结果 ==========")
    print(f"平均得分: {avg_score:.4f}")
    print(f"=====================================\n")
    if log_file:
        log_file.close()


def rag_predict(
    qa_path: str,
    out_path: str,
    persist_dir: str = "/data/xwh/VisRAG/chroma_db",
    top_k: int = 3,
    max_context_chars: int = 6000,
    generator: BaseGenerator = None,
):
    from src.pipeline import VisRAGPipeline

    gen = generator or create_generator()
    pipe = VisRAGPipeline(persist_dir)
    cfg = get_config()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_items = [json.loads(line) for line in f if line.strip()]

    with open(out_path, "w", encoding="utf-8") as f:
        for item in qa_items:
            qid = item["id"]
            question = item.get("question", "")
            text_res = pipe.query_text(question, top_k=top_k)
            text_metas = (text_res.get("metadatas") or [[]])[0]

            text_parts = []
            retrieved_text = []
            for md in text_metas:
                if not isinstance(md, dict):
                    continue
                doc_id = md.get("doc_id", "")
                page = md.get("page", "")
                text = (md.get("text") or "").strip()
                if not text:
                    continue
                retrieved_text.append({"doc_id": doc_id, "page": page})
                text_parts.append(f"[{doc_id} p{page}]\n{text}")

            text_context = "\n\n".join(text_parts)
            if max_context_chars and len(text_context) > max_context_chars:
                text_context = text_context[:max_context_chars]

            vision_res = pipe.query_vision(question, top_k=top_k)
            vision_metas = (vision_res.get("metadatas") or [[]])[0]

            tokens_paths = []
            vision_headers = []
            retrieved_vision = []
            for md in vision_metas:
                if not isinstance(md, dict):
                    continue
                doc_id = md.get("doc_id", "")
                page = md.get("page", "")
                tokens_path = md.get("tokens_path")
                if not tokens_path:
                    continue
                retrieved_vision.append({"doc_id": doc_id, "page": page})
                tokens_paths.append(tokens_path)

            system = "你是一个严谨的问答助手，只能基于给定材料回答。"
            user_text_only = (
                "请根据材料回答问题。若材料不足以确定答案，请回答“无法从给定材料中确定”。\n\n"
                f"问题:\n{question}\n\n"
                f"材料:\n{text_context}\n\n"
                "答案:"
            )

            prediction_text = gen.generate(system, user_text_only)
            if cfg.generator.backend == "deepseek_ocr2" and tokens_paths:
                prediction_with_vision = gen.generate(system, user_text_only, vision_tokens=tokens_paths)
            else:
                vision_texts = pipe.decode_vision_tokens(tokens_paths) if tokens_paths else []
                vision_parts = []
                for header, text in zip([f"[{md.get('doc_id','')} p{md.get('page','')}]" for md in vision_metas if isinstance(md, dict) and md.get("tokens_path")], vision_texts):
                    if not text:
                        continue
                    vision_parts.append(f"{header}\n{text}")
                vision_context = "\n\n".join(vision_parts)
                if max_context_chars and len(vision_context) > max_context_chars:
                    vision_context = vision_context[:max_context_chars]
                combined_context = "\n\n".join([c for c in [text_context, vision_context] if c])
                if max_context_chars and len(combined_context) > max_context_chars:
                    combined_context = combined_context[:max_context_chars]
                user_combined = (
                    "请根据材料回答问题。若材料不足以确定答案，请回答“无法从给定材料中确定”。\n\n"
                    f"问题:\n{question}\n\n"
                    f"材料:\n{combined_context}\n\n"
                    "答案:"
                )
                prediction_with_vision = gen.generate(system, user_combined)

            f.write(
                json.dumps(
                    {
                        "id": qid,
                        "prediction_text": prediction_text,
                        "prediction_with_vision": prediction_with_vision,
                        "retrieved_text": retrieved_text,
                        "retrieved_vision": retrieved_vision,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # 新增：指定配置文件
    p.add_argument("--config", default=None, help="配置文件路径 (默认: config.yaml)")

    g = sub.add_parser("build_qa")
    g.add_argument("--dataset-dir", default="/data/xwh/VisRAG/dataset")
    g.add_argument("--out", default="/data/xwh/VisRAG/qa/qa.jsonl")
    g.add_argument("--qas-per-page", type=int, default=2)
    g.add_argument("--max-pages", type=int, default=2)

    e = sub.add_parser("eval")
    e.add_argument("--qa", required=True)
    e.add_argument("--pred", required=True)
    e.add_argument("--out", default="/data/xwh/VisRAG/qa/metrics.json")
    e.add_argument("--log", default=None)
    e.add_argument("--field", default="prediction_text")

    j = sub.add_parser("judge")
    j.add_argument("--qa", required=True)
    j.add_argument("--pred", required=True)
    j.add_argument("--out", default="/data/xwh/VisRAG/qa/judge.jsonl")
    j.add_argument("--log", default=None)
    j.add_argument("--field", default="prediction_text")

    return p


def main():
    # 先解析全局参数
    import sys
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
    
    # 加载配置并初始化日志
    from src.utils import init_logging_from_config
    if config_path:
        reload_config(config_path)
    init_logging_from_config(config_path)
    
    cfg = get_config()
    logger.info(f"Generator backend: {cfg.generator.backend}")
    
    # 解析其余参数
    args = build_parser().parse_args()

    if args.cmd == "build_qa":
        build_qa(args.dataset_dir, args.out, args.qas_per_page, args.max_pages)
    elif args.cmd == "eval":
        eval_metrics(args.qa, args.pred, args.out, args.log, args.field)
    elif args.cmd == "judge":
        llm_judge(args.qa, args.pred, args.out, None, args.log, args.field)


if __name__ == "__main__":
    main()
