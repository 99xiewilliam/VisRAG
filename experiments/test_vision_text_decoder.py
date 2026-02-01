"""
实验：验证 vision token + 纯文本结合传给 deepseek ocr2 decoder

测试场景：
1. 只使用 vision token 生成（原始方式）
2. vision token + 简单文本 prompt
3. vision token + 复杂指令 prompt
4. vision token + 中文 prompt
5. 多页 vision token + 文本 prompt
"""

import sys
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision import DeepseekTokenDecoder, DeepseekEncoder
from src.utils import get_logger

logger = get_logger(__name__)

# 配置
MODEL_PATH = "/home/xwh/models/DeepSeek-OCR-2"
VISION_TOKENS_DIR = Path("VisRAG/output/vision_tokens")


def test_case_1_pure_vision_token():
    """测试1: 只使用 vision token 生成（baseline）"""
    print("\n" + "="*60)
    print("测试1: 只使用 vision token 生成（baseline）")
    print("="*60)
    
    decoder = DeepseekTokenDecoder(model_path=MODEL_PATH)
    
    # 加载单个页面的 vision tokens
    token_path = VISION_TOKENS_DIR / "DeepSeek-OCR-_Contexts_Optical_Compression/page_1.pt"
    feats = torch.load(token_path, map_location="cpu")
    
    # 使用默认 prompt
    default_prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    
    print(f"Vision token shape: {feats.shape}")
    print(f"Prompt: {default_prompt}")
    
    result = decoder.decode_from_tokens(feats, prompt=default_prompt)
    print(f"\n生成结果:\n{result[:500]}...")
    return result


def test_case_2_vision_with_simple_text():
    """测试2: vision token + 简单文本 prompt"""
    print("\n" + "="*60)
    print("测试2: vision token + 简单文本 prompt")
    print("="*60)
    
    decoder = DeepseekTokenDecoder(model_path=MODEL_PATH)
    
    token_path = VISION_TOKENS_DIR / "DeepSeek-OCR-_Contexts_Optical_Compression/page_1.pt"
    feats = torch.load(token_path, map_location="cpu")
    
    # 自定义简单 prompt
    prompt = "<image>\n请描述这张图片的内容。"
    
    print(f"Vision token shape: {feats.shape}")
    print(f"Prompt: {prompt}")
    
    result = decoder.decode_from_tokens(feats, prompt=prompt)
    print(f"\n生成结果:\n{result[:500]}...")
    return result


def test_case_3_vision_with_complex_instruction():
    """测试3: vision token + 复杂指令 prompt"""
    print("\n" + "="*60)
    print("测试3: vision token + 复杂指令 prompt")
    print("="*60)
    
    decoder = DeepseekTokenDecoder(model_path=MODEL_PATH)
    
    token_path = VISION_TOKENS_DIR / "DeepSeek-OCR-_Contexts_Optical_Compression/page_1.pt"
    feats = torch.load(token_path, map_location="cpu")
    
    # 复杂指令 prompt
    prompt = """<image>
你是一位专业的文档分析助手。请分析这张文档图片并回答以下问题：
1. 这篇论文的标题是什么？
2. 作者有哪些？
3. 这篇论文的主要贡献是什么？
请用简洁的语言回答。"""
    
    print(f"Vision token shape: {feats.shape}")
    print(f"Prompt: {prompt[:100]}...")
    
    result = decoder.decode_from_tokens(feats, prompt=prompt, max_new_tokens=1024)
    print(f"\n生成结果:\n{result}")
    return result


def test_case_4_vision_with_chinese_prompt():
    """测试4: vision token + 中文 prompt"""
    print("\n" + "="*60)
    print("测试4: vision token + 中文 prompt")
    print("="*60)
    
    decoder = DeepseekTokenDecoder(model_path=MODEL_PATH)
    
    token_path = VISION_TOKENS_DIR / "DeepSeek-OCR-_Contexts_Optical_Compression/page_1.pt"
    feats = torch.load(token_path, map_location="cpu")
    
    # 中文 prompt
    prompt = "<image>\n请将这份文档转换为 Markdown 格式。"
    
    print(f"Vision token shape: {feats.shape}")
    print(f"Prompt: {prompt}")
    
    result = decoder.decode_from_tokens(feats, prompt=prompt)
    print(f"\n生成结果:\n{result[:500]}...")
    return result


def test_case_5_multi_page_vision_with_text():
    """测试5: 多页 vision token + 文本 prompt"""
    print("\n" + "="*60)
    print("测试5: 多页 vision token + 文本 prompt")
    print("="*60)
    
    decoder = DeepseekTokenDecoder(model_path=MODEL_PATH)
    
    # 加载多页的 vision tokens
    doc_dir = VISION_TOKENS_DIR / "Domain-Specific_Data_Generation_Framework_for_RAG_Adaptation"
    token_files = sorted(doc_dir.glob("page_*.pt"))[:3]  # 取前3页
    
    all_feats = []
    for token_file in token_files:
        feats = torch.load(token_file, map_location="cpu")
        all_feats.append(feats)
        print(f"  加载 {token_file.name}, shape: {feats.shape}")
    
    # 拼接多页特征
    combined_feats = torch.cat(all_feats, dim=0)
    print(f"Combined vision token shape: {combined_feats.shape}")
    
    # Prompt
    prompt = """<image>
基于上述文档页面，请回答：
这篇论文提出了什么方法来解决领域特定数据的生成问题？
简要概括其核心思想。"""
    
    print(f"Prompt: {prompt}")
    
    result = decoder.decode_from_tokens(combined_feats, prompt=prompt, max_new_tokens=1024)
    print(f"\n生成结果:\n{result}")
    return result


def test_case_6_realtime_encode_and_decode():
    """测试6: 实时编码图片 + 文本 prompt（端到端测试）"""
    print("\n" + "="*60)
    print("测试6: 实时编码图片 + 文本 prompt（端到端测试）")
    print("="*60)
    
    from PIL import Image
    
    encoder = DeepseekEncoder(model_path=MODEL_PATH)
    decoder = DeepseekTokenDecoder(model_path=MODEL_PATH)
    
    # 从 PDF 提取第一页作为图片（这里我们用 PDF 文件直接转换）
    pdf_path = "VisRAG/dataset/Domain-Specific_Data_Generation_Framework_for_RAG_Adaptation.pdf"
    
    # 使用 pdf_to_images 提取第一页
    from src.vision import pdf_to_images
    images = pdf_to_images(pdf_path, max_pages=1)
    
    if not images:
        print("无法提取 PDF 图片")
        return None
    
    print(f"提取到 {len(images)} 页图片")
    
    # 编码图片
    print("编码图片中...")
    feats = encoder.encode_image(images[0])
    print(f"Vision token shape: {feats.shape}")
    
    # 不同 prompt 的对比
    prompts = [
        ("默认 OCR", "<image>\n<|grounding|>Convert the document to markdown. "),
        ("问答模式", "<image>\n这篇论文的标题是什么？"),
        ("摘要模式", "<image>\n请总结这篇论文的主要内容。"),
    ]
    
    results = {}
    for name, prompt in prompts:
        print(f"\n--- {name} ---")
        print(f"Prompt: {prompt}")
        result = decoder.decode_from_tokens(feats, prompt=prompt, max_new_tokens=512)
        print(f"结果: {result[:300]}...")
        results[name] = result
    
    return results


def main():
    """运行所有测试用例"""
    print("="*60)
    print("Vision Token + Text Decoder 实验")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Vision tokens dir: {VISION_TOKENS_DIR}")
    
    results = {}
    
    try:
        # 运行各个测试用例
        results['case1'] = test_case_1_pure_vision_token()
    except Exception as e:
        print(f"测试1失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['case2'] = test_case_2_vision_with_simple_text()
    except Exception as e:
        print(f"测试2失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['case3'] = test_case_3_vision_with_complex_instruction()
    except Exception as e:
        print(f"测试3失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['case4'] = test_case_4_vision_with_chinese_prompt()
    except Exception as e:
        print(f"测试4失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['case5'] = test_case_5_multi_page_vision_with_text()
    except Exception as e:
        print(f"测试5失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['case6'] = test_case_6_realtime_encode_and_decode()
    except Exception as e:
        print(f"测试6失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
