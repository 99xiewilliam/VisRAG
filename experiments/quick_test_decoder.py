"""
快速验证: vision token + 纯文本 传给 deepseek ocr2 decoder
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision import DeepseekTokenDecoder


def main():
    MODEL_PATH = "/home/xwh/models/DeepSeek-OCR-2"
    VISION_TOKENS_DIR = Path("VisRAG/output/vision_tokens")
    
    print("="*60)
    print("快速验证: Vision Token + Text -> Decoder")
    print("="*60)
    
    # 1. 加载 vision tokens
    token_path = VISION_TOKENS_DIR / "DeepSeek-OCR-_Contexts_Optical_Compression/page_1.pt"
    print(f"\n1. 加载 vision tokens: {token_path}")
    feats = torch.load(token_path, map_location="cpu")
    print(f"   Shape: {feats.shape}, Dtype: {feats.dtype}")
    
    # 2. 初始化 decoder
    print(f"\n2. 初始化 DeepseekTokenDecoder")
    print(f"   Model: {MODEL_PATH}")
    decoder = DeepseekTokenDecoder(model_path=MODEL_PATH)
    print(f"   Decoder 初始化完成")
    
    # 3. 测试不同的 prompt
    test_prompts = [
        ("默认 OCR", "<image>\n<|grounding|>Convert the document to markdown. "),
        ("简单中文", "<image>\n请描述这张图片的内容。"),
        ("复杂指令", "<image>\n你是一位专业的文档分析助手。请分析这张文档图片并提取标题和作者信息。"),
    ]
    
    for name, prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"测试: {name}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt[:80]}...")
        print(f"\n生成中...")
        
        try:
            result = decoder.decode_from_tokens(feats, prompt=prompt, max_new_tokens=512)
            print(f"\n结果:\n{result}")
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("验证完成!")
    print("="*60)


if __name__ == "__main__":
    main()
