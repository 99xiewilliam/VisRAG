import json
import os
import sys

import torch
from safetensors.torch import safe_open
from transformers import AutoTokenizer

# Paths
MODEL_DIR = "/home/xwh/models/DeepSeek-OCR-2"
sys.path.insert(0, MODEL_DIR)
WEIGHTS = os.path.join(MODEL_DIR, "model-00001-of-000001.safetensors")


def load_local_package_module(module_basename: str, file_path: str, package_name: str):
    import importlib.util
    import types

    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [MODEL_DIR]
        sys.modules[package_name] = pkg

    full_name = f"{package_name}.{module_basename}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    spec = importlib.util.spec_from_file_location(full_name, file_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def load_text_decoder(model_dir: str, weights_path: str):
    """Load decoder-only model (no vision components)"""
    pkg = "deepseek_ocr2_local"
    mod_cfg = load_local_package_module(
        "configuration_deepseek_v2", os.path.join(model_dir, "configuration_deepseek_v2.py"), pkg
    )
    mod_model = load_local_package_module(
        "modeling_deepseekv2", os.path.join(model_dir, "modeling_deepseekv2.py"), pkg
    )
    DeepseekV2Config = mod_cfg.DeepseekV2Config
    DeepseekV2ForCausalLM = mod_model.DeepseekV2ForCausalLM

    cfg = json.load(open(os.path.join(model_dir, "config.json"), "r"))
    lang_cfg = dict(cfg["language_config"])
    lang_cfg["_attn_implementation"] = "flash_attention_2"
    text_config = DeepseekV2Config(**lang_cfg)

    model = DeepseekV2ForCausalLM(text_config)

    # Load only text weights
    state = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if (
                k == "lm_head.weight"
                or k.startswith("model.embed_tokens.")
                or k.startswith("model.layers.")
                or k.startswith("model.norm.")
            ):
                state[k] = f.get_tensor(k)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print("unexpected keys (trimmed):", unexpected[:10])
    if missing:
        print("missing keys (trimmed):", missing[:10])

    model = model.eval().cuda().to(torch.bfloat16)
    return model


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """Generate text response from prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    eos_id = tokenizer.eos_token_id
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = [next_token]
        past_key_values = outputs.past_key_values

        cur_attn = attention_mask
        for _ in range(max_new_tokens - 1):
            cur_attn = torch.cat([cur_attn, torch.ones_like(next_token)], dim=1)
            outputs = model(
                input_ids=next_token,
                attention_mask=cur_attn,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            past_key_values = outputs.past_key_values
            if eos_id is not None and next_token.item() == eos_id:
                break

    output_ids = torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)
    input_len = input_ids.shape[1]
    gen_ids = output_ids[0, input_len:]
    text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False)
    return text


def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = load_text_decoder(MODEL_DIR, WEIGHTS)
    print("Model loaded successfully!\n")

    # Test cases
    test_cases = [
        {
            "context": "DeepSeek-OCR 是一个端到端的 OCR 模型，由 DeepEncoder 和 DeepSeek-3B 解码器组成。它能够将文档图像压缩成视觉 token，然后解码成 markdown 格式。",
            "question": "DeepSeek-OCR 的主要组成部分是什么？",
        },
        {
            "context": "Python 是一种高级编程语言，由 Guido van Rossum 在 1991 年首次发布。它以其简洁的语法和强大的功能而闻名。",
            "question": "Python 是什么时候发布的？",
        },
        {
            "context": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需明确编程。常见的机器学习方法包括监督学习、无监督学习和强化学习。",
            "question": "机器学习的主要类型有哪些？",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"Test Case {i}")
        print(f"{'='*60}")
        print(f"Context: {test['context']}")
        print(f"Question: {test['question']}")
        print("\nGenerating answer...")

        # Format prompt (you can adjust this format)
        prompt = f"{test['context']}\n\n问题：{test['question']}\n回答："

        answer = generate_text(model, tokenizer, prompt, max_new_tokens=256)
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
