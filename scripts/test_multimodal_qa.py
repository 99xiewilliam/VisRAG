import json
import os
import sys

import torch
from safetensors.torch import safe_open
from transformers import AutoTokenizer

# Paths
MODEL_DIR = "/home/xwh/models/DeepSeek-OCR-2"
sys.path.insert(0, MODEL_DIR)
VISION_PT = "/home/xwh/VisRAG/output_reindex_1536/vision_tokens/DeepSeek-OCR-_Contexts_Optical_Compression/page_1.pt"
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


def load_view_separator(weights_path: str) -> torch.Tensor:
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        if "model.view_seperator" not in f.keys():
            raise KeyError("model.view_seperator not found in safetensors")
        return f.get_tensor("model.view_seperator")


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


def build_input_ids_with_image(tokenizer, prompt: str, num_image_tokens: int) -> torch.Tensor:
    """Build input_ids with <image> tokens expanded"""
    image_token = "<image>"
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    if image_token_id is None or image_token_id < 0:
        raise ValueError("image token id not found in tokenizer")

    splits = prompt.split(image_token)
    tokenized = []
    for i, chunk in enumerate(splits):
        if chunk:
            tokenized += tokenizer.encode(chunk, add_special_tokens=False)
        if i < len(splits) - 1:
            tokenized += [image_token_id] * num_image_tokens

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    tokenized = [bos_id] + tokenized
    return torch.tensor([tokenized], dtype=torch.long)


def generate_with_vision_tokens(model, tokenizer, text_context: str, vision_tokens: torch.Tensor, question: str, max_new_tokens: int = 256):
    """Generate answer using both text context and vision tokens"""
    # Construct prompt: text + <image> + question
    prompt = f"{text_context}\n<image>\n\n问题：{question}\n回答："
    
    num_image_tokens = vision_tokens.shape[0]
    input_ids = build_input_ids_with_image(tokenizer, prompt, num_image_tokens)
    
    # Build inputs_embeds and inject vision tokens
    input_ids = input_ids.cuda()
    inputs_embeds = model.get_input_embeddings()(input_ids).to(torch.bfloat16)
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    mask = (input_ids == image_token_id).squeeze(0)
    
    if mask.sum().item() != num_image_tokens:
        raise ValueError(f"image token count mismatch: {mask.sum().item()} vs {num_image_tokens}")
    
    inputs_embeds[0, mask, :] = vision_tokens.to(inputs_embeds.dtype).cuda()
    attention_mask = torch.ones_like(input_ids)
    
    # Manual greedy decode
    eos_id = tokenizer.eos_token_id
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
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
    
    # Load vision tokens
    print("Loading vision tokens...")
    vision_tokens = torch.load(VISION_PT, map_location="cpu")
    if not isinstance(vision_tokens, torch.Tensor):
        raise TypeError("vision token file is not a tensor")
    
    view_sep = load_view_separator(WEIGHTS).to(vision_tokens.dtype)
    if view_sep.dim() == 1:
        view_sep = view_sep.unsqueeze(0)
    
    # Fix length mismatch (expected +1 for view_seperator)
    if vision_tokens.shape[0] == 1120:
        vision_tokens = torch.cat([vision_tokens, view_sep], dim=0)
    elif vision_tokens.shape[0] != 1121:
        print(f"Warning: vision tokens shape is {vision_tokens.shape[0]}, expected 1121")
    
    print(f"Vision tokens shape: {vision_tokens.shape}\n")
    
    # Test cases: text context + vision token + question
    test_cases = [
        {
            "text_context": "这是一篇关于 DeepSeek-OCR 的论文。论文研究了如何通过光学压缩将长上下文压缩成视觉 token。",
            "question": "这篇论文的标题是什么？",
        },
        {
            "text_context": "DeepSeek-OCR 是一个端到端的 OCR 模型，由 DeepEncoder 和 DeepSeek-3B 解码器组成。",
            "question": "这篇论文的作者是谁？",
        },
        {
            "text_context": "论文讨论了如何将文档图像压缩成视觉 token，然后解码成 markdown 格式。",
            "question": "论文的摘要部分提到了什么？",
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"Test Case {i}")
        print(f"{'='*60}")
        print(f"Text Context: {test['text_context']}")
        print(f"Question: {test['question']}")
        print("\nGenerating answer (using vision tokens)...")
        
        answer = generate_with_vision_tokens(
            model, tokenizer, test['text_context'], vision_tokens, test['question'], max_new_tokens=256
        )
        
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
