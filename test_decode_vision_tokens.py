import json
import os

import torch
from safetensors.torch import safe_open
from transformers import AutoTokenizer

# Paths
MODEL_DIR = "/home/xwh/models/DeepSeek-OCR-2"
VISION_PT = "/home/xwh/VisRAG/output_reindex_1536/vision_tokens/DeepSeek-OCR-_Contexts_Optical_Compression/page_1.pt"
WEIGHTS = os.path.join(MODEL_DIR, "model-00001-of-000001.safetensors")

# Prompt
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def load_view_separator(weights_path: str) -> torch.Tensor:
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        if "model.view_seperator" not in f.keys():
            raise KeyError("model.view_seperator not found in safetensors")
        return f.get_tensor("model.view_seperator")


def load_text_decoder(model_dir: str, weights_path: str):
    # Local imports from the model repo
    from modeling_deepseekv2 import DeepseekV2ForCausalLM
    from configuration_deepseek_v2 import DeepseekV2Config

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


def build_input_ids(tokenizer, prompt: str, num_image_tokens: int) -> torch.Tensor:
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


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = load_text_decoder(MODEL_DIR, WEIGHTS)

    vision_tokens = torch.load(VISION_PT, map_location="cpu")
    if not isinstance(vision_tokens, torch.Tensor):
        raise TypeError("vision token file is not a tensor")

    view_sep = load_view_separator(WEIGHTS).to(vision_tokens.dtype)
    if view_sep.dim() == 1:
        view_sep = view_sep.unsqueeze(0)

    # Fix length mismatch (expected +1 for view_seperator)
    if vision_tokens.shape[0] == view_sep.shape[0] + 1119:
        # 1120 -> 1121
        vision_tokens = torch.cat([vision_tokens, view_sep], dim=0)
    elif vision_tokens.shape[0] == view_sep.shape[0] + 1120:
        # already 1121
        pass
    elif vision_tokens.shape[0] + 1 == 1121:
        vision_tokens = torch.cat([vision_tokens, view_sep], dim=0)

    num_image_tokens = vision_tokens.shape[0]
    input_ids = build_input_ids(tokenizer, PROMPT, num_image_tokens)

    # Build inputs_embeds and inject vision tokens
    input_ids = input_ids.cuda()
    inputs_embeds = model.get_input_embeddings()(input_ids).to(torch.bfloat16)
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    mask = (input_ids == image_token_id).squeeze(0)
    if mask.sum().item() != num_image_tokens:
        raise ValueError(f"image token count mismatch: {mask.sum().item()} vs {num_image_tokens}")

    inputs_embeds[0, mask, :] = vision_tokens.to(inputs_embeds.dtype).cuda()
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            use_cache=True,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(text)


if __name__ == "__main__":
    main()
