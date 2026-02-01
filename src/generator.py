"""LLM Generator 抽象层，支持本地模型和 OpenAI API"""
import os
import re
import json
import base64
import mimetypes
from abc import ABC, abstractmethod
from typing import Optional, Sequence, List
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import safe_open

from .config import GeneratorConfig, get_config
from .utils import get_logger

logger = get_logger(__name__)


class BaseGenerator(ABC):
    """Generator 抽象基类"""
    
    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        images: Optional[Sequence[str]] = None,
        vision_tokens: Optional[Sequence[str]] = None,
    ) -> str:
        """生成回复"""
        pass
    
    def _extract_json_array(self, text: str):
        """辅助方法：从文本中提取 JSON 数组"""
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    
    def _extract_json_object(self, text: str):
        """辅助方法：从文本中提取 JSON 对象"""
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


class LocalGenerator(BaseGenerator):
    """本地 HuggingFace 模型 Generator"""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.cfg = config or get_config().generator
        self.local_cfg = self.cfg.local
        
        # 确定设备
        if self.local_cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.local_cfg.device
        
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        logger.info(f"加载本地模型: {self.local_cfg.model_path}")
        logger.info(f"设备: {self.device}, 精度: {self.dtype}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_cfg.model_path, 
            trust_remote_code=True
        )
        
        # 加载模型
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
            "torch_dtype": self.dtype,
        }
        
        if self.local_cfg.load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.local_cfg.model_path,
            **load_kwargs
        )
        self.model.eval()
        logger.info("本地模型加载完成")
    
    def _build_prompt(self, system: str, user: str) -> str:
        """构建 prompt"""
        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return system + "\n" + user
    
    def generate(
        self,
        system: str,
        user: str,
        images: Optional[Sequence[str]] = None,
        vision_tokens: Optional[Sequence[str]] = None,
    ) -> str:
        """生成回复"""
        if images:
            logger.warning("LocalGenerator 当前为纯文本模型，将忽略 images 输入。若要多模态，请切换 generator.backend=openai 或使用支持图像的本地 VLM。")
        if vision_tokens:
            logger.warning("LocalGenerator 当前为纯文本模型，将忽略 vision_tokens 输入。")
        prompt = self._build_prompt(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.local_cfg.max_new_tokens,
                do_sample=False if self.local_cfg.temperature == 0.0 else True,
                temperature=self.local_cfg.temperature if self.local_cfg.temperature > 0 else None,
                top_p=self.local_cfg.top_p if self.local_cfg.temperature > 0 else None,
            )
        
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if prompt in text:
            return text[len(prompt):].strip()
        return text.strip()


def _load_local_package_module(module_basename: str, file_path: str, package_name: str, package_path: str):
    """加载本地 python 文件为一个“伪 package”子模块，解决其内部的相对 import（from .xxx import yyy）。"""
    import importlib.util
    import types
    import sys

    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [package_path]
        sys.modules[package_name] = pkg

    full_name = f"{package_name}.{module_basename}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    spec = importlib.util.spec_from_file_location(full_name, file_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DecoderOnlyVisionTokenGenerator(BaseGenerator):
    """
    仅加载 DeepSeek-OCR-2 中的文本 decoder（DeepseekV2ForCausalLM）权重。

    - text-only：正常 LM 推理
    - vision_tokens：将 [T, 1280] 的 vision embeddings 注入到 prompt 的 <image> token 展开位置
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.cfg = config or get_config().generator
        self.local_cfg = self.cfg.local
        self.vision_cfg = get_config().vision

        # 优先用 CUDA；若显存不足则自动回退到 CPU（会很慢，但能跑通流程）
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # decoder-only 权重来自 DeepSeek-OCR-2（多模态仓库），而不是 generator.local.model_path
        self.model_dir = self.vision_cfg.model_path
        weights = os.path.join(self.model_dir, "model-00001-of-000001.safetensors")
        if not os.path.exists(weights):
            # fallback: single file name (some exports)
            weights = os.path.join(self.model_dir, "model.safetensors")
        if not os.path.exists(weights):
            raise FileNotFoundError(f"未找到 safetensors 权重文件: {weights}")
        self.weights_path = weights

        # tokenizer（来自 OCR2 repo，包含 <image> / 特殊 token）
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)

        # 读取 language_config，构造 DeepseekV2ForCausalLM（decoder-only）
        cfg_path = os.path.join(self.model_dir, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            full_cfg = json.load(f)
        lang_cfg = dict(full_cfg.get("language_config") or {})
        if not lang_cfg:
            raise ValueError("config.json 中缺少 language_config，无法构造 decoder-only")

        # 用本地模型代码（/home/xwh/models/DeepSeek-OCR-2/*.py）构造 config + model
        pkg = "visrag_deepseek_local"
        mod_cfg = _load_local_package_module(
            "configuration_deepseek_v2",
            os.path.join(self.model_dir, "configuration_deepseek_v2.py"),
            pkg,
            self.model_dir,
        )
        mod_model = _load_local_package_module(
            "modeling_deepseekv2",
            os.path.join(self.model_dir, "modeling_deepseekv2.py"),
            pkg,
            self.model_dir,
        )
        DeepseekV2Config = mod_cfg.DeepseekV2Config
        DeepseekV2ForCausalLM = mod_model.DeepseekV2ForCausalLM

        # Flash-attn 2 在该实现里依赖 dtype，且 warnings 很吵；这里直接走 eager 更稳
        lang_cfg["_attn_implementation"] = "eager"
        text_config = DeepseekV2Config(**lang_cfg)

        logger.info(f"加载 decoder-only: {self.model_dir}")
        # 先在 CPU 上构造+load_state_dict，最后再尝试搬到 GPU，避免 GPU 峰值占用过高/直接 OOM
        self.model = DeepseekV2ForCausalLM(text_config).eval()

        # 只加载文本权重（model.layers/model.embed_tokens/model.norm + lm_head）
        state = {}
        with safe_open(self.weights_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if (
                    k == "lm_head.weight"
                    or k.startswith("model.embed_tokens.")
                    or k.startswith("model.layers.")
                    or k.startswith("model.norm.")
                ):
                    state[k] = f.get_tensor(k)
                # 额外读取 view_seperator 以便补齐旧 tokens
                if k == "model.view_seperator":
                    self._view_sep_cpu = f.get_tensor(k)

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            logger.warning(f"decoder-only unexpected keys (trimmed): {unexpected[:10]}")
        if missing:
            logger.warning(f"decoder-only missing keys (trimmed): {missing[:10]}")

        # move model to runtime device
        try:
            if self.device == "cuda":
                self.model = self.model.to(self.device, dtype=torch.bfloat16)
            else:
                self.model = self.model.to(self.device, dtype=torch.float32)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                logger.warning("CUDA 显存不足，DecoderOnlyVisionTokenGenerator 回退到 CPU（会很慢）")
                self.device = "cpu"
                self.model = self.model.to(self.device, dtype=torch.float32)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                raise

        # cache model dtype for embedding construction
        try:
            self.model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self.model_dtype = torch.float32

        # <image> token id
        self.image_token = "<image>"
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        if image_token_id is None or image_token_id < 0:
            image_token_id = getattr(self.tokenizer, "vocab", {}).get(self.image_token, None)
        if image_token_id is None:
            raise RuntimeError("无法获取 <image> 的 token id")
        self.image_token_id = int(image_token_id)

        # generation params
        self.max_new_tokens = int(getattr(self.local_cfg, "max_new_tokens", 256))
        self.temperature = float(getattr(self.local_cfg, "temperature", 0.0))
        self.top_p = float(getattr(self.local_cfg, "top_p", 1.0))

    def _build_prompt(self, system: str, user: str) -> str:
        # 统一用简单拼接，避免 chat_template 额外引入 role token 影响 stop
        system = (system or "").strip()
        user = (user or "").strip()
        if system and user:
            return system + "\n" + user
        return system or user

    def _maybe_append_view_separator(self, feats: torch.Tensor) -> torch.Tensor:
        view_sep = getattr(self, "_view_sep_cpu", None)
        if view_sep is None:
            return feats
        view_sep = view_sep.to(dtype=feats.dtype)
        if view_sep.dim() == 1:
            view_sep = view_sep.unsqueeze(0)
        if feats.shape[0] == 0:
            return torch.cat([feats, view_sep], dim=0)
        try:
            import torch.nn.functional as F
            sim = F.cosine_similarity(feats[-1], view_sep[0], dim=0)
            if float(sim) < 0.9:
                feats = torch.cat([feats, view_sep], dim=0)
        except Exception:
            feats = torch.cat([feats, view_sep], dim=0)
        return feats

    def _load_vision_feats(self, paths: Sequence[str]) -> torch.Tensor:
        feats_list: list[torch.Tensor] = []
        for p in paths:
            feats = torch.load(p, map_location="cpu")
            if not isinstance(feats, torch.Tensor):
                feats = torch.tensor(feats)
            if feats.ndim != 2:
                raise ValueError(f"vision token shape must be [T,D], got {tuple(feats.shape)} from {p}")
            feats_list.append(feats)
        feats = feats_list[0] if len(feats_list) == 1 else torch.cat(feats_list, dim=0)
        feats = self._maybe_append_view_separator(feats)
        return feats

    def _build_input_ids_with_image(self, prompt: str, num_image_tokens: int) -> torch.LongTensor:
        if self.image_token not in prompt:
            prompt = self.image_token + "\n" + prompt
        parts = prompt.split(self.image_token)
        ids: list[int] = []
        for i, part in enumerate(parts):
            if part:
                ids.extend(self.tokenizer.encode(part, add_special_tokens=False))
            if i != len(parts) - 1:
                ids.extend([self.image_token_id] * int(num_image_tokens))
        if self.tokenizer.bos_token_id is not None:
            ids = [int(self.tokenizer.bos_token_id)] + ids
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _greedy_decode(
        self,
        *,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.Tensor | None,
        attention_mask: torch.LongTensor,
    ) -> str:
        # Stop markers（优先 DeepSeek-OCR 的结束词 + EOS）
        eos_id = self.tokenizer.eos_token_id
        stop_seqs = []
        if eos_id is not None:
            eos_text = self.tokenizer.decode([eos_id], skip_special_tokens=False)
            if eos_text:
                stop_seqs.append(eos_text)
        stop_seqs.append("<｜end▁of▁sentence｜>")
        stop_seqs.extend(["<|User|>", "<|Assistant|>"])

        stop_token_seqs = []
        for s in stop_seqs:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                stop_token_seqs.append(ids)

        # 先跑首步（可能是 inputs_embeds）
        with torch.no_grad():
            # DeepseekV2Model.forward 不允许同时传 input_ids 和 inputs_embeds：
            # - vision token 注入时：首步用 inputs_embeds（input_ids 必须为 None）
            # - 纯文本时：用 input_ids（inputs_embeds 为 None）
            first_input_ids = None if inputs_embeds is not None else input_ids
            out = self.model(
                input_ids=first_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            # 某些提示词下 greedy 可能首 token 就是 EOS，导致空输出。
            # 这里做一个保守兜底：若首 token 为 EOS，则取 top-k 中第一个非 EOS token。
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if eos_id is not None and int(next_token.item()) == int(eos_id):
                topk_ids = torch.topk(logits, k=min(8, logits.shape[-1]), dim=-1).indices[0].tolist()
                fallback = None
                for tid in topk_ids:
                    if int(tid) != int(eos_id):
                        fallback = int(tid)
                        break
                if fallback is not None:
                    next_token = torch.tensor([[fallback]], device=logits.device, dtype=torch.long)
            generated = [next_token]
            past_key_values = out.past_key_values

            recent_tokens = [next_token.item()]
            repetition_threshold = 20

            cur_attn = attention_mask
            for _ in range(self.max_new_tokens - 1):
                token_id = next_token.item()
                if eos_id is not None and token_id == eos_id:
                    break
                cur_attn = torch.cat([cur_attn, torch.ones_like(next_token)], dim=1)
                out = self.model(
                    input_ids=next_token,
                    attention_mask=cur_attn,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                generated.append(next_token)
                past_key_values = out.past_key_values

                recent_tokens.append(next_token.item())
                if len(recent_tokens) > 64:
                    recent_tokens = recent_tokens[-64:]

                # stop seq
                hit = False
                for stop_seq in stop_token_seqs:
                    if len(recent_tokens) >= len(stop_seq) and recent_tokens[-len(stop_seq):] == stop_seq:
                        hit = True
                        break
                if hit:
                    break

                # repetition guard
                if len(generated) >= repetition_threshold * 2:
                    last_n = [g.item() for g in generated[-repetition_threshold:]]
                    prev_n = [g.item() for g in generated[-repetition_threshold * 2:-repetition_threshold]]
                    if last_n == prev_n:
                        break

        gen_ids = torch.cat(generated, dim=1)[0].tolist()
        return self.tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

    def _clean_output(self, text: str) -> str:
        # 与官方脚本一致：去掉显式 stop string
        if not text:
            return ""
        return text.replace("<｜end▁of▁sentence｜>", "").strip()

    def generate(
        self,
        system: str,
        user: str,
        images: Optional[Sequence[str]] = None,
        vision_tokens: Optional[Sequence[str]] = None,
    ) -> str:
        if images:
            logger.warning("DecoderOnlyVisionTokenGenerator 不支持 images，将忽略。请改用 vision_tokens。")

        prompt = self._build_prompt(system, user)

        # text-only
        if not vision_tokens:
            # 与 vision 路径保持一致：不让 tokenizer 自动追加 EOS 等特殊 token，只手动补 BOS
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if self.tokenizer.bos_token_id is not None:
                ids = [int(self.tokenizer.bos_token_id)] + ids
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids, device=self.device)
            # 纯文本用 input_ids（更稳）
            return self._clean_output(
                self._greedy_decode(input_ids=input_ids, inputs_embeds=None, attention_mask=attention_mask)
            )

        # vision tokens injection: 需要 <image> 占位
        feats = self._load_vision_feats(vision_tokens).to(self.device, dtype=getattr(self, "model_dtype", torch.bfloat16))
        # 重要：避免材料/上下文里本身包含 "<image>" 字面量导致被当成占位符多次展开。
        # 这里强制只保留 1 个占位符（放在最前面），其余全部转义成普通文本。
        safe_prompt = prompt.replace(self.image_token, "<image_text>")
        # Place <image> right before the Question block when possible (closer to our verified script)
        if "Question:" in safe_prompt:
            mm_prompt = safe_prompt.replace("Question:", f"{self.image_token}\n\nQuestion:", 1)
        elif "Question：" in safe_prompt:
            mm_prompt = safe_prompt.replace("Question：", f"{self.image_token}\n\nQuestion：", 1)
        elif "问题：" in safe_prompt:
            mm_prompt = safe_prompt.replace("问题：", f"{self.image_token}\n\n问题：", 1)
        elif "问题:" in safe_prompt:
            mm_prompt = safe_prompt.replace("问题:", f"{self.image_token}\n\n问题:", 1)
        else:
            mm_prompt = self.image_token + "\n" + safe_prompt

        input_ids = self._build_input_ids_with_image(mm_prompt, num_image_tokens=feats.shape[0])
        images_seq_mask = (input_ids == int(self.image_token_id))
        if images_seq_mask.sum().item() != feats.shape[0]:
            raise ValueError(
                f"Image token 数量 ({images_seq_mask.sum().item()}) 与 vision tokens 数量 ({feats.shape[0]}) 不匹配"
            )

        attention_mask = torch.ones_like(input_ids, device=self.device)
        with torch.no_grad():
            inputs_embeds = self.model.get_input_embeddings()(input_ids).to(getattr(self, "model_dtype", torch.bfloat16))
            inputs_embeds[images_seq_mask] = feats.to(inputs_embeds.dtype)

        # 关键：首步用 inputs_embeds，但保留 input_ids 以便后续 step 用 token 推进
        return self._clean_output(
            self._greedy_decode(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        )


class OpenAIGenerator(BaseGenerator):
    """OpenAI API Generator"""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "使用 OpenAI API 需要安装 openai 包: "
                "pip install openai"
            )
        
        self.cfg = config or get_config().generator
        self.oai_cfg = self.cfg.openai
        
        # 检查 API Key
        if not self.oai_cfg.api_key:
            raise ValueError(
                "OpenAI API Key 未设置。请通过以下方式之一配置：\n"
                "1. 在 config.yaml 中设置 generator.openai.api_key\n"
                "2. 设置环境变量 OPENAI_API_KEY"
            )
        
        # 初始化 OpenAI 客户端
        client_kwargs = {
            "api_key": self.oai_cfg.api_key,
            "timeout": self.oai_cfg.timeout,
            "max_retries": self.oai_cfg.max_retries,
        }
        if self.oai_cfg.base_url:
            client_kwargs["base_url"] = self.oai_cfg.base_url
        
        self.client = OpenAI(**client_kwargs)
        self.model = self.oai_cfg.model
        
        logger.info(f"使用 OpenAI 模型: {self.model}")
        if self.oai_cfg.base_url:
            logger.info(f"API Base: {self.oai_cfg.base_url}")
    
    def _image_path_to_data_url(self, path: str) -> str:
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = "image/png"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def generate(
        self,
        system: str,
        user: str,
        images: Optional[Sequence[str]] = None,
        vision_tokens: Optional[Sequence[str]] = None,
    ) -> str:
        """生成回复"""
        if vision_tokens:
            logger.warning("OpenAIGenerator 当前不支持 vision_tokens，将忽略。")
        if images:
            content = [{"type": "text", "text": user}]
            for p in images:
                try:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": self._image_path_to_data_url(p)},
                    })
                except Exception:
                    logger.exception(f"读取图片失败，将跳过: {p}")
            user_msg = {"role": "user", "content": content}
        else:
            user_msg = {"role": "user", "content": user}

        messages = [{"role": "system", "content": system}, user_msg]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.oai_cfg.max_tokens,
        }
        
        # OpenAI API 的 temperature 必须 > 0，为 0 时不传
        if self.oai_cfg.temperature > 0:
            kwargs["temperature"] = self.oai_cfg.temperature
            kwargs["top_p"] = self.oai_cfg.top_p
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()


class DeepseekOCR2Generator(BaseGenerator):
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.cfg = config or get_config().generator
        self.vision_cfg = get_config().vision

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise RuntimeError("DeepseekOCR2Generator 需要 CUDA")

        self.model_path = self.vision_cfg.model_path
        logger.info(f"加载 DeepSeek-OCR2: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # 可选：多卡切分权重（需要安装 accelerate）
        # 用法示例：
        #   CUDA_VISIBLE_DEVICES=0,1 VISRAG_OCR2_DEVICE_MAP=auto python ...
        device_map = os.environ.get("VISRAG_OCR2_DEVICE_MAP")
        if device_map:
            load_kwargs = {
                "trust_remote_code": True,
                "device_map": device_map,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }
            self.model = AutoModel.from_pretrained(self.model_path, **load_kwargs)
        else:
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)
        self.model.eval()
        self.embed_tokens = self.model.get_input_embeddings()
        # embedding 所在设备（device_map=auto 时不一定是 cuda:0）
        self._embed_device = self.embed_tokens.weight.device

        self.image_token = "<image>"
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        if image_token_id is None or image_token_id < 0:
            image_token_id = getattr(self.tokenizer, "vocab", {}).get(self.image_token, None)
        if image_token_id is None:
            raise RuntimeError("无法获取 <image> 的 token id")
        self.image_token_id = int(image_token_id)

    def _dummy_images(self):
        # HF 版 DeepSeek-OCR2 的 forward 会在 (len(prompt)>1) 时无条件访问 images[0][1]
        # 用全零 dummy tensor 可让其跳过视觉编码分支（torch.sum(images[0][1]) == 0）。
        z = torch.zeros((1,), device=self._embed_device, dtype=torch.float16)
        return [[z, z]]

    def _maybe_append_view_separator(self, feats: torch.Tensor) -> torch.Tensor:
        # 与官方链路对齐：vision tokens 末尾的 view_seperator
        view_sep = getattr(self.model.model, "view_seperator", None)
        if view_sep is None:
            return feats
        view_sep = view_sep.detach().to(feats.device, dtype=feats.dtype)
        if feats.shape[0] == 0:
            return torch.cat([feats, view_sep[None, :]], dim=0)
        try:
            import torch.nn.functional as F
            sim = F.cosine_similarity(feats[-1], view_sep, dim=0)
            if float(sim) < 0.9:
                feats = torch.cat([feats, view_sep[None, :]], dim=0)
        except Exception:
            feats = torch.cat([feats, view_sep[None, :]], dim=0)
        return feats

    def decode_vision_tokens(self, paths: Sequence[str], prompt: Optional[str] = None) -> List[str]:
        """使用已加载的 OCR2 模型解码 vision tokens，避免额外加载一份模型导致 OOM。"""
        results: List[str] = []
        for p in paths:
            try:
                feats = torch.load(p, map_location="cpu")
                if not isinstance(feats, torch.Tensor):
                    feats = torch.tensor(feats)
                if feats.ndim != 2:
                    raise ValueError(f"vision token shape must be [T,D], got {tuple(feats.shape)} from {p}")
                feats = feats.to(self._embed_device)
                feats = self._maybe_append_view_separator(feats)
                use_prompt = prompt or "<image>\n<|grounding|>Convert the document to markdown. "
                # 与 vision.py::DeepseekTokenDecoder 一致的注入逻辑
                input_ids = self._build_input_ids_with_image(use_prompt, num_image_tokens=feats.shape[0])
                images_seq_mask = (input_ids == int(self.image_token_id))
                attention_mask = torch.ones_like(input_ids, device=self._embed_device)
                with torch.no_grad():
                    inputs_embeds = self.embed_tokens(input_ids)
                    if images_seq_mask.sum().item() != feats.shape[0]:
                        raise ValueError(
                            f"Image token 数量 ({images_seq_mask.sum().item()}) 与 vision tokens 数量 ({feats.shape[0]}) 不匹配"
                        )
                    inputs_embeds[images_seq_mask] = feats.to(inputs_embeds.dtype)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model.generate(
                        input_ids=input_ids,
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        images=self._dummy_images(),
                        temperature=0.0,
                        do_sample=False,
                        max_new_tokens=2048,
                        use_cache=True,
                    )
                gen = out[0, input_ids.shape[1]:]
                text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
            except Exception:
                logger.exception(f"vision tokens 解码失败: {p}")
                text = ""
            results.append(text)
        return results

    def _build_prompt_text(self, system: str, user: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return system + "\n" + user

    def _load_vision_feats(self, paths: Sequence[str]) -> Optional[torch.Tensor]:
        feats_list: list[torch.Tensor] = []
        for p in paths:
            feats = torch.load(p, map_location="cpu")
            if not isinstance(feats, torch.Tensor):
                feats = torch.tensor(feats)
            if feats.ndim != 2:
                raise ValueError(f"vision token shape must be [T,D], got {tuple(feats.shape)} from {p}")
            feats_list.append(feats)
        if not feats_list:
            return None
        if len(feats_list) == 1:
            feats = feats_list[0]
        else:
            feats = torch.cat(feats_list, dim=0)

        # 防止序列过长导致 generate() 在首步产生巨型 logits（[B,seq,vocab]）并 OOM
        # 可通过环境变量调整：VISRAG_OCR2_MAX_VISION_TOKENS
        try:
            max_tokens = int(os.environ.get("VISRAG_OCR2_MAX_VISION_TOKENS", "2048"))
        except Exception:
            max_tokens = 2048
        if max_tokens > 0 and feats.shape[0] > max_tokens:
            # 均匀下采样保留全局信息（比直接截断更稳）
            idx = torch.linspace(0, feats.shape[0] - 1, steps=max_tokens).long()
            feats = feats.index_select(0, idx)
        return feats

    def _build_input_ids_with_image(self, prompt: str, num_image_tokens: int) -> torch.LongTensor:
        if self.image_token not in prompt:
            prompt = self.image_token + "\n" + prompt
        parts = prompt.split(self.image_token)
        ids: list[int] = []
        for i, part in enumerate(parts):
            if part:
                ids.extend(self.tokenizer.encode(part, add_special_tokens=False))
            if i != len(parts) - 1:
                ids.extend([self.image_token_id] * int(num_image_tokens))
        if self.tokenizer.bos_token_id is not None:
            ids = [int(self.tokenizer.bos_token_id)] + ids
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def generate(
        self,
        system: str,
        user: str,
        images: Optional[Sequence[str]] = None,
        vision_tokens: Optional[Sequence[str]] = None,
    ) -> str:
        if images:
            logger.warning("DeepseekOCR2Generator 不支持 images 路径输入，将忽略。请使用 vision_tokens 或直接提供图片给 OCR2 的 infer 接口。")

        prompt = self._build_prompt_text(system, user)

        feats = self._load_vision_feats(vision_tokens or []) if vision_tokens else None

        max_new_tokens = int(getattr(self.cfg.local, "max_new_tokens", 768))
        temperature = float(getattr(self.cfg.local, "temperature", 0.0))
        top_p = float(getattr(self.cfg.local, "top_p", 1.0))

        if feats is None:
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    **inputs,
                    images=self._dummy_images(),
                    max_new_tokens=max_new_tokens,
                    do_sample=False if temperature == 0.0 else True,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    use_cache=True,
                )
            decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if prompt in decoded:
                return decoded[len(prompt):].strip()
            return decoded.strip()

        # vision_tokens: 采用 inputs_embeds 注入（同 src/vision.py::DeepseekTokenDecoder），避免模型内部重新做视觉编码
        feats = feats.to(self.device)
        input_ids = self._build_input_ids_with_image(prompt, num_image_tokens=feats.shape[0])
        images_seq_mask = (input_ids == int(self.image_token_id))

        # 进一步限制 prompt 总 token 长度，避免首步 forward 产生 [B, seq, vocab] 巨型 logits 后
        # transformers 在 _sample 里对 outputs.logits.clone() 造成 OOM
        try:
            max_prompt_tokens = int(os.environ.get("VISRAG_OCR2_MAX_PROMPT_TOKENS", "2300"))
        except Exception:
            max_prompt_tokens = 2300
        if max_prompt_tokens > 0 and input_ids.shape[1] > max_prompt_tokens:
            # 结构是: [BOS] + [<image>]*T + text_tokens...
            bos_len = 1 if (self.tokenizer.bos_token_id is not None) else 0
            img_len = int(feats.shape[0])
            prefix_len = bos_len + img_len
            # 保留全部图像 token，再从文本尾部截取
            keep_text = max_prompt_tokens - prefix_len
            if keep_text <= 0:
                # 极端情况：图像 token 本身就超过限制，进一步下采样图像 token
                new_img_len = max(64, max_prompt_tokens - bos_len)
                if new_img_len < img_len:
                    idx = torch.linspace(0, img_len - 1, steps=new_img_len).long().to(input_ids.device)
                    # 只更新 feats 与 input_ids 中的 image token 数量
                    feats = feats.index_select(0, idx.to(feats.device))
                    input_ids = self._build_input_ids_with_image(prompt, num_image_tokens=feats.shape[0])
                    images_seq_mask = (input_ids == int(self.image_token_id))
            else:
                # 截取: prefix + 最后 keep_text 个 token
                text_tail = input_ids[:, -keep_text:]
                input_ids = torch.cat([input_ids[:, :prefix_len], text_tail], dim=1)
                images_seq_mask = (input_ids == int(self.image_token_id))

        attention_mask = torch.ones_like(input_ids, device=self.device)

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
            if images_seq_mask.sum().item() != feats.shape[0]:
                raise ValueError(
                    f"Image token 数量 ({images_seq_mask.sum().item()}) 与 vision tokens 数量 ({feats.shape[0]}) 不匹配"
                )
            inputs_embeds[images_seq_mask] = feats.to(inputs_embeds.dtype)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(
                # 同时传 input_ids + inputs_embeds：
                # - 首步 generation 会用 inputs_embeds（prepare_inputs_for_generation 里 past_key_values is None 时）
                # - 后续 step 会用 input_ids 推进，避免出现 input_ids 被裁成空导致 RoPE 维度错配
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                images=self._dummy_images(),
                temperature=0.0,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        gen = out[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return text.strip()


def create_generator(config: Optional[GeneratorConfig] = None) -> BaseGenerator:
    """
    工厂函数：根据配置创建对应的 Generator
    
    Usage:
        >>> from src.generator import create_generator
        >>> gen = create_generator()  # 自动读取 config.yaml
        >>> response = gen.generate("你是一个助手", "你好")
    """
    cfg = config or get_config().generator
    
    if cfg.backend == "local":
        return LocalGenerator(cfg)
    elif cfg.backend == "decoder_only":
        return DecoderOnlyVisionTokenGenerator(cfg)
    elif cfg.backend == "openai":
        return OpenAIGenerator(cfg)
    elif cfg.backend == "deepseek_ocr2":
        return DeepseekOCR2Generator(cfg)
    else:
        raise ValueError(f"不支持的 backend 类型: {cfg.backend}，可选: local, decoder_only, openai, deepseek_ocr2")


# 全局 Generator 实例（单例模式）
_generator_instance: Optional[BaseGenerator] = None


def get_generator() -> BaseGenerator:
    """获取全局 Generator 实例"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = create_generator()
    return _generator_instance


def reset_generator():
    """重置 Generator 实例（用于重新加载配置后）"""
    global _generator_instance
    _generator_instance = None
