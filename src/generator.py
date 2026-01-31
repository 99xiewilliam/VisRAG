"""LLM Generator 抽象层，支持本地模型和 OpenAI API"""
import os
import re
import json
import base64
import mimetypes
from abc import ABC, abstractmethod
from typing import Optional, Sequence
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

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
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(self.device)
        self.model.eval()

        self.image_token = "<image>"
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        if image_token_id is None or image_token_id < 0:
            image_token_id = getattr(self.tokenizer, "vocab", {}).get(self.image_token, None)
        if image_token_id is None:
            raise RuntimeError("无法获取 <image> 的 token id")
        self.image_token_id = int(image_token_id)

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
            return feats_list[0]
        return torch.cat(feats_list, dim=0)

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

        feats = feats.to(self.device)
        input_ids = self._build_input_ids_with_image(prompt, num_image_tokens=feats.shape[0])
        images_seq_mask = (input_ids == int(self.image_token_id))
        attention_mask = torch.ones_like(input_ids, device=self.device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images_seq_mask=images_seq_mask,
                image_embeds=[feats.to(torch.bfloat16)],
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
    elif cfg.backend == "openai":
        return OpenAIGenerator(cfg)
    elif cfg.backend == "deepseek_ocr2":
        return DeepseekOCR2Generator(cfg)
    else:
        raise ValueError(f"不支持的 backend 类型: {cfg.backend}，可选: local, openai, deepseek_ocr2")


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
