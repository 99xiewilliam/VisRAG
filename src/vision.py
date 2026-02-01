import os
import io
import fitz
import torch
from PIL import Image, ImageOps
from typing import List, Dict, Any, Tuple, Optional
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
from .utils import get_logger

logger = get_logger(__name__)


BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True


def _ensure_dynamic_cache_compat():
    try:
        from transformers import DynamicCache
    except Exception:
        return
    if hasattr(DynamicCache, "from_legacy_cache"):
        if not hasattr(DynamicCache, "get_usable_length") and hasattr(DynamicCache, "get_seq_length"):
            def get_usable_length(self, seq_length: int | None = None, layer_idx: int = 0) -> int:
                return int(self.get_seq_length(layer_idx=layer_idx))

            DynamicCache.get_usable_length = get_usable_length
        return

    @classmethod
    def from_legacy_cache(cls, past_key_values=None, config=None, **kwargs):
        if past_key_values is None:
            return cls(config=config, **kwargs)
        if isinstance(past_key_values, cls):
            return past_key_values
        return cls(ddp_cache_data=past_key_values, config=config, **kwargs)

    DynamicCache.from_legacy_cache = from_legacy_cache
    if not hasattr(DynamicCache, "get_usable_length") and hasattr(DynamicCache, "get_seq_length"):
        def get_usable_length(self, seq_length: int | None = None, layer_idx: int = 0) -> int:
            return int(self.get_seq_length(layer_idx=layer_idx))

        DynamicCache.get_usable_length = get_usable_length


def pdf_to_images(pdf_path: str, dpi: int = 144, max_pages: int | None = None) -> List[Image.Image]:
    images: List[Image.Image] = []
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i in range(doc.page_count):
        if max_pages is not None and i >= max_pages:
            break
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        data = pix.tobytes("png")
        img = Image.open(io.BytesIO(data))
        images.append(img)
    doc.close()
    return images


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: List[Tuple[int, int]], width: int, height: int, image_size: int):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def count_tiles(orig_width: int, orig_height: int, min_num: int = 2, max_num: int = 6, image_size: int = 768):
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    return find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)


def dynamic_preprocess(image: Image.Image, image_size: int = 768, min_num: int = 2, max_num: int = 6):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    return processed_images, target_aspect_ratio


class ImageTransform:
    def __init__(self, mean: Tuple[float, float, float] = (0.5, 0.5, 0.5), std: Tuple[float, float, float] = (0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __call__(self, pil_img: Image.Image):
        return self.transform(pil_img)


class DeepseekEncoder:
    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        if model_path is None:
            # 延迟导入，避免循环依赖；优先使用 config.yaml 的 vision.model_path
            try:
                from .config import get_config
                model_path = get_config().vision.model_path
            except Exception:
                model_path = "/home/xwh/models/DeepSeek-OCR-2"

        logger.info(f"初始化 DeepseekEncoder: device={self.device}, dtype={self.dtype}, model_path={model_path}")
        _ensure_dynamic_cache_compat()
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        self.sam = model.model.sam_model.to(self.device, dtype=self.dtype)
        self.qwen2 = model.model.qwen2_model.to(self.device, dtype=self.dtype)
        self.projector = model.model.projector.to(self.device, dtype=self.dtype)
        # 与官方链路对齐：视觉 token 序列末尾的 view_seperator
        self.view_seperator = getattr(model.model, "view_seperator", None)
        if self.view_seperator is not None:
            self.view_seperator = self.view_seperator.detach().to(self.device, dtype=self.dtype)
        self.image_transform = ImageTransform()
        del model

    def _build_views(self, image: Image.Image):
        if image.size[0] <= 768 and image.size[1] <= 768:
            crop_ratio = (1, 1)
            images_crop_raw = []
        else:
            if CROP_MODE:
                images_crop_raw, crop_ratio = dynamic_preprocess(image, image_size=IMAGE_SIZE)
            else:
                crop_ratio = (1, 1)
                images_crop_raw = []

        if IMAGE_SIZE <= 768 and not CROP_MODE:
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        global_view = ImageOps.pad(image, (BASE_SIZE, BASE_SIZE), color=tuple(int(x * 255) for x in self.image_transform.mean))
        global_tensor = self.image_transform(global_view).to(self.dtype)

        images_crop_list = []
        if crop_ratio[0] > 1 or crop_ratio[1] > 1:
            for img in images_crop_raw:
                images_crop_list.append(self.image_transform(img).to(self.dtype))

        if images_crop_list:
            crops = torch.stack(images_crop_list, dim=0)
        else:
            crops = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=self.dtype)

        return global_tensor, crops

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        global_tensor, crops = self._build_views(image)
        image_ori = global_tensor.unsqueeze(0).to(self.device)
        patches = crops.to(self.device)

        with torch.no_grad():
            if torch.sum(patches).item() != 0:
                local_feat_1 = self.sam(patches)
                local_feat_2 = self.qwen2(local_feat_1)
                local_features = self.projector(local_feat_2)
                global_feat_1 = self.sam(image_ori)
                global_feat_2 = self.qwen2(global_feat_1)
                global_features = self.projector(global_feat_2)
                global_features = global_features.view(-1, global_features.shape[-1])
                local_features = local_features.view(-1, local_features.shape[-1])
                features = torch.cat([local_features, global_features], dim=0)
            else:
                global_feat_1 = self.sam(image_ori)
                global_feat_2 = self.qwen2(global_feat_1)
                global_features = self.projector(global_feat_2)
                features = global_features.view(-1, global_features.shape[-1])
        # 按官方 vLLM 实现：在末尾追加 view_seperator
        if self.view_seperator is not None:
            features = torch.cat([features, self.view_seperator[None, :]], dim=0)
        return features


class DeepseekTokenDecoder:
    """使用预计算的 vision tokens（[T, 1280]）进行生成，跳过视觉编码。

    通过直接构造 inputs_embeds 并将 vision tokens 注入到对应位置。
    """

    def __init__(self, model_path: str = "/home/xwh/models/DeepSeek-OCR-2", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not str(self.device).startswith("cuda"):
            raise RuntimeError("DeepseekTokenDecoder requires CUDA for generation")
        logger.info(f"初始化 DeepseekTokenDecoder: device={self.device}")
        _ensure_dynamic_cache_compat()
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.image_token = "<image>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        if self.image_token_id is None or self.image_token_id < 0:
            # fallback: some tokenizers store in vocab
            self.image_token_id = getattr(self.tokenizer, "vocab", {}).get(self.image_token, None)
        if self.image_token_id is None:
            raise RuntimeError("无法获取 <image> 的 token id")
        
        # 获取文本 embedding 层
        self.embed_tokens = self.model.get_input_embeddings()

    def _build_input_ids(self, prompt: str, num_image_tokens: int) -> torch.LongTensor:
        if self.image_token not in prompt:
            raise ValueError("prompt must contain <image> placeholder")
        parts = prompt.split(self.image_token)
        ids: list[int] = []
        for i, part in enumerate(parts):
            if part:
                ids.extend(self.tokenizer.encode(part, add_special_tokens=False))
            if i != len(parts) - 1:
                ids.extend([int(self.image_token_id)] * int(num_image_tokens))
        # add BOS if available
        if self.tokenizer.bos_token_id is not None:
            ids = [int(self.tokenizer.bos_token_id)] + ids
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def decode_from_tokens(
        self,
        feats: torch.Tensor,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown. ",
        max_new_tokens: int = 2048,
    ) -> str:
        """
        使用预计算的 vision tokens 进行解码。
        
        Args:
            feats: 预计算的 vision tokens, shape [T, D] (D=1280 for DeepSeek-OCR-2)
            prompt: 包含 <image> 占位符的 prompt
            max_new_tokens: 最大生成 token 数
        """
        # feats: [T, 1280]
        if feats.ndim != 2:
            raise ValueError(f"feats must be 2D [T, D], got {tuple(feats.shape)}")
        feats = feats.to(self.device)
        # 兼容旧 token（可能缺 view_seperator）：若末 token 不像 view_seperator，则追加
        view_sep = getattr(self.model.model, "view_seperator", None)
        if view_sep is not None:
            view_sep = view_sep.detach().to(feats.device, dtype=feats.dtype)
            if feats.shape[0] > 0:
                # 简单相似度检查，避免重复追加
                import torch.nn.functional as F
                sim = F.cosine_similarity(feats[-1], view_sep, dim=0)
                if float(sim) < 0.9:
                    feats = torch.cat([feats, view_sep[None, :]], dim=0)

        input_ids = self._build_input_ids(prompt, num_image_tokens=feats.shape[0])
        images_seq_mask = (input_ids == int(self.image_token_id))
        attention_mask = torch.ones_like(input_ids, device=self.device)

        # 构建 inputs_embeds: 先获取文本 embedding，然后在 image token 位置替换为 vision tokens
        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)
            # 将 vision tokens 注入到 image token 对应的位置
            # feats 需要与 image token 数量匹配
            if images_seq_mask.sum().item() != feats.shape[0]:
                raise ValueError(
                    f"Image token 数量 ({images_seq_mask.sum().item()}) 与 vision tokens 数量 ({feats.shape[0]}) 不匹配"
                )
            inputs_embeds[images_seq_mask] = feats.to(inputs_embeds.dtype)

        # 构造 dummy images 参数来绕过模型内部的视觉编码
        # 模型会检查 torch.sum(images[0][1]).item() != 0 来决定是否进行视觉编码
        # 传入全零 tensor 可以跳过视觉编码，直接使用我们提供的 inputs_embeds
        dummy_image = torch.zeros((1, 3, 1024, 1024), device=self.device, dtype=torch.float16)
        dummy_patches = torch.zeros((1, 3, 768, 768), device=self.device, dtype=torch.float16)
        # images 参数格式: list of [patches, image_ori]
        dummy_images = [[dummy_patches, dummy_image]]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(
                # 同时传 input_ids + inputs_embeds，避免后续 step input_ids 被裁空导致 RoPE 维度错配
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                images=dummy_images,  # 传入 dummy images 绕过视觉编码检查
                temperature=0.0,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        # out includes prompt tokens; slice by input length
        gen = out[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return text.strip()


def decode_tokens_list(
    tokens_paths: List[str],
    prompt: str | None = None,
    device: str | None = None,
) -> List[str]:
    global _decoder_instance, _decoder_device
    if _decoder_instance is None or (device is not None and device != _decoder_device):
        _decoder_device = device
        _decoder_instance = DeepseekTokenDecoder(device=device)
    decoder = _decoder_instance
    results: List[str] = []
    for path in tokens_paths:
        try:
            feats = torch.load(path, map_location="cpu")
            if not isinstance(feats, torch.Tensor):
                feats = torch.tensor(feats)
            if prompt:
                text = decoder.decode_from_tokens(feats, prompt=prompt)
            else:
                text = decoder.decode_from_tokens(feats)
        except Exception:
            logger.exception(f"vision tokens 解码失败: {path}")
            text = ""
        results.append(text)
    return results


_decoder_instance: Optional[DeepseekTokenDecoder] = None
_decoder_device: Optional[str] = None

def mean_pool(feats: torch.Tensor) -> torch.Tensor:
    return feats.mean(dim=0)


def extract_pdf_vision_tokens(
    pdf_path: str,
    out_dir: str,
    device: str | None = None,
    max_pages: int | None = None,
) -> List[Dict[str, Any]]:
    logger.info(f"提取 PDF 视觉 tokens: {pdf_path} -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    encoder = DeepseekEncoder(device=device)
    pages = pdf_to_images(pdf_path, max_pages=max_pages)
    logger.info(f"PDF 共 {len(pages)} 页")
    results: List[Dict[str, Any]] = []
    for idx, img in enumerate(pages):
        feats = encoder.encode_image(img)  # [T, 1280]
        pooled = mean_pool(feats)  # [1280]
        tokens_path = os.path.join(out_dir, f"page_{idx+1}.pt")
        torch.save(feats.cpu(), tokens_path)
        results.append({
            "page": idx + 1,
            "vector": pooled.cpu().numpy().tolist(),
            "tokens_path": tokens_path,
        })
    return results
