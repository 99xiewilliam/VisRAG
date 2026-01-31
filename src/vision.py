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
    def __init__(self, model_path: str = "/data/xwh/models/DeepSeek-OCR-2", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"初始化 DeepseekEncoder: device={self.device}, dtype={self.dtype}")
        _ensure_dynamic_cache_compat()
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        self.sam = model.model.sam_model.to(self.device, dtype=self.dtype)
        self.qwen2 = model.model.qwen2_model.to(self.device, dtype=self.dtype)
        self.projector = model.model.projector.to(self.device, dtype=self.dtype)
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
        return features


class DeepseekTokenDecoder:
    """使用预计算的 vision tokens（[T, 1280]）进行生成，跳过视觉编码。

    依赖 /data/xwh/models/DeepSeek-OCR-2/modeling_deepseekocr2.py 已支持 image_embeds 注入。
    """

    def __init__(self, model_path: str = "/data/xwh/models/DeepSeek-OCR-2", device: str | None = None):
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
        # feats: [T, 1280]
        if feats.ndim != 2:
            raise ValueError(f"feats must be 2D [T, D], got {tuple(feats.shape)}")
        feats = feats.to(self.device)

        input_ids = self._build_input_ids(prompt, num_image_tokens=feats.shape[0])
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
