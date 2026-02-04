import os
import sys
from typing import Optional
from PIL import Image

from ..config import get_app_config
from ..utils import get_logger

logger = get_logger(__name__)


class OCRService:
    def __init__(self):
        self.cfg = get_app_config().ocr
        # vLLM path
        self._llm = None
        self._processor = None
        self._sampling = None

        # HF infer() path (kept for reference)
        # self._tokenizer = None
        # self._model = None

    def _lazy_init(self):
        if self._llm is not None:
            return
        vllm_dir = self.cfg.vllm_code_dir or os.environ.get("DEEPSEEK_OCR2_VLLM_DIR")
        if vllm_dir and os.path.isdir(vllm_dir) and vllm_dir not in sys.path:
            sys.path.insert(0, vllm_dir)
        from deepseek_ocr2 import DeepseekOCR2ForCausalLM
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm import LLM, SamplingParams
        from process.image_process import DeepseekOCR2Processor

        ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)
        self._llm = LLM(
            model=self.cfg.model_path,
            hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
            block_size=256,
            enforce_eager=False,
            trust_remote_code=True,
            max_model_len=8192,
            swap_space=0,
            max_num_seqs=1,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            disable_mm_preprocessor_cache=True,
        )
        self._processor = DeepseekOCR2Processor()
        self._sampling = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

    def _build_prompt(self, question: Optional[str], context_text: Optional[str]) -> str:
        if self.cfg.prompt_template:
            # Use custom prompt template from config
            prompt = self.cfg.prompt_template
            if question:
                prompt = prompt.replace("{question}", question)
            if context_text:
                prompt = prompt.replace("{context}", context_text)
            return prompt
        
        # Default QA template (HF infer() prefers <|grounding|>)
        base = (
            "<image>\n"
            "<|grounding|>\n"
            "You are a helpful document QA assistant. "
            "Use the image and the provided context to answer the question. "
            "If the context is insufficient, answer based on the image.\n"
        )
        if question:
            base += f"Question: {question}\n"
        if context_text:
            base += f"Context:\n{context_text}\n"
        base += "Answer:"
        return base

    def run(self, image_path: str, question: Optional[str] = None, context_text: Optional[str] = None) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        self._lazy_init()
        prompt = self._build_prompt(question, context_text)
        # vLLM path (active)
        image = Image.open(image_path).convert("RGB")
        # Some installs don't accept `conversation` arg. Fallback: set PROMPT dynamically.
        try:
            import inspect

            sig = inspect.signature(self._processor.tokenize_with_images)
            if "conversation" in sig.parameters:
                image_tokens = self._processor.tokenize_with_images(
                    images=[image],
                    conversation=prompt,
                    bos=True,
                    eos=True,
                    cropping=True,
                )
            else:
                # Fallback to global PROMPT override
                from process import image_process as _img_proc

                _img_proc.PROMPT = prompt
                image_tokens = self._processor.tokenize_with_images(
                    images=[image],
                    bos=True,
                    eos=True,
                    cropping=True,
                )
        except Exception:
            # Last-resort: call without conversation
            image_tokens = self._processor.tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=True,
            )

        cache_item = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_tokens
            },
        }
        logger.info("Running OCR2 vLLM generation")
        outputs = self._llm.generate([cache_item], sampling_params=self._sampling)
        if not outputs:
            return ""
        text = outputs[0].outputs[0].text
        return (text or "").replace("<｜end▁of▁sentence｜>", "").strip()

        # HF infer() path (disabled)
        # out_dir = self.cfg.output_dir
        # if not out_dir:
        #     from ..config import get_app_config as _get_cfg
        #     out_dir = os.path.join(_get_cfg().indexing.assets_dir, "ocr_infer")
        # ensure_dir(out_dir)
        # logger.info("Running OCR2 HF infer()")
        # res = self._model.infer(
        #     self._tokenizer,
        #     prompt=prompt,
        #     image_file=image_path,
        #     output_path=out_dir,
        #     base_size=int(getattr(self.cfg, "base_size", 1024)),
        #     image_size=int(getattr(self.cfg, "image_size", 768)),
        #     crop_mode=bool(getattr(self.cfg, "crop_mode", True)),
        #     save_results=bool(getattr(self.cfg, "save_results", False)),
        #     eval_mode=True,
        # )
        # return (res or "").strip()
