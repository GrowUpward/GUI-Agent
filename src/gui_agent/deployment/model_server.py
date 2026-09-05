"""FastAPI inference service for a Qwen3.5 GUI agent and an optional PEFT adapter."""

from __future__ import annotations

import argparse
import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from gui_agent.training.grpo import SYSTEM_PROMPT, parse_action_text, truncate_at_balanced_dict


class PredictRequest(BaseModel):
    instruction: str
    image_path: str | None = None
    image_base64: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


def _image_from_request(request: PredictRequest) -> tuple[Path, bool]:
    if request.image_path:
        path = Path(request.image_path).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"image_path does not exist: {path}")
        return path, False

    if request.image_base64:
        try:
            payload = request.image_base64.split(",", 1)[-1]
            raw = base64.b64decode(payload, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise HTTPException(status_code=400, detail="image_base64 is invalid") from exc
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        handle.write(raw)
        handle.close()
        return Path(handle.name), True

    raise HTTPException(status_code=400, detail="Provide image_path or image_base64")


def _history_text(history: list[dict[str, Any]]) -> str:
    if not history:
        return "None"
    lines = []
    for index, item in enumerate(history[-6:], start=1):
        lines.append(f"{index}. {item}")
    return "\n".join(lines)


def _build_messages(instruction: str, image_path: Path, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user_prompt = (
        f"Task: {instruction}\n"
        f"Previous actions:\n{_history_text(history)}\n\n"
        "Inspect the screenshot and choose the next action. Coordinates must use the normalized "
        "0-1000 system. Return exactly one Python dictionary and no reasoning."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def _normalized_to_pixels(action: dict[str, Any] | None, width: int, height: int) -> dict[str, Any] | None:
    if not action or "coordinate" not in action:
        return None
    coordinate = action.get("coordinate")
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
        return None
    try:
        nx = max(0.0, min(1000.0, float(coordinate[0])))
        ny = max(0.0, min(1000.0, float(coordinate[1])))
    except (TypeError, ValueError):
        return None
    return {
        "pixel": [round(nx * width / 1000), round(ny * height / 1000)],
        "normalized": [nx, ny],
    }


class ModelRuntime:
    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        processor_path: str | None = None,
        device: str = "auto",
        attn_implementation: str = "sdpa",
        load_in_4bit: bool = False,
        legacy_yx_output: bool = False,
        max_image_side: int = 672,
        max_new_tokens: int = 64,
    ) -> None:
        if load_in_4bit and not torch.cuda.is_available():
            raise ValueError("--load-in-4bit requires CUDA")

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        device_map: str | dict[str, str] = "auto" if device == "auto" else {"": device}
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
        )
        if adapter_path:
            adapter = Path(adapter_path).expanduser().resolve()
            if not (adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(f"Invalid PEFT adapter directory: {adapter}")
            self.model = PeftModel.from_pretrained(self.model, str(adapter))
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            processor_path or model_path,
            trust_remote_code=True,
        )
        self.max_image_side = max_image_side
        self.max_new_tokens = max_new_tokens
        self.legacy_yx_output = legacy_yx_output

    @torch.inference_mode()
    def predict(self, request: PredictRequest, image_path: Path) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        if max(image.size) > self.max_image_side:
            image.thumbnail((self.max_image_side, self.max_image_side), Image.Resampling.LANCZOS)

        messages = _build_messages(request.instruction, image_path, request.history)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generated = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[1]
        response = self.processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        action_text = truncate_at_balanced_dict(response).strip()
        action, format_valid = parse_action_text(action_text)
        if self.legacy_yx_output and action and isinstance(action.get("coordinate"), (list, tuple)):
            raw_coordinate = action["coordinate"]
            if len(raw_coordinate) == 2:
                action = dict(action)
                action["coordinate"] = [raw_coordinate[1], raw_coordinate[0]]
        coordinate = _normalized_to_pixels(action, original_width, original_height)

        result: dict[str, Any] = {
            "format_valid": format_valid,
            "parsed_action": action,
            "response": action_text,
            "image_size": [original_width, original_height],
            "legacy_yx_output_corrected": self.legacy_yx_output,
        }
        if action:
            result.update(action)
        if coordinate:
            result["coordinate"] = coordinate["pixel"]
            result["normalized_coordinate"] = coordinate["normalized"]
        return result


def create_app(runtime: ModelRuntime) -> FastAPI:
    app = FastAPI(title="GUI-Agent inference service", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        image_path, temporary = _image_from_request(request)
        started = time.perf_counter()
        try:
            result = runtime.predict(request, image_path)
            result["latency_seconds"] = round(time.perf_counter() - started, 4)
            return result
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if temporary:
                image_path.unlink(missing_ok=True)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("BASE_MODEL"), required=not os.environ.get("BASE_MODEL"))
    parser.add_argument("--adapter-path", default=os.environ.get("SFT_ADAPTER"))
    parser.add_argument("--processor-path", default=os.environ.get("PROCESSOR_PATH"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--legacy-yx-output",
        action="store_true",
        help="Swap legacy adapter [y, x] predictions to the corrected [x, y] convention.",
    )
    parser.add_argument("--max-image-side", type=int, default=672)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = ModelRuntime(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        processor_path=args.processor_path,
        device=args.device,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        legacy_yx_output=args.legacy_yx_output,
        max_image_side=args.max_image_side,
        max_new_tokens=args.max_new_tokens,
    )
    uvicorn.run(create_app(runtime), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
