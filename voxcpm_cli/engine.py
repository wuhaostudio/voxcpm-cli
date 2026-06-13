"""VoxCPM2 OpenVINO engine boundary."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import soundfile as sf

from . import paths


ORIGINAL_REQUIRED = ("config.json", "model.safetensors", "tokenizer.json")
OPENVINO_REQUIRED = (
    "openvino_embed_tokens.xml",
    "openvino_embed_tokens.bin",
    "openvino_feat_encoder.xml",
    "openvino_feat_encoder.bin",
    "openvino_base_lm.xml",
    "openvino_base_lm.bin",
    "openvino_residual_lm.xml",
    "openvino_residual_lm.bin",
    "openvino_decode_heads.xml",
    "openvino_decode_heads.bin",
    "openvino_dit_estimator.xml",
    "openvino_dit_estimator.bin",
    "openvino_audio_vae_encoder.xml",
    "openvino_audio_vae_encoder.bin",
    "openvino_audio_vae_decoder.xml",
    "openvino_audio_vae_decoder.bin",
    "tokenizer.json",
)


class VoxCPMError(RuntimeError):
    def __init__(self, code: str, message: str, exit_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code


def _dir_has_files(directory: Path, filenames: tuple[str, ...]) -> bool:
    return all((directory / name).is_file() for name in filenames)


def _openvino_devices() -> tuple[list[str], str | None]:
    try:
        import openvino as ov

        core = ov.Core()
        devices = list(core.available_devices)
        return devices, None
    except Exception as exc:  # pragma: no cover - depends on local runtime
        return [], str(exc)


def select_device(requested: str | None, available: list[str]) -> str:
    requested = (requested or "AUTO").upper()
    if requested == "AUTO":
        if "GPU" in available:
            return "GPU"
        if "CPU" in available:
            return "CPU"
        return "AUTO"
    if requested not in {"CPU", "GPU"}:
        raise VoxCPMError("INVALID_ARGUMENT", "Device must be CPU, GPU, or AUTO.", 1)
    if available and requested not in available:
        raise VoxCPMError("OPENVINO_DEVICE_UNAVAILABLE", f"OpenVINO device is unavailable: {requested}", 2)
    return requested


def status(
    *,
    model_dir: str | None = None,
    ov_model_dir: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    root = paths.project_root()
    model_path = Path(model_dir).expanduser().resolve() if model_dir else paths.original_model_dir(root).resolve()
    ov_path = Path(ov_model_dir).expanduser().resolve() if ov_model_dir else paths.openvino_model_dir(root).resolve()
    devices, openvino_error = _openvino_devices()

    original_ready = _dir_has_files(model_path, ORIGINAL_REQUIRED)
    ov_ready = _dir_has_files(ov_path, OPENVINO_REQUIRED)
    selected_device = select_device(device, devices)

    result: dict[str, Any] = {
        "ok": True,
        "ready": original_ready and ov_ready and openvino_error is None,
        "model_ready": original_ready,
        "ov_model_ready": ov_ready,
        "model_dir": str(model_path),
        "ov_model_dir": str(ov_path),
        "available_devices": devices,
        "selected_device": selected_device,
    }
    if openvino_error:
        result["openvino_error"] = openvino_error
    return result


def prepare_model(
    *,
    model_dir: str | None = None,
    ov_model_dir: str | None = None,
    force_convert: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    state = status(model_dir=model_dir, ov_model_dir=ov_model_dir, device=device)
    model_path = Path(state["model_dir"])
    ov_path = Path(state["ov_model_dir"])

    if not state["model_ready"]:
        print("Downloading VoxCPM2 model files...", file=sys.stderr)
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="openbmb/VoxCPM2",
                local_dir=str(model_path),
                allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
            )
        except Exception as exc:
            raise VoxCPMError("MODEL_DOWNLOAD_FAILED", str(exc), 2) from exc

    state = status(model_dir=str(model_path), ov_model_dir=str(ov_path), device=device)
    if not state["model_ready"]:
        raise VoxCPMError("MODEL_DOWNLOAD_FAILED", "Downloaded model files are incomplete.", 2)

    if force_convert or not state["ov_model_ready"]:
        print("Converting VoxCPM2 model to OpenVINO IR...", file=sys.stderr)
        try:
            from .voxcpm2_tts_helper import convert_voxcpm2_model

            convert_voxcpm2_model(str(model_path), str(ov_path))
        except Exception as exc:
            raise VoxCPMError("MODEL_CONVERSION_FAILED", str(exc), 2) from exc

    state = status(model_dir=str(model_path), ov_model_dir=str(ov_path), device=device)
    if not state["ov_model_ready"]:
        raise VoxCPMError("MODEL_CONVERSION_FAILED", "OpenVINO model files are incomplete.", 2)

    return {
        "ok": True,
        "model_ready": state["model_ready"],
        "ov_model_ready": state["ov_model_ready"],
        "message": "VoxCPM2 OpenVINO model is ready." if state["ready"] else "VoxCPM2 model is not ready.",
    }


def synthesize(
    *,
    text: str,
    output: str | None = None,
    voice_instruction: str | None = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    max_len: int = 2000,
    device: str | None = None,
    model_dir: str | None = None,
    ov_model_dir: str | None = None,
) -> dict[str, Any]:
    try:
        output_path = paths.resolve_output_path(output)
    except ValueError as exc:
        raise VoxCPMError("INVALID_OUTPUT_PATH", str(exc), 1) from exc

    state = status(model_dir=model_dir, ov_model_dir=ov_model_dir, device=device)
    if not state["ready"]:
        raise VoxCPMError("MODEL_NOT_READY", "Run prepare before synth.", 2)

    final_text = f"({voice_instruction}){text}" if voice_instruction else text
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    try:
        from .voxcpm2_tts_helper import OVVoxCPM2Model

        pipe = OVVoxCPM2Model(model_dir=state["ov_model_dir"], device=state["selected_device"])
    except Exception as exc:
        raise VoxCPMError("MODEL_LOAD_FAILED", str(exc), 2) from exc

    try:
        wav, sample_rate = pipe.generate(
            text=final_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            max_len=max_len,
        )
    except Exception as exc:
        raise VoxCPMError("TTS_GENERATION_FAILED", str(exc), 3) from exc

    try:
        sf.write(str(output_path), wav, sample_rate)
    except Exception as exc:
        raise VoxCPMError("AUDIO_WRITE_FAILED", str(exc), 4) from exc

    duration_ms = int((time.perf_counter() - started) * 1000)
    return {
        "ok": True,
        "path": str(output_path),
        "sample_rate": sample_rate,
        "format": "wav",
        "model": "VoxCPM2 OpenVINO",
        "device": state["selected_device"],
        "duration_ms": duration_ms,
    }
