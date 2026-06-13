"""Command line interface for VoxCPM2 TTS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__, engine


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        _print_json(_error("INVALID_ARGUMENT", message))
        raise SystemExit(1)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _read_text(args: argparse.Namespace) -> str:
    if bool(args.text) == bool(args.text_file):
        raise engine.VoxCPMError("NO_TEXT_INPUT", "Provide exactly one of --text or --text-file.", 1)
    if args.text_file:
        try:
            return Path(args.text_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise engine.VoxCPMError("INVALID_ARGUMENT", str(exc), 1) from exc
    return args.text


def _add_common_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=["CPU", "GPU", "AUTO"], default="AUTO")
    parser.add_argument("--model-dir")
    parser.add_argument("--ov-model-dir")


def build_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(prog="voxcpm")
    parser.add_argument("--version", action="store_true", help="Print package version as JSON.")

    subparsers = parser.add_subparsers(dest="command", parser_class=JsonArgumentParser)

    status_parser = subparsers.add_parser("status", help="Check local model and OpenVINO status.")
    status_parser.add_argument("--json", action="store_true", help="Output JSON.")
    _add_common_model_options(status_parser)

    prepare_parser = subparsers.add_parser("prepare", help="Download and convert VoxCPM2 model files.")
    prepare_parser.add_argument("--json", action="store_true", help="Output JSON.")
    prepare_parser.add_argument("--force-convert", action="store_true")
    _add_common_model_options(prepare_parser)

    synth_parser = subparsers.add_parser("synth", help="Synthesize text into a WAV file.")
    synth_parser.add_argument("--json", action="store_true", help="Output JSON.")
    synth_parser.add_argument("--text")
    synth_parser.add_argument("--text-file")
    synth_parser.add_argument("--output")
    synth_parser.add_argument("--voice-instruction")
    synth_parser.add_argument("--cfg-value", type=float, default=2.0)
    synth_parser.add_argument("--inference-timesteps", type=int, default=10)
    synth_parser.add_argument("--max-len", type=int, default=2000)
    _add_common_model_options(synth_parser)

    return parser


def main() -> int:
    parser = build_parser()

    try:
        args = parser.parse_args()

        if args.version:
            _print_json({"ok": True, "name": "voxcpm-cli", "version": __version__})
            return 0

        if args.command == "status":
            _print_json(engine.status(model_dir=args.model_dir, ov_model_dir=args.ov_model_dir, device=args.device))
            return 0

        if args.command == "prepare":
            _print_json(
                engine.prepare_model(
                    model_dir=args.model_dir,
                    ov_model_dir=args.ov_model_dir,
                    force_convert=args.force_convert,
                    device=args.device,
                )
            )
            return 0

        if args.command == "synth":
            text = _read_text(args)
            if not text.strip():
                raise engine.VoxCPMError("NO_TEXT_INPUT", "Text input is empty.", 1)
            _print_json(
                engine.synthesize(
                    text=text,
                    output=args.output,
                    voice_instruction=args.voice_instruction,
                    cfg_value=args.cfg_value,
                    inference_timesteps=args.inference_timesteps,
                    max_len=args.max_len,
                    device=args.device,
                    model_dir=args.model_dir,
                    ov_model_dir=args.ov_model_dir,
                )
            )
            return 0

        parser.print_help(file=sys.stderr)
        _print_json(_error("INVALID_ARGUMENT", "Provide a command."))
        return 1

    except engine.VoxCPMError as exc:
        _print_json(_error(exc.code, exc.message))
        return exc.exit_code
    except Exception as exc:
        _print_json(_error("INVALID_ARGUMENT", str(exc)))
        return 1
