# VoxCPM CLI

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white" />
  <img alt="Platform" src="https://img.shields.io/badge/Platform-Windows-lightgrey" />
  <img alt="Runtime" src="https://img.shields.io/badge/Runtime-OpenVINO-685B4C" />
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-4DAB8A" />
</p>

A slim local command-line tool for [VoxCPM2](https://github.com/OpenBMB/VoxCPM)
text-to-speech, running entirely on your machine through
[OpenVINO](https://openvino.ai/) — no server, no PyTorch required at runtime.

The project ships a JSON-first CLI, machine-friendly exit codes, and an agent
Skill, so it works equally well from a terminal, a script, or an LLM agent.

## Why

- **Local & private** — speech is generated on your machine; nothing is sent
  to a third-party API.
- **Agent-friendly** — every command emits structured JSON on stdout and
  human logs on stderr, with meaningful exit codes.
- **Inference without PyTorch** — the model is converted to OpenVINO IR, so
  synthesis runs on a lightweight OpenVINO runtime (CPU or GPU) instead of a
  full PyTorch stack.
- **Slim repo** — model weights and converted artifacts are *not* stored in
  git; `prepare` downloads and converts them explicitly when you approve.

## How It Works

```
┌────────────┐  download/convert  ┌───────────────────────┐
│ HF + GitHub ──────────────────► │ cache/  models/       │
└────────────┘                     │ (gitignored artifacts)│
                                   └───────────┬───────────┘
                                               │
                          python -m voxcpm_cli│ synth
                                               ▼
                                   ┌───────────────────────┐
                                   │ OpenVINO pipeline      │
                                   │ (no PyTorch at runtime)│
                                   └───────────┬───────────┘
                                               ▼
                                   output/*.wav  +  JSON result
```

`prepare` downloads the upstream VoxCPM source and VoxCPM2 weights, then
converts them into OpenVINO IR. `synth` loads the IR artifacts with OpenVINO
and writes a 48 kHz WAV to `output/`.

## Requirements

| Item | Requirement |
|---|---|
| OS | Windows |
| Python | 3.10, 3.11, or 3.12 |
| Network | First `prepare` run only (model download + source download) |
| Disk | Enough for upstream source, original weights, and OpenVINO IR |

Runtime dependencies (declared in `pyproject.toml`): `huggingface-hub`,
`librosa`, `numpy`, `openvino`, `soundfile`, `tokenizers`.

Conversion-only extras, needed when running `prepare` from a slim checkout:
`nncf`, `safetensors`, `torch>=2.5.0`, `transformers>=4.36.2`.

## Installation

From the project root:

```bash
python -m pip install -e .
```

To also be able to download and convert the model locally:

```bash
python -m pip install -e ".[convert]"
```

## Quick Start

```bash
# 1. Check whether models and OpenVINO are ready (never downloads anything)
python -m voxcpm_cli status --json

# 2. If not ready, download & convert (network + disk use — approve first)
python -m voxcpm_cli prepare --json

# 3. Synthesize
python -m voxcpm_cli synth --text "你好" --json
```

The CLI is also installed as the `voxcpm` console script after `pip install`.

## CLI Reference

All commands accept `--json` for JSON output on stdout. Progress and logs go
to stderr, so stdout stays machine-parseable.

### `status`

Report whether the expected artifacts exist and which OpenVINO device would
be used:

```bash
python -m voxcpm_cli status --json
```

Expected directories:

```text
cache/source/VoxCPM/
models/original/VoxCPM2/
models/openvino/VoxCPM2/
```

### `prepare`

Download the upstream source, the VoxCPM2 weights (Hugging Face), and
convert them into OpenVINO IR under `models/openvino/VoxCPM2/`:

```bash
python -m voxcpm_cli prepare --json
```

Useful options:

```bash
python -m voxcpm_cli prepare --json --device AUTO
python -m voxcpm_cli prepare --json --force-convert
python -m voxcpm_cli prepare --json \
  --model-dir models/original/VoxCPM2 \
  --ov-model-dir models/openvino/VoxCPM2
```

### `synth`

Generate a WAV file. Provide exactly one of `--text` or `--text-file`:

```bash
# Short text
python -m voxcpm_cli synth --text "你好" --json

# Long text from a file
python -m voxcpm_cli synth --text-file input.txt --json

# Explicit output path (must stay inside output/)
python -m voxcpm_cli synth --text-file input.txt --output output/demo.wav --json

# Voice instruction
python -m voxcpm_cli synth --text-file input.txt \
  --voice-instruction "温柔、自然、语速适中" --json

# Generation parameters
python -m voxcpm_cli synth --text-file input.txt \
  --cfg-value 2.0 --inference-timesteps 10 --max-len 2000 \
  --device AUTO --json
```

On success, stdout is JSON:

```json
{
  "ok": true,
  "path": "C:\\project\\voxcpm-cli\\output\\demo.wav",
  "sample_rate": 48000,
  "format": "wav",
  "model": "VoxCPM2 OpenVINO",
  "device": "GPU",
  "duration_ms": 12345
}
```

### `--version`

```bash
python -m voxcpm_cli --version
```

## Error Handling

Failures are JSON too, and each maps to a process exit code:

```json
{
  "ok": false,
  "error": { "code": "MODEL_NOT_READY", "message": "Run prepare before synth." }
}
```

| Exit code | Meaning |
|---:|---|
| `0` | success |
| `1` | validation error |
| `2` | model prepare/load error |
| `3` | synthesis error |
| `4` | file write error |

Common error codes:

- `NO_TEXT_INPUT`
- `INVALID_ARGUMENT`
- `INVALID_OUTPUT_PATH`
- `MODEL_NOT_READY`
- `VOXCPM_SOURCE_DOWNLOAD_FAILED`
- `MODEL_DOWNLOAD_FAILED`
- `MODEL_CONVERSION_FAILED`
- `MODEL_LOAD_FAILED`
- `OPENVINO_DEVICE_UNAVAILABLE`
- `TTS_GENERATION_FAILED`
- `AUDIO_WRITE_FAILED`

## Using an Agent

The repo includes an agent Skill in `skills/voxcpm-tts/SKILL.md`. The
workflow it describes:

1. Run `python -m voxcpm_cli status --json`.
2. If not ready, ask the user before running `python -m voxcpm_cli prepare --json`.
3. Write long text to a temporary `.txt` file.
4. Run `python -m voxcpm_cli synth --text-file <file> --json`.
5. Return the WAV path, sample rate, and format.

Rules: use the local CLI (never a server); do not expose sensitive text in
final replies; do not download or convert models without explicit approval.

## Project Structure

```text
voxcpm-cli/
├── pyproject.toml            # Package + dependencies (and the convert extra)
├── LICENSE                   # Apache-2.0
├── README.md
├── voxcpm_cli/
│   ├── __main__.py          # python -m voxcpm_cli entry
│   ├── cli.py               # argparse + JSON output, exit codes
│   ├── engine.py            # status / prepare / synthesize orchestration
│   ├── paths.py             # Artifact & output-path resolution
│   └── voxcpm2_tts_helper.py# Conversion + OpenVINO inference pipeline
├── skills/voxcpm-tts/       # Agent Skill instructions
├── cache/                   # Downloaded upstream source (gitignored)
├── models/original/         # Downloaded weights (gitignored)
├── models/openvino/         # Converted OpenVINO IR (gitignored)
└── output/                  # Generated WAVs (gitignored)
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
