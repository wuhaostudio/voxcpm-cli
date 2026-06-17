# VoxCPM CLI

Slim local CLI for VoxCPM2 text-to-speech with OpenVINO.

The first stable interface is a JSON command line tool plus an agent Skill:

```bash
python -m voxcpm_cli synth --text-file input.txt --json
```

Model weights, upstream VoxCPM source scripts, and converted OpenVINO artifacts are
not stored in this slim tree. `prepare` downloads and converts them explicitly.

## Current Status

Implemented and locally verified:

- JSON CLI entrypoint.
- `status`, `prepare`, and `synth` command contracts.
- OpenVINO device discovery.
- Structured JSON errors and meaningful process exit codes.
- Output path restriction to the project `output/` directory.
- Agent Skill instructions in `skills/voxcpm-tts/SKILL.md`.

Not yet end-to-end verified:

- Real Hugging Face model download.
- Real OpenVINO conversion.
- Real WAV generation from VoxCPM2.

## Requirements

- Windows.
- Python 3.10, 3.11, or 3.12.
- Enough disk space for the upstream VoxCPM source, original VoxCPM2 weights, and converted OpenVINO files.
- Network access for the first `prepare` run.

Runtime dependencies are declared in `pyproject.toml`:

- `huggingface-hub`
- `librosa`
- `numpy`
- `openvino`
- `soundfile`
- `tokenizers`

Conversion dependencies are optional and needed when running `prepare` from a slim checkout:

- `nncf`
- `safetensors`
- `torch>=2.5.0`
- `transformers>=4.36.2`

## Install

From the project root:

```bash
python -m pip install -e .
```

If you need to download and convert the model locally, install the conversion extras:

```bash
python -m pip install -e ".[convert]"
```

## Check The CLI

```bash
python -m voxcpm_cli --version
python -m voxcpm_cli status --json
```

`status` never downloads or converts model files. It reports whether the expected model directories are ready:

```text
cache/source/VoxCPM/
models/original/VoxCPM2/
models/openvino/VoxCPM2/
```

## Prepare The Model

Run this only after approving the network and disk use:

```bash
python -m voxcpm_cli prepare --json
```

Useful options:

```bash
python -m voxcpm_cli prepare --json --device AUTO
python -m voxcpm_cli prepare --json --force-convert
python -m voxcpm_cli prepare --json --model-dir models/original/VoxCPM2 --ov-model-dir models/openvino/VoxCPM2
```

`prepare` downloads the upstream VoxCPM source from `https://github.com/OpenBMB/VoxCPM`
into `cache/source/VoxCPM/`, downloads VoxCPM2 files from Hugging Face into
`models/original/VoxCPM2/`, and converts them into OpenVINO IR under
`models/openvino/VoxCPM2/`.

## Synthesize Speech

Short text:

```bash
python -m voxcpm_cli synth --text "你好" --json
```

Long text:

```bash
python -m voxcpm_cli synth --text-file input.txt --json
```

Specific output path inside `output/`:

```bash
python -m voxcpm_cli synth --text-file input.txt --output output/demo.wav --json
```

Voice instruction:

```bash
python -m voxcpm_cli synth --text-file input.txt --voice-instruction "温柔、自然、语速适中" --json
```

Generation options:

```bash
python -m voxcpm_cli synth --text-file input.txt --cfg-value 2.0 --inference-timesteps 10 --max-len 2000 --device AUTO --json
```

Successful output is JSON on stdout:

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

Logs and progress are written to stderr. Machine-readable results are written to stdout.

## Error Codes

Failure output is also JSON:

```json
{
  "ok": false,
  "error": {
    "code": "MODEL_NOT_READY",
    "message": "Run prepare before synth."
  }
}
```

Exit codes:

```text
0 success
1 validation error
2 model prepare/load error
3 synthesis error
4 file write error
```

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

## Agent Usage

Agents should use the local CLI, not a server.

1. Run `python -m voxcpm_cli status --json`.
2. If not ready, ask before running `python -m voxcpm_cli prepare --json`.
3. Write long text to a temporary `.txt` file.
4. Run `python -m voxcpm_cli synth --text-file <file> --json`.
5. Return the WAV path, sample rate, and format.

Do not expose full sensitive text in final replies. Do not download or convert models without explicit approval.
