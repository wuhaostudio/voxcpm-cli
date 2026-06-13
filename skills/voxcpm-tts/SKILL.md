---
name: voxcpm-tts
description: Use when Codex needs to convert text to speech, generate a WAV audio file, read selected text aloud, or call the local VoxCPM CLI from a repo or host such as AetherCursor.
---

# VoxCPM TTS

Use the local CLI, not a server.

1. Locate the `voxcpm-cli` root.
2. Run `python -m voxcpm_cli status --json`.
3. If the model is not ready, ask before running `python -m voxcpm_cli prepare --json`.
4. For long text, write it to a temporary `.txt` file and call `python -m voxcpm_cli synth --text-file <file> --json`.
5. Parse stdout JSON and return the WAV path, sample rate, and format.

Keep logs and final replies short. Do not expose full sensitive text. Do not download or convert models without user approval.
