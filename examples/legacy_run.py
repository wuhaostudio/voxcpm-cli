import os
from pathlib import Path
import openvino as ov
from huggingface_hub import snapshot_download
from voxcpm_cli.voxcpm2_tts_helper import convert_voxcpm2_model, OVVoxCPM2Model
import soundfile as sf


def voxcpm2_tts(
    text: str,
    output_path: str = "../output/demo.wav",
    model_dir: str = "../models/original/VoxCPM2",
    ov_model_dir: str = "../models/openvino/VoxCPM2",
    device: str = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    quantization_config: dict = None,
    sample_rate: int = 48000,
):
    """
    使用 VoxCPM2 生成语音并保存为音频文件。

    Parameters
    ----------
    text : str
        要合成的文本。
    output_path : str
        输出音频文件路径。
    model_dir : str
        原始模型存放目录（用于下载/检查）。
    ov_model_dir : str
        OpenVINO 转换模型输出目录。
    device : str, optional
        推理设备，默认自动选择 GPU > CPU。
    cfg_value : float
        CFG 引导强度。
    inference_timesteps : int
        扩散步数，越大稳定性越好。
    quantization_config : dict or None
        量化配置，例如 {"mode": "INT8_SYM"} 进行 INT8 权重量化。
    sample_rate : int
        输出音频采样率。
    """
    # 1. 检查并下载模型
    required_modules = ["config.json", "model.safetensors", "tokenizer.json"]
    if not all(os.path.exists(os.path.join(model_dir, f)) for f in required_modules):
        print("模型未安装或文件不完整，开始下载...")
        snapshot_download(
            repo_id="openbmb/VoxCPM2",
            local_dir=model_dir,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
        )
    else:
        print("模型已存在，跳过下载。")

    # 2. 转换为 OpenVINO 格式（若已存在则跳过）
    model_path = Path(model_dir)
    ov_path = Path(ov_model_dir)
    if not ov_path.exists() or not any(ov_path.iterdir()):
        print("开始转换为 OpenVINO 模型...")
        convert_voxcpm2_model(
            model_path=str(model_path),
            output_dir=str(ov_path),
            quantization_config=quantization_config,
        )
    else:
        print("OpenVINO 模型已存在，跳过转换。")

    # 3. 初始化推理设备
    core = ov.Core()
    if device is None:
        device = "GPU" if "GPU" in core.available_devices else "CPU"
    print(f"使用设备: {device}")

    # 4. 加载模型并生成音频
    pipe = OVVoxCPM2Model(model_dir=str(ov_path), device=device)
    audio, generated_sample_rate = pipe.generate(
        text=text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
    )

    # 5. 保存音频
    sf.write(output_path, audio, generated_sample_rate or sample_rate)
    print(f"语音已保存至: {output_path}")


if __name__ == "__main__":
    voxcpm2_tts(
        text="今天天气真不错，我们去西湖边走走吧。",
        output_path="demo.wav",
    )
