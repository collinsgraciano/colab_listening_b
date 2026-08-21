"""
Qwen3-TTS-12Hz-0.6B-Base 测试脚本
- 使用本地模型路径
- 不依赖 flash-attn（Windows 不可用）
- 不需要参考音频（测试纯文本生成）
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL_PATH = r"H:\models\Qwen3-TTS-12Hz-0.6B-Base"

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# Reference audio for voice cloning
ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
ref_text = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

print("Generating speech (English)...")
wavs, sr = model.generate_voice_clone(
    text="Hello, this is a test of Qwen3 TTS. The quick brown fox jumps over the lazy dog.",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)

output_path = "test_qwen3_tts_output.wav"
sf.write(output_path, wavs[0], sr)
print(f"Saved to {output_path}")
print(f"Sample rate: {sr}, Duration: {len(wavs[0])/sr:.2f}s")

# Check VRAM usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"VRAM allocated: {allocated:.2f} GB")
    print(f"VRAM reserved:  {reserved:.2f} GB")

print("Done!")
