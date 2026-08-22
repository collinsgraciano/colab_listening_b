"""
Qwen3-TTS-12Hz-0.6B-CustomVoice 测试脚本
- 使用预设声音 (predefined speakers)
- 查询支持的 speaker 和 language 列表
- 生成中英文测试音频
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL_PATH = r"H:\models\Qwen3-TTS-12Hz-0.6B-CustomVoice"

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# 查询支持的 speaker 和 language
print("\n=== Supported Speakers ===")
speakers = model.get_supported_speakers()
for s in speakers:
    print(f"  {s}")

print(f"\n=== Supported Languages ===")
languages = model.get_supported_languages()
for lang in languages:
    print(f"  {lang}")

# 测试 1: 中文 - Vivian (明亮年轻女声)
print("\n--- Test 1: Vivian (Chinese) ---")
wavs, sr = model.generate_custom_voice(
    text="你好，我是 Qwen3 语音合成模型，很高兴认识你！",
    language="Chinese",
    speaker="Vivian",
)
sf.write("test_cv_vivian_zh.wav", wavs[0], sr)
print(f"  Saved test_cv_vivian_zh.wav ({len(wavs[0])/sr:.2f}s)")

# 测试 2: 英文 - Serena (温柔年轻女声)
print("\n--- Test 2: Serena (English) ---")
wavs, sr = model.generate_custom_voice(
    text="Hello, this is a test of Qwen3 TTS with Serena's voice.",
    language="English",
    speaker="Serena",
)
sf.write("test_cv_serena_en.wav", wavs[0], sr)
print(f"  Saved test_cv_serena_en.wav ({len(wavs[0])/sr:.2f}s)")

# 测试 3: 英文 - Eric (成都男声，略带沙哑)
print("\n--- Test 3: Eric (English) ---")
wavs, sr = model.generate_custom_voice(
    text="The quick brown fox jumps over the lazy dog.",
    language="English",
    speaker="Eric",
)
sf.write("test_cv_eric_en.wav", wavs[0], sr)
print(f"  Saved test_cv_eric_en.wav ({len(wavs[0])/sr:.2f}s)")

# 测试 4: 带 instruct 风格指令
print("\n--- Test 4: Vivian with instruct (whisper) ---")
try:
    wavs, sr = model.generate_custom_voice(
        text="今天天气真好，我们出去走走吧。",
        language="Chinese",
        speaker="Vivian",
        instruct="Speak in a whisper tone.",
    )
    sf.write("test_cv_vivian_whisper.wav", wavs[0], sr)
    print(f"  Saved test_cv_vivian_whisper.wav ({len(wavs[0])/sr:.2f}s)")
except Exception as e:
    print(f"  instruct not supported on 0.6B: {e}")

# VRAM 使用情况
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"\nVRAM allocated: {allocated:.2f} GB")
    print(f"VRAM reserved:  {reserved:.2f} GB")

print("\nDone!")
