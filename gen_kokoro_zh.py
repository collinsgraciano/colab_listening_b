#!/usr/bin/env python3
"""Standalone Kokoro Chinese TTS generator using zf_xiaoxiao.pt voice.

Generates Chinese (Mandarin) audio via Kokoro KPipeline with the
zf_xiaoxiao.pt voice (Mandarin Chinese female), applies loudnorm.

Usage:
    pip install pypinyin cn2an ordered_set jieba  # required for Chinese TTS

    python gen_kokoro_zh.py --text "你好，世界。" --output out.mp3
    python gen_kokoro_zh.py --voice voices/zf_xiaoxiao.pt --text "..." --output out.wav
    echo "你好" | python gen_kokoro_zh.py --output out.mp3   # read stdin
"""
import argparse
import os
import sys
import subprocess
import json
from pathlib import Path

# Fix Windows console encoding to avoid UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# HF_ENDPOINT: Windows local uses hf-mirror (HuggingFace blocked in China),
# Colab/Linux uses direct huggingface.co. Allow override via env var.
if "HF_ENDPOINT" not in os.environ:
    if sys.platform == "win32":
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    else:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"

# Ensure all Chinese pipeline dependencies are installed
_REQUIRED_MODULES = ["pypinyin", "cn2an", "ordered_set", "jieba"]
_missing = []
for _m in _REQUIRED_MODULES:
    try:
        __import__(_m)
    except ImportError:
        _missing.append(_m)
if _missing:
    print("Missing dependencies for Chinese Kokoro TTS:")
    print(f"   Run: pip install {' '.join(_missing)}")
    sys.exit(1)

DEFAULT_VOICE = "zf_xiaoxiao.pt"


def find_voice_in_cache(filename: str) -> str | None:
    """Search HF cache for a Kokoro voice .pt file by filename."""
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    if not cache_dir.exists():
        return None
    for root, _dirs, files in os.walk(cache_dir):
        if filename in files:
            path = os.path.join(root, filename)
            if os.path.getsize(path) > 1000:
                return path
    return None


def download_voice(filename: str, out_dir: str | None = None) -> str:
    """Download a Kokoro voice .pt file from HF mirror."""
    if out_dir is None:
        cache_voices = Path(os.path.expanduser(
            "~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/voices"))
        cache_voices.mkdir(parents=True, exist_ok=True)
        out_dir = str(cache_voices)
    dest = os.path.join(out_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest
    urls = [
        f"https://hf-mirror.com/hexgrad/Kokoro-82M/resolve/main/voices/{filename}",
        f"https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{filename}",
    ]
    for url in urls:
        print(f"  [Download] {url}")
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) > 1000:
                Path(dest).write_bytes(data)
                print(f"  [OK] Downloaded voice: {dest} ({len(data)//1024}KB)")
                return dest
        except Exception as e:
            print(f"  [Warn] Download failed: {str(e)[:80]}")
    raise FileNotFoundError(
        f"Voice '{filename}' could not be downloaded. "
        f"Please manually place the .pt file at: {dest}")


def resolve_voice(voice: str) -> str:
    """Resolve a voice name/path to a valid .pt file, downloading if needed."""
    if os.path.exists(voice) and os.path.getsize(voice) > 1000:
        return voice
    filename = os.path.basename(voice)
    cached = find_voice_in_cache(filename)
    if cached:
        return cached
    return download_voice(filename)


def get_duration(audio_path: str) -> float:
    """Get audio duration via ffprobe."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "json", audio_path], text=True)
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return 0.0


def loudnorm(input_path: str):
    """Normalize volume with loudnorm."""
    norm_path = input_path.replace(".mp3", "_norm.mp3")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
         "-c:a", "aac", "-b:a", "128k", norm_path],
        capture_output=True, timeout=30)
    if result.returncode == 0 and os.path.exists(norm_path) and os.path.getsize(norm_path) > 1000:
        os.replace(norm_path, input_path)
    return input_path


def generate(text: str, voice: str, out_path: str,
             speed: float = 1.0, normalize: bool = True) -> str:
    """Generate Chinese speech with Kokoro Chinese voice.

    Args:
        text: Chinese text to speak.
        voice: path to .pt voice file, or voice name to look up in cache.
        out_path: output file (.mp3 or .wav).
        speed: speaking speed (1.0 normal, 0.8 slow, 1.2 fast).
        normalize: apply loudnorm after synthesis.

    Returns:
        Output audio path.
    """
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np

    voice_path = resolve_voice(voice)
    pipeline = KPipeline(lang_code="z")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    all_audio = []
    for result in pipeline(text, voice=voice_path, speed=speed):
        if result.audio is not None and len(result.audio) > 0:
            all_audio.append(result.audio)

    if not all_audio:
        raise RuntimeError(f"Kokoro produced no audio for: {text[:50]}")

    final_audio = all_audio[0] if len(all_audio) == 1 else np.concatenate(all_audio)

    wav_path = out_path.rsplit(".", 1)[0] + "_tmp.wav"
    sf.write(wav_path, final_audio, 24000)

    if out_path.endswith(".wav"):
        os.replace(wav_path, out_path)
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libmp3lame", "-b:a", "128k",
             "-ar", "24000", "-ac", "1", out_path],
            check=True, capture_output=True)
        os.remove(wav_path)

    if normalize:
        loudnorm(out_path)

    dur = get_duration(out_path)
    print(f"Generated: {out_path} ({dur:.1f}s, voice={voice_path})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Kokoro Chinese TTS")
    parser.add_argument("--text", default=None, help="Chinese text to speak")
    parser.add_argument("--voice", default=DEFAULT_VOICE,
                        help=f"Voice .pt path or name (default {DEFAULT_VOICE})")
    parser.add_argument("--output", default="out.mp3", help="Output file (.mp3 or .wav)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed (1.0 normal, 0.8 slow)")
    parser.add_argument("--no-normalize", action="store_true", help="Skip loudnorm")
    args = parser.parse_args()

    text = args.text
    if text is None:
        text = sys.stdin.read().strip()
    if not text:
        print("No text provided. Use --text or pipe via stdin.")
        sys.exit(1)

    generate(text, args.voice, args.output,
             speed=args.speed, normalize=not args.no_normalize)


if __name__ == "__main__":
    main()