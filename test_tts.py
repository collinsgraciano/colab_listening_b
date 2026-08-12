#!/usr/bin/env python3
"""Quick test: verify TTS engines work (Kokoro EN + edge-tts ZH).

Usage:
    python test_tts.py                     # Test both EN and ZH
    python test_tts.py --en-only           # Test English only
    python test_tts.py --zh-only           # Test Chinese only
    python test_tts.py --text "Hello"      # Custom text for English
    python test_tts.py --zh-text "你好"    # Custom text for Chinese
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("tts_test_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def test_kokoro_en(text: str = "Hello, welcome to English listening practice.") -> bool:
    """Test Kokoro English TTS. Returns True if successful."""
    print("\n" + "=" * 60)
    print("Testing Kokoro English TTS (af_sky)...")
    sys.stdout.flush()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    try:
        from kokoro import KPipeline
        print("  [1/3] KPipeline imported OK")
        sys.stdout.flush()

        pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
        print("  [2/3] Model loaded OK")
        sys.stdout.flush()

        out_path = str(OUTPUT_DIR / "test_en.mp3")
        import soundfile as sf
        import numpy as np

        all_audio = []
        for gs, ps, audio in pipeline(text, voice='af_sky', speed=1.0):
            if audio is not None and len(audio) > 0:
                all_audio.append(audio)

        if not all_audio:
            print("  ❌ FAILED: No audio generated")
            return False

        final = all_audio[0] if len(all_audio) == 1 else np.concatenate(all_audio)
        wav_tmp = out_path.replace('.mp3', '_tmp.wav')
        sf.write(wav_tmp, final, 24000)
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_tmp, "-c:a", "libmp3lame", "-b:a", "128k",
             "-ar", "24000", "-ac", "1", out_path],
            check=True, capture_output=True)
        os.remove(wav_tmp)

        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", out_path], text=True).strip())
        print(f"  [3/3] ✅ Generated: {out_path} ({dur:.1f}s)")
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_edge_zh(text: str = "你好，欢迎来到英语听力练习频道。") -> bool:
    """Test edge-tts Chinese TTS. Returns True if successful."""
    print("\n" + "=" * 60)
    print("Testing edge-tts Chinese TTS (zh-CN-XiaoxiaoNeural)...")
    sys.stdout.flush()

    try:
        import asyncio
        import edge_tts
        from edge_tts.exceptions import NoAudioReceived

        print("  [1/3] edge-tts imported OK")
        sys.stdout.flush()

        out_path = str(OUTPUT_DIR / "test_zh.mp3")

        async def _gen():
            comm = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural", rate="-10%")
            await asyncio.wait_for(comm.save(out_path), timeout=30)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(_gen())
        print("  [2/3] Audio synthesized OK")
        sys.stdout.flush()

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            print("  ❌ FAILED: Empty audio")
            return False

        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", out_path], text=True).strip())
        print(f"  [3/3] ✅ Generated: {out_path} ({dur:.1f}s)")
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test TTS engines")
    parser.add_argument("--en-only", action="store_true", help="Test English only")
    parser.add_argument("--zh-only", action="store_true", help="Test Chinese only")
    parser.add_argument("--text", default="Hello, welcome to English listening practice.",
                        help="Custom English text")
    parser.add_argument("--zh-text", default="你好，欢迎来到英语听力练习频道。",
                        help="Custom Chinese text")
    args = parser.parse_args()

    test_en = not args.zh_only
    test_zh = not args.en_only

    results = []
    if test_en:
        results.append(("Kokoro EN", test_kokoro_en(args.text)))
    if test_zh:
        results.append(("edge-tts ZH", test_edge_zh(args.zh_text)))

    print("\n" + "=" * 60)
    print("RESULTS:")
    all_ok = True
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        if not ok:
            all_ok = False
        print(f"  {status}  {name}")

    print(f"\nOutput files: {OUTPUT_DIR.resolve()}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()