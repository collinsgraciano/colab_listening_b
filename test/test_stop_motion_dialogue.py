#!/usr/bin/env python3
"""End-to-end stop_motion dialogue segment test.

Uses real generated assets:
  - pose_0_0.png : character speaking (mouth open, gesturing)
  - pose_0_1.png : character listening (slight smile, hands resting)
  - background.png : coffee shop interior (no characters)
  - dialogue_test.wav : TTS audio

Renders a single dialogue segment with:
  - White-bg removal + pose normalization
  - Multi-pose stop-motion (pose A → optical flow transition → pose B)
  - Landing transform (scale/pan/bounce)
  - Subtitle overlay
  - Final encoded MP4
"""
import os
import sys
import math
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from PIL import Image, ImageDraw, ImageFont
from media_utils import TARGET_W, TARGET_H, FONT_EN, FONT_ZH, get_duration
from stop_motion import (
    remove_bg, normalize_pose, generate_morph_frames,
    render_frame, compute_landing, quantize,
    MOTION_FPS, DELIVERY_FPS, POSE_BOTTOM,
)

TEST_DIR = Path(__file__).parent
POSE_A = str(TEST_DIR / "pose_0_0.png")
POSE_B = str(TEST_DIR / "pose_0_1.png")
BG_PATH = str(TEST_DIR / "background.png")
AUDIO_PATH = str(TEST_DIR / "dialogue_test.wav")
OUT_DIR = TEST_DIR / "sm_output"
OUT_VIDEO = str(TEST_DIR / "stop_motion_dialogue.mp4")

EN_TEXT = "Hi there! Welcome to our coffee shop. What can I get for you today?"
ZH_TEXT = "你好！歡迎來到我們的咖啡店。今天想喝點什麼？"


def render_subtitle(en, zh, w=TARGET_W, h=TARGET_H):
    """Render subtitle overlay PNG."""
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    en_font = ImageFont.truetype(FONT_EN, 44)
    zh_font = ImageFont.truetype(FONT_ZH, 36)
    en_bbox = draw.textbbox((0, 0), en, font=en_font)
    en_w, en_h = en_bbox[2]-en_bbox[0], en_bbox[3]-en_bbox[1]
    zh_bbox = draw.textbbox((0, 0), zh, font=zh_font)
    zh_w, zh_h = zh_bbox[2]-zh_bbox[0], zh_bbox[3]-zh_bbox[1]
    box_h = en_h + zh_h + 30
    box_top = h - box_h - 40
    box = Image.new("RGBA", (w, box_h), (0, 0, 0, 140))
    overlay.alpha_composite(box, (0, box_top))
    en_x = (w - en_w) // 2
    en_y = box_top + 10
    draw.text((en_x, en_y), en, font=en_font, fill="white",
              stroke_width=3, stroke_fill="black")
    zh_x = (w - zh_w) // 2
    zh_y = en_y + en_h + 8
    draw.text((zh_x, zh_y), zh, font=zh_font,
              fill="rgb(255,220,0)", stroke_width=2, stroke_fill="black")
    return overlay


def main():
    # Verify assets
    for label, path in [("Pose A", POSE_A), ("Pose B", POSE_B),
                         ("Background", BG_PATH), ("Audio", AUDIO_PATH)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)
        print(f"  {label}: {os.path.getsize(path)/1024:.0f}KB")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Process poses: white-bg removal + normalization
    print("\n1. Processing character poses...")
    raw_a = Image.open(POSE_A)
    raw_b = Image.open(POSE_B)
    print(f"   Raw sizes: A={raw_a.size}, B={raw_b.size}")

    alpha_a = remove_bg(raw_a)
    alpha_b = remove_bg(raw_b)
    norm_a = normalize_pose(alpha_a)
    norm_b = normalize_pose(alpha_b)
    print(f"   Normalized: A={norm_a.size}, B={norm_b.size}")

    # Count transparent pixels
    for name, img in [("A", norm_a), ("B", norm_b)]:
        px = list(img.getdata())
        transp = sum(1 for p in px if p[3] == 0)
        print(f"   Pose {name}: {100*transp/len(px):.1f}% transparent")

    # 2. Generate morph frames (optical flow transition)
    print("\n2. Generating optical-flow morph frames (A→B)...")
    morph_frames = generate_morph_frames(norm_a, norm_b, n_frames=7)
    print(f"   Generated {len(morph_frames)} morph frames")
    for i, f in enumerate(morph_frames):
        f.save(str(OUT_DIR / f"morph_{i}.png"))
    print(f"   Saved to {OUT_DIR}/morph_0..6.png")

    # 3. Load background
    print("\n3. Loading background...")
    bg = Image.open(BG_PATH).convert("RGBA").resize((TARGET_W, TARGET_H))

    # 4. Render subtitle overlay
    print("4. Rendering subtitle overlay...")
    sub_overlay = render_subtitle(EN_TEXT, ZH_TEXT)

    # 5. Render full dialogue segment frames
    audio_dur = get_duration(AUDIO_PATH)
    pad = 0.4
    duration = audio_dur + pad
    total_frames = round(duration * DELIVERY_FPS)
    mid_frame = total_frames // 2
    morph_dur = 0.2  # 200ms transition
    morph_n = len(morph_frames)
    direction = 1  # even line index

    char_x = TARGET_W / 2
    char_bottom = POSE_BOTTOM

    print(f"\n5. Rendering {total_frames} frames (dur={duration:.2f}s, audio={audio_dur:.2f}s)...")
    frames_dir = OUT_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)

    for idx in range(total_frames):
        t = idx / DELIVERY_FPS

        # Determine pose + morph state
        if t < duration / 2 - morph_dur / 2:
            # First half: pose A with landing
            current_pose = norm_a
            local_time = t
            landing = compute_landing(local_time, direction=direction)
        elif t < duration / 2 + morph_dur / 2:
            # Transition zone: morph
            morph_progress = (t - (duration / 2 - morph_dur / 2)) / morph_dur
            morph_idx = min(morph_n - 1, max(0, int(morph_progress * morph_n)))
            current_pose = morph_frames[morph_idx]
            landing = {"scale": 0.0, "x": 0.0, "y": 0.0, "rotation": 0.0}
        else:
            # Second half: pose B with landing
            current_pose = norm_b
            local_time = t - (duration / 2 + morph_dur / 2)
            landing = compute_landing(local_time, direction=-direction)

        scale = 1.0 + landing["scale"]
        x = char_x + landing["x"]
        bottom = char_bottom + landing["y"]
        rotation = landing["rotation"]

        frame = render_frame(bg, current_pose, x, bottom, scale, rotation)
        # Overlay subtitle
        frame_rgba = frame.convert("RGBA")
        frame_rgba.alpha_composite(sub_overlay, (0, 0))
        frame = frame_rgba.convert("RGB")

        frame.save(str(frames_dir / f"frame-{idx:04d}.png"), compress_level=2)

        if idx % 24 == 0:
            print(f"   Frame {idx}/{total_frames} (t={t:.2f}s)")

    print(f"   All {total_frames} frames rendered.")

    # 6. Encode with FFmpeg
    print(f"\n6. Encoding video ({DELIVERY_FPS}fps)...")
    fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05"
    frame_pattern = str(frames_dir / "frame-%04d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(DELIVERY_FPS), "-i", frame_pattern,
        "-i", AUDIO_PATH,
        "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(DELIVERY_FPS),
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
        OUT_VIDEO,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"   ERROR: {r.stderr[-300:]}")
        sys.exit(1)

    size_kb = os.path.getsize(OUT_VIDEO) / 1024
    print(f"   Video saved: {OUT_VIDEO} ({size_kb:.0f}KB)")

    # 7. Verify output
    print("\n7. Verifying output...")
    r2 = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=width,height,nb_frames,codec_name", "-of", "csv=p=0", OUT_VIDEO],
        capture_output=True, text=True)
    print(f"   {r2.stdout.strip()}")

    # Also save a comparison static version
    static_out = str(TEST_DIR / "static_dialogue.mp4")
    cmd_static = [
        "ffmpeg", "-y", "-loop", "1", "-i", POSE_A, "-i", AUDIO_PATH,
        "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
               f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2",
        "-r", "24",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
        static_out,
    ]
    subprocess.run(cmd_static, capture_output=True, text=True, timeout=120)
    print(f"   Static comparison: {static_out} ({os.path.getsize(static_out)/1024:.0f}KB)")

    # Side-by-side comparison
    cmp_out = str(TEST_DIR / "sm_comparison.mp4")
    cmd_cmp = [
        "ffmpeg", "-y",
        "-i", static_out, "-i", OUT_VIDEO,
        "-filter_complex",
        f"[0:v]scale={TARGET_W//2}:{TARGET_H//2}[left];"
        f"[1:v]scale={TARGET_W//2}:{TARGET_H//2}[right];"
        f"[left][right]hstack=inputs=2[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-t", f"{duration:.3f}",
        cmp_out,
    ]
    r3 = subprocess.run(cmd_cmp, capture_output=True, text=True, timeout=120)
    if r3.returncode == 0:
        print(f"   Comparison: {cmp_out} ({os.path.getsize(cmp_out)/1024:.0f}KB)")
    else:
        print(f"   Comparison failed: {r3.stderr[-200:]}")

    print(f"\nDone! Output files in {TEST_DIR}:")
    for f in ["stop_motion_dialogue.mp4", "static_dialogue.mp4", "sm_comparison.mp4"]:
        p = TEST_DIR / f
        if p.exists():
            print(f"  {f}: {p.stat().st_size/1024:.0f}KB")


if __name__ == "__main__":
    main()
