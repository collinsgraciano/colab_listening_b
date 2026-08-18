#!/usr/bin/env python3
"""Test2: stop_motion with per-character reference + 1280x720 + rembg.

Flow:
  1. char_a_ref.png, char_b_ref.png — single-character reference sheets
  2. background.png — scene without characters
  3. pose_0_0.png — char_a speaking (referencing char_a_ref CDN)
  4. pose_0_1.png — char_a listening (referencing char_a_ref CDN)
  5. dialogue_test.wav — TTS audio
  6. stop_motion render → MP4
"""
import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from PIL import Image, ImageDraw, ImageFont
from media_utils import TARGET_W, TARGET_H, FONT_EN, FONT_ZH, get_duration
from stop_motion import (
    remove_bg, normalize_pose, generate_morph_frames,
    render_frame, compute_landing, _HAS_REMBG,
    POSE_CENTER_Y, DELIVERY_FPS,
)

T2 = Path(__file__).parent
POSE_A = str(T2 / "pose_0_0.png")
POSE_B = str(T2 / "pose_0_1.png")
BG_PATH = str(T2 / "background.png")
AUDIO = str(T2 / "dialogue_test.wav")
OUT_VIDEO = str(T2 / "stop_motion_dialogue.mp4")
STATIC_VIDEO = str(T2 / "static_dialogue.mp4")
CMP_VIDEO = str(T2 / "comparison.mp4")

EN = "Hi there! Welcome to our coffee shop. What can I get for you today?"
ZH = "你好！歡迎來到我們的咖啡店。今天想喝點什麼？"


def render_subtitle(en, zh, w=TARGET_W, h=TARGET_H):
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    en_font = ImageFont.truetype(FONT_EN, 44)
    zh_font = ImageFont.truetype(FONT_ZH, 36)
    en_bb = draw.textbbox((0, 0), en, font=en_font)
    en_w, en_h = en_bb[2]-en_bb[0], en_bb[3]-en_bb[1]
    zh_bb = draw.textbbox((0, 0), zh, font=zh_font)
    zh_w, zh_h = zh_bb[2]-zh_bb[0], zh_bb[3]-zh_bb[1]
    box_h = en_h + zh_h + 30
    box_top = h - box_h - 40
    box = Image.new("RGBA", (w, box_h), (0, 0, 0, 140))
    overlay.alpha_composite(box, (0, box_top))
    draw.text(((w-en_w)//2, box_top+10), en, font=en_font, fill="white",
              stroke_width=3, stroke_fill="black")
    draw.text(((w-zh_w)//2, box_top+10+en_h+8), zh, font=zh_font,
              fill="rgb(255,220,0)", stroke_width=2, stroke_fill="black")
    return overlay


def main():
    for label, path in [("Pose A", POSE_A), ("Pose B", POSE_B),
                         ("Background", BG_PATH), ("Audio", AUDIO)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)
        if label != "Audio":
            print(f"  {label}: {os.path.getsize(path)//1024}KB ({Image.open(path).size})")
        else:
            print(f"  {label}: {os.path.getsize(path)//1024}KB")

    out_dir = T2 / "sm_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n1. AI background removal (rembg={_HAS_REMBG})...")
    raw_a = Image.open(POSE_A)
    raw_b = Image.open(POSE_B)
    print(f"   Raw: A={raw_a.size}, B={raw_b.size}")
    alpha_a = remove_bg(raw_a)
    alpha_b = remove_bg(raw_b)
    for name, img in [("A", alpha_a), ("B", alpha_b)]:
        px = list(img.getdata())
        t = sum(1 for p in px if p[3] == 0)
        print(f"   Pose {name}: {100*t/len(px):.1f}% transparent")

    norm_a = normalize_pose(alpha_a)
    norm_b = normalize_pose(alpha_b)
    print(f"   Normalized: A={norm_a.size}, B={norm_b.size}")

    # Save cutout previews
    norm_a.save(str(out_dir / "cutout_a.png"))
    norm_b.save(str(out_dir / "cutout_b.png"))
    print(f"   Saved cutouts: {out_dir}/cutout_a.png, cutout_b.png")

    print("\n2. Optical-flow morph (A→B)...")
    morph = generate_morph_frames(norm_a, norm_b, n_frames=7)
    print(f"   {len(morph)} frames")
    for i, f in enumerate(morph):
        f.save(str(out_dir / f"morph_{i}.png"))

    print("\n3. Rendering frames...")
    bg = Image.open(BG_PATH).convert("RGBA").resize((TARGET_W, TARGET_H))
    sub = render_subtitle(EN, ZH)
    audio_dur = get_duration(AUDIO)
    pad = 0.4
    duration = audio_dur + pad
    total = round(duration * DELIVERY_FPS)
    mid = total // 2
    morph_dur = 0.2
    morph_n = len(morph)
    direction = 1
    cx = TARGET_W / 2
    bottom = POSE_CENTER_Y

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for idx in range(total):
        t = idx / DELIVERY_FPS
        if t < duration/2 - morph_dur/2:
            pose = norm_a
            landing = compute_landing(t, direction=direction)
        elif t < duration/2 + morph_dur/2:
            mp = (t - (duration/2 - morph_dur/2)) / morph_dur
            mi = min(morph_n-1, max(0, int(mp*morph_n)))
            pose = morph[mi]
            landing = {"scale":0,"x":0,"y":0,"rotation":0}
        else:
            pose = norm_b
            landing = compute_landing(t-(duration/2+morph_dur/2), direction=-direction)
        scale = 1.0 + landing["scale"]
        frame = render_frame(bg, pose, cx+landing["x"], bottom+landing["y"], scale, landing["rotation"], centered=True)
        rgba = frame.convert("RGBA")
        rgba.alpha_composite(sub, (0, 0))
        rgba.convert("RGB").save(str(frames_dir / f"frame-{idx:04d}.png"), compress_level=2)
        if idx % 24 == 0:
            print(f"   Frame {idx}/{total} (t={t:.2f}s)")

    print(f"\n4. Encoding ({DELIVERY_FPS}fps)...")
    fade = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0,audio_dur-0.05):.2f}:d=0.05"
    cmd = ["ffmpeg","-y","-framerate",str(DELIVERY_FPS),"-i",str(frames_dir/"frame-%04d.png"),
           "-i",AUDIO,"-t",f"{duration:.3f}","-map","0:v:0","-map","1:a:0",
           "-c:v","libx264","-pix_fmt","yuv420p","-r",str(DELIVERY_FPS),
           "-c:a","aac","-b:a","128k","-ar","44100","-ac","2",
           "-af",f"{fade},apad=whole_dur={duration:.3f}",OUT_VIDEO]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"   ERROR: {r.stderr[-300:]}")
        sys.exit(1)
    print(f"   {OUT_VIDEO} ({os.path.getsize(OUT_VIDEO)//1024}KB)")

    # Static comparison
    cmd_s = ["ffmpeg","-y","-loop","1","-i",POSE_A,"-i",AUDIO,
             "-t",f"{duration:.3f}","-map","0:v:0","-map","1:a:0",
             "-c:v","libx264","-pix_fmt","yuv420p",
             "-vf",f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2",
             "-r","24","-c:a","aac","-b:a","128k","-ar","44100","-ac","2",
             "-af",f"{fade},apad=whole_dur={duration:.3f}",STATIC_VIDEO]
    subprocess.run(cmd_s, capture_output=True, text=True, timeout=120)
    print(f"   {STATIC_VIDEO} ({os.path.getsize(STATIC_VIDEO)//1024}KB)")

    # Side-by-side
    cmd_c = ["ffmpeg","-y","-i",STATIC_VIDEO,"-i",OUT_VIDEO,
             "-filter_complex",
             f"[0:v]scale={TARGET_W//2}:{TARGET_H//2}[l];[1:v]scale={TARGET_W//2}:{TARGET_H//2}[r];[l][r]hstack=inputs=2[v]",
             "-map","[v]","-map","1:a","-c:v","libx264","-pix_fmt","yuv420p","-r","24",
             "-c:a","aac","-b:a","128k","-ar","44100","-ac","2","-t",f"{duration:.3f}",CMP_VIDEO]
    subprocess.run(cmd_c, capture_output=True, text=True, timeout=120)
    if os.path.exists(CMP_VIDEO):
        print(f"   {CMP_VIDEO} ({os.path.getsize(CMP_VIDEO)//1024}KB)")

    # Verify
    r = subprocess.run(["ffprobe","-v","error","-show_entries",
                        "stream=width,height,nb_frames,codec_name","-of","csv=p=0",OUT_VIDEO],
                       capture_output=True, text=True)
    print(f"\n5. Verify: {r.stdout.strip()}")

    # Cleanup frames
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)

    print("\nDone!")
    for f in ["stop_motion_dialogue.mp4","static_dialogue.mp4","comparison.mp4"]:
        p = T2 / f
        if p.exists():
            print(f"  {f}: {p.stat().st_size//1024}KB")


if __name__ == "__main__":
    main()
