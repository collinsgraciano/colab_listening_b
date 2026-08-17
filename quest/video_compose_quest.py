"""Quest FFmpeg video composition — task-hook slow listening, all-image mode.

Structure:
  Ch1: Title Card    (scene image + big title overlay, 5s silence)
  Ch2: Hook / Intro  (scene image + listening-task card + narrator audio)
  Ch3: Slow Dialogue (one static image per line + slow TTS + burned subtitles)
  Ch4: Outro & CTA   (scene image + answer card + narrator audio)

Reuses render helpers from parent video_compose.py (same pattern as enhanced/).
No video clips are generated — this is the static (credit-light) variant.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

_PARENT = str(Path(__file__).parent.parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from video_compose import (
    _render_title_card,
    _render_practice_intro,
)
from media_utils import (
    FONT_EN, FONT_ZH, VF_NORM,
    concat_segments, burn_subtitles, apply_final_loudnorm,
    make_silent_fallback_cmd,
)


def _wrap_lines(draw, text, font_path, size, max_w, min_size=24):
    """Word-wrap text at the largest font size (>= min_size) that fits max_w.

    Returns (font, lines).
    """
    from PIL import ImageFont
    while size >= min_size:
        font = ImageFont.truetype(font_path, size)
        words = text.split()
        lines = []
        cur = ""
        for word in words:
            test = (cur + " " + word).strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_w or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        ok = all(draw.textbbox((0, 0), ln, font=font)[2] -
                 draw.textbbox((0, 0), ln, font=font)[0] <= max_w for ln in lines)
        if ok and lines:
            return font, lines
        size -= 2
    font = ImageFont.truetype(font_path, min_size)
    return font, [text]


def _fit_font(draw, text, font_path, start_size, min_size, max_w):
    """Shrink font size until text fits max_w. Returns (font, w, h)."""
    from PIL import ImageFont
    size = start_size
    font = ImageFont.truetype(font_path, size)
    while size > min_size:
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_w:
            break
        size -= 2
        font = ImageFont.truetype(font_path, size)
    bbox = draw.textbbox((0, 0), text, font=font)
    return font, bbox[2] - bbox[0], bbox[3] - bbox[1]


def _render_hook_frame(hook_en, question_en, question_zh, scene_img_path,
                       out_path, w=1280, h=720):
    """Render the listening-task hook card PNG (transparent bg for overlay).

    Layout: task badge -> narrator text (wrapped, small) -> question EN (big
    white) -> question ZH (gold) -> bottom hint bar. All plain text (no emoji —
    CJK fonts render emoji as tofu).
    """
    from PIL import Image, ImageDraw, ImageFont

    MARGIN = 70
    frame = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)

    # Semi-transparent dark panel
    panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel)
    pd.rectangle([40, 30, w - 40, h - 110], fill=(0, 0, 0, 185))
    frame = Image.alpha_composite(frame, panel)
    draw = ImageDraw.Draw(frame)

    cur_y = 55

    # Badge: LISTENING TASK · 聽力任務
    badge_txt = "LISTENING TASK · 聽力任務"
    badge_font, bw, bh = _fit_font(draw, badge_txt, FONT_EN, 46, 28, w - 2 * MARGIN)
    draw.text(((w - bw) // 2, cur_y), badge_txt, font=badge_font,
              fill=(255, 220, 0, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))
    cur_y += bh + 28

    # Narrator text (small, wrapped) — auto-shrink until <= 9 lines
    if hook_en:
        for start in (40, 36, 32, 28, 24):
            n_font, n_lines = _wrap_lines(draw, hook_en, FONT_EN, start,
                                          w - 2 * MARGIN, min_size=24)
            if len(n_lines) <= 9:
                break
        for ln in n_lines:
            lb = draw.textbbox((0, 0), ln, font=n_font)
            lw, lh = lb[2] - lb[0], lb[3] - lb[1]
            draw.text(((w - lw) // 2, cur_y), ln, font=n_font,
                      fill=(235, 235, 235, 255))
            cur_y += lh + 6
        cur_y += 22

    # Question EN (big white)
    if question_en:
        q_font, qw, qh = _fit_font(draw, question_en, FONT_EN, 64, 34, w - 2 * MARGIN)
        draw.text(((w - qw) // 2, cur_y), question_en, font=q_font,
                  fill=(255, 255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0, 255))
        cur_y += qh + 18

    # Question ZH (gold)
    if question_zh:
        z_font, zw, zh = _fit_font(draw, question_zh, FONT_ZH, 50, 26, w - 2 * MARGIN)
        draw.text(((w - zw) // 2, cur_y), question_zh, font=z_font,
                  fill=(255, 220, 0, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))
        cur_y += zh + 14

    frame.save(out_path, "PNG")


def _render_outro_frame(question_en, question_zh, scene_img_path,
                        out_path, w=1280, h=720):
    """Render the closing answer/CTA card PNG (transparent bg for overlay)."""
    from PIL import Image, ImageDraw

    MARGIN = 70
    frame = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel)
    pd.rectangle([40, 40, w - 40, h - 40], fill=(0, 0, 0, 190))
    frame = Image.alpha_composite(frame, panel)
    draw = ImageDraw.Draw(frame)

    # Vertical stack, centered as a block
    blocks = []

    t1 = "你的答案是什麼？"
    blocks.append((t1, FONT_ZH, 56, (255, 220, 0, 255), 5))
    if question_en:
        blocks.append((question_en, FONT_EN, 58, (255, 255, 255, 255), 6))
    if question_zh:
        blocks.append((question_zh, FONT_ZH, 44, (255, 220, 0, 255), 4))
    blocks.append(("在評論區用英文寫下你的答案！", FONT_ZH, 44, (255, 255, 255, 255), 4))
    blocks.append(("哪怕一句話也很棒！", FONT_ZH, 38, (200, 200, 200, 255), 3))
    blocks.append(("Like · Subscribe · 每天慢速聽英文", FONT_ZH, 36, (130, 200, 255, 255), 3))

    # Measure all blocks with fitted fonts
    fitted = []
    for text, fpath, start_size, color, stroke in blocks:
        font, tw, th = _fit_font(draw, text, fpath, start_size, 24, w - 2 * MARGIN)
        fitted.append((text, font, tw, th, color, stroke))

    GAP = 26
    total_h = sum(f[3] for f in fitted) + GAP * (len(fitted) - 1)
    cur_y = max(50, (h - total_h) // 2)
    for text, font, tw, th, color, stroke in fitted:
        draw.text(((w - tw) // 2, cur_y), text, font=font,
                  fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        cur_y += th + GAP

    frame.save(out_path, "PNG")


def compose_quest(
    work_dir: str,
    dialogue_images: list[str],
    timeline: list[dict],
    script: dict,
    narration: dict,
    normal_paths: list[str],
    scene_img: str,
    srt_dir: str,
    pad: float = 5.0,
    progress_cb=None,
) -> str:
    """Compose the final quest (task-hook slow listening) video — all images.

    Args:
        work_dir: Working directory for temp files and output.
        dialogue_images: Per-line dialogue image paths (dialogue_img_{i}.png).
        timeline: From build_quest_timeline (audio_dur enriched).
        script: Quest script dict.
        narration: {"hook": path, "outro": path}.
        normal_paths: English dialogue audio paths (slow speed).
        scene_img: Scene background image path.
        srt_dir: Directory used as FFmpeg cwd for the subtitle burn.
        pad: Silence pad after dialogue lines (seconds).
        progress_cb: callback(percent, message).

    Returns:
        Path to final video (named by YouTube title).
    """
    def _cb(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    work = Path(work_dir)
    tmp_dir = work / "tmp_segments"
    static_dir = work / "static_frames"
    vid_dir = work / "videos"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    question_en = script.get("listening_question_en", "")
    question_zh = script.get("listening_question_zh", "")
    hook_en = script.get("hook_intro_en", "")

    segments = []
    seg_idx = 0
    total_segs = len(timeline)

    for seg in timeline:
        seg_type = seg["type"]
        duration = seg["duration"]
        audio_idx = seg.get("audio_index", 0)
        out_path = str(tmp_dir / f"seg_{seg_idx:03d}.mp4")
        seg_idx += 1

        audio_file = None
        audio_dur = seg.get("audio_dur", duration - pad)

        if seg_type == "dialogue":
            audio_file = normal_paths[audio_idx] if audio_idx < len(normal_paths) else None
        elif seg_type == "hook_intro":
            audio_file = narration.get("hook")
        elif seg_type == "outro":
            audio_file = narration.get("outro")
        elif seg_type == "title_card":
            audio_dur = duration

        fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05"

        # --- Build FFmpeg command per segment type ---
        if seg_type == "title_card":
            # Static scene + big title overlay + silence (static-mode pattern)
            title_en = seg.get("subtitle_en", "")
            title_zh = seg.get("subtitle_zh", "")
            title_overlay = str(static_dir / "title_overlay.png")
            _render_title_card(title_en, title_zh, "", scene_img, title_overlay)
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                   "-i", title_overlay,
                   "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                   "-t", f"{duration:.3f}",
                   "-filter_complex", f"[0:v]{VF_NORM}[bg];[bg][1:v]overlay=0:0[v]",
                   "-map", "[v]", "-map", "2:a",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   out_path]

        elif seg_type == "hook_intro":
            # Static scene + listening-task card + narrator audio
            hook_overlay = str(static_dir / "hook_overlay.png")
            _render_hook_frame(hook_en, question_en, question_zh, scene_img,
                               hook_overlay)
            out_dur = audio_dur + pad
            hook_audio = narration.get("hook")
            if hook_audio and os.path.exists(hook_audio):
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                       "-i", hook_audio, "-i", hook_overlay,
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex",
                       f"[0:v]{VF_NORM}[bg];[bg][2:v]overlay=0:0[v];"
                       f"[1:a]{fade_af},apad=whole_dur={out_dur:.3f}[a]",
                       "-map", "[v]", "-map", "[a]",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                       "-i", hook_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}",
                       "-filter_complex", f"[0:v]{VF_NORM}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        elif seg_type == "outro":
            # Static scene + answer/CTA card + narrator audio
            outro_overlay = str(static_dir / "outro_overlay.png")
            _render_outro_frame(question_en, question_zh, scene_img, outro_overlay)
            out_dur = audio_dur + pad
            outro_audio = narration.get("outro")
            if outro_audio and os.path.exists(outro_audio):
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                       "-i", outro_audio, "-i", outro_overlay,
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex",
                       f"[0:v]{VF_NORM}[bg];[bg][2:v]overlay=0:0[v];"
                       f"[1:a]{fade_af},apad=whole_dur={out_dur:.3f}[a]",
                       "-map", "[v]", "-map", "[a]",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                       "-i", outro_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}",
                       "-filter_complex", f"[0:v]{VF_NORM}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        elif seg_type == "dialogue":
            # Per-line static image + slow TTS audio
            idx = min(audio_idx, len(dialogue_images) - 1) if dialogue_images else 0
            d_img = dialogue_images[idx] if dialogue_images and idx < len(dialogue_images) else scene_img
            if not os.path.exists(d_img):
                d_img = scene_img
            if audio_file and os.path.exists(audio_file):
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", d_img, "-i", audio_file,
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", VF_NORM, "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", d_img,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", VF_NORM, "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
        else:
            # Unknown segment type — silent static placeholder keeps timeline intact
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                   "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                   "-t", f"{duration:.3f}", "-vf", f"{VF_NORM},fps=24",
                   "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   out_path]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            print(f"  FFmpeg TIMEOUT (300s) on seg {seg_idx} ({seg_type}), using fallback")
            r = None
        if r is None or r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            print(f"  FFmpeg error seg {seg_idx}: {r.stderr[-200:] if r else 'timeout'}")
            fallback_cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                            "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                            "-t", f"{duration:.3f}", "-vf", f"{VF_NORM},fps=24",
                            "-map", "0:v:0", "-map", "1:a:0",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                            out_path]
            try:
                r2 = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                print(f"  Fallback also timed out, skipping segment {seg_idx}")
                continue
            if r2.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
                print(f"  Fallback also failed, skipping segment: {r2.stderr[-200:]}")
                continue
        segments.append(out_path)
        _cb(int(seg_idx / total_segs * 80),
            f"  Segment {seg_idx}/{total_segs} ({seg_type})")

    # --- Concat all segments ---
    _cb(80, "Concatenating segments...")
    no_sub = str(vid_dir / "final_no_sub.mp4")
    concat_segments(segments, no_sub, tmp_dir=tmp_dir)

    # --- Burn subtitles via Pillow overlay (dialogue entries only) ---
    _cb(90, "Burning subtitles (Pillow overlay)...")
    final_path = burn_subtitles(no_sub, timeline, script, str(work), srt_dir, pad, _cb)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- Final loudnorm pass ---
    _cb(95, "Final loudnorm pass (normalize volume)...")
    apply_final_loudnorm(final_path, str(vid_dir))

    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    _cb(100, f"Quest video done: {final_path} ({size_mb:.1f}MB)")
    return final_path
