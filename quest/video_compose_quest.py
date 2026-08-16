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
    _get_duration,
    _render_title_card,
    _render_practice_intro,
    _VF_NORM,
    _probe_resolution,
    FONT_EN, FONT_ZH,
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
                   "-filter_complex", f"[0:v]{_VF_NORM}[bg];[bg][1:v]overlay=0:0[v]",
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
                       f"[0:v]{_VF_NORM}[bg];[bg][2:v]overlay=0:0[v];"
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
                       "-filter_complex", f"[0:v]{_VF_NORM}[bg];[bg][1:v]overlay=0:0[v]",
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
                       f"[0:v]{_VF_NORM}[bg];[bg][2:v]overlay=0:0[v];"
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
                       "-filter_complex", f"[0:v]{_VF_NORM}[bg];[bg][1:v]overlay=0:0[v]",
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
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", _VF_NORM, "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", d_img,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", _VF_NORM, "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
        else:
            # Unknown segment type — silent static placeholder keeps timeline intact
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                   "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                   "-t", f"{duration:.3f}", "-vf", f"{_VF_NORM},fps=24",
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
                            "-t", f"{duration:.3f}", "-vf", f"{_VF_NORM},fps=24",
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
    concat_list = tmp_dir / "concat.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"file '{s}'\n")

    no_sub = str(vid_dir / "final_no_sub.mp4")
    result = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", no_sub,
    ], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Concat failed: {result.stderr.decode()[-2000:]}")

    # --- Burn subtitles via Pillow overlay (dialogue entries only) ---
    _cb(90, "Burning subtitles (Pillow overlay)...")
    import re as _re
    _yt = script.get("youtube_title", script.get("title", "final_video"))
    _safe = _re.sub(r'[\U0001F000-\U0001FFFF]', '', _yt)
    _safe = _re.sub(r'[\\/:*?"<>|]', '', _safe).strip()
    _safe = _re.sub(r'\s+', '_', _safe)[:80] or "final_video"
    final_path = str(work / f"{_safe}.mp4")

    subtitle_entries = []
    t_cursor = 0.0
    for seg in timeline:
        dur = seg["duration"]
        seg_type = seg.get("type", "")
        if seg_type == "dialogue":
            en = seg.get("subtitle_en", "")
            zh = seg.get("subtitle_zh", "")
            audio_d = seg.get("audio_dur", dur - pad)
            if en or zh:
                subtitle_entries.append({
                    "start": t_cursor,
                    "end": t_cursor + audio_d,
                    "en": en,
                    "zh": zh,
                })
        t_cursor += dur

    if subtitle_entries:
        from PIL import Image, ImageDraw, ImageFont
        w, h = _probe_resolution(no_sub)
        sub_overlay_dir = tmp_dir / "subtitles"
        sub_overlay_dir.mkdir(exist_ok=True)

        BOTTOM_MARGIN = 36
        for i, entry in enumerate(subtitle_entries):
            overlay_path = str(sub_overlay_dir / f"sub_{i:03d}.png")
            bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(bg)

            en_text = entry["en"]
            zh_text = entry["zh"]

            en_font = en_w = en_h = None
            if en_text:
                en_size = 50
                en_font = ImageFont.truetype(FONT_EN, en_size)
                while en_size > 28:
                    bbox = draw.textbbox((0, 0), en_text, font=en_font)
                    if bbox[2] - bbox[0] <= w - 80:
                        break
                    en_size -= 2
                    en_font = ImageFont.truetype(FONT_EN, en_size)
                bbox = draw.textbbox((0, 0), en_text, font=en_font)
                en_w, en_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            zh_font = zh_w = zh_h = None
            if zh_text:
                zh_size = 50
                zh_font = ImageFont.truetype(FONT_ZH, zh_size)
                while zh_size > 24:
                    bbox = draw.textbbox((0, 0), zh_text, font=zh_font)
                    if bbox[2] - bbox[0] <= w - 80:
                        break
                    zh_size -= 2
                    zh_font = ImageFont.truetype(FONT_ZH, zh_size)
                bbox = draw.textbbox((0, 0), zh_text, font=zh_font)
                zh_w, zh_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # Stack bottom-up (ZH lowest, EN above it)
            if en_text and zh_text:
                zh_y = h - BOTTOM_MARGIN - zh_h
                en_y = zh_y - 15 - en_h
            elif en_text:
                en_y = h - BOTTOM_MARGIN - en_h
                zh_y = 0
            else:
                en_y = 0
                zh_y = h - BOTTOM_MARGIN - zh_h

            if en_text and en_font is not None:
                draw.text(((w - en_w) // 2, en_y), en_text, font=en_font,
                          fill=(255, 255, 255, 255), stroke_width=5, stroke_fill=(0, 0, 0, 255))
            if zh_text and zh_font is not None:
                draw.text(((w - zh_w) // 2, zh_y), zh_text, font=zh_font,
                          fill=(255, 215, 0, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))

            bg.save(overlay_path, "PNG")
            entry["overlay_path"] = overlay_path

        filter_parts = []
        prev_label = "0:v"
        for i, entry in enumerate(subtitle_entries):
            overlay_path = entry["overlay_path"].replace("\\", "/").replace(":", "\\:")
            start = entry["start"]
            end = entry["end"]
            filter_parts.append(
                f"[{prev_label}][{i+1}:v]overlay=0:0:enable='between(t,{start:.3f},{end:.3f})'[v{i}]"
            )
            prev_label = f"v{i}"

        input_args = ["-i", no_sub]
        for entry in subtitle_entries:
            input_args.extend(["-i", entry["overlay_path"]])

        filter_complex = ";".join(filter_parts)
        final_label = prev_label

        cmd = ["ffmpeg", "-y"] + input_args + [
            "-filter_complex", filter_complex,
            "-map", f"[{final_label}]",
            "-map", "0:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            final_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, cwd=str(srt_dir), timeout=1800)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Subtitle overlay burn timed out after 1800s (too many subtitle PNGs?)")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Subtitle overlay burn failed: {e.stderr.decode(errors='replace')[-500:] if e.stderr else e}")
    else:
        shutil.copy2(no_sub, final_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- Final loudnorm pass ---
    _cb(95, "Final loudnorm pass (normalize volume)...")
    norm_path = str(vid_dir / "final_video_norm.mp4")
    norm_result = subprocess.run(
        ["ffmpeg", "-y", "-i", final_path,
         "-c:v", "copy",
         "-c:a", "aac", "-b:a", "128k",
         "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
         norm_path],
        capture_output=True, timeout=600,
    )
    if norm_result.returncode == 0 and os.path.exists(norm_path) and os.path.getsize(norm_path) > 1000:
        os.replace(norm_path, final_path)
    else:
        if os.path.exists(norm_path):
            try:
                os.remove(norm_path)
            except OSError:
                pass
        try:
            vol_path = str(vid_dir / "final_video_vol.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-i", final_path,
                 "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "128k",
                 "-af", "volume=6dB",
                 vol_path],
                capture_output=True, timeout=600,
            )
            if os.path.exists(vol_path) and os.path.getsize(vol_path) > 1000:
                os.replace(vol_path, final_path)
        except Exception:
            pass

    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    _cb(100, f"Quest video done: {final_path} ({size_mb:.1f}MB)")
    return final_path
