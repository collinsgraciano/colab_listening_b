"""Standalone FFmpeg video composition for listening practice videos.

Adapts compose_listening to take parameters directly (no meta.json, no DATA_DIR).
Includes inline Pillow rendering functions for static frames and subtitle overlays.

No dependency on any external project.
"""
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

# Fonts: auto-detect Windows vs Linux (Colab) paths
import platform
_IS_WINDOWS = platform.system() == "Windows"
if _IS_WINDOWS:
    FONT_EN = r"C:\Windows\Fonts\msyhbd.ttc"
    FONT_ZH = r"C:\Windows\Fonts\msyh.ttc"
    FONT_PH = r"C:\Windows\Fonts\cambria.ttc"
else:
    FONT_EN = "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"
    FONT_ZH = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
    FONT_PH = "/usr/share/fonts/truetype/cambria/cambria.ttc"
    import os as _os
    if not _os.path.exists(FONT_EN):
        FONT_EN = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not _os.path.exists(FONT_ZH):
        FONT_ZH = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
    if not _os.path.exists(FONT_PH):
        FONT_PH = FONT_EN


def _get_duration(path: str) -> float:
    """Get media duration in seconds via ffprobe."""
    try:
        return float(subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            text=True,
        ).strip())
    except Exception:
        return 0.0


def _has_audio(video_path: str) -> bool:
    """Check if a video file has an audio stream."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        return "audio" in r.stdout.strip()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pillow rendering functions
# ---------------------------------------------------------------------------

def _render_static_frame(en_text, phonetic, zh_text, scene_img_path,
                          out_path, idx, total, w=1280, h=720):
    """Render a static PNG with English + phonetic + Chinese text over scene."""
    from PIL import Image, ImageDraw, ImageFont

    bg = Image.open(scene_img_path).convert("RGBA").resize((w, h))
    draw_on_bg = ImageDraw.Draw(bg)

    # Sentence number (top-right, yellow on black circle)
    num_font = ImageFont.truetype(FONT_EN, 36)
    num_text = f"{idx+1}/{total}"
    num_bbox = draw_on_bg.textbbox((0, 0), num_text, font=num_font)
    num_tw, num_th = num_bbox[2]-num_bbox[0], num_bbox[3]-num_bbox[1]
    num_pad = 12
    num_radius = max(num_tw, num_th) // 2 + num_pad
    cx, cy = w - num_radius - 20, num_radius + 20

    def _fit_font(text, font_path, start_size=56, min_size=16, max_w=w-80):
        size = start_size
        font = ImageFont.truetype(font_path, size)
        while size > min_size:
            bbox = draw_on_bg.textbbox((0, 0), text, font=font)
            if bbox[2] - bbox[0] <= max_w:
                break
            size -= 2
            font = ImageFont.truetype(font_path, size)
        return size, font

    en_fit, _ = _fit_font(en_text, FONT_EN)
    ph_fit = _fit_font(phonetic, FONT_PH)[0] if phonetic else en_fit
    zh_fit = _fit_font(zh_text, FONT_ZH)[0] if zh_text else en_fit
    final_size = min(en_fit, ph_fit, zh_fit)

    en_font = ImageFont.truetype(FONT_EN, final_size)
    en_bbox = draw_on_bg.textbbox((0, 0), en_text, font=en_font)
    en_w, en_h = en_bbox[2]-en_bbox[0], en_bbox[3]-en_bbox[1]
    target_h = en_h

    def _match_height(text, font_path, target_h, base_size, max_w=w-40):
        size = base_size
        font = ImageFont.truetype(font_path, size)
        bb = draw_on_bg.textbbox((0, 0), text, font=font)
        actual_h = bb[3] - bb[1]
        if actual_h > 0:
            size = int(size * target_h / actual_h)
            size = max(size, 14)
        font = ImageFont.truetype(font_path, size)
        while size > 14:
            bb = draw_on_bg.textbbox((0, 0), text, font=font)
            if bb[2] - bb[0] <= max_w:
                break
            size -= 2
            font = ImageFont.truetype(font_path, size)
        return size, font

    ph_font_final = None
    zh_font_final = None
    ph_w_final = ph_h_final = 0
    zh_w_final = zh_h_final = 0

    if phonetic:
        ph_size, ph_font_final = _match_height(phonetic, FONT_PH, target_h, final_size)
        bbox = draw_on_bg.textbbox((0, 0), phonetic, font=ph_font_final)
        ph_w_final, ph_h_final = bbox[2]-bbox[0], bbox[3]-bbox[1]
    if zh_text:
        zh_size, zh_font_final = _match_height(zh_text, FONT_ZH, target_h, final_size)
        bbox = draw_on_bg.textbbox((0, 0), zh_text, font=zh_font_final)
        zh_w_final, zh_h_final = bbox[2]-bbox[0], bbox[3]-bbox[1]

    gap = 15
    total_text_h = en_h + (ph_h_final + gap if phonetic else 0) + (zh_h_final + gap if zh_text else 0)
    box_padding = 30
    box_h = int(total_text_h + box_padding * 2)
    box_y = (h - box_h) // 2

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([0, box_y, w, box_y + box_h], fill=(0, 0, 0, 175))
    bg = Image.alpha_composite(bg, overlay)
    draw = ImageDraw.Draw(bg)

    draw.ellipse([cx-num_radius, cy-num_radius, cx+num_radius, cy+num_radius], fill=(0, 0, 0, 200))
    draw.text((cx-num_tw//2, cy-num_th//2-2), num_text, font=num_font, fill=(255, 220, 0, 255))

    en_y = box_y + box_padding
    draw.text(((w-en_w)//2, en_y), en_text, font=en_font, fill=(255, 255, 255, 255))

    if phonetic and ph_font_final:
        ph_y = en_y + en_h + gap
        draw.text(((w-ph_w_final)//2, ph_y), phonetic, font=ph_font_final, fill=(130, 200, 255, 255))

    if zh_text and zh_font_final:
        zh_y = en_y + en_h + gap + (ph_h_final + gap if phonetic else 0)
        draw.text(((w-zh_w_final)//2, zh_y), zh_text, font=zh_font_final, fill=(255, 220, 0, 255))

    bg.convert("RGB").save(out_path, "PNG")


def _render_title_card(title_en, title_zh, scene_zh, scene_img_path,
                       out_path, w=1280, h=720):
    """Render a title card overlay PNG with TRANSPARENT background (for video overlay).

    Text: 大字英文標題 (center) + 繁中標題 (below), thick stroke for readability.
    """
    from PIL import Image, ImageDraw, ImageFont

    STROKE = 8
    MARGIN = 80  # safe margin from frame edge

    bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bg)

    en_y = 0
    en_h = 0
    if title_en:
        display_text = title_en.upper()  # measure what we actually draw
        en_size = 120
        en_font = ImageFont.truetype(FONT_EN, en_size)
        while en_size > 40:
            bbox = draw.textbbox((0, 0), display_text, font=en_font)
            # Account for stroke_width: rendered width = bbox + 2*stroke
            rendered_w = (bbox[2]-bbox[0]) + STROKE * 2
            if rendered_w <= w - MARGIN:
                break
            en_size -= 2
            en_font = ImageFont.truetype(FONT_EN, en_size)
        bbox = draw.textbbox((0, 0), display_text, font=en_font)
        en_w = bbox[2]-bbox[0]
        en_h = bbox[3]-bbox[1]
        en_y = int(h * 0.30)
        draw.text(((w-en_w)//2, en_y), display_text, font=en_font,
                  fill=(255, 255, 255, 255), stroke_width=STROKE, stroke_fill=(0, 0, 0, 255))

    if title_zh:
        ZH_STROKE = 6
        zh_size = 120
        zh_font = ImageFont.truetype(FONT_ZH, zh_size)
        while zh_size > 28:
            bbox = draw.textbbox((0, 0), title_zh, font=zh_font)
            rendered_w = (bbox[2]-bbox[0]) + ZH_STROKE * 2
            if rendered_w <= w - MARGIN:
                break
            zh_size -= 2
            zh_font = ImageFont.truetype(FONT_ZH, zh_size)
        bbox = draw.textbbox((0, 0), title_zh, font=zh_font)
        zh_w = bbox[2]-bbox[0]
        zh_y = en_y + en_h + 30 if title_en else int(h * 0.45)
        draw.text(((w-zh_w)//2, zh_y), title_zh, font=zh_font,
                  fill=(255, 220, 0, 255), stroke_width=ZH_STROKE, stroke_fill=(0, 0, 0, 255))

    bg.save(out_path, "PNG")


def _render_practice_intro(intro_en, intro_zh, scene_img_path,
                            out_path, w=1280, h=720):
    """Render practice intro / outro overlay PNG — large text, multi-line English layout.

    Uses proper word-wrapping to fit long sentences into multiple lines.
    """
    from PIL import Image, ImageDraw, ImageFont

    STROKE = 7
    MARGIN = 80

    bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bg)

    # Word-wrap: split into lines that fit within w - MARGIN at given font size
    def _wrap_text(text, font_path, start_size, min_size, max_w, stroke):
        """Find font size + line breaks so every line fits within max_w."""
        size = start_size
        while size >= min_size:
            font = ImageFont.truetype(font_path, size)
            available_w = max_w - stroke * 2
            words = text.split()
            lines = []
            cur_line = ""
            for word in words:
                test = (cur_line + " " + word).strip()
                bbox = draw.textbbox((0, 0), test, font=font)
                if bbox[2]-bbox[0] <= available_w or not cur_line:
                    cur_line = test
                else:
                    lines.append(cur_line)
                    cur_line = word
            if cur_line:
                lines.append(cur_line)
            # Check all lines fit
            all_fit = True
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                if bbox[2]-bbox[0] > available_w:
                    all_fit = False
                    break
            if all_fit and lines:
                return size, font, lines
            size -= 2
        # Fallback: min size, force-split by width
        font = ImageFont.truetype(font_path, min_size)
        available_w = max_w - stroke * 2
        words = text.split()
        lines = []
        cur_line = ""
        for word in words:
            test = (cur_line + " " + word).strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2]-bbox[0] <= available_w or not cur_line:
                cur_line = test
            else:
                lines.append(cur_line)
                cur_line = word
        if cur_line:
            lines.append(cur_line)
        return min_size, font, lines

    en_lines = []
    en_font = None
    if intro_en:
        en_size, en_font, en_lines = _wrap_text(
            intro_en, FONT_EN, 96, 36, w - MARGIN, STROKE)

    # Calculate total height for vertical centering
    line_heights = []
    for line_text in en_lines:
        bbox = draw.textbbox((0, 0), line_text, font=en_font)
        line_heights.append(bbox[3]-bbox[1])

    zh_h = 0
    zh_font_final = None
    zh_w = 0
    if intro_zh:
        zh_size = 80
        zh_font_final = ImageFont.truetype(FONT_ZH, zh_size)
        while zh_size > 28:
            bbox = draw.textbbox((0, 0), intro_zh, font=zh_font_final)
            rendered_w = (bbox[2]-bbox[0]) + 5 * 2
            if rendered_w <= w - MARGIN:
                break
            zh_size -= 2
            zh_font_final = ImageFont.truetype(FONT_ZH, zh_size)
        bbox = draw.textbbox((0, 0), intro_zh, font=zh_font_final)
        zh_w = bbox[2]-bbox[0]
        zh_h = bbox[3]-bbox[1]

    total_h = sum(line_heights) + max(0, len(line_heights) - 1) * 20
    if intro_zh:
        total_h += 30 + zh_h
    start_y = max(int(h * 0.25), (h - total_h) // 2)

    current_y = start_y
    for i, line_text in enumerate(en_lines):
        bbox = draw.textbbox((0, 0), line_text, font=en_font)
        en_w = bbox[2]-bbox[0]
        en_h = bbox[3]-bbox[1]
        draw.text(((w-en_w)//2, current_y), line_text, font=en_font,
                  fill=(255, 255, 255, 255), stroke_width=STROKE, stroke_fill=(0, 0, 0, 255))
        current_y += en_h + 20

    if intro_zh and zh_font_final:
        zh_y = current_y + 10
        draw.text(((w-zh_w)//2, zh_y), intro_zh, font=zh_font_final,
                  fill=(255, 220, 0, 255), stroke_width=5, stroke_fill=(0, 0, 0, 255))

    bg.save(out_path, "PNG")


# ---------------------------------------------------------------------------
# Main compose function
# ---------------------------------------------------------------------------

def compose_listening(
    work_dir: str,
    clip_paths: list[str],
    timeline: list[dict],
    script: dict,
    narration: dict,
    normal_paths: list[str],
    zh_paths: list[str],
    scene_img: str,
    srt_dir: str,
    pad: float = 0.4,
    progress_cb=None,
    group_info: list[dict] | None = None,
    line_to_group: dict | None = None,
) -> str:
    """Compose final listening practice video.

    Args:
        work_dir: Working directory for temp files and output.
        clip_paths: List of video clip paths (index 0 = scene/HOOK, 1+ = dialogue groups).
                    May contain None for failed clips (handled gracefully).
        timeline: Timeline segments from build_listening_timeline (with audio_dur added).
        script: Lesson script dict.
        narration: {"intro": path, "outro": path, "practice_intro": path}.
        normal_paths: English dialogue audio paths.
        zh_paths: Chinese dialogue audio paths.
        scene_img: Scene background image path.
        srt_dir: Directory for SRT file (cwd for FFmpeg subtitle burn).
        pad: Audio pad between segments (seconds).
        progress_cb: callback(percent, message).
        group_info: [{clip_path, audio_path, total_dur, lines}] per group.
        line_to_group: {line_idx: group_idx} mapping for group-based dialogue.

    Returns:
        Path to final video.
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

    dialogue = script.get("dialogue", [])
    n = len(dialogue)
    HOOK_CLIP = clip_paths[0] if clip_paths else None
    DIALOGUE_CLIPS = clip_paths[1:] if len(clip_paths) > 1 else []

    # --- Render static frames for Ch3 ---
    if os.path.exists(scene_img):
        _cb(5, f"Rendering {n} static frames...")
        for i, line in enumerate(dialogue):
            p = str(static_dir / f"zh_{i}.png")
            _render_static_frame(
                line.get("text", ""), line.get("phonetic", ""),
                line.get("zh", ""), scene_img, p, i, n)
        _cb(10, "Static frames done.")

    # --- Build each segment ---
    segments = []
    seg_idx = 0
    total_segs = len(timeline)
    processed_groups = set()  # track which groups have been rendered as a single segment
    skipped_segs = 0  # count skipped segments for accurate progress

    for seg in timeline:
        seg_type = seg["type"]
        duration = seg["duration"]
        audio_idx = seg.get("audio_index", 0)

        # Group-based dialogue: skip lines that belong to an already-processed group
        if seg_type == "dialogue" and group_info and line_to_group:
            gi = line_to_group.get(audio_idx)
            if gi is not None and gi in processed_groups:
                # This line's group already rendered — skip
                skipped_segs += 1
                continue
            if gi is not None:
                processed_groups.add(gi)
                # Render entire group as ONE segment (no -ss slicing)
                ginfo = group_info[gi]
                group_audio = ginfo["audio_path"]
                group_clip = ginfo["clip_path"]
                group_dur = ginfo["total_dur"]
                out_path = str(tmp_dir / f"seg_{seg_idx:03d}.mp4")
                seg_idx += 1
                # group_dur = actual concat audio duration (includes pad silence)
                # Slow down clip to match audio duration via setpts (correct direction: >1 = slower)
                fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, group_dur-0.05):.2f}:d=0.05"
                if group_clip and os.path.exists(group_clip) and group_audio and os.path.exists(group_audio):
                    vid_dur = _get_duration(group_clip)
                    if vid_dur > 0 and group_dur > 0 and abs(vid_dur - group_dur) > 0.01:
                        # CORRECT: group_dur/vid_dur (>1 = slow down, <1 = speed up)
                        # e.g. clip=13s, audio=14.3s → 14.3/13=1.0898 → PTS stretched → slow down
                        # fps=24 then duplicates frames evenly throughout (no freeze)
                        vf = f"setpts={group_dur/vid_dur:.4f}*PTS,fps=24"
                    else:
                        vf = "fps=24"
                    cmd = ["ffmpeg", "-y", "-i", group_clip, "-i", group_audio,
                           "-t", f"{group_dur:.3f}", "-vf", vf,
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           "-af", fade_af,
                           out_path]
                else:
                    cmd = ["ffmpeg", "-y", "-i", group_clip,
                           "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                           "-t", f"{group_dur:.3f}", "-vf", "fps=24",
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           out_path]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
                    print(f"  FFmpeg error group seg {gi}: {r.stderr[-200:]}")
                    # Fallback: silent segment with scene image to maintain timeline
                    fallback_cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                                   "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                                   "-t", f"{group_dur:.3f}", "-vf", "fps=24",
                                   "-map", "0:v:0", "-map", "1:a:0",
                                   "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                                   out_path]
                    r2 = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=60)
                    if r2.returncode != 0:
                        print(f"  Fallback also failed: {r2.stderr[-200:]}")
                        continue
                segments.append(out_path)
                _cb(int(seg_idx / (total_segs - skipped_segs) * 80), f"  Segment {seg_idx}/{total_segs - skipped_segs} (group {gi})")
                continue
        d_idx = seg.get("dialogue_idx", -1)
        out_path = str(tmp_dir / f"seg_{seg_idx:03d}.mp4")
        seg_idx += 1

        # Determine audio file and audio_dur
        audio_file = None
        audio_dur = seg.get("audio_dur", duration - pad)

        if seg_type == "title_card":
            audio_file = None
            audio_dur = duration
        elif seg_type == "practice_intro":
            audio_file = narration.get("practice_intro")
            if audio_file and os.path.exists(audio_file):
                audio_dur = _get_duration(audio_file)
            else:
                audio_dur = duration - pad
        elif seg_type == "dialogue":
            audio_file = normal_paths[audio_idx] if audio_idx < len(normal_paths) else None
            audio_dur = seg.get("audio_dur", duration - pad)
        elif seg_type == "listen_en":
            audio_file = normal_paths[audio_idx] if audio_idx < len(normal_paths) else None
            audio_dur = seg.get("audio_dur", duration - pad)
        elif seg_type == "listen_zh":
            audio_file = zh_paths[audio_idx] if audio_idx < len(zh_paths) and zh_paths[audio_idx] else None
            audio_dur = seg.get("audio_dur", duration - pad)
        elif seg_type == "outro":
            audio_file = narration.get("outro")
            audio_dur = seg.get("audio_dur", duration - pad)

        # Determine video source
        if seg_type in ("listen_en", "listen_zh", "practice"):
            video_src = str(static_dir / f"zh_{d_idx}.png") if d_idx >= 0 else scene_img
            if not os.path.exists(video_src):
                video_src = scene_img
            is_static = True
        elif seg_type == "title_card":
            video_src = HOOK_CLIP if HOOK_CLIP and os.path.exists(HOOK_CLIP) else scene_img
            is_static = False
        elif seg_type == "practice_intro":
            video_src = clip_paths[0] if clip_paths else (HOOK_CLIP or scene_img)
            is_static = False
        elif seg_type == "outro":
            video_src = HOOK_CLIP if HOOK_CLIP and os.path.exists(HOOK_CLIP) else scene_img
            is_static = False
        else:
            idx = min(audio_idx, len(DIALOGUE_CLIPS)-1) if DIALOGUE_CLIPS else 0
            video_src = DIALOGUE_CLIPS[idx] if DIALOGUE_CLIPS else (HOOK_CLIP or scene_img)
            is_static = False

        fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05"

        # --- Build FFmpeg command ---
        if is_static:
            # Static image: loop image + audio
            if audio_file and os.path.exists(audio_file):
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", video_src, "-i", audio_file,
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", video_src,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        elif seg_type == "title_card":
            # Title card: video clip + title overlay
            title_en = seg.get("subtitle_en", "")
            title_zh = seg.get("subtitle_zh", "")
            title_overlay = str(static_dir / "title_overlay.png")
            _render_title_card(title_en, title_zh, "", scene_img, title_overlay)
            vid_dur = _get_duration(video_src) if os.path.exists(video_src) else 0
            vf = f"setpts={duration/vid_dur:.4f}*PTS,fps=24" if vid_dur > 0 else "fps=24"
            fade_af_title = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, duration-0.05):.2f}:d=0.05"
            if _has_audio(video_src):
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", title_overlay,
                       "-t", f"{duration:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "0:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       "-af", fade_af_title,
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", title_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        elif seg_type == "practice_intro":
            # Practice intro: video clip + text overlay + narration ONLY (no video audio)
            intro_en = seg.get("subtitle_en", "")
            intro_zh = seg.get("subtitle_zh", "")
            intro_overlay = str(static_dir / "practice_intro_overlay.png")
            _render_practice_intro(intro_en, intro_zh, scene_img, intro_overlay)
            vid_dur = _get_duration(video_src) if os.path.exists(video_src) else 0
            vf = f"setpts={audio_dur/vid_dur:.4f}*PTS,fps=24" if vid_dur > 0 and audio_dur > 0 else "fps=24"
            out_dur = audio_dur + pad
            fade_af_pi = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05"
            narration_audio = narration.get("practice_intro")
            if narration_audio and os.path.exists(narration_audio):
                # Use narration audio ONLY (discard video's audio entirely)
                # Audio filters go INSIDE filter_complex (not -af, which conflicts with filter_complex)
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", narration_audio, "-i", intro_overlay,
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][2:v]overlay=0:0[v];[1:a]afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05,apad=whole_dur={out_dur:.3f}[a]",
                       "-map", "[v]", "-map", "[a]",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            else:
                # No narration: use silence (no video audio)
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", intro_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        elif seg_type == "outro":
            # Outro: video clip + text overlay + narration ONLY (no video audio)
            outro_en = seg.get("subtitle_en", "")
            outro_zh = seg.get("subtitle_zh", "")
            outro_overlay = str(static_dir / "outro_overlay.png")
            _render_practice_intro(outro_en, outro_zh, scene_img, outro_overlay)
            vid_dur = _get_duration(video_src) if os.path.exists(video_src) else 0
            out_dur = audio_dur + pad
            vf = f"setpts={audio_dur/vid_dur:.4f}*PTS,fps=24" if vid_dur > 0 and audio_dur > 0 else "fps=24"
            fade_af_outro = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05"
            outro_audio = narration.get("outro")
            if outro_audio and os.path.exists(outro_audio):
                # Use narration audio ONLY (discard video's audio entirely)
                # Audio filters go INSIDE filter_complex (not -af, which conflicts with filter_complex)
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", outro_audio, "-i", outro_overlay,
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][2:v]overlay=0:0[v];[1:a]afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05,apad=whole_dur={out_dur:.3f}[a]",
                       "-map", "[v]", "-map", "[a]",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            else:
                # No narration: use silence (no video audio)
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", outro_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        else:
            # Dialogue: video clip + audio
            # 方案 B: grouped dialogue handled above (group_info+line_to_group).
            # This else branch is the no-grouping fallback (uses whole clip per line).
            ginfo = None  # group_clip_map no longer used — grouped dialogue handled above
            if ginfo and ginfo.get("clip_path") and os.path.exists(ginfo["clip_path"]):
                # Use group clip with accurate seek (-ss AFTER -i for frame-accurate slicing)
                video_src = ginfo["clip_path"]
                clip_start = ginfo["clip_start"]
                clip_segment_dur = ginfo["clip_segment_dur"]
                vf = "fps=24"
                if audio_file and os.path.exists(audio_file):
                    # -ss after -i = accurate seek (slower but no keyframe alignment issues)
                    cmd = ["ffmpeg", "-y",
                           "-i", video_src, "-i", audio_file,
                           "-ss", f"{clip_start:.3f}", "-t", f"{clip_segment_dur:.3f}",
                           "-vf", vf,
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-t", f"{duration:.3f}",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                           out_path]
                else:
                    cmd = ["ffmpeg", "-y",
                           "-i", video_src,
                           "-ss", f"{clip_start:.3f}", "-t", f"{clip_segment_dur:.3f}",
                           "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                           "-vf", vf,
                           "-map", "0:v:0", "-map", "2:a",
                           "-t", f"{duration:.3f}",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           out_path]
            else:
                # Fallback: whole clip behavior (no grouping)
                vid_dur = _get_duration(video_src) if os.path.exists(video_src) else 0
                if vid_dur > 0 and audio_dur > 0:
                    vf = f"setpts={audio_dur/vid_dur:.4f}*PTS,fps=24"
                else:
                    vf = "fps=24"
                if audio_file and os.path.exists(audio_file):
                    cmd = ["ffmpeg", "-y", "-i", video_src, "-i", audio_file,
                           "-t", f"{duration:.3f}", "-vf", vf,
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                           out_path]
                else:
                    cmd = ["ffmpeg", "-y", "-i", video_src,
                           "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                           "-t", f"{duration:.3f}", "-vf", vf,
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           out_path]

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            print(f"  FFmpeg error seg {seg_idx}: {r.stderr[-200:]}")
            # Fallback: silent segment with scene image
            fallback_cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                           "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                           "-t", f"{duration:.3f}", "-vf", "fps=24",
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           out_path]
            r2 = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=60)
            if r2.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
                print(f"  Fallback also failed, skipping segment: {r2.stderr[-200:]}")
                continue
        segments.append(out_path)
        _cb(int(seg_idx / (total_segs - skipped_segs) * 80),
            f"  Segment {seg_idx}/{total_segs - skipped_segs} ({seg_type})")

    # --- Concat all segments ---
    _cb(80, "Concatenating segments...")
    concat_list = tmp_dir / "concat.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"file '{s}'\n")

    no_sub = str(vid_dir / "final_no_sub.mp4")
    # Use -c copy (all segments already have uniform format: libx264/yuv420p/24fps/aac/44100Hz/stereo)
    result = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", no_sub,
    ], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Concat failed: {result.stderr.decode()[-2000:]}")

    # --- Burn subtitles via Pillow overlay ---
    _cb(90, "Burning subtitles (Pillow overlay)...")
    final_path = str(vid_dir / "final_video.mp4")

    # Extract dialogue subtitle entries from timeline
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

    # Render subtitle PNGs and overlay
    if subtitle_entries:
        from PIL import Image, ImageDraw, ImageFont
        w, h = 1280, 720
        sub_overlay_dir = tmp_dir / "subtitles"
        sub_overlay_dir.mkdir(exist_ok=True)

        for i, entry in enumerate(subtitle_entries):
            overlay_path = str(sub_overlay_dir / f"sub_{i:03d}.png")
            bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(bg)

            en_text = entry["en"]
            zh_text = entry["zh"]

            en_y = 0
            en_h = 0
            if en_text:
                en_size = 50
                en_font = ImageFont.truetype(FONT_EN, en_size)
                while en_size > 28:
                    bbox = draw.textbbox((0, 0), en_text, font=en_font)
                    if bbox[2]-bbox[0] <= w-80:
                        break
                    en_size -= 2
                    en_font = ImageFont.truetype(FONT_EN, en_size)
                bbox = draw.textbbox((0, 0), en_text, font=en_font)
                en_w = bbox[2]-bbox[0]
                en_h = bbox[3]-bbox[1]
                en_y = h - 140
                draw.text(((w-en_w)//2, en_y), en_text, font=en_font,
                          fill=(255, 255, 255, 255), stroke_width=5, stroke_fill=(0, 0, 0, 255))

            if zh_text:
                zh_size = 50
                zh_font = ImageFont.truetype(FONT_ZH, zh_size)
                while zh_size > 24:
                    bbox = draw.textbbox((0, 0), zh_text, font=zh_font)
                    if bbox[2]-bbox[0] <= w-80:
                        break
                    zh_size -= 2
                    zh_font = ImageFont.truetype(FONT_ZH, zh_size)
                bbox = draw.textbbox((0, 0), zh_text, font=zh_font)
                zh_w = bbox[2]-bbox[0]
                zh_y = en_y + en_h + 15 if en_text else h - 80
                draw.text(((w-zh_w)//2, zh_y), zh_text, font=zh_font,
                          fill=(255, 215, 0, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))

            bg.save(overlay_path, "PNG")
            entry["overlay_path"] = overlay_path

        # Build FFmpeg overlay filter
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
        subprocess.run(cmd, check=True, capture_output=True, cwd=str(srt_dir))
    else:
        shutil.copy2(no_sub, final_path)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Final loudnorm pass: normalize the entire video to consistent volume
    # Use -c:v copy (video passthrough, only re-encode audio) for speed
    _cb(95, "Final loudnorm pass (normalize volume)...")
    norm_path = str(vid_dir / "final_video_norm.mp4")
    norm_result = subprocess.run(
        ["ffmpeg", "-y", "-i", final_path,
         "-c:v", "copy",  # video passthrough — fast, no re-encode
         "-c:a", "aac", "-b:a", "128k",
         "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
         norm_path],
        capture_output=True, timeout=600,
    )
    if norm_result.returncode == 0 and os.path.exists(norm_path) and os.path.getsize(norm_path) > 1000:
        os.replace(norm_path, final_path)
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        _cb(100, f"Listening video done: {final_path} ({size_mb:.1f}MB)")
    else:
        # Fallback: simple volume boost with video passthrough
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
            pass  # Keep original loudnorm-free video if both fail
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        _cb(100, f"Listening video done: {final_path} ({size_mb:.1f}MB)")

    return final_path
