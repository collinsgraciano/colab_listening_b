"""Enhanced FFmpeg video composition for 7-chapter listening practice videos.

Adds:
- Vocabulary frame rendering (word + IPA + 繁中 + example)
- Quiz frame rendering (question + 4 options + answer reveal)
- Slow-speed dialogue handling (video speed-adjusted to slow audio)
- 7-segment shadowing (down from 9)

Reuses all render functions from parent video_compose.py.
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
    _get_duration, _has_audio, _render_static_frame,
    _render_title_card, _render_practice_intro,
    FONT_EN, FONT_ZH, FONT_PH,
)


def _render_vocab_frame(word, phonetic, zh, example, scene_img_path,
                         out_path, idx, total, w=1280, h=720):
    """Render a vocabulary card PNG: word + IPA + 繁中 + example sentence."""
    from PIL import Image, ImageDraw, ImageFont

    bg = Image.open(scene_img_path).convert("RGBA").resize((w, h))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    # Semi-transparent dark box in center
    box_h = 400
    box_y = (h - box_h) // 2
    ov_draw.rectangle([80, box_y, w - 80, box_y + box_h], fill=(0, 0, 0, 180))
    bg = Image.alpha_composite(bg, overlay)
    draw = ImageDraw.Draw(bg)

    # Number badge (top-left of box)
    num_font = ImageFont.truetype(FONT_EN, 28)
    draw.text((100, box_y + 15), f"Vocabulary {idx+1}/{total}", font=num_font,
              fill=(255, 220, 0, 255))

    # Word (large, white)
    word_font = ImageFont.truetype(FONT_EN, 72)
    wb = draw.textbbox((0, 0), word, font=word_font)
    ww, wh = wb[2] - wb[0], wb[3] - wb[1]
    word_y = box_y + 60
    draw.text(((w - ww) // 2, word_y), word, font=word_font,
              fill=(255, 255, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))

    # Phonetic (medium, light blue)
    cur_y = word_y + wh + 10
    if phonetic:
        ph_font = ImageFont.truetype(FONT_PH, 36)
        pb = draw.textbbox((0, 0), phonetic, font=ph_font)
        pw = pb[2] - pb[0]
        draw.text(((w - pw) // 2, cur_y), phonetic, font=ph_font,
                  fill=(130, 200, 255, 255))
        cur_y += pb[3] - pb[1] + 10

    # Chinese meaning (large, gold)
    if zh:
        zh_font = ImageFont.truetype(FONT_ZH, 48)
        zb = draw.textbbox((0, 0), zh, font=zh_font)
        zw = zb[2] - zb[0]
        draw.text(((w - zw) // 2, cur_y), zh, font=zh_font,
                  fill=(255, 220, 0, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
        cur_y += zb[3] - zb[1] + 15

    # Example sentence (smaller, light gray, word-wrapped)
    if example:
        ex_font = ImageFont.truetype(FONT_EN, 28)
        max_w = w - 200
        words_list = example.split()
        lines = []
        cur_line = ""
        for word_t in words_list:
            test = (cur_line + " " + word_t).strip()
            tb = draw.textbbox((0, 0), test, font=ex_font)
            if tb[2] - tb[0] <= max_w or not cur_line:
                cur_line = test
            else:
                lines.append(cur_line)
                cur_line = word_t
        if cur_line:
            lines.append(cur_line)

        for line in lines:
            lb = draw.textbbox((0, 0), line, font=ex_font)
            lw = lb[2] - lb[0]
            draw.text(((w - lw) // 2, cur_y), line, font=ex_font,
                      fill=(200, 200, 200, 255))
            cur_y += lb[3] - lb[1] + 5

    bg.convert("RGB").save(out_path, "PNG")


def _render_quiz_frame(question, options, answer, scene_img_path,
                        out_path, is_answer, idx, total, w=1280, h=720):
    """Render a comprehension quiz card PNG.

    is_answer=False: show question + 4 options (all white)
    is_answer=True:  show question + 4 options with correct option highlighted green
    """
    from PIL import Image, ImageDraw, ImageFont

    bg = Image.open(scene_img_path).convert("RGBA").resize((w, h))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    box_h = 520
    box_y = (h - box_h) // 2
    ov_draw.rectangle([80, box_y, w - 80, box_y + box_h], fill=(0, 0, 0, 180))
    bg = Image.alpha_composite(bg, overlay)
    draw = ImageDraw.Draw(bg)

    # Header
    hdr_font = ImageFont.truetype(FONT_EN, 28)
    draw.text((100, box_y + 15), f"Question {idx+1}/{total}", font=hdr_font,
              fill=(255, 220, 0, 255))

    # Question (wrapped, white)
    q_font = ImageFont.truetype(FONT_EN, 36)
    max_w = w - 200
    words_list = question.split()
    lines = []
    cur_line = ""
    for word_t in words_list:
        test = (cur_line + " " + word_t).strip()
        tb = draw.textbbox((0, 0), test, font=q_font)
        if tb[2] - tb[0] <= max_w or not cur_line:
            cur_line = test
        else:
            lines.append(cur_line)
            cur_line = word_t
    if cur_line:
        lines.append(cur_line)

    cur_y = box_y + 60
    for line in lines:
        lb = draw.textbbox((0, 0), line, font=q_font)
        lw = lb[2] - lb[0]
        draw.text(((w - lw) // 2, cur_y), line, font=q_font,
                  fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
        cur_y += lb[3] - lb[1] + 5

    cur_y += 15

    # Options (A/B/C/D)
    opt_font = ImageFont.truetype(FONT_EN, 32)
    for oi, opt in enumerate(options):
        is_correct = is_answer and opt.startswith(answer)
        color = (100, 255, 100, 255) if is_correct else (255, 255, 255, 255)
        ob = draw.textbbox((0, 0), opt, font=opt_font)
        ow = ob[2] - ob[0]
        draw.text(((w - ow) // 2, cur_y), opt, font=opt_font,
                  fill=color, stroke_width=2, stroke_fill=(0, 0, 0, 255))
        if is_correct:
            # Green checkmark indicator
            draw.text((120, cur_y), ">>", font=opt_font, fill=(100, 255, 100, 255))
        cur_y += ob[3] - ob[1] + 12

    bg.convert("RGB").save(out_path, "PNG")


def compose_listening_enhanced(
    work_dir: str,
    clip_paths: list[str],
    timeline: list[dict],
    script: dict,
    narration: dict,
    normal_paths: list[str],
    zh_paths: list[str],
    slow_paths: list[str],
    vocab_paths: list[str],
    quiz_paths: list[str],
    scene_img: str,
    srt_dir: str,
    pad: float = 0.4,
    progress_cb=None,
    group_info: list[dict] | None = None,
    line_to_group: dict | None = None,
) -> str:
    """Compose enhanced 7-chapter listening practice video.

    Args:
        slow_paths: slow-speed (75%) dialogue audio paths.
        vocab_paths: vocabulary word+example TTS audio paths.
        quiz_paths: comprehension question TTS audio paths.
        See compose_listening for other parameters.
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
    vocabulary = script.get("vocabulary", [])
    questions = script.get("comprehension_questions", [])
    n = len(dialogue)
    HOOK_CLIP = clip_paths[0] if clip_paths else None
    DIALOGUE_CLIPS = clip_paths[1:] if len(clip_paths) > 1 else []

    # --- Render static frames for Ch1 (vocab), Ch5 (quiz), Ch6 (shadowing) ---
    if os.path.exists(scene_img):
        _cb(5, f"Rendering static frames...")
        # Vocabulary frames
        for vi, vocab in enumerate(vocabulary):
            p = str(static_dir / f"vocab_{vi}.png")
            _render_vocab_frame(
                vocab.get("word", ""), vocab.get("phonetic", ""),
                vocab.get("zh", ""), vocab.get("example", ""),
                scene_img, p, vi, len(vocabulary))
        # Quiz frames (question + answer for each)
        for qi, quiz in enumerate(questions):
            p = str(static_dir / f"quiz_q_{qi}.png")
            _render_quiz_frame(
                quiz.get("question", ""), quiz.get("options", []),
                quiz.get("answer", ""), scene_img, p,
                is_answer=False, idx=qi, total=len(questions))
            p = str(static_dir / f"quiz_a_{qi}.png")
            _render_quiz_frame(
                quiz.get("question", ""), quiz.get("options", []),
                quiz.get("answer", ""), scene_img, p,
                is_answer=True, idx=qi, total=len(questions))
        # Shadowing frames (same as original)
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
    processed_groups = set()
    skipped_segs = 0

    for seg in timeline:
        seg_type = seg["type"]
        duration = seg["duration"]
        audio_idx = seg.get("audio_index", 0)

        # Group-based dialogue (Ch3): skip lines in already-processed groups
        if seg_type == "dialogue" and group_info and line_to_group:
            gi = line_to_group.get(audio_idx)
            if gi is not None and gi in processed_groups:
                skipped_segs += 1
                continue
            if gi is not None:
                processed_groups.add(gi)
                ginfo = group_info[gi]
                group_audio = ginfo["audio_path"]
                group_clip = ginfo["clip_path"]
                group_dur = ginfo["total_dur"]
                out_path = str(tmp_dir / f"seg_{seg_idx:03d}.mp4")
                seg_idx += 1
                fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, group_dur-0.05):.2f}:d=0.05"
                if group_clip and os.path.exists(group_clip) and group_audio and os.path.exists(group_audio):
                    vid_dur = _get_duration(group_clip)
                    if vid_dur > 0 and group_dur > 0 and abs(vid_dur - group_dur) > 0.01:
                        vf = f"setpts={group_dur/vid_dur:.4f}*PTS,fps=24"
                    else:
                        vf = "fps=24"
                    cmd = ["ffmpeg", "-y", "-i", group_clip, "-i", group_audio,
                           "-t", f"{group_dur:.3f}", "-vf", vf,
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           "-af", fade_af, out_path]
                else:
                    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                           "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                           "-t", f"{group_dur:.3f}", "-vf", "fps=24",
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           out_path]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
                    print(f"  FFmpeg error group seg {gi}: {r.stderr[-200:]}")
                    fb_cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                              "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                              "-t", f"{group_dur:.3f}", "-vf", "fps=24",
                              "-map", "0:v:0", "-map", "1:a:0",
                              "-c:v", "libx264", "-pix_fmt", "yuv420p",
                              "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                              out_path]
                    r2 = subprocess.run(fb_cmd, capture_output=True, text=True, timeout=60)
                    if r2.returncode != 0:
                        continue
                segments.append(out_path)
                eff_total = total_segs - skipped_segs
                _cb(int(seg_idx / max(eff_total, 1) * 80), f"  Segment {seg_idx}/{eff_total} (group {gi})")
                continue

        # Ch4: Slow-speed dialogue — uses slow audio + same group clip (re-speeded)
        if seg_type == "dialogue_slow" and group_info and line_to_group:
            gi = line_to_group.get(audio_idx)
            if gi is not None:
                # Group all slow dialogue lines together (same grouping as normal)
                if gi in processed_groups:
                    skipped_segs += 1
                    continue
                processed_groups.add(gi)
                # Re-use group clip but with slow audio
                # We need to build slow group audio from slow_paths
                ginfo = group_info[gi]
                group_clip = ginfo["clip_path"]
                group_lines = ginfo["lines"]
                slow_lines_audio = [slow_paths[li] for li in group_lines
                                    if li < len(slow_paths) and os.path.exists(slow_paths[li])]
                if not slow_lines_audio:
                    skipped_segs += 1
                    continue

                # Build slow group audio (concat with pad)
                slow_group_path = str(tmp_dir / f"slow_group_{gi}.mp3")
                if len(slow_lines_audio) == 1:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", slow_lines_audio[0],
                         "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                         slow_group_path],
                        capture_output=True, timeout=30)
                else:
                    inputs = []
                    for la in slow_lines_audio:
                        inputs.extend(["-i", la])
                    filter_parts = []
                    for j in range(len(slow_lines_audio)):
                        line_idx = group_lines[j]
                        # Use slow_durations for padding
                        from pipeline import _get_audio_duration as _gad
                        slow_dur = _gad(slow_lines_audio[j]) if j < len(slow_lines_audio) else 4.0
                        pad_dur = slow_dur + pad
                        filter_parts.append(f"[{j}:a]apad=whole_dur={pad_dur:.3f}[a{j}]")
                    concat_inputs = "".join(f"[a{j}]" for j in range(len(slow_lines_audio)))
                    filter_parts.append(f"{concat_inputs}concat=n={len(slow_lines_audio)}:v=0:a=1[a]")
                    filter_complex = ";".join(filter_parts)
                    subprocess.run(
                        ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex,
                         "-map", "[a]",
                         "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                         slow_group_path],
                        capture_output=True, timeout=60)

                if not os.path.exists(slow_group_path) or os.path.getsize(slow_group_path) < 1000:
                    skipped_segs += 1
                    continue

                group_dur = _get_duration(slow_group_path)
                out_path = str(tmp_dir / f"seg_{seg_idx:03d}.mp4")
                seg_idx += 1
                fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, group_dur-0.05):.2f}:d=0.05"
                if group_clip and os.path.exists(group_clip):
                    vid_dur = _get_duration(group_clip)
                    if vid_dur > 0 and group_dur > 0:
                        vf = f"setpts={group_dur/vid_dur:.4f}*PTS,fps=24"
                    else:
                        vf = "fps=24"
                    cmd = ["ffmpeg", "-y", "-i", group_clip, "-i", slow_group_path,
                           "-t", f"{group_dur:.3f}", "-vf", vf,
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           "-af", fade_af, out_path]
                else:
                    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                           "-i", slow_group_path,
                           "-t", f"{group_dur:.3f}", "-vf", "fps=24",
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           out_path]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                    segments.append(out_path)
                else:
                    print(f"  FFmpeg error slow group {gi}: {r.stderr[-200:]}")
                eff_total = total_segs - skipped_segs
                _cb(int(seg_idx / max(eff_total, 1) * 80), f"  Segment {seg_idx}/{eff_total} (slow group {gi})")
                continue

        d_idx = seg.get("dialogue_idx", -1)
        out_path = str(tmp_dir / f"seg_{seg_idx:03d}.mp4")
        seg_idx += 1

        # Determine audio file and audio_dur
        audio_file = None
        audio_dur = seg.get("audio_dur", duration - pad)

        if seg_type == "vocab":
            audio_file = vocab_paths[audio_idx] if audio_idx < len(vocab_paths) else None
            audio_dur = seg.get("audio_dur", duration - pad)
        elif seg_type == "quiz":
            audio_file = quiz_paths[audio_idx] if audio_idx < len(quiz_paths) else None
            audio_dur = seg.get("audio_dur", duration - pad)
        elif seg_type == "title_card":
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
        if seg_type == "vocab":
            vi = seg.get("vocab_idx", d_idx)
            video_src = str(static_dir / f"vocab_{vi}.png") if vi >= 0 else scene_img
            if not os.path.exists(video_src):
                video_src = scene_img
            is_static = True
        elif seg_type == "quiz":
            qi = seg.get("quiz_idx", d_idx)
            # Use answer frame if audio_dur > 50% through (rough heuristic)
            # Actually: use question frame always; answer is shown via a second segment
            video_src = str(static_dir / f"quiz_q_{qi}.png") if qi >= 0 else scene_img
            if not os.path.exists(video_src):
                video_src = scene_img
            is_static = True
        elif seg_type in ("listen_en", "listen_zh", "practice"):
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
            idx = min(audio_idx, len(DIALOGUE_CLIPS) - 1) if DIALOGUE_CLIPS else 0
            video_src = DIALOGUE_CLIPS[idx] if DIALOGUE_CLIPS else (HOOK_CLIP or scene_img)
            is_static = False

        fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05"

        # --- Build FFmpeg command ---
        if is_static:
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
                       "-af", fade_af_title, out_path]
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
            intro_en = seg.get("subtitle_en", "")
            intro_zh = seg.get("subtitle_zh", "")
            intro_overlay = str(static_dir / "practice_intro_overlay.png")
            _render_practice_intro(intro_en, intro_zh, scene_img, intro_overlay)
            vid_dur = _get_duration(video_src) if os.path.exists(video_src) else 0
            vf = f"setpts={audio_dur/vid_dur:.4f}*PTS,fps=24" if vid_dur > 0 and audio_dur > 0 else "fps=24"
            out_dur = audio_dur + pad
            narration_audio = narration.get("practice_intro")
            if narration_audio and os.path.exists(narration_audio):
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", narration_audio, "-i", intro_overlay,
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][2:v]overlay=0:0[v];[1:a]afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05,apad=whole_dur={out_dur:.3f}[a]",
                       "-map", "[v]", "-map", "[a]",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", intro_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        elif seg_type == "outro":
            outro_en = seg.get("subtitle_en", "")
            outro_zh = seg.get("subtitle_zh", "")
            outro_overlay = str(static_dir / "outro_overlay.png")
            _render_practice_intro(outro_en, outro_zh, scene_img, outro_overlay)
            vid_dur = _get_duration(video_src) if os.path.exists(video_src) else 0
            out_dur = audio_dur + pad
            vf = f"setpts={audio_dur/vid_dur:.4f}*PTS,fps=24" if vid_dur > 0 and audio_dur > 0 else "fps=24"
            outro_audio = narration.get("outro")
            if outro_audio and os.path.exists(outro_audio):
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", outro_audio, "-i", outro_overlay,
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][2:v]overlay=0:0[v];[1:a]afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05,apad=whole_dur={out_dur:.3f}[a]",
                       "-map", "[v]", "-map", "[a]",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", outro_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        else:
            # Fallback: dialogue (non-grouped)
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
            fb_cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                      "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                      "-t", f"{duration:.3f}", "-vf", "fps=24",
                      "-map", "0:v:0", "-map", "1:a:0",
                      "-c:v", "libx264", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                      out_path]
            r2 = subprocess.run(fb_cmd, capture_output=True, text=True, timeout=60)
            if r2.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
                print(f"  Fallback also failed, skipping: {r2.stderr[-200:]}")
                continue
        segments.append(out_path)
        eff_total = total_segs - skipped_segs
        _cb(int(seg_idx / max(eff_total, 1) * 80),
            f"  Segment {seg_idx}/{eff_total} ({seg_type})")

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

    # --- Burn subtitles via Pillow overlay ---
    _cb(90, "Burning subtitles (Pillow overlay)...")
    import re as _re
    _yt = script.get("youtube_title", script.get("title", "final_video"))
    _safe = _re.sub(r'[\U0001F000-\U0001FFFF]', '', _yt)
    _safe = _re.sub(r'[\\/:*?"<>|]', '', _safe).strip()
    _safe = _re.sub(r'\s+', '_', _safe)[:80] or "final_video"
    final_path = str(vid_dir / f"{_safe}.mp4")

    subtitle_entries = []
    t_cursor = 0.0
    for seg in timeline:
        dur = seg["duration"]
        seg_type = seg.get("type", "")
        if seg_type in ("dialogue", "dialogue_slow"):
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
                    if bbox[2] - bbox[0] <= w - 80:
                        break
                    en_size -= 2
                    en_font = ImageFont.truetype(FONT_EN, en_size)
                bbox = draw.textbbox((0, 0), en_text, font=en_font)
                en_w = bbox[2] - bbox[0]
                en_h = bbox[3] - bbox[1]
                en_y = h - 140
                draw.text(((w - en_w) // 2, en_y), en_text, font=en_font,
                          fill=(255, 255, 255, 255), stroke_width=5, stroke_fill=(0, 0, 0, 255))

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
                zh_w = bbox[2] - bbox[0]
                zh_y = en_y + en_h + 15 if en_text else h - 80
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
        subprocess.run(cmd, check=True, capture_output=True, cwd=str(srt_dir))
    else:
        shutil.copy2(no_sub, final_path)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Final loudnorm pass
    _cb(95, "Final loudnorm pass...")
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
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                 "-af", "volume=6dB", vol_path],
                capture_output=True, timeout=600,
            )
            if os.path.exists(vol_path) and os.path.getsize(vol_path) > 1000:
                os.replace(vol_path, final_path)
        except Exception:
            pass

    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    _cb(100, f"Enhanced video done: {final_path} ({size_mb:.1f}MB)")
    return final_path
