"""YouTube thumbnail generator — generates thumbnail with text baked into the AI prompt.

Instead of generating a background then overlaying text via Pillow, this module
puts the title text, level badge, and Chinese subtitle directly into the generate_image
prompt so the AI renders everything in one step.

Fallback: if AI image gen fails, falls back to Pillow text overlay on scene image.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

_PARENT = str(Path(__file__).parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from video_compose import FONT_EN, FONT_ZH

# YouTube thumbnail specs
THUMB_W = 1280
THUMB_H = 720


def _build_thumbnail_prompt(script: dict, structure: str) -> str:
    """Build a prompt that generates a YouTube thumbnail in the reference style.

    Reference style elements:
    - Top-left orange banner: "A1-A2 LEVEL"
    - Top-right green icon: "中英對照"
    - Top center: "沉浸式聽力動畫" (listening) or "沉浸式英文動畫" (original)
    - Main scene: 3D Pixar characters + background + props
    - Large text below characters: Traditional Chinese title (e.g. "在藥房")
    - Smaller text below: e.g. "18句聽力練習"
    - Bottom row of circular icons with bilingual text (scene keywords)
    """
    title_zh = script.get("title_zh", script.get("intro_zh", ""))
    cefr = script.get("cefr", "A2")
    char_a_desc = script.get("char_a_description", "friendly young person")
    char_b_desc = script.get("char_b_description", "friendly young person")
    scene_zh = script.get("scene_zh", script.get("title", "everyday life"))
    scene_en = script.get("scene", script.get("title", "everyday life"))

    # Character expression & action from new LLM fields (fallback to defaults)
    expression = script.get("thumbnail_expression", "surprised and excited")
    action = script.get("thumbnail_action", "looking toward the camera and gesturing naturally")

    # Subtitle text
    subtitle = script.get("thumbnail_subtitle", "18句聽力練習")

    # Bottom icons from new LLM field (fallback to generic)
    icons = script.get("thumbnail_icons", [
        {"en": "Dialogue", "zh": "會話"},
        {"en": "Listening", "zh": "聽力"},
        {"en": "Shadowing", "zh": "跟讀"},
        {"en": "Practice", "zh": "練習"},
    ])
    icon_lines = "  ".join(f"{i['zh']} {i['en']}" for i in icons[:5])

    if structure == "enhanced":
        top_center = "沉浸式聽力動畫"
    else:
        top_center = "沉浸式英文動畫"

    return f"""A highly complex 3D Pixar-style YouTube thumbnail for {scene_en} English listening practice, complete with an orange banner at the top left reading "{cefr} LEVEL" and a green icon at the top right with the text "中英對照". At the top center, the text "{top_center}" is integrated.

The main scene features a detailed view of {scene_en} with {char_a_desc} and {char_b_desc}, both with a {expression} expression, {action}. The background shows a detailed {scene_en} setting with relevant props and environment.

The large text below the characters reads "{title_zh}" in bold yellow font with black outline, and below that, smaller text reads "{subtitle}".

At the very bottom, a precise row of circular icons is rendered with legible text associated: {icon_lines}.

Clean legible text, bright studio lighting, vibrant colors, highly detailed, professional composition, 3D animated movie style, Pixar quality rendering, soft shadows, subsurface scattering, cinematic lighting."""


def generate_thumbnail(script: dict, scene_img: str, output_path: str,
                        mcp_call_tool=None, mcp_parse_task_id=None,
                        mcp_poll_task=None, mcp_download_file=None,
                        structure: str = "original") -> str:
    """Generate a YouTube thumbnail with text baked into the AI prompt (one step).

    Falls back to Pillow text overlay on scene_img if AI generation fails.
    """
    prompt = _build_thumbnail_prompt(script, structure)

    # Try AI generation with text baked in
    if mcp_call_tool and mcp_parse_task_id and mcp_poll_task and mcp_download_file:
        print("  [Thumbnail] Generating thumbnail with baked-in text via MCP...")
        try:
            result = mcp_call_tool("generate_image", {
                "prompt": prompt,
                "provider": "frontier",
                "quality": "high",
                "image_size": '{"width": 1280, "height": 720}',
                "output_format": "jpeg",
            })
            task_id = mcp_parse_task_id(result)
            if task_id:
                data = mcp_poll_task(task_id, interval=10)
                url = data.get("url", "")
                if url and mcp_download_file(url, output_path):
                    size_kb = os.path.getsize(output_path) // 1024
                    if size_kb > 10:
                        print(f"  [Thumbnail] Saved (AI baked-in): {output_path} ({size_kb}KB)")
                        return output_path
                    else:
                        print(f"  [Thumbnail] AI image too small ({size_kb}KB), falling back")
                else:
                    print("  [Thumbnail] AI generation returned no URL, falling back to Pillow")
        except Exception as e:
            print(f"  [Thumbnail] AI generation failed: {e}, falling back to Pillow")

    # Fallback: Pillow text overlay on scene image
    print("  [Thumbnail] Using Pillow fallback (text overlay on scene image)...")
    return _pillow_fallback(script, scene_img, output_path, structure)


def _pillow_fallback(script: dict, scene_img: str, output_path: str,
                      structure: str) -> str:
    """Pillow fallback: overlay text on scene image."""
    from PIL import Image, ImageDraw, ImageFont

    if not os.path.exists(scene_img):
        print(f"  [Thumbnail] ERROR: No scene image at {scene_img}")
        return None

    bg = Image.open(scene_img).convert("RGBA").resize((THUMB_W, THUMB_H))

    # Dark gradient on right side
    overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    for x in range(THUMB_W // 2, THUMB_W):
        alpha = int((x - THUMB_W // 2) / (THUMB_W // 2) * 160)
        ov_draw.line([(x, 0), (x, THUMB_H)], fill=(0, 0, 0, alpha))
    ov_draw.rectangle([0, THUMB_H - 80, THUMB_W, THUMB_H], fill=(0, 0, 0, 200))
    bg = Image.alpha_composite(bg, overlay)
    draw = ImageDraw.Draw(bg)

    title_en = script.get("title", "").upper() or "ENGLISH LISTENING"
    title_zh = script.get("title_zh", script.get("intro_zh", ""))
    cefr = script.get("cefr", "A2")

    STROKE = 8
    MARGIN = 40

    # English title
    en_size = 120
    en_font = ImageFont.truetype(FONT_EN, en_size)
    while en_size > 40:
        bbox = draw.textbbox((0, 0), title_en, font=en_font)
        if (bbox[2] - bbox[0]) + STROKE * 2 <= THUMB_W // 2 - MARGIN:
            break
        en_size -= 4
        en_font = ImageFont.truetype(FONT_EN, en_size)
    bbox = draw.textbbox((0, 0), title_en, font=en_font)
    en_w, en_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    en_y = int(THUMB_H * 0.25)
    draw.text((THUMB_W // 2 + (THUMB_W // 2 - en_w) // 2, en_y), title_en,
              font=en_font, fill=(255, 220, 0, 255),
              stroke_width=STROKE, stroke_fill=(0, 0, 0, 255))

    # Chinese title
    if title_zh:
        zh_size = 60
        zh_font = ImageFont.truetype(FONT_ZH, zh_size)
        while zh_size > 24:
            bbox = draw.textbbox((0, 0), title_zh, font=zh_font)
            if (bbox[2] - bbox[0]) + 5 * 2 <= THUMB_W // 2 - MARGIN:
                break
            zh_size -= 2
            zh_font = ImageFont.truetype(FONT_ZH, zh_size)
        bbox = draw.textbbox((0, 0), title_zh, font=zh_font)
        zh_w = bbox[2] - bbox[0]
        draw.text((THUMB_W // 2 + (THUMB_W // 2 - zh_w) // 2, en_y + en_h + 15),
                  title_zh, font=zh_font, fill=(255, 215, 0, 255),
                  stroke_width=5, stroke_fill=(0, 0, 0, 255))

    # CEFR badge
    badge_r = 50
    cx, cy = THUMB_W - badge_r - 30, badge_r + 30
    draw.ellipse([cx - badge_r, cy - badge_r, cx + badge_r, cy + badge_r],
                 fill=(220, 50, 50, 255), outline=(255, 255, 255, 255), width=3)
    bfont = ImageFont.truetype(FONT_EN, 42)
    bb = draw.textbbox((0, 0), cefr, font=bfont)
    draw.text((cx - (bb[2] - bb[0]) // 2, cy - (bb[3] - bb[1]) // 2 - 2),
              cefr, font=bfont, fill=(255, 255, 255, 255))

    # Bottom bar
    label = "Vocabulary + Quiz + Slow Speed + Shadowing" if structure == "enhanced" else "Listen + Repeat + Shadowing"
    lfont = ImageFont.truetype(FONT_EN, 28)
    lb = draw.textbbox((0, 0), label, font=lfont)
    draw.text(((THUMB_W - (lb[2] - lb[0])) // 2, THUMB_H - 55),
              label, font=lfont, fill=(255, 255, 255, 255))

    bg.convert("RGB").save(output_path, "JPEG", quality=90)
    print(f"  [Thumbnail] Saved (Pillow fallback): {output_path}")
    return output_path


def save_youtube_metadata(script: dict, timeline: list[dict],
                           output_path: str, structure: str = "original") -> str:
    """Save YouTube metadata (title, description, tags) as JSON.

    Also post-processes the description to insert real timestamps from the timeline.
    """
    # Calculate real timestamps from timeline
    timestamps = {}
    t_cursor = 0.0
    for seg in timeline:
        seg_type = seg.get("type", "")
        dur = seg.get("duration", 0)

        if structure == "enhanced":
            if seg_type == "vocab" and "vocabulary" not in timestamps:
                timestamps["vocabulary"] = t_cursor
            elif seg_type == "title_card" and "title" not in timestamps:
                timestamps["title"] = t_cursor
            elif seg_type == "dialogue" and "dialogue" not in timestamps:
                timestamps["dialogue"] = t_cursor
            elif seg_type == "dialogue_slow" and "slow" not in timestamps:
                timestamps["slow"] = t_cursor
            elif seg_type == "quiz" and "quiz" not in timestamps:
                timestamps["quiz"] = t_cursor
            elif seg_type == "practice_intro" and "shadowing" not in timestamps:
                timestamps["shadowing"] = t_cursor
            elif seg_type == "outro" and "outro" not in timestamps:
                timestamps["outro"] = t_cursor
        else:
            if seg_type == "title_card" and "title" not in timestamps:
                timestamps["title"] = t_cursor
            elif seg_type == "dialogue" and "dialogue" not in timestamps:
                timestamps["dialogue"] = t_cursor
            elif seg_type == "practice_intro" and "shadowing" not in timestamps:
                timestamps["shadowing"] = t_cursor
            elif seg_type == "outro" and "outro" not in timestamps:
                timestamps["outro"] = t_cursor

        t_cursor += dur

    def _fmt_ts(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    chapters = []
    if structure == "enhanced":
        if "vocabulary" in timestamps:
            chapters.append(f"{_fmt_ts(timestamps['vocabulary'])} Vocabulary Preview")
        if "title" in timestamps:
            chapters.append(f"{_fmt_ts(timestamps['title'])} Title Card")
        if "dialogue" in timestamps:
            chapters.append(f"{_fmt_ts(timestamps['dialogue'])} Immersive Dialogue")
        if "slow" in timestamps:
            chapters.append(f"{_fmt_ts(timestamps['slow'])} Slow Speed Replay")
        if "quiz" in timestamps:
            chapters.append(f"{_fmt_ts(timestamps['quiz'])} Comprehension Quiz")
        if "shadowing" in timestamps:
            chapters.append(f"{_fmt_ts(timestamps['shadowing'])} Shadowing Practice")
        if "outro" in timestamps:
            chapters.append(f"{_fmt_ts(timestamps['outro'])} Outro")
    else:
        if "title" in timestamps:
            chapters.append(f"{_fmt_ts(timestamps['title'])} Title Card")
        if "dialogue" in timestamps:
            chapters.append(f"{_fmt_ts(timestamps['dialogue'])} Immersive Dialogue")
        if "shadowing" in timestamps:
            chapters.append(f"{_fmt_ts(timestamps['shadowing'])} Shadowing Practice")
        if "outro" in timestamps:
            chapters.append(f"{_fmt_ts(timestamps['outro'])} Outro")

    description = script.get("youtube_description", "")
    if chapters and "⏱️" not in description:
        chapters_text = "\n⏱️ Chapters:\n" + "\n".join(chapters) + "\n"
        description = description + "\n" + chapters_text

    metadata = {
        "title": script.get("youtube_title", script.get("title", "")),
        "description": description,
        "tags": script.get("youtube_tags", []),
        "chapters": chapters,
    }

    Path(output_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [YouTube] Metadata saved: {output_path}")
    return output_path
