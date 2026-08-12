"""YouTube thumbnail generator — generates thumbnail image via MCP + overlays text via Pillow.

Workflow:
1. Generate a thumbnail background image via generate_image (frontier)
2. Download the image
3. Overlay title text, level badge, and Chinese subtitle via Pillow
4. Save as thumbnail.jpg (1280x720, <2MB)
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


def generate_thumbnail(script: dict, scene_img: str, output_path: str,
                        mcp_call_tool=None, mcp_parse_task_id=None,
                        mcp_poll_task=None, mcp_download_file=None,
                        structure: str = "original") -> str:
    """Generate a YouTube thumbnail image.

    Uses the scene image as background, overlays:
    - Large English title (right side, yellow with black stroke)
    - Chinese subtitle (below English title, smaller)
    - CEFR level badge (top-right, red circle)
    - Structure-specific label (bottom bar)

    Args:
        script: LLM-generated script with youtube_title, title_zh, etc.
        scene_img: Path to existing scene image (fallback if image gen fails)
        output_path: Where to save the final thumbnail.jpg
        mcp_*: MCP client functions (call_tool, parse_task_id, poll_task, download_file)
        structure: "original" or "enhanced" — affects bottom label

    Returns:
        Path to the saved thumbnail, or None on failure.
    """
    from PIL import Image, ImageDraw, ImageFont

    # Step 1: Try to generate a thumbnail-specific image via MCP
    thumb_prompt = script.get("thumbnail_prompt", "")
    bg_path = scene_img  # fallback to scene image

    if thumb_prompt and mcp_call_tool and mcp_parse_task_id and mcp_poll_task and mcp_download_file:
        print("  [Thumbnail] Generating background image via MCP...")
        try:
            result = mcp_call_tool("generate_image", {
                "prompt": thumb_prompt,
                "provider": "frontier",
                "quality": "high",
                "image_size": "landscape_16_9",
                "output_format": "png",
            })
            task_id = mcp_parse_task_id(result)
            if task_id:
                data = mcp_poll_task(task_id, interval=10)
                url = data.get("url", "")
                if url:
                    tmp_bg = output_path.replace(".jpg", "_bg.png")
                    if mcp_download_file(url, tmp_bg) and os.path.exists(tmp_bg):
                        bg_path = tmp_bg
                        print(f"  [Thumbnail] Background image downloaded: {bg_path}")
        except Exception as e:
            print(f"  [Thumbnail] MCP image gen failed, using scene image: {e}")

    if not os.path.exists(bg_path):
        print(f"  [Thumbnail] ERROR: No background image available")
        return None

    # Step 2: Compose thumbnail with Pillow
    print("  [Thumbnail] Composing thumbnail with text overlay...")
    bg = Image.open(bg_path).convert("RGBA").resize((THUMB_W, THUMB_H))

    # Dark gradient overlay on right side for text readability
    overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    # Right-side dark gradient (simulated with rectangles of increasing alpha)
    for x in range(THUMB_W // 2, THUMB_W):
        alpha = int((x - THUMB_W // 2) / (THUMB_W // 2) * 160)
        ov_draw.line([(x, 0), (x, THUMB_H)], fill=(0, 0, 0, alpha))
    # Bottom bar for structure label
    ov_draw.rectangle([0, THUMB_H - 80, THUMB_W, THUMB_H], fill=(0, 0, 0, 200))
    bg = Image.alpha_composite(bg, overlay)
    draw = ImageDraw.Draw(bg)

    # --- English title (large, yellow, right-aligned area) ---
    title_en = script.get("title", "").upper()
    if not title_en:
        title_en = script.get("youtube_title", "ENGLISH LISTENING PRACTICE")[:40]

    STROKE = 8
    MARGIN = 40

    en_size = 120
    en_font = ImageFont.truetype(FONT_EN, en_size)
    while en_size > 40:
        bbox = draw.textbbox((0, 0), title_en, font=en_font)
        rendered_w = (bbox[2] - bbox[0]) + STROKE * 2
        if rendered_w <= THUMB_W // 2 - MARGIN:
            break
        en_size -= 4
        en_font = ImageFont.truetype(FONT_EN, en_size)

    bbox = draw.textbbox((0, 0), title_en, font=en_font)
    en_w = bbox[2] - bbox[0]
    en_h = bbox[3] - bbox[1]
    en_x = THUMB_W // 2 + (THUMB_W // 2 - en_w) // 2
    en_y = int(THUMB_H * 0.25)
    draw.text((en_x, en_y), title_en, font=en_font,
              fill=(255, 220, 0, 255), stroke_width=STROKE, stroke_fill=(0, 0, 0, 255))

    # --- Chinese title (below English, gold) ---
    title_zh = script.get("title_zh", script.get("intro_zh", ""))
    if title_zh:
        zh_stroke = 5
        zh_size = 60
        zh_font = ImageFont.truetype(FONT_ZH, zh_size)
        while zh_size > 24:
            bbox = draw.textbbox((0, 0), title_zh, font=zh_font)
            rendered_w = (bbox[2] - bbox[0]) + zh_stroke * 2
            if rendered_w <= THUMB_W // 2 - MARGIN:
                break
            zh_size -= 2
            zh_font = ImageFont.truetype(FONT_ZH, zh_size)

        bbox = draw.textbbox((0, 0), title_zh, font=zh_font)
        zh_w = bbox[2] - bbox[0]
        zh_x = THUMB_W // 2 + (THUMB_W // 2 - zh_w) // 2
        zh_y = en_y + en_h + 15
        draw.text((zh_x, zh_y), title_zh, font=zh_font,
                  fill=(255, 215, 0, 255), stroke_width=zh_stroke, stroke_fill=(0, 0, 0, 255))

    # --- CEFR level badge (top-right, red circle) ---
    cefr = script.get("cefr", "A2")
    badge_r = 50
    badge_cx = THUMB_W - badge_r - 30
    badge_cy = badge_r + 30
    draw.ellipse([badge_cx - badge_r, badge_cy - badge_r,
                   badge_cx + badge_r, badge_cy + badge_r],
                  fill=(220, 50, 50, 255), outline=(255, 255, 255, 255), width=3)
    badge_font = ImageFont.truetype(FONT_EN, 42)
    bbox = draw.textbbox((0, 0), cefr, font=badge_font)
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((badge_cx - bw // 2, badge_cy - bh // 2 - 2), cefr,
              font=badge_font, fill=(255, 255, 255, 255))

    # --- Bottom bar: structure label ---
    if structure == "enhanced":
        label = "Vocabulary + Quiz + Slow Speed + Shadowing"
    else:
        label = "Listen + Repeat + Shadowing"

    label_font = ImageFont.truetype(FONT_EN, 28)
    bbox = draw.textbbox((0, 0), label, font=label_font)
    lw = bbox[2] - bbox[0]
    draw.text(((THUMB_W - lw) // 2, THUMB_H - 55), label,
              font=label_font, fill=(255, 255, 255, 255))

    # --- Save as JPG (<2MB) ---
    bg.convert("RGB").save(output_path, "JPEG", quality=90)
    size_kb = os.path.getsize(output_path) // 1024
    print(f"  [Thumbnail] Saved: {output_path} ({size_kb}KB)")

    # Cleanup temp bg
    tmp_bg = output_path.replace(".jpg", "_bg.png")
    if os.path.exists(tmp_bg) and tmp_bg != scene_img:
        try:
            os.remove(tmp_bg)
        except OSError:
            pass

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

    # Build chapters string
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

    # Replace placeholder timestamps in description with real ones
    description = script.get("youtube_description", "")
    # Insert real chapters at the end of description
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
