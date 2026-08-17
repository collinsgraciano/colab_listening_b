"""Quest timeline builder + SRT generator for task-hook slow listening videos.

Structure (mirrors the reference slow-listening video):
  Ch1: Title Card      (5s, scene image + big title overlay)
  Ch2: Hook / Intro    (narrator ~60-90s: reassure slow speech + listening task)
  Ch3: Slow Dialogue   (3 phases in one continuous flow: buildup -> core -> review)
  Ch4: Outro & CTA     (narrator ~45-60s: repeat question + comment + subscribe)
"""
import sys
from pathlib import Path

_PARENT = str(Path(__file__).parent.parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from media_utils import build_srt


def build_quest_timeline(script: dict, dialogue_durations: list[float],
                         pad: float = 5.0) -> list[dict]:
    """Build a timeline for the quest (task-hook slow listening) lesson.

    Args:
        script: LLM-generated quest script (dialogue lines carry "phase").
        dialogue_durations: per-line TTS durations (slow speed).
        pad: silence pad between dialogue lines (long by default — thinking time).

    Returns:
        List of timeline segment dicts: title_card, hook_intro, dialogue×N, outro.
    """
    dialogue = script.get("dialogue", [])
    outro = script.get("outro", "That's all for today. Keep practicing!")
    outro_zh = script.get("outro_zh", "")

    timeline = []

    def _add(seg_type, dur, sub_en="", sub_zh="", audio_idx=0, image_idx=0, d_idx=-1):
        timeline.append({
            "type": seg_type,
            "duration": dur,
            "subtitle_en": sub_en,
            "subtitle_zh": sub_zh,
            "speaker": "",
            "audio_index": audio_idx,
            "image_idx": image_idx,
            "dialogue_idx": d_idx,
        })

    # 1. Title card (5s)
    title_en = script.get("title", "")
    title_zh = script.get("title_zh", script.get("intro_zh", ""))
    scene_zh = script.get("scene_zh", "")
    if title_en:
        _add("title_card", 5.0, title_en, title_zh, audio_idx=0, image_idx=0)
        timeline[-1]["scene_zh"] = scene_zh

    # 2. Hook / intro (narrator) — duration placeholder, enriched from narration
    #    audio later by _enrich_timeline ("hook_intro")
    hook_en = script.get("hook_intro_en", "")
    if hook_en:
        _add("hook_intro", 10.0, hook_en, script.get("hook_intro_zh", ""),
             audio_idx=-1, image_idx=0)

    # 3. Slow dialogue — all lines in order (phases stay contiguous)
    for i, line in enumerate(dialogue):
        dur = dialogue_durations[i] if i < len(dialogue_durations) else 3.0
        _add("dialogue", dur, line.get("text", ""), line.get("zh", ""),
             audio_idx=i, image_idx=i + 1)
        seg = timeline[-1]
        seg["phase"] = line.get("phase", "buildup")
        seg["speaker"] = line.get("speaker", "")

    # 4. Outro & CTA (narrator)
    if outro:
        outro_dur = min(max(len(outro) * 0.08, 6.0), 10.0)
        _add("outro", outro_dur, outro, outro_zh, audio_idx=0, image_idx=0)

    return timeline


def build_srt_from_timeline_quest(timeline: list[dict], gap: float = 0.0) -> str:
    """Build SRT from quest timeline. Timestamps match video exactly.

    Skips title_card / hook_intro / outro — their text is baked into big
    overlay cards in compose. Only dialogue lines become subtitles.
    """
    return build_srt(timeline, skip_types={"title_card", "hook_intro", "outro"}, gap=gap)
