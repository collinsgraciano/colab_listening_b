"""Unified structure dispatch: maps --structure to the correct
LLM client, timeline builder, SRT builder, and compose function.

This replaces scattered if/elif chains in pipeline.py with a single lookup.
Each structure variant registers its four domain functions here.
"""
import sys
from pathlib import Path

_PARENT = str(Path(__file__).parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# --- Original ---
from llm_client import generate_listening_script
from timeline import build_listening_timeline, build_srt_from_timeline
from video_compose import compose_listening, compose_static

# --- Enhanced (lazy import to avoid circular deps) ---
def _enhanced_script(topic, cefr, lessons_dir, num_lines):
    from enhanced.llm_client_enhanced import generate_listening_script_enhanced
    return generate_listening_script_enhanced(topic, cefr, lessons_dir=lessons_dir, num_lines=num_lines)

def _enhanced_timeline(script, dialogue_durations, slow_durations,
                        vocab_durations, quiz_durations, pad, practice_duration):
    from enhanced.timeline_enhanced import build_enhanced_timeline
    return build_enhanced_timeline(script, dialogue_durations, slow_durations,
                                   vocab_durations, quiz_durations,
                                   pad=pad, practice_duration=practice_duration)

def _enhanced_srt(timeline, gap=0.0):
    from enhanced.timeline_enhanced import build_srt_from_timeline_enhanced
    return build_srt_from_timeline_enhanced(timeline, gap=gap)

def _enhanced_compose(**kwargs):
    from enhanced.video_compose_enhanced import compose_listening_enhanced
    return compose_listening_enhanced(**kwargs)

# --- Quest (lazy import) ---
def _quest_script(topic, cefr, lessons_dir, num_lines):
    from quest.llm_client_quest import generate_quest_script
    return generate_quest_script(topic, cefr, lessons_dir=lessons_dir, num_lines=num_lines)

def _quest_timeline(script, dialogue_durations, pad):
    from quest.timeline_quest import build_quest_timeline
    return build_quest_timeline(script, dialogue_durations, pad=pad)

def _quest_srt(timeline, gap=0.0):
    from quest.timeline_quest import build_srt_from_timeline_quest
    return build_srt_from_timeline_quest(timeline, gap=gap)

def _quest_compose(**kwargs):
    from quest.video_compose_quest import compose_quest
    return compose_quest(**kwargs)


STRUCTURES = {
    "original": {
        "generate_script": generate_listening_script,
        "build_timeline": lambda script, ddur, pad, pd: build_listening_timeline(
            script, ddur, pad=pad, practice_duration=pd),
        "build_srt": lambda tl, gap=0.0: build_srt_from_timeline(tl, gap=gap),
        "compose": compose_listening,
        "needs_video_clips": True,
        "needs_zh_tts": True,
        "needs_dialogue_images": False,
    },
    "enhanced": {
        "generate_script": _enhanced_script,
        "build_timeline": lambda script, ddur, pad, pd: _enhanced_timeline(
            script, ddur, [], [], [], pad, pd),  # slow/vocab/quiz durations filled by caller
        "build_srt": _enhanced_srt,
        "compose": _enhanced_compose,
        "needs_video_clips": True,
        "needs_zh_tts": True,
        "needs_dialogue_images": False,
    },
    "static": {
        "generate_script": generate_listening_script,
        "build_timeline": lambda script, ddur, pad, pd: build_listening_timeline(
            script, ddur, pad=pad, practice_duration=pd),
        "build_srt": lambda tl, gap=0.0: build_srt_from_timeline(tl, gap=gap),
        "compose": compose_static,
        "needs_video_clips": False,
        "needs_zh_tts": True,
        "needs_dialogue_images": True,
    },
    "quest": {
        "generate_script": _quest_script,
        "build_timeline": lambda script, ddur, pad, pd: _quest_timeline(
            script, ddur, pad=pad),
        "build_srt": _quest_srt,
        "compose": _quest_compose,
        "needs_video_clips": False,
        "needs_zh_tts": False,
        "needs_dialogue_images": True,
    },
}


def get_structure(name: str) -> dict:
    """Return the structure dispatch dict for the given structure name."""
    return STRUCTURES[name]


def list_structures() -> list[str]:
    """Return all available structure names."""
    return list(STRUCTURES.keys())
