"""Enhanced timeline builder + SRT generator for 7-chapter listening practice videos.

Structure:
  Ch1: Vocabulary Preview  (6 keywords, static cards + TTS)
  Ch2: Title Card           (5s)
  Ch3: Immersive Dialogue   (normal speed, grouped video, subtitles)
  Ch4: Slow Speed Replay    (same dialogue at 75% speed, subtitles)
  Ch5: Comprehension Quiz   (3 MCQs, static cards + TTS)
  Ch6: Shadowing Practice   (per line: 1EN→sil→1EN→sil→1ZH→1EN→sil = 7 segs)
  Ch7: Outro                (narration)
"""
import sys
from pathlib import Path

_PARENT = str(Path(__file__).parent.parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from timeline import _format_srt_time


def build_enhanced_timeline(script: dict, dialogue_durations: list[float],
                            slow_durations: list[float],
                            vocab_durations: list[float],
                            quiz_durations: list[float],
                            pad: float = 0.4,
                            practice_duration: float = 3.0) -> list[dict]:
    """Build a timeline for the enhanced 7-chapter listening lesson.

    Args:
        script: LLM-generated script with vocabulary + comprehension_questions.
        dialogue_durations: per-line normal-speed TTS durations.
        slow_durations: per-line slow-speed (75%) TTS durations.
        vocab_durations: per-vocabulary-word TTS durations.
        quiz_durations: per-question TTS durations (question + answer combined).
        pad: audio pad between segments (seconds).
        practice_duration: silence duration in Ch6 shadowing.

    Returns:
        List of timeline segment dicts.
    """
    dialogue = script.get("dialogue", [])
    n = len(dialogue)
    vocabulary = script.get("vocabulary", [])
    questions = script.get("comprehension_questions", [])
    outro = script.get("outro", "That's all for today. Keep practicing!")
    outro_zh = script.get("outro_zh", "")

    timeline = []

    def _add(seg_type, dur, sub_en="", sub_zh="", audio_idx=0,
             image_idx=0, d_idx=-1, extra=None):
        seg = {
            "type": seg_type,
            "duration": dur,
            "subtitle_en": sub_en,
            "subtitle_zh": sub_zh,
            "speaker": "",
            "audio_index": audio_idx,
            "image_idx": image_idx,
            "dialogue_idx": d_idx,
        }
        if extra:
            seg.update(extra)
        timeline.append(seg)

    # === Ch1: Vocabulary Preview ===
    for vi, vocab in enumerate(vocabulary):
        dur = vocab_durations[vi] if vi < len(vocab_durations) else 4.0
        _add("vocab", dur + pad,
             sub_en=vocab.get("word", ""),
             sub_zh=vocab.get("zh", ""),
             audio_idx=vi, d_idx=vi,
             extra={"phonetic": vocab.get("phonetic", ""),
                    "example": vocab.get("example", ""),
                    "vocab_idx": vi})

    # === Ch2: Title Card ===
    title_en = script.get("title", "")
    title_zh = script.get("title_zh", script.get("intro_zh", ""))
    scene_zh = script.get("scene_zh", "")
    if title_en:
        _add("title_card", 5.0, title_en, title_zh, audio_idx=0, image_idx=0)
        timeline[-1]["scene_zh"] = scene_zh

    # === Ch3: Immersive Dialogue (normal speed) ===
    for i, line in enumerate(dialogue):
        dur = dialogue_durations[i] if i < len(dialogue_durations) else 3.0
        _add("dialogue", dur, line.get("text", ""), line.get("zh", ""),
             audio_idx=i, image_idx=i + 1)

    # === Ch4: Slow Speed Replay ===
    for i, line in enumerate(dialogue):
        dur = slow_durations[i] if i < len(slow_durations) else 4.0
        _add("dialogue_slow", dur, line.get("text", ""), line.get("zh", ""),
             audio_idx=i, image_idx=i + 1)

    # === Ch5: Comprehension Quiz ===
    for qi, quiz in enumerate(questions):
        q_dur = quiz_durations[qi] if qi < len(quiz_durations) else 5.0
        _add("quiz", q_dur + pad,
             sub_en=quiz.get("question", ""),
             sub_zh="",
             audio_idx=qi, d_idx=qi,
             extra={"quiz_options": quiz.get("options", []),
                    "quiz_answer": quiz.get("answer", ""),
                    "quiz_idx": qi})

    # === Ch6: Shadowing Practice ===
    practice_intro_en = script.get("practice_intro_en",
                                   "Now let's practice. Listen and repeat each sentence.")
    practice_intro_zh = script.get("practice_intro_zh", "現在來練習。請跟著朗讀每一句。")
    _add("practice_intro", 4.0, practice_intro_en, practice_intro_zh,
         audio_idx=-1, image_idx=0)

    for i in range(n):
        line = dialogue[i]
        en_text = line.get("text", "")
        zh_text = line.get("zh", "")
        dur = dialogue_durations[i] if i < len(dialogue_durations) else 3.0

        # 7 segments: 1EN → sil → 1EN → sil → 1ZH → 1EN → sil
        _add("listen_en", dur, en_text, "", audio_idx=i, image_idx=-1, d_idx=i)
        _add("practice", practice_duration, "", "", audio_idx=-1, image_idx=-1, d_idx=i)
        _add("listen_en", dur, en_text, "", audio_idx=i, image_idx=-1, d_idx=i)
        _add("practice", practice_duration, "", "", audio_idx=-1, image_idx=-1, d_idx=i)
        _add("listen_zh", dur, en_text, zh_text, audio_idx=i, image_idx=-1, d_idx=i)
        _add("listen_en", dur, en_text, "", audio_idx=i, image_idx=-1, d_idx=i)
        _add("practice", practice_duration, "", "", audio_idx=-1, image_idx=-1, d_idx=i)

    # === Ch7: Outro ===
    if outro:
        outro_dur = min(max(len(outro) * 0.08, 3.0), 6.0)
        _add("outro", outro_dur, outro, outro_zh, audio_idx=0, image_idx=0)

    return timeline


def build_srt_from_timeline_enhanced(timeline: list[dict], gap: float = 0.0) -> str:
    """Build SRT from enhanced timeline. Timestamps match video exactly.

    Skips: vocab, listen_en, listen_zh, practice, title_card, practice_intro, outro, quiz
    Text is on static images, not subtitles.
    """
    srt_lines = []
    idx = 1
    current_time = 0.0

    skip_types = {"vocab", "listen_en", "listen_zh", "practice",
                  "title_card", "practice_intro", "outro", "quiz"}

    for seg in timeline:
        dur = seg["duration"]
        start = current_time
        end = start + dur

        text_en = seg.get("subtitle_en", "")
        text_zh = seg.get("subtitle_zh", "")
        seg_type = seg.get("type", "")

        if seg_type in skip_types:
            current_time = end + gap
            continue

        if not text_en:
            current_time = end + gap
            continue

        audio_dur = seg.get("audio_dur", dur)
        srt_end = start + audio_dur

        srt_lines.append(str(idx))
        srt_lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(srt_end)}")
        srt_lines.append(text_en)
        if text_zh:
            srt_lines.append(text_zh)
        srt_lines.append("")
        idx += 1
        current_time = end + gap

    return "\n".join(srt_lines)
