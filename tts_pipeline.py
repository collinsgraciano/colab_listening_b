"""Batch TTS generation for all dialogue/narration audio."""
import os

from tts_engine import TTSEngine, build_voice_map, get_zh_voice


def generate_tts(script, dialogue, audio_dir, results, quest=False, tts_rate=None):
    """Generate all TTS audio. Runs in a thread.

    Produces narration and dialogue EN/ZH audio.
    Writes results into the *results* dict (shared with caller).

    Args:
        tts_rate: Override dialogue English TTS rate (e.g. '-15%', '0%').
                  If None, uses mode default (quest: '0%', non-quest: '-15%').
    """
    tts = TTSEngine()
    voice_map = build_voice_map(script)

    narration = {}
    if quest:
        welcome_text = script.get("welcome_en", "")
        hook_text = script.get("hook_intro_en", "")
        outro_text = script.get("outro", "That's all for today. Keep practicing!")
        for name, text, rate in [("welcome", welcome_text, "-10%"),
                                 ("hook", hook_text, "-10%"),
                                 ("outro", outro_text, "-10%")]:
            if text:
                path = str(audio_dir / f"{name}.mp3")
                dur = tts.synth_english(text, "af_sky", path, rate=rate)
                narration[name] = path
                print(f"  [TTS] {name}: {dur:.1f}s")
    else:
        intro_text = script.get("story_hook", "")
        outro_text = script.get("outro", "That's all for today. Keep practicing!")
        practice_intro_text = script.get("practice_intro_en", "Now let's practice. Listen and repeat each sentence.")

        for name, text in [("intro", intro_text), ("outro", outro_text), ("practice_intro", practice_intro_text)]:
            if text:
                path = str(audio_dir / f"{name}.mp3")
                dur = tts.synth_english(text, "af_sky", path, rate="+0%")
                narration[name] = path
                print(f"  [TTS] {name}: {dur:.1f}s")

    # Dialogue English (character voices; tts_rate overrides mode default)
    dialogue_rate = tts_rate if tts_rate else ("0%" if quest else "-15%")
    normal_paths = []
    dialogue_durations = []
    for i, line in enumerate(dialogue):
        text = line.get("text", "")
        speaker = line.get("speaker", "char_a")
        voice = voice_map.get(speaker, "af_sarah")
        path = str(audio_dir / f"dialogue_{i}.mp3")
        dur = tts.synth_english(text, voice, path, rate=dialogue_rate)
        normal_paths.append(path)
        dialogue_durations.append(dur)
        print(f"  [TTS] dialogue_{i}: {dur:.1f}s ({voice})")

    # Dialogue Chinese (quest skips entirely)
    zh_paths = []
    if not quest:
        for i, line in enumerate(dialogue):
            text = line.get("zh", "")
            if not text:
                zh_paths.append("")
                continue
            speaker = line.get("speaker", "char_a")
            voice = get_zh_voice(speaker, script)
            path = str(audio_dir / f"zh_{i}.mp3")
            dur = tts.synth_chinese(text, voice, path, rate="-10%")
            zh_paths.append(path)
            print(f"  [TTS] zh_{i}: {dur:.1f}s")

    results["narration"] = narration
    results["normal_paths"] = normal_paths
    results["dialogue_durations"] = dialogue_durations
    results["zh_paths"] = zh_paths
    results["vocab_paths"] = []
    results["slow_paths"] = []
    results["slow_durations"] = []
    results["quiz_paths"] = []
    print("  [TTS] All TTS generation complete.")
