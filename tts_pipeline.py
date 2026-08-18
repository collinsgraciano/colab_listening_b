"""Batch TTS generation for all dialogue/narration/vocab/quiz audio."""
import os
import subprocess

from tts_engine import TTSEngine, build_voice_map, get_zh_voice
from media_utils import get_duration as _get_audio_duration


def generate_tts(script, dialogue, audio_dir, results, enhanced=False, quest=False):
    """Generate all TTS audio. Runs in a thread.

    Produces narration, dialogue EN/ZH, and (enhanced only) vocab/slow/quiz audio.
    Writes results into the *results* dict (shared with caller).
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

    # Dialogue English (character voices; quest: -25% extra slow for beginners)
    dialogue_rate = "-25%" if quest else "-15%"
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

    # Enhanced: vocabulary + slow dialogue + quiz TTS
    vocab_paths = []
    slow_paths = []
    quiz_paths = []
    slow_durations = []

    if enhanced:
        vocabulary = script.get("vocabulary", [])
        questions = script.get("comprehension_questions", [])

        for vi, vocab in enumerate(vocabulary):
            word = vocab.get("word", "")
            example = vocab.get("example", "")
            text = f"{word}. {example}" if example else word
            path = str(audio_dir / f"vocab_{vi}.mp3")
            dur = tts.synth_english(text, "af_sky", path, rate="+0%")
            vocab_paths.append(path)
            print(f"  [TTS] vocab_{vi}: {dur:.1f}s ({word})")

        for i, normal_path in enumerate(normal_paths):
            if not os.path.exists(normal_path):
                slow_paths.append("")
                slow_durations.append(0.0)
                continue
            slow_path = str(audio_dir / f"dialogue_slow_{i}.mp3")
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", normal_path,
                 "-filter:a", "atempo=0.75",
                 "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                 slow_path],
                capture_output=True, timeout=30)
            if r.returncode == 0 and os.path.exists(slow_path):
                dur = _get_audio_duration(slow_path)
                slow_paths.append(slow_path)
                slow_durations.append(dur)
                print(f"  [TTS] dialogue_slow_{i}: {dur:.1f}s")
            else:
                slow_paths.append(normal_path)
                slow_durations.append(dialogue_durations[i] if i < len(dialogue_durations) else 3.0)
                print(f"  [TTS] dialogue_slow_{i}: FAILED, using normal audio")

        for qi, quiz in enumerate(questions):
            question = quiz.get("question", "")
            options = quiz.get("options", [])
            answer = quiz.get("answer", "")
            opts_text = " ".join(options)
            text = f"{question}. {opts_text}. The answer is {answer}."
            path = str(audio_dir / f"quiz_{qi}.mp3")
            dur = tts.synth_english(text, "af_sky", path, rate="+0%")
            quiz_paths.append(path)
            print(f"  [TTS] quiz_{qi}: {dur:.1f}s")

    results["narration"] = narration
    results["normal_paths"] = normal_paths
    results["dialogue_durations"] = dialogue_durations
    results["zh_paths"] = zh_paths
    results["vocab_paths"] = vocab_paths
    results["slow_paths"] = slow_paths
    results["slow_durations"] = slow_durations
    results["quiz_paths"] = quiz_paths
    print("  [TTS] All TTS generation complete.")
