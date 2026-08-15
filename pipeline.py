#!/usr/bin/env python3
"""Standalone listening video generation pipeline — Colab version.

Usage:
    python pipeline.py --topic "At the Pharmacy" --cefr A2 --output ./output --mcp-token YOUR_TOKEN

Steps:
1. LLM script generation (SenseNova DeepSeek V4 Flash)
2. Concurrent: image generation + TTS audio (Kokoro EN, edge-tts ZH)
3. Group consecutive dialogue lines, generate one video clip per group
4. Timeline + SRT building
5. Static frame rendering (Pillow, inside compose)
6. Final video composition (FFmpeg, group audio concat + setpts)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import threading
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

from mcp_client import initialize, call_tool, parse_task_id, poll_task, download_file
from llm_client import generate_listening_script
from tts_engine import TTSEngine, build_voice_map, get_zh_voice
from timeline import build_listening_timeline, build_srt_from_timeline
from video_compose import compose_listening
from grouping_b import build_dialogue_groups, merge_group_prompt
from topic_manager import pick_random_topic, mark_topic_used


def _save_checkpoint(work_dir: Path, step: str, **extra):
    """Save progress to checkpoint.json for resume support."""
    cp_path = work_dir / "checkpoint.json"
    cp = {}
    if cp_path.exists():
        try:
            cp = json.loads(cp_path.read_text(encoding="utf-8"))
        except Exception:
            cp = {}
    if step not in cp.get("completed_steps", []):
        cp.setdefault("completed_steps", []).append(step)
    cp.update(extra)
    cp["_run_dir"] = str(work_dir)  # resume uses this to relocate the run directory
    cp["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cp_path.write_text(json.dumps(cp, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [Checkpoint] Saved: {step}")


def _load_checkpoint(work_dir: Path) -> dict:
    """Load checkpoint from work_dir, or scan subdirectories for incomplete runs.

    A run is considered complete when step6_4k is done (so a crash during 4K
    upscale can still be resumed). Among multiple incomplete runs, the most
    recently updated one wins.
    """
    candidates = []

    def _check(cp_path: Path):
        try:
            cp = json.loads(cp_path.read_text(encoding="utf-8"))
            if "step6_4k" not in cp.get("completed_steps", []):
                candidates.append((cp_path.stat().st_mtime, cp_path.parent, cp))
            else:
                # Completed — delete checkpoint so future scans skip it
                cp_path.unlink()
        except Exception:
            pass

    # First check work_dir/checkpoint.json directly (backward compat)
    _check(work_dir / "checkpoint.json")
    # Scan subdirectories for incomplete checkpoints
    if not candidates and work_dir.exists():
        for sub in work_dir.iterdir():
            if sub.is_dir():
                _check(sub / "checkpoint.json")

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, run_dir, cp = candidates[0]
        print(f"  [Resume] Found incomplete run in: {run_dir.name}")
        return cp
    return {}


def _step_done(checkpoint: dict, step: str) -> bool:
    return step in checkpoint.get("completed_steps", [])


def _validate_script(script: dict, num_lines: int, enhanced: bool = False) -> tuple[bool, str]:
    """Validate a generated script. Returns (is_valid, error_message)."""
    dialogue = script.get("dialogue", [])
    if len(dialogue) < num_lines:
        return False, f"Dialogue count {len(dialogue)} < required {num_lines}"
    for i, line in enumerate(dialogue):
        text = line.get("text", "").strip()
        if not text:
            return False, f"Dialogue line {i} has empty 'text'"
        if not line.get("zh", "").strip():
            return False, f"Dialogue line {i} has empty 'zh'"
        if not line.get("phonetic", "").strip():
            return False, f"Dialogue line {i} has empty 'phonetic'"
        if not line.get("speaker", ""):
            return False, f"Dialogue line {i} has empty 'speaker'"
    if enhanced:
        vocab = script.get("vocabulary", [])
        if len(vocab) < 3:
            return False, f"Vocabulary count {len(vocab)} < required 3"
        questions = script.get("comprehension_questions", [])
        if len(questions) < 1:
            return False, f"Comprehension questions count {len(questions)} < required 1"
    return True, ""


def _generate_script_with_retry(topic, cefr, lessons_dir, num_lines,
                                enhanced=False, max_attempts=5) -> dict:
    """Generate and validate script, retrying on failure."""
    for attempt in range(max_attempts):
        try:
            print(f"  [Script] Attempt {attempt+1}/{max_attempts}...")
            if enhanced:
                from enhanced.llm_client_enhanced import generate_listening_script_enhanced
                script = generate_listening_script_enhanced(topic, cefr,
                                                             lessons_dir=lessons_dir,
                                                             num_lines=num_lines)
            else:
                script = generate_listening_script(topic, cefr, lessons_dir=lessons_dir,
                                                   num_lines=num_lines)
            valid, msg = _validate_script(script, num_lines, enhanced=enhanced)
            if valid:
                print(f"  [Script] Valid: {len(script['dialogue'])} lines")
                return script
            print(f"  [Script] Invalid: {msg}")
        except Exception as e:
            if "quota exceeded" in str(e):
                print(f"  [Script] FATAL: LLM quota exceeded — aborting retries.")
                print(f"  [Script] Try: --model {'glm-5.2' if os.environ.get('SENSENOVA_MODEL', 'deepseek-v4-flash') == 'deepseek-v4-flash' else 'deepseek-v4-flash'}")
                raise
            print(f"  [Script] Error: {e}")
        if attempt < max_attempts - 1:
            time.sleep(3)
    raise RuntimeError(f"Script generation failed after {max_attempts} attempts")


def _get_audio_duration(path: str) -> float:
    """Get audio duration via ffprobe."""
    try:
        return float(subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path], text=True).strip())
    except Exception:
        return 0.0


def _file_ok(path: str, min_size: int) -> bool:
    """Check a file exists and is larger than min_size (guards partial downloads)."""
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > min_size


def _generate_video_clips(video_tasks, clips_dir, clip_paths, offset=0):
    """Generate video clips in batches. Runs in a thread.
    Each task has its own 'duration' (dynamic per-group, not fixed).
    clip_paths[offset + i] corresponds to video_tasks[i]; entries already set
    (per-clip resume) are skipped so no credits are re-spent.
    """
    batch_size = 12

    for batch_start in range(0, len(video_tasks), batch_size):
        batch = video_tasks[batch_start:batch_start + batch_size]
        task_ids = []

        for j, task in enumerate(batch):
            idx = offset + batch_start + j
            if clip_paths[idx] is not None:
                continue  # already downloaded (per-clip resume)
            print(f"  [Video] Creating task {idx+1}/{len(video_tasks)}: {task['filename']}...")
            try:
                result = call_tool("generate_video", {
                    "mode": "reference_image",
                    "image_urls": task["image_urls"],
                    "prompt": task["prompt"],
                    "duration": task["duration"],
                    "ratio": "16:9",
                    "resolution": "720p",
                    "generate_audio": task.get("generate_audio", False),
                })
                task_id = parse_task_id(result)
                if not task_id:
                    print(f"    [Video] WARNING: no task_id for {task['filename']}")
                    if result and "result" in result:
                        print(f"    [Video] API response content: {json.dumps(result['result'].get('content', []), ensure_ascii=False)[:1000]}")
                    elif result:
                        print(f"    [Video] Full API response: {json.dumps(result, ensure_ascii=False)[:1000]}")
                task_ids.append((idx, task_id, task["filename"]))
            except Exception as e:
                print(f"    [Video] ERROR creating task: {e}")
                task_ids.append((idx, "", task["filename"]))
            if j < len(batch) - 1:
                time.sleep(5)

        for idx, task_id, filename in task_ids:
            if not task_id:
                continue
            print(f"  [Video] Polling: {filename}...")
            data = poll_task(task_id, interval=40)
            url = data.get("url", "")
            if url:
                dest = str(clips_dir / filename)
                if download_file(url, dest) and _file_ok(dest, 500000):
                    clip_paths[idx] = dest
                    print(f"    [Video] Downloaded: {filename} ({os.path.getsize(dest)//1024}KB)")
                else:
                    print(f"    [Video] WARNING: File too small or download failed, re-downloading...")
                    if download_file(url, dest) and _file_ok(dest, 500000):
                        clip_paths[idx] = dest
                    else:
                        print(f"    [Video] ERROR: Downloaded file still too small/missing")
            else:
                print(f"    [Video] ERROR: No video URL. Status={data.get('status')}")
                raw_json = data.get("raw_json", "")
                raw_text = data.get("raw_text", "")
                error_msg = data.get("error", "")
                if error_msg:
                    print(f"    [Video] Error message: {error_msg}")
                if raw_json:
                    print(f"    [Video] Raw resource JSON: {raw_json[:1000]}")
                if raw_text and not raw_json:
                    print(f"    [Video] Raw text block: {raw_text[:1000]}")

    # Retry failed clips one by one — each retry creates a BRAND NEW video task
    MAX_RETRY = 5
    failed = [offset + i for i in range(len(video_tasks)) if clip_paths[offset + i] is None]
    for idx in failed:
        task = video_tasks[idx - offset]
        attempt = 0
        while attempt < MAX_RETRY:
            attempt += 1
            print(f"  [Video] Retrying clip {idx+1} ({task['filename']}) attempt {attempt}/{MAX_RETRY}...")
            try:
                # 1. Create a NEW video generation request (not re-querying old one)
                result = call_tool("generate_video", {
                    "mode": "reference_image",
                    "image_urls": task["image_urls"],
                    "prompt": task["prompt"],
                    "duration": task["duration"],
                    "ratio": "16:9",
                    "resolution": "720p",
                    "generate_audio": task.get("generate_audio", False),
                })
                # 2. Extract task_id — if empty, the request itself failed
                task_id = parse_task_id(result)
                if not task_id:
                    print(f"    [Video] FAILED: could not create task (parse_task_id returned empty)")
                    # Print the actual request parameters
                    print(f"    [Video] Request params:")
                    print(f"      mode: reference_image")
                    print(f"      image_urls: {task['image_urls'][:120]}...")
                    print(f"      prompt: {task['prompt'][:200]}...")
                    print(f"      duration: {task['duration']}")
                    print(f"      ratio: 16:9")
                    print(f"      generate_audio: True")
                    # Print the raw API response
                    if result and "result" in result:
                        raw_content = json.dumps(result["result"].get("content", []), ensure_ascii=False)[:1500]
                        print(f"    [Video] API response content: {raw_content}")
                    elif result:
                        print(f"    [Video] Full API response: {json.dumps(result, ensure_ascii=False)[:1500]}")
                    time.sleep(10)
                    continue  # ← creates another NEW task
                # 3. Poll the NEW task until it completes or fails
                data = poll_task(task_id, interval=40)
                url = data.get("url", "")
                if not url:
                    print(f"    [Video] FAILED: task {task_id[:16]}... returned no URL. Status={data.get('status')}")
                    # Print the FULL raw error info
                    raw_json = data.get("raw_json", "")
                    raw_text = data.get("raw_text", "")
                    raw_response = data.get("raw_response", "")
                    error_msg = data.get("error", "")
                    if error_msg:
                        print(f"    [Video] Error message: {error_msg}")
                    if raw_json:
                        print(f"    [Video] Raw resource JSON: {raw_json[:1000]}")
                    if raw_text and not raw_json:
                        print(f"    [Video] Raw text block: {raw_text[:1000]}")
                    if raw_response and not raw_json and not raw_text:
                        print(f"    [Video] Full check_task response: {raw_response[:1000]}")
                    time.sleep(10)
                    continue  # ← creates another NEW task
                # 4. Download the result
                dest = str(clips_dir / task["filename"])
                if download_file(url, dest) and _file_ok(dest, 500000):
                    clip_paths[idx] = dest
                    print(f"    [Video] Retry successful: {task['filename']}")
                    break
                else:
                    print(f"    [Video] FAILED: file too small or missing, creating new task...")
                    time.sleep(10)
                    continue
            except Exception as e:
                print(f"    [Video] FAILED (attempt {attempt}): {type(e).__name__}: {e}")
                time.sleep(10)
                continue

        if clip_paths[idx] is None:
            print(f"    [Video] GIVING UP on clip {idx+1} after {MAX_RETRY} retries: {task['filename']}")

    total = sum(1 for p in clip_paths[offset:offset + len(video_tasks)] if p is not None)
    print(f"  [Video] Clips for this batch: {total}/{len(video_tasks)}")


def _generate_tts(script, dialogue, audio_dir, results, enhanced=False):
    """Generate all TTS audio. Runs in a thread."""
    tts = TTSEngine()
    voice_map = build_voice_map(script)

    narration = {}
    intro_text = script.get("story_hook", "")
    outro_text = script.get("outro", "That's all for today. Keep practicing!")
    practice_intro_text = script.get("practice_intro_en", "Now let's practice. Listen and repeat each sentence.")

    for name, text in [("intro", intro_text), ("outro", outro_text), ("practice_intro", practice_intro_text)]:
        if text:
            path = str(audio_dir / f"{name}.mp3")
            dur = tts.synth_english(text, "af_sky", path, rate="+0%")
            narration[name] = path
            print(f"  [TTS] {name}: {dur:.1f}s")

    # Dialogue English (character voices, -15% slow)
    normal_paths = []
    dialogue_durations = []
    for i, line in enumerate(dialogue):
        text = line.get("text", "")
        speaker = line.get("speaker", "char_a")
        voice = voice_map.get(speaker, "af_sarah")
        path = str(audio_dir / f"dialogue_{i}.mp3")
        dur = tts.synth_english(text, voice, path, rate="-15%")
        normal_paths.append(path)
        dialogue_durations.append(dur)
        print(f"  [TTS] dialogue_{i}: {dur:.1f}s ({voice})")

    # Dialogue Chinese (edge-tts primary with retries, Kokoro fallback, -10% slow)
    zh_paths = []
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

        # Vocabulary TTS: read word + example sentence
        for vi, vocab in enumerate(vocabulary):
            word = vocab.get("word", "")
            example = vocab.get("example", "")
            text = f"{word}. {example}" if example else word
            path = str(audio_dir / f"vocab_{vi}.mp3")
            dur = tts.synth_english(text, "af_sky", path, rate="+0%")
            vocab_paths.append(path)
            print(f"  [TTS] vocab_{vi}: {dur:.1f}s ({word})")

        # Slow-speed dialogue: FFmpeg atempo=0.75 on existing dialogue audio
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

        # Quiz TTS: read question + options + answer
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


def main():
    parser = argparse.ArgumentParser(description="Generate English listening practice video")
    parser.add_argument("--topic", default=None, help="Topic (e.g. 'At the Pharmacy'). If not specified, picks randomly from topics.json")
    parser.add_argument("--cefr", default="A2", choices=["A1", "A2", "B1", "B2", "C1", "C2"], help="CEFR level (default A2)")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--clip-duration", type=int, default=15, help="Video clip duration in seconds")
    parser.add_argument("--practice-duration", type=float, default=3.0, help="Silence duration in Ch3")
    parser.add_argument("--pad", type=float, default=0.4, help="Audio pad between segments")
    parser.add_argument("--lessons-dir", default=None, help="Lessons dir for anti-duplicate check")
    parser.add_argument("--topics-file", default=str(Path(__file__).parent / "topics.json"), help="Path to topics.json")
    parser.add_argument("--used-topics-file", default=None, help="Path to used_topics.json (default: <output>/used_topics.json — persists on Drive across Colab sessions)")
    parser.add_argument("--num-lines", type=int, default=18, help="Number of dialogue lines (default 18)")
    parser.add_argument("--mcp-tokens", default=None, help="TJGenerators MCP OAuth tokens, comma-separated for multi-token rotation")
    parser.add_argument("--mcp-token", default=None, help="(Deprecated) Single MCP token. Use --mcp-tokens instead.")
    parser.add_argument("--api-key", default=None, help="SenseNova API key (or set SENSENOVA_API_KEY env var)")
    parser.add_argument("--model", default=None, choices=["deepseek-v4-flash", "glm-5.2"],
                        help="SenseNova LLM model: 'deepseek-v4-flash' (default) or 'glm-5.2'")
    parser.add_argument("--structure", default="original", choices=["original", "enhanced", "static"],
                        help="Video structure: 'original' (4-chapter, video clips), 'enhanced' (7-chapter with vocab+quiz+slow), or 'static' (all images, no video generation)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint in output dir")
    parser.add_argument("--no-4k", dest="no_4k", action="store_true", help="Skip the final 4K upscaling step")
    parser.add_argument("--upscale-timeout", type=int, default=3600, help="Timeout in seconds for 4K upscale (default 3600)")
    args = parser.parse_args()

    # Set API key from arg if provided
    if args.api_key:
        os.environ["SENSENOVA_API_KEY"] = args.api_key
    if args.model:
        os.environ["SENSENOVA_MODEL"] = args.model
    if not os.environ.get("SENSENOVA_API_KEY"):
        print("ERROR: SENSENOVA_API_KEY not set. Pass --api-key or set env var.")
        sys.exit(1)

    parent_dir = Path(args.output).resolve()
    parent_dir.mkdir(parents=True, exist_ok=True)

    # used_topics.json defaults to the OUTPUT dir (persists on Drive across
    # Colab sessions; the repo dir is wiped on every re-clone)
    used_topics_file = args.used_topics_file or str(parent_dir / "used_topics.json")

    # Resume: scan for incomplete checkpoint in parent dir or subdirectories
    if args.resume:
        checkpoint = _load_checkpoint(parent_dir)
        if checkpoint:
            print(f"  [Resume] Found checkpoint: {checkpoint.get('completed_steps', [])}")
        else:
            print(f"  [Resume] No incomplete checkpoint, starting fresh.")
            checkpoint = {}
    else:
        checkpoint = {}

    # Resolve topic (priority): checkpoint topic > --topic > random pick.
    # A randomly-picked topic is only marked "used" AFTER the script is saved,
    # so failed runs and repeated --resume runs don't burn the topic pool.
    if checkpoint.get("topic"):
        topic = checkpoint["topic"]
        print(f"  [Topic] Resuming topic from checkpoint: '{topic}'")
    elif args.topic:
        topic = args.topic
        print(f"  [Topic] Using specified topic: '{topic}'")
    else:
        print("  [Topic] No topic specified, picking randomly from topics.json...")
        topic = pick_random_topic(args.topics_file, used_topics_file, mark=False)
        if not topic:
            print("  [Topic] ERROR: No topics found. Please specify --topic or provide topics.json.")
            sys.exit(1)
    args.topic = topic

    # ===== Step 0: Generate script (with validation + retry) =====
    print("=" * 60)
    print("Step 0: Generating script via LLM (SenseNova DeepSeek V4 Flash)...")

    # Determine run directory:
    # - On resume: use the directory recorded in the checkpoint (or newest subfolder)
    # - On fresh start: create a subfolder named by YouTube title AFTER script generation
    if checkpoint and checkpoint.get("_run_dir") and Path(checkpoint["_run_dir"]).exists():
        work_dir = Path(checkpoint["_run_dir"])
    elif checkpoint:
        # Fallback: newest subfolder containing a checkpoint.json (by mtime)
        cands = [(p.stat().st_mtime, p) for p in parent_dir.iterdir()
                 if p.is_dir() and (p / "checkpoint.json").exists()]
        if cands:
            cands.sort(reverse=True)
            work_dir = cands[0][1]
        else:
            work_dir = parent_dir
    else:
        work_dir = parent_dir  # temporary, script goes here first

    # Create subdirs
    img_dir = work_dir / "images"
    clips_dir = work_dir / "clips"
    audio_dir = work_dir / "audio"
    sub_dir = work_dir / "subtitles"
    vid_dir = work_dir / "videos"

    script_path = work_dir / "script.json"
    if _step_done(checkpoint, "step0_script") and script_path.exists():
        print("  [Resume] Loading existing script...")
        script = json.loads(script_path.read_text(encoding="utf-8"))
        mark_topic_used(used_topics_file, topic)  # idempotent re-mark on resume
    else:
        script = _generate_script_with_retry(args.topic, args.cefr, args.lessons_dir,
                                             args.num_lines, enhanced=(args.structure == "enhanced"))
        # After script generation, create a subfolder named by YouTube title
        yt_title = script.get("youtube_title", script.get("title", topic))
        safe_title = re.sub(r'[\U0001F000-\U0001FFFF]', '', yt_title)  # remove emoji
        safe_title = re.sub(r'[\\/:*?"<>|]', '', safe_title).strip()
        safe_title = re.sub(r'\s+', '_', safe_title)[:80]  # limit length
        if not safe_title:
            safe_title = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
        work_dir = parent_dir / safe_title
        work_dir.mkdir(parents=True, exist_ok=True)
        img_dir = work_dir / "images"
        clips_dir = work_dir / "clips"
        audio_dir = work_dir / "audio"
        sub_dir = work_dir / "subtitles"
        vid_dir = work_dir / "videos"
        script_path = work_dir / "script.json"
        for d in [img_dir, clips_dir, audio_dir, sub_dir, vid_dir]:
            d.mkdir(parents=True, exist_ok=True)
        script_path.write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
        _save_checkpoint(work_dir, "step0_script", topic=topic, cefr=args.cefr, structure=args.structure)
        # Only burn the topic once its script is safely saved to disk
        mark_topic_used(used_topics_file, topic)
    print(f"  Script saved: {script_path}")
    print(f"  Title: {script.get('title', '')}")
    print(f"  Dialogue lines: {len(script.get('dialogue', []))}")

    # ===== Step 1: Initialize MCP =====
    print("\n" + "=" * 60)
    print("Step 1: Initializing TJGenerators MCP...")
    # Parse tokens: --mcp-tokens takes priority, fall back to --mcp-token (deprecated)
    raw_tokens = args.mcp_tokens or args.mcp_token or ""
    tokens = [t.strip() for t in raw_tokens.split(",") if t.strip()]
    if not tokens:
        # Windows local: auto-detect
        initialize()
    else:
        initialize(tokens=tokens)
    print("  MCP connected.")

    # ===== Step 2: Concurrent image generation + TTS audio =====
    print("\n" + "=" * 60)
    print("Step 2: Concurrent generation — images + TTS audio...")

    dialogue = script.get("dialogue", [])
    n = len(dialogue)
    is_enhanced = (args.structure == "enhanced")
    is_static = (args.structure == "static")
    tts_thread = None  # set when a fresh TTS run is started (non-resume path)

    char_a_desc = script.get("char_a_description", "friendly young man")
    char_b_desc = script.get("char_b_description", "friendly young woman")
    # Prefer the LLM-generated scene name (e.g. "pharmacy") — more precise than
    # the raw topic string, and works for outdoor scenes too
    scene = script.get("scene") or args.topic

    image_prompts = [
        (f"Character design sheet, {char_a_desc} on the left, {char_b_desc} on the right, plain white background, full body, front view, 3D cartoon style, no text, no background, 16:9", "char_scene.png"),
        (f"Scene background, {scene}, wide shot, showing all key elements of the scene, 3D cartoon style, no characters, no text, 16:9", "scene.png"),
    ]

    # Check if Step 2 already completed (images + audio exist)
    step2_done = _step_done(checkpoint, "step2_images_tts")
    char_scene_file = img_dir / "char_scene.png"
    scene_file = img_dir / "scene.png"
    all_audio_exist = all((audio_dir / f"dialogue_{i}.mp3").exists() for i in range(n))
    all_zh_exist = all((audio_dir / f"zh_{i}.mp3").exists() for i in range(n))
    narration_files = ["intro.mp3", "outro.mp3", "practice_intro.mp3"]
    narration_exist = all((audio_dir / f).exists() for f in narration_files)
    # Static mode: also check per-line dialogue images
    all_dialogue_imgs_exist = (not is_static) or all(
        (img_dir / f"dialogue_img_{i}.png").exists() for i in range(n))

    if step2_done and char_scene_file.exists() and scene_file.exists() and all_audio_exist and all_zh_exist and narration_exist and all_dialogue_imgs_exist:
        print("  [Resume] Step 2 already done, loading existing images + audio...")
        # Re-upload images to get CDN URLs (needed for video generation)
        image_urls = {}
        for _, filename in [("a", "char_scene.png"), ("b", "scene.png")]:
            filepath = str(img_dir / filename)
            print(f"  [Image] Re-uploading {filename} for CDN URL...")
            try:
                upload_result = call_tool("file_upload", {"file_path": filepath})
                if "result" in upload_result:
                    for item in upload_result["result"].get("content", []):
                        if item.get("type") == "resource":
                            res_json = json.loads(item.get("resource", {}).get("text", ""))
                            image_urls[filename] = res_json.get("file_url", "")
                        elif item.get("type") == "text":
                            m = re.search(r"(https?://[^\s`'\")]+)", item.get("text", ""))
                            if m:
                                image_urls[filename] = m.group(1)
            except Exception as e:
                print(f"    [Image] Re-upload failed: {e}")

        normal_paths = [str(audio_dir / f"dialogue_{i}.mp3") for i in range(n)]
        zh_paths = [str(audio_dir / f"zh_{i}.mp3") for i in range(n)]
        dialogue_durations = [_get_audio_duration(p) for p in normal_paths]
        narration = {}
        for name in ["intro", "outro", "practice_intro"]:
            narration[name] = str(audio_dir / f"{name}.mp3")
        tts_results = {
            "narration": narration,
            "normal_paths": normal_paths,
            "dialogue_durations": dialogue_durations,
            "zh_paths": zh_paths,
            "vocab_paths": [],
            "slow_paths": [],
            "slow_durations": [],
            "quiz_paths": [],
        }
        if is_enhanced:
            vocab_count = len(script.get("vocabulary", []))
            quiz_count = len(script.get("comprehension_questions", []))
            tts_results["vocab_paths"] = [str(audio_dir / f"vocab_{i}.mp3") for i in range(vocab_count) if (audio_dir / f"vocab_{i}.mp3").exists()]
            tts_results["quiz_paths"] = [str(audio_dir / f"quiz_{i}.mp3") for i in range(quiz_count) if (audio_dir / f"quiz_{i}.mp3").exists()]
            tts_results["slow_paths"] = [str(audio_dir / f"dialogue_slow_{i}.mp3") for i in range(n) if (audio_dir / f"dialogue_slow_{i}.mp3").exists()]
            tts_results["slow_durations"] = [_get_audio_duration(p) if p else 0.0 for p in tts_results["slow_paths"]]
        print("  [Resume] Images + audio loaded.")
    else:
        # Start TTS thread immediately (only depends on script, not images)
        tts_results = {}

        def _tts_worker():
            try:
                _generate_tts(script, dialogue, audio_dir, tts_results, is_enhanced)
            except Exception as e:
                import traceback
                traceback.print_exc()
                tts_results["fatal_error"] = f"{type(e).__name__}: {e}"

        tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        tts_thread.start()
        print("  [TTS] Started TTS generation in background thread.")

        # Generate images in main thread (TTS runs concurrently)
        image_urls = {}
        image_failed = False
        for prompt, filename in image_prompts:
            print(f"  [Image] Generating: {filename}...")
            try:
                result = call_tool("generate_image", {
                    "prompt": prompt,
                    "provider": "frontier",
                    "quality": "high",
                    "image_size": "landscape_16_9",
                    "output_format": "png",
                })
                task_id = parse_task_id(result)
                data = poll_task(task_id, interval=10, max_wait=600)
                url = data.get("url", "")
                if url:
                    dest = str(img_dir / filename)
                    download_file(url, dest)
                    image_urls[filename] = url
                    print(f"    [Image] Downloaded: {dest}")
                else:
                    print(f"    [Image] FATAL: No URL for {filename}")
                    image_failed = True
            except RuntimeError as e:
                if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                    print("\n  [FATAL] 所有 MCP Token 积分已耗尽！请充值后重新运行（--resume）继续。")
                    tts_thread.join(timeout=5)
                    sys.exit(1)
                raise

        if image_failed:
            missing = [fn for fn, _ in image_prompts if fn not in image_urls]
            print(f"  [Image] ABORTING: Missing required images: {missing}")
            tts_thread.join(timeout=5)
            sys.exit(1)

        # Static mode: generate one dialogue image per line (no video clips)
        if is_static:
            char_scene_cdn = image_urls.get("char_scene.png", "")
            print(f"  [Image] Generating {n} dialogue line images (static mode)...")
            for i, line in enumerate(dialogue):
                d_img_path = str(img_dir / f"dialogue_img_{i}.png")
                if os.path.exists(d_img_path):
                    print(f"    [Image] dialogue_img_{i} already exists, skipping")
                    continue
                img_prompt = line.get("image_prompt", f"{char_a_desc} and {char_b_desc} at {scene}, 3D cartoon style, 16:9")
                print(f"    [Image] Generating dialogue_img_{i}/{n}...")
                try:
                    gen_params = {
                        "prompt": img_prompt,
                        "provider": "frontier",
                        "quality": "high",
                        "image_size": "landscape_16_9",
                        "output_format": "png",
                    }
                    if char_scene_cdn:
                        gen_params["image_urls"] = char_scene_cdn
                    result = call_tool("generate_image", gen_params)
                    task_id = parse_task_id(result)
                    data = poll_task(task_id, interval=10, max_wait=600)
                    url = data.get("url", "")
                    if url:
                        download_file(url, d_img_path)
                        print(f"      [Image] Downloaded: dialogue_img_{i}.png")
                    else:
                        print(f"      [Image] WARNING: No URL for dialogue_img_{i}, will use scene image as fallback")
                except RuntimeError as e:
                    if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                        print("\n  [FATAL] 所有 MCP Token 积分已耗尽！请充值后重新运行（--resume）继续。")
                        tts_thread.join(timeout=5)
                        sys.exit(1)
                    print(f"      [Image] ERROR generating dialogue_img_{i}: {e}")
                except Exception as e:
                    print(f"      [Image] ERROR generating dialogue_img_{i}: {e}")

        print("  [Image] All images done. Waiting for TTS...")

    # clip_0 (scene establishing pan) doesn't depend on TTS — launch it now so
    # it generates in parallel while the TTS thread finishes
    scene_url = image_urls.get("scene.png", "")
    char_scene_url = image_urls.get("char_scene.png", "")
    clips_dir.mkdir(parents=True, exist_ok=True)
    scene_clip_thread = None

    if is_static:
        # Static mode: no video clips at all — all segments use static images
        clip_paths = []
        print("  [Static] Skipping clip_0 generation (static mode — no video clips)")
    else:
        scene_clip_task = {
            "image_urls": scene_url,
            "prompt": f"{scene}, slow camera pan, establishing shot, no characters, 3D cartoon style. The video MUST closely reference the uploaded reference image. CRITICAL: do NOT show any characters in this establishing shot.",
            "filename": "clip_0.mp4",
            "duration": 5,  # scene pan: 5s fixed
            "generate_audio": True,  # clip_0 generates ambient sound
        }
        clip_paths = [str(clips_dir / "clip_0.mp4")
                      if _file_ok(str(clips_dir / "clip_0.mp4"), 500000) else None]
        if clip_paths[0] is None:
            scene_clip_thread = threading.Thread(
                target=_generate_video_clips,
                args=([scene_clip_task], clips_dir, clip_paths),
                daemon=True,
            )
            scene_clip_thread.start()
            print("  [Video] Scene clip (clip_0) generation started in parallel with TTS.")
        else:
            print("  [Resume] clip_0 already exists, reusing.")

    # Wait for TTS if it was started (not in resume mode)
    if tts_thread is not None:
        tts_thread.join()

    # Validate TTS results BEFORE saving the step2 checkpoint — otherwise a
    # silently-failed TTS run gets persisted and grouping uses wrong durations
    _tts_err = tts_results.get("fatal_error")
    if _tts_err:
        print("  [TTS] FATAL: TTS generation thread crashed:")
        print(f"    {_tts_err}")
        raise RuntimeError(f"TTS generation failed: {_tts_err}. Fix the issue and re-run with --resume.")
    _got_en = len(tts_results.get("normal_paths", []))
    if _got_en < n:
        raise RuntimeError(
            f"TTS incomplete: got {_got_en}/{n} English dialogue audio files. "
            f"Re-run with --resume to continue.")

    # Save Step 2 checkpoint
    _save_checkpoint(work_dir, "step2_images_tts")

    # ===== Step 3: Generate video clips =====
    print("\n" + "=" * 60)

    # Collect TTS results (shared by both paths)
    narration = tts_results.get("narration", {})
    normal_paths = tts_results.get("normal_paths", [])
    dialogue_durations = tts_results.get("dialogue_durations", [])
    zh_paths = tts_results.get("zh_paths", [])

    if is_static:
        # Static mode: no video clips, no grouping — all segments use static images
        print("Step 3: Skipped (static mode — no video generation)")
        clip_paths_final = []
        group_info = []
        line_to_group = {}
        _save_checkpoint(work_dir, "step3_video")
        print(f"  TTS: {len(normal_paths)} EN + {sum(1 for p in zh_paths if p)} ZH")
    else:
        print(f"Step 3: Generating video clips (Seedance2, up to {args.clip_duration}s per group)...")

        # Scene clip thread was launched in Step 2 — join it if still running
        # (it usually finishes while TTS is still going)
        if scene_clip_thread is not None:
            scene_clip_thread.join()

        # 方案 B: group consecutive lines regardless of speaker (TTS durations ready)
        groups = build_dialogue_groups(dialogue, dialogue_durations, args.clip_duration)
        n_groups = len(groups)
        print(f"  [Group] {len(dialogue)} dialogue lines -> {n_groups} video groups:")
        for gi, g in enumerate(groups):
            print(f"    Group {gi}: lines={g['lines']} speakers={g['speakers']} total_audio={g['total_audio']:.1f}s")

        # Group clip tasks (clip_0 scene pan was already launched in Step 2)
        group_tasks = []
        for gi, group in enumerate(groups):
            # Use combined character-scene image as reference (both characters in one image)
            combined_prompt = merge_group_prompt(group, dialogue)
            video_prompt = f"{scene}. {combined_prompt} 3D cartoon style. The video MUST closely reference the uploaded reference image — the characters' appearance, clothing, and the scene must match the reference image exactly. CRITICAL: only ONE instance of each character should appear on screen — do NOT create duplicate characters or clones. Each character appears exactly once, no mirror images, no doubling."
            # Dynamic duration: match group's total TTS audio + pad between/after lines
            n_lines_in_group = len(group["lines"])
            group_total_with_pad = group["total_audio"] + n_lines_in_group * args.pad
            group_dur = round(group_total_with_pad)  # Seedance2 requires integer seconds
            group_dur = max(4, min(group_dur, 15))  # Seedance2 API: duration must be 4-15
            group_tasks.append({
                "image_urls": char_scene_url,
                "prompt": video_prompt,
                "filename": f"clip_{gi+1}.mp4",
                "duration": group_dur,
                "generate_audio": False,
            })
        video_tasks = [scene_clip_task] + group_tasks

        # Per-clip resume: reuse any existing clip file (saves credits on partial runs)
        for gi in range(len(group_tasks)):
            p = str(clips_dir / f"clip_{gi+1}.mp4")
            clip_paths.append(p if _file_ok(p, 500000) else None)
        reused = sum(1 for p in clip_paths if p is not None)
        if reused:
            print(f"  [Resume] Reusing {reused}/{len(video_tasks)} existing clips.")

        if all(p is not None for p in clip_paths):
            print("  [Resume] Step 3 already done — all clips present.")
        else:
            video_thread = threading.Thread(
                target=_generate_video_clips,
                args=(group_tasks, clips_dir, clip_paths, 1),
                daemon=True,
            )
            video_thread.start()

            print("  Waiting for group video clips to complete...")
            video_thread.join()
            print("  >> Video clips done.")
            _save_checkpoint(work_dir, "step3_video")

        clip_paths_final = clip_paths  # preserve index alignment for group_info
        ok_count = sum(1 for p in clip_paths if p is not None)
        print(f"  Clips: {ok_count}/{len(video_tasks)}, TTS: {len(normal_paths)} EN + {sum(1 for p in zh_paths if p)} ZH")

        # Build group info: concat each group's TTS audio into a single file
        group_info = []
        for gi, group in enumerate(groups):
            clip_idx = gi + 1  # scene is index 0
            clip_path = clip_paths[clip_idx] if clip_idx < len(clip_paths) else None

            group_audio_path = str(audio_dir / f"group_audio_{gi}.mp3")
            lines_audio = [normal_paths[li] for li in group["lines"]
                           if li < len(normal_paths) and os.path.exists(normal_paths[li])]
            kept_lines = [li for li in group["lines"]
                          if li < len(normal_paths) and os.path.exists(normal_paths[li])]
            n_lines = len(lines_audio)
            if not lines_audio:
                continue

            if n_lines == 1:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", lines_audio[0],
                     "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                     group_audio_path],
                    capture_output=True, timeout=30)
            else:
                inputs = []
                for la in lines_audio:
                    inputs.extend(["-i", la])
                filter_parts = []
                for j in range(n_lines):
                    line_idx = kept_lines[j]
                    line_dur = dialogue_durations[line_idx] if line_idx < len(dialogue_durations) else 3.0
                    pad_dur = line_dur + args.pad
                    filter_parts.append(f"[{j}:a]apad=whole_dur={pad_dur:.3f}[a{j}]")
                concat_inputs = "".join(f"[a{j}]" for j in range(n_lines))
                filter_parts.append(f"{concat_inputs}concat=n={n_lines}:v=0:a=1[a]")
                filter_complex = ";".join(filter_parts)
                subprocess.run(
                    ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex,
                     "-map", "[a]",
                     "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                     group_audio_path],
                    capture_output=True, timeout=60)

            if os.path.exists(group_audio_path) and os.path.getsize(group_audio_path) > 1000:
                group_total_dur = _get_audio_duration(group_audio_path)
            else:
                group_total_dur = group["total_audio"]
                group_audio_path = normal_paths[group["lines"][0]] if normal_paths else None

            group_info.append({
                "clip_path": clip_path,
                "audio_path": group_audio_path,
                "total_dur": group_total_dur,
                "lines": list(group["lines"]),
            })
            print(f"  [Group {gi}] audio concat: {group_total_dur:.1f}s, lines={group['lines']}")

        # Map: line_idx -> group_idx
        line_to_group = {}
        for gi, g in enumerate(groups):
            for li in g["lines"]:
                line_to_group[li] = gi

    # ===== Step 4: Build timeline + SRT =====
    print("\n" + "=" * 60)
    print("Step 4: Building timeline + SRT...")
    srt_path = sub_dir / "output.srt"
    meta_path = sub_dir / "meta.json"
    timeline_resumed = False
    if _step_done(checkpoint, "step4_timeline") and srt_path.exists() and meta_path.exists():
        print("  [Resume] Loading existing timeline + SRT...")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        timeline = meta["timeline"]
        # Load vars from meta for downstream steps
        narration = meta.get("narration", {})
        normal_paths = meta.get("normal_paths", [])
        zh_paths = meta.get("zh_paths", [])
        timeline_resumed = True
    if not timeline_resumed:
        tts = TTSEngine()
        if is_enhanced:
            from enhanced.timeline_enhanced import build_enhanced_timeline, build_srt_from_timeline_enhanced
            slow_paths = tts_results.get("slow_paths", [])
            slow_durations = tts_results.get("slow_durations", [])
            vocab_paths = tts_results.get("vocab_paths", [])
            quiz_paths = tts_results.get("quiz_paths", [])
            # Get vocab and quiz durations
            vocab_durations = [tts.get_duration(p) if p and os.path.exists(p) else 4.0 for p in vocab_paths]
            quiz_durations = [tts.get_duration(p) if p and os.path.exists(p) else 6.0 for p in quiz_paths]

            timeline = build_enhanced_timeline(
                script, dialogue_durations, slow_durations,
                vocab_durations, quiz_durations,
                pad=args.pad, practice_duration=args.practice_duration,
            )
            # Add audio_dur for enhanced segment types
            for seg in timeline:
                seg_type = seg.get("type", "")
                audio_idx = seg.get("audio_index", 0)
                if seg_type == "vocab":
                    ad = vocab_durations[audio_idx] if audio_idx < len(vocab_durations) else 4.0
                    seg["audio_dur"] = ad
                    seg["duration"] = ad + args.pad
                elif seg_type == "quiz":
                    ad = quiz_durations[audio_idx] if audio_idx < len(quiz_durations) else 6.0
                    seg["audio_dur"] = ad
                    seg["duration"] = ad + args.pad
                elif seg_type == "dialogue_slow":
                    ad = slow_durations[audio_idx] if audio_idx < len(slow_durations) else 4.0
                    seg["audio_dur"] = ad
                    seg["duration"] = ad + args.pad
                elif seg_type in ("dialogue", "listen_en"):
                    ad = dialogue_durations[audio_idx] if audio_idx < len(dialogue_durations) else 3.0
                    seg["audio_dur"] = ad
                    seg["duration"] = ad + args.pad
                elif seg_type == "listen_zh":
                    if audio_idx < len(zh_paths) and zh_paths[audio_idx]:
                        ad = tts.get_duration(zh_paths[audio_idx])
                    else:
                        ad = dialogue_durations[audio_idx] if audio_idx < len(dialogue_durations) else 3.0
                    seg["audio_dur"] = ad
                    seg["duration"] = ad + args.pad
                elif seg_type == "practice":
                    seg["audio_dur"] = 0
                elif seg_type == "title_card":
                    seg["audio_dur"] = seg["duration"]
                elif seg_type == "practice_intro":
                    pi_path = narration.get("practice_intro", "")
                    ad = tts.get_duration(pi_path) if pi_path and os.path.exists(pi_path) else seg["duration"] - args.pad
                    seg["audio_dur"] = ad
                    seg["duration"] = ad + args.pad
                elif seg_type == "outro":
                    out_path = narration.get("outro", "")
                    ad = tts.get_duration(out_path) if out_path and os.path.exists(out_path) else seg["duration"] - args.pad
                    seg["audio_dur"] = ad
                    seg["duration"] = ad + args.pad

            srt = build_srt_from_timeline_enhanced(timeline, gap=0.0)
        else:
            timeline = build_listening_timeline(
                script, dialogue_durations,
                pad=args.pad, practice_duration=args.practice_duration,
            )
            # Add audio_dur and adjust duration to include pad
            for seg in timeline:
                seg_type = seg.get("type", "")
                audio_idx = seg.get("audio_index", 0)
                if seg_type in ("dialogue", "listen_en"):
                    ad = dialogue_durations[audio_idx] if audio_idx < len(dialogue_durations) else 3.0
                    seg["audio_dur"] = ad
                    seg["duration"] = ad + args.pad
                elif seg_type == "listen_zh":
                    if audio_idx < len(zh_paths) and zh_paths[audio_idx]:
                        ad = tts.get_duration(zh_paths[audio_idx])
                    else:
                        ad = dialogue_durations[audio_idx] if audio_idx < len(dialogue_durations) else 3.0
                    seg["audio_dur"] = ad
                    seg["duration"] = ad + args.pad
                elif seg_type == "practice":
                    seg["audio_dur"] = 0
                elif seg_type == "title_card":
                    seg["audio_dur"] = seg["duration"]
                elif seg_type == "practice_intro":
                    pi_path = narration.get("practice_intro", "")
                    ad = tts.get_duration(pi_path) if pi_path and os.path.exists(pi_path) else seg["duration"] - args.pad
                    seg["audio_dur"] = ad
                    seg["duration"] = ad + args.pad
                elif seg_type == "outro":
                    out_path = narration.get("outro", "")
                    ad = tts.get_duration(out_path) if out_path and os.path.exists(out_path) else seg["duration"] - args.pad
                    seg["audio_dur"] = ad
                    seg["duration"] = ad + args.pad
            srt = build_srt_from_timeline(timeline, gap=0.0)
        srt_path.write_text(srt, encoding="utf-8")
        print(f"  SRT saved: {srt_path}")

        # Save meta.json
        meta = {
            "timeline": timeline,
            "script": script,
            "pad": args.pad,
            "narration": narration,
            "normal_paths": normal_paths,
            "zh_paths": zh_paths,
        }
        if is_enhanced:
            meta["slow_paths"] = tts_results.get("slow_paths", [])
            meta["vocab_paths"] = tts_results.get("vocab_paths", [])
            meta["quiz_paths"] = tts_results.get("quiz_paths", [])
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Meta saved: {meta_path}")
    _save_checkpoint(work_dir, "step4_timeline")

    # ===== Step 4.5: Generate YouTube metadata + thumbnail =====
    print("\n" + "-" * 60)
    print("Step 4.5: Generating YouTube metadata + thumbnail...")
    from thumbnail_gen import generate_thumbnail, save_youtube_metadata

    scene_img_full = str(img_dir / "scene.png")
    thumb_path = str(work_dir / "thumbnail.jpg")
    yt_meta_path = str(work_dir / "youtube_metadata.json")
    if _step_done(checkpoint, "step4.5_thumbnail") and os.path.exists(thumb_path) and os.path.exists(yt_meta_path):
        print("  [Resume] Thumbnail + YouTube metadata already exist, skipping...")
    else:
        # Use char_scene.png CDN URL as reference for thumbnail character consistency
        char_scene_cdn = image_urls.get("char_scene.png", "")
        generate_thumbnail(
            script=script,
            scene_img=scene_img_full,
            output_path=thumb_path,
            mcp_call_tool=call_tool,
            mcp_parse_task_id=parse_task_id,
            mcp_poll_task=poll_task,
            mcp_download_file=download_file,
            structure=args.structure,
            char_scene_url=char_scene_cdn,
        )

        save_youtube_metadata(
            script=script,
            timeline=timeline,
            output_path=yt_meta_path,
            structure=args.structure,
        )
        _save_checkpoint(work_dir, "step4.5_thumbnail")

    # ===== Step 5: Compose final video =====
    print("\n" + "=" * 60)
    print("Step 5: Composing final video...")
    scene_img = str(img_dir / "scene.png")

    def progress_cb(pct, msg):
        print(f"  [{pct}%] {msg}")

    # Build video filename from YouTube title
    yt_title = script.get("youtube_title", script.get("title", "final"))
    safe_vid_name = re.sub(r'[\U0001F000-\U0001FFFF]', '', yt_title)  # remove emoji
    safe_vid_name = re.sub(r'[\\/:*?"<>|]', '', safe_vid_name).strip()
    safe_vid_name = re.sub(r'\s+', '_', safe_vid_name)[:80] or "final_video"
    final_video_path = work_dir / f"{safe_vid_name}.mp4"
    if _step_done(checkpoint, "step5_compose") and final_video_path.exists():
        print("  [Resume] Final video already exists, skipping compose...")
        final_path = str(final_video_path)
    else:
        if is_enhanced:
            from enhanced.video_compose_enhanced import compose_listening_enhanced
            final_path = compose_listening_enhanced(
                work_dir=str(work_dir),
                clip_paths=clip_paths_final,
                timeline=timeline,
                script=script,
                narration=narration,
                normal_paths=normal_paths,
                zh_paths=zh_paths,
                slow_paths=tts_results.get("slow_paths", []),
                vocab_paths=tts_results.get("vocab_paths", []),
                quiz_paths=tts_results.get("quiz_paths", []),
                scene_img=scene_img,
                srt_dir=str(sub_dir),
                pad=args.pad,
                progress_cb=progress_cb,
                group_info=group_info,
                line_to_group=line_to_group,
            )
        elif is_static:
            from video_compose import compose_static
            dialogue_images = [str(img_dir / f"dialogue_img_{i}.png") for i in range(n)]
            final_path = compose_static(
                work_dir=str(work_dir),
                dialogue_images=dialogue_images,
                timeline=timeline,
                script=script,
                narration=narration,
                normal_paths=normal_paths,
                zh_paths=zh_paths,
                scene_img=scene_img,
                srt_dir=str(sub_dir),
                pad=args.pad,
                progress_cb=progress_cb,
            )
        else:
            final_path = compose_listening(
                work_dir=str(work_dir),
                clip_paths=clip_paths_final,
                timeline=timeline,
                script=script,
                narration=narration,
                normal_paths=normal_paths,
                zh_paths=zh_paths,
                scene_img=scene_img,
                srt_dir=str(sub_dir),
                pad=args.pad,
                progress_cb=progress_cb,
                group_info=group_info,
                line_to_group=line_to_group,
            )
        _save_checkpoint(work_dir, "step5_compose")

    # ===== Step 6: Upscale to 4K =====
    print("\n" + "=" * 60)
    print("Step 6: Upscaling to 4K...")
    final_4k_path = work_dir / f"{safe_vid_name}_4K.mp4"
    if args.no_4k:
        print("  [4K] Skipped (--no-4k).")
    elif _step_done(checkpoint, "step6_4k") and final_4k_path.exists():
        print("  [Resume] 4K video already exists, skipping...")
    else:
        # preset medium (not slow) — Colab CPUs can't finish preset slow for a
        # 12-min 4K encode inside any reasonable timeout
        try:
            r = subprocess.run(
                ["ffmpeg", "-i", final_path,
                 "-vf", "scale=3840:2160:flags=lanczos",
                 "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-threads", "0",
                 "-c:a", "copy",
                 str(final_4k_path), "-y"],
                capture_output=True, timeout=args.upscale_timeout)
        except subprocess.TimeoutExpired:
            print(f"  [4K] Upscaling timed out — 720p version is still available.")
            r = None
            final_4k_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"  [4K] Upscaling error: {e}")
            r = None
            final_4k_path.unlink(missing_ok=True)
        if r is not None and r.returncode == 0 and final_4k_path.exists():
            size_4k = os.path.getsize(final_4k_path) / (1024 * 1024)
            print(f"  4K video saved: {final_4k_path} ({size_4k:.1f}MB)")
            _save_checkpoint(work_dir, "step6_4k")
        elif r is not None:
            print(f"  [4K] Upscaling failed, 720p version is still available.")
            stderr = r.stderr.decode("utf-8", errors="replace")[-500:] if r.stderr else ""
            if stderr:
                print(f"  [4K] FFmpeg stderr: {stderr}")

    # Clean up checkpoint — next run will start fresh with a new topic
    cp_path = work_dir / "checkpoint.json"
    if cp_path.exists():
        cp_path.unlink()
        print("  [Checkpoint] Cleared — next run will start fresh.")

    print("\n" + "=" * 60)
    print(f"DONE! Final video: {final_path}")
    print(f"Size: {os.path.getsize(final_path) / (1024*1024):.1f}MB")
    if final_4k_path.exists():
        print(f"4K video: {final_4k_path}")
        print(f"4K Size: {os.path.getsize(final_4k_path) / (1024*1024):.1f}MB")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
            print("\n" + "=" * 60)
            print("FATAL: 所有 MCP Token 积分已耗尽！")
            print("请充值积分后重新运行（加 --resume 参数）继续上次未完成的视频。")
            print("=" * 60)
            sys.exit(1)
        raise
    except KeyboardInterrupt:
        print("\n\n用户中断。下次运行加 --resume 可继续。")
        sys.exit(0)
