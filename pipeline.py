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
import sys
import time
import threading
import subprocess
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


def _get_audio_duration(path):
    """Get audio duration via ffprobe."""
    import subprocess as _sp
    try:
        return float(_sp.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path], text=True).strip())
    except Exception:
        return 0.0
from topic_manager import pick_random_topic


def _generate_video_clips(video_tasks, clips_dir, clip_paths):
    """Generate video clips in batches. Runs in a thread.
    Each task has its own 'duration' (dynamic per-group, not fixed).
    """
    batch_size = 12

    for batch_start in range(0, len(video_tasks), batch_size):
        batch = video_tasks[batch_start:batch_start + batch_size]
        task_ids = []

        for j, task in enumerate(batch):
            idx = batch_start + j
            print(f"  [Video] Creating task {idx+1}/{len(video_tasks)}: {task['filename']}...")
            try:
                result = call_tool("generate_video", {
                    "mode": "reference_image",
                    "image_urls": task["image_urls"],
                    "prompt": task["prompt"],
                    "duration": task["duration"],
                    "ratio": "16:9",
                    "resolution": "720p",
                    "generate_audio": True,
                })
                task_id = parse_task_id(result)
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
                if download_file(url, dest) and os.path.getsize(dest) > 500000:
                    clip_paths[idx] = dest
                    print(f"    [Video] Downloaded: {filename} ({os.path.getsize(dest)//1024}KB)")
                else:
                    print(f"    [Video] WARNING: File too small, re-downloading...")
                    download_file(url, dest)
                    if os.path.getsize(dest) > 500000:
                        clip_paths[idx] = dest
                    else:
                        print(f"    [Video] ERROR: Downloaded file still too small")
            else:
                print(f"    [Video] ERROR: No video URL")

    # Retry failed clips one by one
    failed = [i for i, p in enumerate(clip_paths) if p is None]
    for idx in failed:
        print(f"  [Video] Retrying clip {idx+1}...")
        task = video_tasks[idx]
        try:
            result = call_tool("generate_video", {
                "mode": "reference_image",
                "image_urls": task["image_urls"],
                "prompt": task["prompt"],
                "duration": task["duration"],
                "ratio": "16:9",
                "resolution": "720p",
                "generate_audio": True,
            })
            task_id = parse_task_id(result)
            data = poll_task(task_id, interval=40)
            url = data.get("url", "")
            if url:
                dest = str(clips_dir / task["filename"])
                if download_file(url, dest) and os.path.getsize(dest) > 500000:
                    clip_paths[idx] = dest
                    print(f"    [Video] Retry successful: {task['filename']}")
        except Exception as e:
            print(f"    [Video] Retry failed: {e}")

    total = sum(1 for p in clip_paths if p is not None)
    print(f"  [Video] Total clips downloaded: {total}/{len(video_tasks)}")


def _generate_tts(script, dialogue, audio_dir, results):
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
            try:
                dur = tts.synth_english(text, "af_sky", path, rate="+0%")
                narration[name] = path
                print(f"  [TTS] {name}: {dur:.1f}s")
            except Exception as e:
                print(f"  [TTS] {name} FAILED: {e}")

    # Dialogue English (character voices, -15% slow)
    normal_paths = []
    dialogue_durations = []
    for i, line in enumerate(dialogue):
        text = line.get("text", "")
        speaker = line.get("speaker", "char_a")
        voice = voice_map.get(speaker, "af_sarah")
        path = str(audio_dir / f"dialogue_{i}.mp3")
        try:
            dur = tts.synth_english(text, voice, path, rate="-15%")
        except Exception as e:
            print(f"  [TTS] dialogue_{i} FAILED, using fallback: {e}")
            dur = max(len(text) * 0.07, 2.0)
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
        try:
            # synth_chinese tries edge-tts first (5 retries, 30s timeout),
            # then falls back to Kokoro automatically
            dur = tts.synth_chinese(text, voice, path, rate="-10%")
        except Exception as e:
            print(f"  [TTS] zh_{i} ALL FAILED (edge-tts + Kokoro): {e}")
            dur = max(len(text) * 0.15, 2.0)
        zh_paths.append(path)
        print(f"  [TTS] zh_{i}: {dur:.1f}s")

    results["narration"] = narration
    results["normal_paths"] = normal_paths
    results["dialogue_durations"] = dialogue_durations
    results["zh_paths"] = zh_paths
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
    parser.add_argument("--used-topics-file", default=str(Path(__file__).parent / "used_topics.json"), help="Path to used_topics.json")
    parser.add_argument("--mcp-token", required=True, help="TJGenerators MCP OAuth token")
    args = parser.parse_args()

    # Resolve topic: use --topic if given, otherwise pick randomly from topics.json
    if args.topic:
        topic = args.topic
        print(f"  [Topic] Using specified topic: '{topic}'")
    else:
        print("  [Topic] No topic specified, picking randomly from topics.json...")
        topic = pick_random_topic(args.topics_file, args.used_topics_file)
        if not topic:
            print("  [Topic] ERROR: No topics found. Please specify --topic or provide topics.json.")
            sys.exit(1)
    args.topic = topic

    work_dir = Path(args.output).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    img_dir = work_dir / "images"
    clips_dir = work_dir / "clips"
    audio_dir = work_dir / "audio"
    sub_dir = work_dir / "subtitles"
    vid_dir = work_dir / "videos"
    for d in [img_dir, clips_dir, audio_dir, sub_dir, vid_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ===== Step 0: Generate script =====
    print("=" * 60)
    print("Step 0: Generating script via LLM (SenseNova DeepSeek V4 Flash)...")
    script = generate_listening_script(args.topic, args.cefr, lessons_dir=args.lessons_dir)
    script_path = work_dir / "script.json"
    script_path.write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Script saved: {script_path}")
    print(f"  Title: {script.get('title', '')}")
    print(f"  Dialogue lines: {len(script.get('dialogue', []))}")

    # ===== Step 1: Initialize MCP =====
    print("\n" + "=" * 60)
    print("Step 1: Initializing TJGenerators MCP...")
    initialize(token=args.mcp_token)
    print("  MCP connected.")

    # ===== Step 2: Concurrent image generation + TTS audio =====
    print("\n" + "=" * 60)
    print("Step 2: Concurrent generation — images + TTS audio...")

    dialogue = script.get("dialogue", [])
    n = len(dialogue)

    char_a_desc = script.get("char_a_description", "friendly young man")
    char_b_desc = script.get("char_b_description", "friendly young woman")
    scene = args.topic

    image_prompts = [
        (f"Character design sheet, {char_a_desc} on the left, {char_b_desc} on the right, plain white background, full body, front view, 3D cartoon style, no text, no background, 16:9", "char_scene.png"),
        (f"Scene background, {scene} interior, wide shot, showing all key elements of the scene, 3D cartoon style, no characters, no text, 16:9", "scene.png"),
    ]

    # Start TTS thread immediately (only depends on script, not images)
    tts_results = {}
    tts_thread = threading.Thread(
        target=_generate_tts,
        args=(script, dialogue, audio_dir, tts_results),
        daemon=True,
    )
    tts_thread.start()
    print("  [TTS] Started TTS generation in background thread.")

    # Generate images in main thread (TTS runs concurrently)
    image_urls = {}
    for prompt, filename in image_prompts:
        print(f"  [Image] Generating: {filename}...")
        result = call_tool("generate_image", {
            "prompt": prompt,
            "provider": "frontier",
            "quality": "high",
            "image_size": "landscape_16_9",
            "output_format": "png",
        })
        task_id = parse_task_id(result)
        data = poll_task(task_id, interval=10)
        url = data.get("url", "")
        if url:
            dest = str(img_dir / filename)
            download_file(url, dest)
            image_urls[filename] = url
            print(f"    [Image] Downloaded: {dest}")
        else:
            print(f"    [Image] ERROR: No URL for {filename}")

    print("  [Image] All images done. Waiting for TTS...")

    # ===== Step 3: Generate video clips (images ready, TTS may still be running) =====
    print("\n" + "=" * 60)
    print(f"Step 3: Generating video clips (Seedance2, {args.clip_duration}s each, generate_audio=True)...")

    scene_url = image_urls.get("scene.png", "")
    char_scene_url = image_urls.get("char_scene.png", "")

    # 方案 B: group consecutive lines regardless of speaker (need TTS durations first)
    if tts_thread.is_alive():
        print("  [Group] Waiting for TTS to finish so we can group by audio durations...")
        tts_thread.join()
    print("  >> TTS audio done (needed for grouping).")

    dialogue_durations = tts_results.get("dialogue_durations", [])
    groups = build_dialogue_groups(dialogue, dialogue_durations, args.clip_duration)
    n_groups = len(groups)
    print(f"  [Group] {len(dialogue)} dialogue lines -> {n_groups} video groups:")
    for gi, g in enumerate(groups):
        print(f"    Group {gi}: lines={g['lines']} speakers={g['speakers']} total_audio={g['total_audio']:.1f}s")

    # Build video generation tasks: 1 scene clip + N group clips
    video_tasks = []
    video_tasks.append({
        "image_urls": scene_url,
        "prompt": f"{scene} interior, slow camera pan, establishing shot, no characters, 3D cartoon style. The video MUST closely reference the uploaded reference image.",
        "filename": "clip_0.mp4",
        "duration": 5,  # scene pan: 5s fixed
    })
    for gi, group in enumerate(groups):
        # Use combined character-scene image as reference (both characters in one image)
        combined_prompt = merge_group_prompt(group, dialogue)
        video_prompt = f"{scene} interior. {combined_prompt} 3D cartoon style. The video MUST closely reference the uploaded reference image — the characters' appearance, clothing, and the scene must match the reference image exactly."
        # Dynamic duration: match group's total TTS audio + pad between/after lines
        # Each line has pad seconds of silence after it, so total = sum(audio_dur) + n_lines * pad
        n_lines_in_group = len(group["lines"])
        group_total_with_pad = group["total_audio"] + n_lines_in_group * args.pad
        group_dur = round(group_total_with_pad)  # Seedance2 requires integer seconds
        video_tasks.append({
            "image_urls": char_scene_url,
            "prompt": video_prompt,
            "filename": f"clip_{gi+1}.mp4",
            "duration": group_dur,
        })

    # Start video generation in a thread (TTS may still be running)
    clip_paths = [None] * len(video_tasks)
    video_thread = threading.Thread(
        target=_generate_video_clips,
        args=(video_tasks, clips_dir, clip_paths),
        daemon=True,
    )
    video_thread.start()

    # Wait for video to complete (TTS already joined above)
    print("  Waiting for video clips to complete...")
    video_thread.join()
    print("  >> Video clips done.")

    # Collect results
    narration = tts_results.get("narration", {})
    normal_paths = tts_results.get("normal_paths", [])
    dialogue_durations = tts_results.get("dialogue_durations", [])
    zh_paths = tts_results.get("zh_paths", [])
    clip_paths_final = [p for p in clip_paths if p is not None]
    print(f"  Clips: {len(clip_paths_final)}/{len(video_tasks)}, TTS: {len(normal_paths)} EN + {sum(1 for p in zh_paths if p)} ZH")

    # Build group info: concat each group's TTS audio into a single file
    # Insert pad seconds of silence between lines so audio timeline matches SRT timeline.
    # Timeline per line: audio_dur + pad. So group audio = line0 + pad + line1 + pad + ... + lineN + pad
    group_audio_paths = []
    group_info = []
    for gi, group in enumerate(groups):
        clip_idx = gi + 1  # scene is index 0
        clip_path = clip_paths[clip_idx] if clip_idx < len(clip_paths) else None

        # Build concat list with silence between lines
        # Use FFmpeg filter_complex to insert silence between audio files
        group_audio_path = str(audio_dir / f"group_audio_{gi}.mp3")
        n_lines = len(group["lines"])
        lines_audio = [normal_paths[li] for li in group["lines"]
                       if li < len(normal_paths) and os.path.exists(normal_paths[li])]
        if not lines_audio:
            continue

        if n_lines == 1:
            # Single line: just copy
            subprocess.run(
                ["ffmpeg", "-y", "-i", lines_audio[0],
                 "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                 group_audio_path],
                capture_output=True, timeout=30)
        else:
            # Multiple lines: insert pad silence between each + pad at end
            # Build filter: [0:a][silence0][1:a][silence1]...[Na]concat=N+(N-1):v=0:a=1
            inputs = []
            for la in lines_audio:
                inputs.extend(["-i", la])
            # Build filter_complex: adelay for silence, then concat
            # Simpler approach: use anullsrc + concat demuxer
            concat_input = str(audio_dir / f"group_audio_{gi}_list.txt")
            silence_files = []
            with open(concat_input, "w", encoding="utf-8") as f:
                for j, la in enumerate(lines_audio):
                    f.write(f"file '{la}'\n")
                    # Add silence after each line (including last, to match timeline pad)
                    sil_path = str(audio_dir / f"silence_{gi}_{j}.mp3")
                    subprocess.run(
                        ["ffmpeg", "-y", "-f", "lavfi", "-i",
                         f"anullsrc=stereo:44100", "-t", f"{args.pad}",
                         "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                         sil_path],
                        capture_output=True, timeout=10)
                    silence_files.append(sil_path)
                    f.write(f"file '{sil_path}'\n")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_input,
                 "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                 group_audio_path],
                capture_output=True, timeout=30)
            # Cleanup silence files
            for sf in silence_files:
                try:
                    os.remove(sf)
                except OSError:
                    pass

        if os.path.exists(group_audio_path) and os.path.getsize(group_audio_path) > 1000:
            group_total_dur = _get_audio_duration(group_audio_path)
        else:
            group_total_dur = group["total_audio"]
            group_audio_path = normal_paths[group["lines"][0]] if normal_paths else None

        group_audio_paths.append(group_audio_path)
        group_info.append({
            "clip_path": clip_path,
            "audio_path": group_audio_path,
            "total_dur": group_total_dur,
            "lines": list(group["lines"]),
        })
        print(f"  [Group {gi}] audio concat: {group_total_dur:.1f}s, lines={group['lines']}")

    # Map: line_idx -> group_idx (for compose to know which group a line belongs to)
    line_to_group = {}
    for gi, g in enumerate(groups):
        for li in g["lines"]:
            line_to_group[li] = gi

    # ===== Step 4: Build timeline + SRT =====
    print("\n" + "=" * 60)
    print("Step 4: Building timeline + SRT...")
    timeline = build_listening_timeline(
        script, dialogue_durations,
        pad=args.pad, practice_duration=args.practice_duration,
    )

    # Add audio_dur and adjust duration to include pad
    tts = TTSEngine()
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

    # Build SRT
    srt = build_srt_from_timeline(timeline, gap=0.0)
    srt_path = sub_dir / "output.srt"
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
    meta_path = sub_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Meta saved: {meta_path}")

    # ===== Step 5: Compose final video =====
    print("\n" + "=" * 60)
    print("Step 5: Composing final video...")
    scene_img = str(img_dir / "scene.png")

    def progress_cb(pct, msg):
        print(f"  [{pct}%] {msg}")

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

    print("\n" + "=" * 60)
    print(f"DONE! Final video: {final_path}")
    print(f"Size: {os.path.getsize(final_path) / (1024*1024):.1f}MB")


if __name__ == "__main__":
    main()
