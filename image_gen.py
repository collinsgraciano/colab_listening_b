"""Image generation: character/scene images, per-line dialogue images, resume check."""
import json
import os
import re
import sys
from pathlib import Path

from mcp_client import call_tool, parse_task_id, poll_task, download_file
from checkpoint import step_done
from media_utils import get_duration as _get_audio_duration


def check_step2_resume(checkpoint, script, dirs, n, is_quest, is_stop_motion=False):
    """Check if Step 2 can be resumed from existing files. Returns (tts_results, image_urls) or None."""
    img_dir, audio_dir = dirs["images"], dirs["audio"]
    step2_done = step_done(checkpoint, "step2_images_tts")
    char_scene_file = img_dir / "char_scene.png"
    scene_file = img_dir / "scene.png"
    all_audio_exist = all((audio_dir / f"dialogue_{i}.mp3").exists() for i in range(n))
    if is_quest:
        all_zh_exist = True
    else:
        all_zh_exist = all((audio_dir / f"zh_{i}.mp3").exists() for i in range(n))
    narration_files = (["welcome.mp3", "hook.mp3", "outro.mp3"] if is_quest
                       else ["intro.mp3", "outro.mp3", "practice_intro.mp3"])
    narration_exist = all((audio_dir / f).exists() for f in narration_files)
    # For image mode: need dialogue images unless stop_motion (which needs pose images)
    struct_val = checkpoint.get("structure", "")
    needs_dialogue_imgs = (struct_val not in ("image", "quest")) or (
        struct_val == "image" and not is_stop_motion)
    all_dialogue_imgs_exist = (not needs_dialogue_imgs) or all(
        (img_dir / f"dialogue_img_{i}.png").exists() for i in range(n))
    # For stop_motion: need pose images
    if is_stop_motion:
        all_pose_imgs_exist = all(
            os.path.exists(str(img_dir / f"pose_{i}_{j}.png"))
            for i in range(n) for j in range(4))
    else:
        all_pose_imgs_exist = True

    if not (step2_done and char_scene_file.exists() and scene_file.exists()
            and all_audio_exist and all_zh_exist
            and narration_exist and all_dialogue_imgs_exist and all_pose_imgs_exist):
        return None

    print("  [Resume] Step 2 already done, loading existing images + audio...")
    reupload_files = ["char_scene.png", "scene.png"]
    image_urls = {}
    for filename in reupload_files:
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
    if is_quest:
        zh_paths = [""] * n
    else:
        zh_paths = [str(audio_dir / f"zh_{i}.mp3") for i in range(n)]
    dialogue_durations = [_get_audio_duration(p) for p in normal_paths]
    narration = {}
    for name in (["welcome", "hook", "outro"] if is_quest
                 else ["intro", "outro", "practice_intro"]):
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
    print("  [Resume] Images + audio loaded.")
    return tts_results, image_urls


def generate_images(image_prompts, img_dir, tts_thread):
    """Generate character/scene images via MCP. Returns image_urls dict."""
    image_urls = {}
    image_failed = False
    for prompt, filename in image_prompts:
        print(f"  [Image] Generating: {filename}...")
        try:
            result = call_tool("generate_image", {
                "prompt": prompt,
                "provider": "seedream",
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
                if tts_thread:
                    tts_thread.join(timeout=5)
                sys.exit(1)
            raise

    if image_failed:
        missing = [fn for fn, _ in image_prompts if fn not in image_urls]
        print(f"  [Image] ABORTING: Missing required images: {missing}")
        if tts_thread:
            tts_thread.join(timeout=5)
        sys.exit(1)
    return image_urls


def generate_dialogue_images(dialogue, img_dir, char_a_desc, char_b_desc, scene,
                               is_quest, char_scene_cdn, char_scene_c_cdn,
                               tts_thread):
    """Generate per-line dialogue images for static/quest modes (5 concurrent)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    n = len(dialogue)
    mode_label = "quest" if is_quest else "image"
    print(f"  [Image] Generating {n} dialogue line images ({mode_label} mode, 5 concurrent)...")

    def _gen_one(i, line):
        d_img_path = str(img_dir / f"dialogue_img_{i}.png")
        if os.path.exists(d_img_path):
            print(f"    [Image] dialogue_img_{i} already exists, skipping")
            return i, True
        img_prompt = line.get("image_prompt",
                                f"{char_a_desc} and {char_b_desc} at {scene}, 3D cartoon style, 16:9")
        if is_quest and line.get("phase") == "core" and char_scene_c_cdn:
            ref_cdn = char_scene_c_cdn
        else:
            ref_cdn = char_scene_cdn
        print(f"    [Image] Generating dialogue_img_{i}/{n}...")
        try:
            gen_params = {
                "prompt": img_prompt,
                "provider": "frontier",
                "quality": "high",
                "image_size": {"width": 1280, "height": 720},
                "output_format": "png",
            }
            if ref_cdn:
                gen_params["image_urls"] = ref_cdn
            result = call_tool("generate_image", gen_params)
            task_id = parse_task_id(result)
            data = poll_task(task_id, interval=10, max_wait=600)
            url = data.get("url", "")
            if url:
                download_file(url, d_img_path)
                print(f"      [Image] Downloaded: dialogue_img_{i}.png")
                return i, True
            print(f"      [Image] WARNING: No URL for dialogue_img_{i}, will use scene image as fallback")
            return i, False
        except RuntimeError as e:
            if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                print("\n  [FATAL] 所有 MCP Token 积分已耗尽！请充值后重新运行（--resume）继续。")
                if tts_thread:
                    tts_thread.join(timeout=5)
                sys.exit(1)
            print(f"      [Image] ERROR generating dialogue_img_{i}: {e}")
            return i, False
        except Exception as e:
            print(f"      [Image] ERROR generating dialogue_img_{i}: {e}")
            return i, False

    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(_gen_one, i, line) for i, line in enumerate(dialogue)]
        for fut in as_completed(futs):
            fut.result()


def generate_pose_images(dialogue, img_dir, char_a_desc, char_b_desc, scene,
                          char_a_ref_cdn, char_b_ref_cdn, tts_thread):
    """Generate per-line character pose atlas (2×2 grid) and split into 4 poses.

    For each dialogue line, generates ONE image containing 4 poses arranged in
    a 2×2 grid (top-left: speaking, top-right: listening, bottom-left:
    thinking, bottom-right: reacting). The image is then split into 4 separate
    pose_{i}_{j}.png files. This guarantees character consistency across poses
    and reduces API calls by 4×.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image as PILImage
    n = len(dialogue)
    print(f"  [PoseAtlas] Generating {n} pose atlases (2×2 grid → 4 poses each)...")

    def _gen_one(line_idx, line):
        # Check if split poses already exist
        all_exist = all(
            os.path.exists(str(img_dir / f"pose_{line_idx}_{j}.png"))
            for j in range(4)
        )
        if all_exist:
            print(f"    [PoseAtlas] line {line_idx} poses already exist, skipping")
            return (line_idx, True)

        atlas_path = str(img_dir / f"pose_atlas_{line_idx}.png")
        poses = line.get("poses", [])
        if not poses:
            img_prompt = line.get("image_prompt", f"{char_a_desc} at {scene}")
            poses = [img_prompt, img_prompt]

        # Pick the correct character reference based on speaker
        speaker = line.get("speaker", "char_a")
        if speaker == "char_b":
            ref_cdn = char_b_ref_cdn
            char_desc = char_b_desc
        else:
            ref_cdn = char_a_ref_cdn
            char_desc = char_a_desc

        # Build 2×2 atlas prompt: 4 poses in one image
        pose_a = poses[0] if len(poses) > 0 else "speaking with mouth open"
        pose_b = poses[1] if len(poses) > 1 else "listening with slight smile"
        # Pad to 4 poses with variations
        pose_c = "thinking with hand on chin"
        pose_d = "surprised with raised eyebrows"

        # Ensure character description is in each pose
        def _with_desc(p):
            if char_desc and char_desc.lower() not in p.lower():
                return f"{char_desc}, {p}"
            return p

        atlas_prompt = (
            f"2×2 grid character pose sheet, four poses of the same character, "
            f"top-left: {_with_desc(pose_a)}, "
            f"top-right: {_with_desc(pose_b)}, "
            f"bottom-left: {_with_desc(pose_c)}, "
            f"bottom-right: {_with_desc(pose_d)}, "
            f"half-body close-up, waist up, all four poses same character same outfit, "
            f"plain white background, 3D cartoon style, "
            f"cel-shaded with thin clean black outline tightly hugging the character silhouette, "
            f"no props, no objects, no scene, no text"
        )

        print(f"    [PoseAtlas] Generating atlas for line {line_idx}...")
        try:
            gen_params = {
                "prompt": atlas_prompt,
                "provider": "seedream",
                "image_size": {"width": 1280, "height": 1280},
                "output_format": "png",
            }
            if ref_cdn:
                gen_params["image_urls"] = ref_cdn
            result = call_tool("generate_image", gen_params)
            task_id = parse_task_id(result)
            data = poll_task(task_id, interval=10, max_wait=600)
            url = data.get("url", "")
            if not url:
                print(f"      [PoseAtlas] WARNING: No URL for line {line_idx}")
                return (line_idx, False)

            download_file(url, atlas_path)
            print(f"      [PoseAtlas] Downloaded atlas: pose_atlas_{line_idx}.png")

            # Split into 4 quadrants
            atlas = PILImage.open(atlas_path).convert("RGBA")
            w, h = atlas.size
            half_w, half_h = w // 2, h // 2
            quads = [
                (0, 0, half_w, half_h),          # top-left
                (half_w, 0, w, half_h),           # top-right
                (0, half_h, half_w, h),           # bottom-left
                (half_w, half_h, w, h),           # bottom-right
            ]
            for j, (left, top, right, bottom) in enumerate(quads):
                cell = atlas.crop((left, top, right, bottom))
                out_path = str(img_dir / f"pose_{line_idx}_{j}.png")
                cell.save(out_path)
                print(f"      [PoseAtlas] Split: pose_{line_idx}_{j}.png ({cell.size})")

            # Clean up atlas
            if os.path.exists(atlas_path):
                os.remove(atlas_path)

            return (line_idx, True)
        except RuntimeError as e:
            if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                print("\n  [FATAL] 所有 MCP Token 积分已耗尽！")
                if tts_thread:
                    tts_thread.join(timeout=5)
                sys.exit(1)
            print(f"      [PoseAtlas] ERROR line {line_idx}: {e}")
            return (line_idx, False)
        except Exception as e:
            print(f"      [PoseAtlas] ERROR line {line_idx}: {e}")
            return (line_idx, False)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(_gen_one, i, line) for i, line in enumerate(dialogue)]
        for fut in as_completed(futs):
            fut.result()

    print(f"  [PoseAtlas] Done — {n} atlases → {n*4} pose images.")


def generate_quest_atlases(script, img_dir, tts_thread):
    """Generate character pose atlases for quest mode.

    All characters (char_a, char_b, char_c, host): 4×2 grid (8 poses each).
    Shared style prefix ensures visual consistency across all characters.
    """
    from PIL import Image as PILImage

    _STYLE = ("3D cartoon style, Pixar-like, warm soft lighting, "
              "cel-shaded with thin clean black outline, "
              "vibrant saturated colors, smooth surfaces")

    chars = [
        ("char_a", script.get("char_a_description", "friendly young man"), 8),
        ("char_b", script.get("char_b_description", "friendly young woman"), 8),
        ("char_c", script.get("char_c_description", "friendly staff member"), 8),
        ("host", script.get("host_description", "friendly young woman with short brown hair, wearing a smart blue blazer, warm smile, professional TV host appearance"), 8),
    ]
    # Also generate per-character half-body reference images
    ref_urls = {}
    for char_key, char_desc, _ in chars:
        ref_path = str(img_dir / f"{char_key}_ref.png")
        if not os.path.exists(ref_path):
            print(f"  [QuestRef] Generating {char_key}_ref...")
            try:
                result = call_tool("generate_image", {
                    "prompt": (f"Character reference, {char_desc}, single character, "
                               f"plain white background, half-body close-up, waist up, "
                               f"front view, 3D cartoon style, no text, no background scene"),
                    "provider": "seedream",
                    "image_size": {"width": 1280, "height": 720},
                    "output_format": "png",
                })
                task_id = parse_task_id(result)
                data = poll_task(task_id, interval=10, max_wait=600)
                url = data.get("url", "")
                if url:
                    download_file(url, ref_path)
                    ref_urls[char_key] = url
                    print(f"    [QuestRef] Downloaded: {char_key}_ref.png")
            except Exception as e:
                if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                    if tts_thread:
                        tts_thread.join(timeout=5)
                    sys.exit(1)
                print(f"    [QuestRef] ERROR: {e}")
        else:
            # Re-upload for CDN URL
            try:
                upload_result = call_tool("file_upload", {"file_path": ref_path})
                for item in upload_result.get("result", {}).get("content", []):
                    if item.get("type") == "resource":
                        import json
                        res_json = json.loads(item["resource"]["text"])
                        ref_urls[char_key] = res_json.get("file_url", "")
            except Exception:
                pass

    # Generate atlases — all characters use 4×2 (8 poses)
    for char_key, char_desc, n_poses in chars:
        all_exist = all(
            os.path.exists(str(img_dir / f"pose_{char_key}_{j}.png"))
            for j in range(n_poses)
        )
        if all_exist:
            print(f"  [QuestAtlas] {char_key} poses already exist, skipping")
            continue

        atlas_path = str(img_dir / f"pose_atlas_{char_key}.png")
        atlas_prompt = (
            f"4x2 grid character pose sheet, eight poses of the same character, "
            f"{char_desc}, "
            f"top row left to right: speaking with mouth open, listening with slight smile, "
            f"thinking with hand on chin, surprised with raised eyebrows, "
            f"bottom row left to right: nodding in agreement, waving right hand, "
            f"pointing forward, laughing with eyes closed, "
            f"half-body close-up, waist up, all eight poses same character same outfit, "
            f"plain white background, {_STYLE}, "
            f"no props, no objects, no scene, no text"
        )
        grid_w, grid_h = 4, 2
        img_size = {"width": 2560, "height": 1280}

        print(f"  [QuestAtlas] Generating {grid_w}×{grid_h} atlas for {char_key} ({n_poses} poses)...")
        try:
            gen_params = {
                "prompt": atlas_prompt,
                "provider": "seedream",
                "image_size": img_size,
                "output_format": "png",
            }
            ref_cdn = ref_urls.get(char_key, "")
            if ref_cdn:
                gen_params["image_urls"] = ref_cdn
            result = call_tool("generate_image", gen_params)
            task_id = parse_task_id(result)
            data = poll_task(task_id, interval=10, max_wait=600)
            url = data.get("url", "")
            if not url:
                # Debug: parse raw_json and print only output.data section
                import json as _json
                raw = data.get("raw_json", "")
                if raw:
                    try:
                        rj = _json.loads(raw)
                        out = rj.get("output") or {}
                        d = out.get("data") or {}
                        print(f"    [QuestAtlas] DEBUG {char_key} output.data keys={list(d.keys())}")
                        print(f"    [QuestAtlas] DEBUG {char_key} output.data={_json.dumps(d, ensure_ascii=False)[:600]}")
                        res = d.get("result") or {}
                        print(f"    [QuestAtlas] DEBUG {char_key} result={_json.dumps(res, ensure_ascii=False)[:400]}")
                    except Exception:
                        pass
                print(f"    [QuestAtlas] WARNING: No URL for {char_key}")
                continue
            download_file(url, atlas_path)
            print(f"    [QuestAtlas] Downloaded: pose_atlas_{char_key}.png")

            atlas = PILImage.open(atlas_path).convert("RGBA")
            w, h = atlas.size
            cw, ch = w // grid_w, h // grid_h
            idx = 0
            for row in range(grid_h):
                for col in range(grid_w):
                    cell = atlas.crop((col * cw, row * ch, (col + 1) * cw, (row + 1) * ch))
                    out_path = str(img_dir / f"pose_{char_key}_{idx}.png")
                    cell.save(out_path)
                    print(f"    [QuestAtlas] Split: pose_{char_key}_{idx}.png ({cell.size})")
                    idx += 1

            if os.path.exists(atlas_path):
                os.remove(atlas_path)
        except RuntimeError as e:
            if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                if tts_thread:
                    tts_thread.join(timeout=5)
                sys.exit(1)
            print(f"    [QuestAtlas] ERROR {char_key}: {e}")
        except Exception as e:
            print(f"    [QuestAtlas] ERROR {char_key}: {e}")

    print(f"  [QuestAtlas] Done — 4 characters × 8 poses = 32 pose images.")
