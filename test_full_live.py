"""Full live test: LLM script → MCP images → TTS → compose video."""
import sys, os, json, shutil, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ["SENSENOVA_API_KEY"] = "sk-8Tr86c17YvA5jBEoem2uYYAQGXGzmpDU"

from quest.llm_client_quest import generate_quest_script, split_phase_lines
from pipeline import _validate_script

# ============================================================
# Step 1: Generate LLM script (10 lines for speed)
# ============================================================
print("=" * 60)
print("STEP 1: Generate LLM script")
print("=" * 60)
topic = "At the Ice Cream Shop"
script = generate_quest_script(topic, cefr="A1", num_lines=10)
Path("test_quest_script.json").write_text(
    json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")

ok, msg = _validate_script(script, 10, quest=True)
if not ok:
    print(f"VALIDATION FAILED: {msg}")
    sys.exit(1)
print(f"Validation: PASS ({msg})")

dl = script.get("dialogue", [])
print(f"Title: {script.get('title','')}")
print(f"Host: {script.get('host_description','')}")
print(f"Host BG: {script.get('host_bg_prompt','')}")
print(f"Scene images: {[s.get('label','') for s in script.get('scene_images',[])]}")
print(f"Lines: {len(dl)}")
for i, line in enumerate(dl):
    print(f"  [{i}] {line.get('phase',''):8s} {line.get('speaker',''):6s} on_screen={line.get('on_screen',[])} | {line.get('text','')}")

# ============================================================
# Step 2: Generate images via MCP
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Generate images via MCP")
print("=" * 60)
from mcp_client import initialize as _mcp_init
_mcp_init()  # auto-detect token from ~/.codely-cli/mcp-oauth-tokens.json
from mcp_client import call_tool, parse_task_id, poll_task, download_file

test_dir = Path("test_output")
img_dir = test_dir / "images"
# Clean old images
if img_dir.exists():
    shutil.rmtree(img_dir)
img_dir.mkdir(parents=True, exist_ok=True)

# Collect all image generation tasks
tasks = []

# Scene backgrounds
scene_images = script.get("scene_images", [])
if not scene_images:
    scene_images = [{"prompt": f"a {script.get('scene','shop')} interior, 3D cartoon style, 16:9, no people", "label": "main"}]
for si, si_data in enumerate(scene_images[:5]):
    prompt = si_data.get("prompt", f"a {script.get('scene','shop')} interior, 3D cartoon style, 16:9, no people")
    fname = f"scene_{si}.png" if si > 0 else "scene.png"
    print(f"  Submitting: {fname} ({si_data.get('label','')})")
    result = call_tool("generate_image", {
        "prompt": prompt, "provider": "seedream", "image_size": "landscape_16_9",
    })
    tid = parse_task_id(result)
    tasks.append((tid, str(img_dir / fname), f"scene_{si}"))

# Host background
host_bg_prompt = script.get("host_bg_prompt", "a bright modern TV studio set, 3D cartoon style, 16:9, no people")
print(f"  Submitting: host_bg.png")
result = call_tool("generate_image", {
    "prompt": host_bg_prompt, "provider": "seedream", "image_size": "landscape_16_9",
})
tid = parse_task_id(result)
tasks.append((tid, str(img_dir / "host_bg.png"), "host_bg"))

# Character atlases: char_a/b/c = 2x2 (4 poses), host = 4x2 (8 poses)
chars = [
    ("char_a", script.get("char_a_description", "friendly young man"), 4, 2, 2),
    ("char_b", script.get("char_b_description", "friendly young woman"), 4, 2, 2),
    ("char_c", script.get("char_c_description", "friendly staff member"), 4, 2, 2),
    ("host", script.get("host_description", "friendly TV host"), 8, 4, 2),
]
for char_key, char_desc, n_poses, grid_w, grid_h in chars:
    if n_poses == 8:
        atlas_prompt = (
            f"4x2 grid character pose sheet, eight poses of the same character, {char_desc}, "
            f"top row: speaking with mouth open, listening with slight smile, "
            f"thinking with hand on chin, surprised with raised eyebrows, "
            f"bottom row: nodding, waving right hand, pointing forward, laughing, "
            f"half-body close-up, waist up, all same outfit, plain white background, "
            f"3D cartoon style, cel-shaded, no props, no scene, no text"
        )
        img_size = {"width": 2560, "height": 1280}
    else:
        atlas_prompt = (
            f"2x2 grid character pose sheet, four poses of the same character, {char_desc}, "
            f"top-left: speaking with mouth open, top-right: listening with slight smile, "
            f"bottom-left: thinking with hand on chin, bottom-right: surprised, "
            f"half-body close-up, waist up, all same outfit, plain white background, "
            f"3D cartoon style, cel-shaded, no props, no scene, no text"
        )
        img_size = {"width": 1280, "height": 1280}

    print(f"  Submitting: pose_atlas_{char_key}.png ({grid_w}x{grid_h}, {n_poses} poses)")
    result = call_tool("generate_image", {
        "prompt": atlas_prompt, "provider": "seedream", "image_size": img_size,
    })
    tid = parse_task_id(result)
    tasks.append((tid, str(img_dir / f"pose_atlas_{char_key}.png"), f"atlas_{char_key}"))

# Poll and download all images
print(f"\n  Waiting for {len(tasks)} images...")
from PIL import Image as PILImage
for tid, out_path, label in tasks:
    data = poll_task(tid, interval=10, max_wait=600)
    url = data.get("url", "")
    if url:
        download_file(url, out_path)
        print(f"    Downloaded: {label} -> {Path(out_path).name} ({os.path.getsize(out_path)//1024}KB)")
    else:
        print(f"    FAILED: {label}")

# Split atlases into individual poses
print("\n  Splitting atlases...")
for char_key, _, n_poses, grid_w, grid_h in chars:
    atlas_path = img_dir / f"pose_atlas_{char_key}.png"
    if not atlas_path.exists():
        print(f"    WARNING: atlas not found: {atlas_path}")
        continue
    atlas = PILImage.open(atlas_path).convert("RGBA")
    w, h = atlas.size
    cw, ch = w // grid_w, h // grid_h
    idx = 0
    for row in range(grid_h):
        for col in range(grid_w):
            cell = atlas.crop((col*cw, row*ch, (col+1)*cw, (row+1)*ch))
            cell.save(str(img_dir / f"pose_{char_key}_{idx}.png"))
            idx += 1
    print(f"    {char_key}: {n_poses} poses split OK")

# ============================================================
# Step 3: Generate TTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Generate TTS")
print("=" * 60)
from tts_engine import TTSEngine, build_voice_map
tts = TTSEngine()
voice_map = build_voice_map(script)
audio_dir = test_dir / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)

narration = {}
for name, text in [("welcome", script.get("welcome_en","")),
                   ("hook", script.get("hook_intro_en","")),
                   ("outro", script.get("outro",""))]:
    if text:
        path = str(audio_dir / f"{name}.mp3")
        dur = tts.synth_english(text, "af_sky", path, rate="0%")
        narration[name] = path
        print(f"  {name}: {dur:.1f}s")

normal_paths = []
dialogue_durations = []
for i, line in enumerate(dl):
    text = line.get("text", "")
    speaker = line.get("speaker", "char_a")
    voice = voice_map.get(speaker, "af_sarah")
    path = str(audio_dir / f"dialogue_{i}.mp3")
    dur = tts.synth_english(text, voice, path, rate="0%")
    normal_paths.append(path)
    dialogue_durations.append(dur)
    print(f"  dialogue_{i}: {dur:.1f}s ({voice})")

# ============================================================
# Step 4: Build timeline + compose
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Build timeline + compose video")
print("=" * 60)
from quest.timeline_quest import build_quest_timeline
from timeline_enrich import enrich_timeline

pad = 0.4
timeline = build_quest_timeline(script, dialogue_durations, pad=pad)
enrich_timeline(timeline, tts, pad, dialogue_durations, [], narration)
print(f"Timeline: {len(timeline)} segments")

scene_img = str(img_dir / "scene.png")
host_bg = str(img_dir / "host_bg.png")
if not os.path.exists(host_bg):
    host_bg = scene_img

scene_bg_list = [scene_img]
for si in range(1, 5):
    p = str(img_dir / f"scene_{si}.png")
    if os.path.exists(p):
        scene_bg_list.append(p)
    else:
        scene_bg_list.append(scene_img)
print(f"Scene backgrounds: {len(scene_bg_list)}")

# Build char_pose_map
char_pose_map = {}
for ck in ("char_a", "char_b", "char_c"):
    poses = [str(img_dir / f"pose_{ck}_{j}.png") for j in range(4)]
    if all(os.path.exists(p) for p in poses):
        char_pose_map[ck] = poses
host_poses = [str(img_dir / f"pose_host_{j}.png") for j in range(8)]
if not all(os.path.exists(p) for p in host_poses):
    host_poses = [str(img_dir / f"pose_host_{j}.png") for j in range(4)]
char_pose_map["host"] = host_poses
print(f"Pose map: {list(char_pose_map.keys())} (host={len(host_poses)} poses)")

from quest.video_compose_quest import compose_quest
final_path = compose_quest(
    work_dir=str(test_dir),
    pose_images=[],
    char_pose_map=char_pose_map,
    host_poses=host_poses,
    host_bg=host_bg,
    scene_bg_list=scene_bg_list,
    timeline=timeline,
    script=script,
    narration=narration,
    normal_paths=normal_paths,
    scene_img=scene_img,
    srt_dir=str(test_dir),
    pad=pad,
    progress_cb=lambda pct, msg: print(f"  [{pct:3d}%] {msg}") if pct % 25 == 0 else None,
)

print(f"\n{'=' * 60}")
print(f"DONE: {final_path}")
print(f"Size: {os.path.getsize(final_path) / (1024*1024):.1f}MB")
