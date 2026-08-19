"""test2: Full live test with all fixes — subtitle, pad, slow poses, 8 host poses."""
import sys, os, json, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ["SENSENOVA_API_KEY"] = "sk-8Tr86c17YvA5jBEoem2uYYAQGXGzmpDU"

base = Path(__file__).parent.resolve()
test_dir = base / "test2"
img_dir = test_dir / "images"
audio_dir = test_dir / "audio"

# Clean old test2
if test_dir.exists():
    shutil.rmtree(test_dir)
img_dir.mkdir(parents=True, exist_ok=True)
audio_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Step 1: LLM script
# ============================================================
print("=" * 60)
print("STEP 1: LLM script")
print("=" * 60)
from quest.llm_client_quest import generate_quest_script
from pipeline import _validate_script

script = generate_quest_script("At the Ice Cream Shop", cefr="A1", num_lines=10)
ok, msg = _validate_script(script, 10, quest=True)
print(f"Validation: {ok} ({msg})")
dl = script.get("dialogue", [])
print(f"Host: {script.get('host_description','')}")
print(f"Host BG: {script.get('host_bg_prompt','')[:80]}")
print(f"Scene images: {[s.get('label','') for s in script.get('scene_images',[])]}")
for i, line in enumerate(dl):
    print(f"  [{i}] {line.get('phase',''):8s} {line.get('speaker',''):6s} on_screen={line.get('on_screen',[])} | {line.get('text','')}")

# ============================================================
# Step 2: Generate images via TJGenerators MCP tools
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Generate images")
print("=" * 60)
from mcp_client import initialize as _mcp_init
_mcp_init()
from mcp_client import call_tool, parse_task_id, poll_task, download_file
from PIL import Image as PILImage

tasks = []

# Scene backgrounds (multiple for variety)
scene_images = script.get("scene_images", [])
if not scene_images:
    scene_images = [{"prompt": f"a {script.get('scene','shop')} interior, 3D cartoon style, 16:9, no people", "label": "main"}]
for si, si_data in enumerate(scene_images[:5]):
    prompt = si_data.get("prompt", "")
    fname = "scene.png" if si == 0 else f"scene_{si}.png"
    print(f"  Submit: {fname} ({si_data.get('label','')})")
    result = call_tool("generate_image", {"prompt": prompt, "provider": "seedream", "image_size": "landscape_16_9"})
    tasks.append((parse_task_id(result), str(img_dir / fname), f"scene_{si}"))

# Host background (TV studio)
host_bg_prompt = script.get("host_bg_prompt", "a bright modern TV studio, 3D cartoon, no people, 16:9")
print("  Submit: host_bg.png")
result = call_tool("generate_image", {"prompt": host_bg_prompt, "provider": "seedream", "image_size": "landscape_16_9"})
tasks.append((parse_task_id(result), str(img_dir / "host_bg.png"), "host_bg"))

# Character atlases
chars = [
    ("char_a", script.get("char_a_description", ""), 4, 2, 2),
    ("char_b", script.get("char_b_description", ""), 4, 2, 2),
    ("char_c", script.get("char_c_description", ""), 4, 2, 2),
    ("host", script.get("host_description", ""), 8, 4, 2),
]
for ck, desc, n_poses, gw, gh in chars:
    if n_poses == 8:
        p = (f"4x2 grid, eight poses of the same character: {desc}. "
             f"Top row: speaking, listening, thinking, surprised. "
             f"Bottom row: nodding, waving, pointing, laughing. "
             f"Half-body close-up, waist up, plain white background, 3D cartoon, cel-shaded, no props, no text")
        sz = {"width": 2560, "height": 1280}
    else:
        p = (f"2x2 grid, four poses of the same character: {desc}. "
             f"Top-left: speaking, top-right: listening, "
             f"bottom-left: thinking, bottom-right: surprised. "
             f"Half-body close-up, waist up, plain white background, 3D cartoon, cel-shaded, no props, no text")
        sz = {"width": 1280, "height": 1280}
    print(f"  Submit: pose_atlas_{ck}.png ({gw}x{gh}, {n_poses} poses)")
    result = call_tool("generate_image", {"prompt": p, "provider": "seedream", "image_size": sz})
    tasks.append((parse_task_id(result), str(img_dir / f"pose_atlas_{ck}.png"), f"atlas_{ck}"))

# Poll + download
print(f"\n  Waiting for {len(tasks)} images...")
for tid, out_path, label in tasks:
    data = poll_task(tid, interval=10, max_wait=600)
    url = data.get("url", "")
    if url:
        download_file(url, out_path)
        print(f"    OK: {label} ({os.path.getsize(out_path)//1024}KB)")
    else:
        print(f"    FAILED: {label}")

# Split atlases
print("\n  Splitting atlases...")
for ck, _, n_poses, gw, gh in chars:
    atlas_path = img_dir / f"pose_atlas_{ck}.png"
    if not atlas_path.exists():
        print(f"    WARNING: {atlas_path} not found")
        continue
    atlas = PILImage.open(atlas_path).convert("RGBA")
    w, h = atlas.size
    cw, ch = w // gw, h // gh
    idx = 0
    for row in range(gh):
        for col in range(gw):
            atlas.crop((col*cw, row*ch, (col+1)*cw, (row+1)*ch)).save(str(img_dir / f"pose_{ck}_{idx}.png"))
            idx += 1
    print(f"    {ck}: {n_poses} poses split OK")

# ============================================================
# Step 3: TTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: TTS (normal speed)")
print("=" * 60)
from tts_engine import TTSEngine, build_voice_map
tts = TTSEngine()
voice_map = build_voice_map(script)

narration = {}
for name, text in [("welcome", script.get("welcome_en", "")),
                   ("hook", script.get("hook_intro_en", "")),
                   ("outro", script.get("outro", ""))]:
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
# Step 4: Timeline + Compose
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Timeline + Compose")
print("=" * 60)
from quest.timeline_quest import build_quest_timeline
from timeline_enrich import enrich_timeline

pad = 0.4
timeline = build_quest_timeline(script, dialogue_durations, pad=pad)
enrich_timeline(timeline, tts, pad, dialogue_durations, [], narration)
print(f"Timeline: {len(timeline)} segments, pad={pad}s")

scene_img = str(img_dir / "scene.png")
host_bg = str(img_dir / "host_bg.png")
if not os.path.exists(host_bg):
    host_bg = scene_img

scene_bg_list = [scene_img]
for si in range(1, 5):
    p = str(img_dir / f"scene_{si}.png")
    scene_bg_list.append(p if os.path.exists(p) else scene_img)

# Build pose map: dialogue chars 4 poses, host 8 poses
char_pose_map = {}
for ck in ("char_a", "char_b", "char_c"):
    poses = [str(img_dir / f"pose_{ck}_{j}.png") for j in range(4)]
    if all(os.path.exists(p) for p in poses):
        char_pose_map[ck] = poses
host_poses = [str(img_dir / f"pose_host_{j}.png") for j in range(8)]
if not all(os.path.exists(p) for p in host_poses):
    host_poses = [str(img_dir / f"pose_host_{j}.png") for j in range(4)]
char_pose_map["host"] = host_poses
print(f"Pose map: {[(k, len(v)) for k, v in char_pose_map.items()]}")

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

print(f"\n{'='*60}")
print(f"DONE: {final_path}")
print(f"Size: {os.path.getsize(final_path) / (1024*1024):.1f}MB")
