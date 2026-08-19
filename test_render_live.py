"""Live render test: use real LLM script + MCP-generated images to render a short quest video."""
import sys
import os
import json
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load the generated script
script = json.loads(Path("test_quest_script.json").read_text(encoding="utf-8"))
dialogue = script.get("dialogue", [])

# --- Split atlases into per-pose images ---
from PIL import Image as PILImage

test_dir = Path("test_output")
img_dir = test_dir / "images"

char_keys = [
    ("char_a", script.get("char_a_description", "")),
    ("char_b", script.get("char_b_description", "")),
    ("char_c", script.get("char_c_description", "")),
    ("host", script.get("host_description", "")),
]

for char_key, _ in char_keys:
    atlas_path = img_dir / f"pose_atlas_{char_key}.png"
    if not atlas_path.exists():
        print(f"WARNING: atlas not found: {atlas_path}")
        continue
    atlas = PILImage.open(atlas_path).convert("RGBA")
    w, h = atlas.size
    hw, hh = w // 2, h // 2
    quads = [(0, 0, hw, hh), (hw, 0, w, hh), (0, hh, hw, h), (hw, hh, w, h)]
    for j, (l, t, r, b) in enumerate(quads):
        cell = atlas.crop((l, t, r, b))
        out = img_dir / f"pose_{char_key}_{j}.png"
        cell.save(str(out))
        print(f"  Split: {out.name} ({cell.size})")

# --- Build pose map ---
scene_img = str(img_dir / "scene.png")
char_pose_map = {}
for char_key, _ in char_keys:
    poses = [str(img_dir / f"pose_{char_key}_{j}.png") for j in range(4)]
    if all(os.path.exists(p) for p in poses):
        char_pose_map[char_key] = poses
        print(f"  Pose map: {char_key} -> 4 poses OK")
    else:
        print(f"  Pose map: {char_key} -> MISSING poses")

host_poses = char_pose_map.get("host", [scene_img] * 4)

# --- Generate TTS audio ---
print("\n--- Generating TTS ---")
os.environ["SENSENOVA_API_KEY"] = "sk-8Tr86c17YvA5jBEoem2uYYAQGXGzmpDU"

from tts_engine import TTSEngine
tts = TTSEngine()
from tts_engine import build_voice_map
voice_map = build_voice_map(script)

audio_dir = test_dir / "audio"
audio_dir.mkdir(exist_ok=True)

# Narration
narration = {}
for name, text, rate in [
    ("welcome", script.get("welcome_en", ""), "0%"),
    ("hook", script.get("hook_intro_en", ""), "0%"),
    ("outro", script.get("outro", ""), "0%"),
]:
    if text:
        path = str(audio_dir / f"{name}.mp3")
        dur = tts.synth_english(text, "af_sky", path, rate=rate)
        narration[name] = path
        print(f"  TTS {name}: {dur:.1f}s")

# Dialogue TTS (-25% slow)
normal_paths = []
dialogue_durations = []
for i, line in enumerate(dialogue):
    text = line.get("text", "")
    speaker = line.get("speaker", "char_a")
    voice = voice_map.get(speaker, "af_sarah")
    path = str(audio_dir / f"dialogue_{i}.mp3")
    dur = tts.synth_english(text, voice, path, rate="0%")
    normal_paths.append(path)
    dialogue_durations.append(dur)
    print(f"  TTS dialogue_{i}: {dur:.1f}s ({voice})")

# --- Build timeline ---
print("\n--- Building timeline ---")
from quest.timeline_quest import build_quest_timeline
pad = 0.4
timeline = build_quest_timeline(script, dialogue_durations, pad=pad)
print(f"  Timeline segments: {len(timeline)}")
for seg in timeline:
    print(f"    {seg['type']:12s} dur={seg.get('duration',0):.1f}s speaker={seg.get('speaker','')}")

# Enrich
from timeline_enrich import enrich_timeline
enrich_timeline(timeline, tts, pad, dialogue_durations, [], narration)

# --- Compose ---
print("\n--- Composing video ---")
from quest.video_compose_quest import compose_quest

final_path = compose_quest(
    work_dir=str(test_dir),
    pose_images=[],  # not used when char_pose_map is provided
    char_pose_map=char_pose_map,
    host_poses=host_poses,
    timeline=timeline,
    script=script,
    narration=narration,
    normal_paths=normal_paths,
    scene_img=scene_img,
    srt_dir=str(test_dir),
    pad=pad,
    progress_cb=lambda pct, msg: print(f"  [{pct:3d}%] {msg}") if pct % 20 == 0 else None,
)

print(f"\n=== DONE ===")
print(f"Final video: {final_path}")
size_mb = os.path.getsize(final_path) / (1024 * 1024)
print(f"Size: {size_mb:.1f}MB")
