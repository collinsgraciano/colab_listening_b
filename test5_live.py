"""test5 render: use multi-round generated script + test4 images."""
import sys, os, json, shutil, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ["SENSENOVA_API_KEY"] = "sk-8Tr86c17YvA5jBEoem2uYYAQGXGzmpDU"

base = Path(__file__).parent.resolve()
test_dir = base / "test5"
src_img = Path(r"H:\2026_main_project\test4\images")
img_dir = test_dir / "images"
audio_dir = test_dir / "audio"

if test_dir.exists():
    shutil.rmtree(test_dir)
img_dir.mkdir(parents=True, exist_ok=True)
audio_dir.mkdir(parents=True, exist_ok=True)

# Copy images from test4
for f in src_img.glob("*.png"):
    shutil.copy2(str(f), str(img_dir / f.name))
print(f"Images: {len(list(img_dir.glob('*.png')))} files copied")

# Generate multi-round script
print("\n=== Multi-round script generation ===")
from quest.llm_client_quest import generate_quest_script
from pipeline import _validate_script

script = generate_quest_script("At the Bubble Tea Shop", cefr="A2", num_lines=10)
ok, msg = _validate_script(script, 10, quest=True)
print(f"Validation: {ok} ({msg})")
if not ok:
    print(f"FAILED: {msg}"); sys.exit(1)

# Save script
(test_dir / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")

dl = script.get("dialogue", [])
print(f"Question: {script.get('listening_question_en','')}")
for i, line in enumerate(dl):
    print(f"  [{i}] {line.get('phase',''):8s} {line.get('speaker',''):6s} on_screen={line.get('on_screen',[])} | {line.get('text','')}")

# TTS
print("\n=== TTS ===")
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

# Timeline
print("\n=== Timeline ===")
from quest.timeline_quest import build_quest_timeline
from timeline_enrich import enrich_timeline

pad = 0.4
timeline = build_quest_timeline(script, dialogue_durations, pad=pad)
enrich_timeline(timeline, tts, pad, dialogue_durations, [], narration)
print(f"Segments: {len(timeline)}, pad={pad}s")

scene_img = str(img_dir / "scene.png")
host_bg = str(img_dir / "host_bg.png")
if not os.path.exists(host_bg):
    host_bg = scene_img
scene_bg_list = [scene_img]
for si in range(1, 5):
    p = str(img_dir / f"scene_{si}.png")
    scene_bg_list.append(p if os.path.exists(p) else scene_img)

# Pose map
char_pose_map = {}
for ck in ("char_a", "char_b", "char_c", "host"):
    poses = [str(img_dir / f"pose_{ck}_{j}.png") for j in range(8)]
    if all(os.path.exists(p) for p in poses):
        char_pose_map[ck] = poses
host_poses = char_pose_map.get("host", [scene_img] * 8)
print(f"Pose map: {[(k, len(v)) for k, v in char_pose_map.items()]}")

# Compose
print("\n=== Composing video ===")
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

print(f"\nDONE: {final_path}")
print(f"Size: {os.path.getsize(final_path) / (1024*1024):.1f}MB")
