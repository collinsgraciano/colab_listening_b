"""test4 render: TTS + compose with Phase 1 animation (flow morph + breathing + audio-driven)."""
import sys, os, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ["SENSENOVA_API_KEY"] = "sk-8Tr86c17YvA5jBEoem2uYYAQGXGzmpDU"

base = Path(__file__).parent.resolve()
test_dir = base / "test4"
img_dir = test_dir / "images"
audio_dir = test_dir / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)

script = json.loads((base / "test_quest_script.json").read_text(encoding="utf-8"))
dl = script.get("dialogue", [])

# TTS
print("--- TTS (normal speed) ---")
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

# Timeline + Compose
print("\n--- Timeline + Compose ---")
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

# All chars: 8 poses each
char_pose_map = {}
for ck in ("char_a", "char_b", "char_c", "host"):
    poses = [str(img_dir / f"pose_{ck}_{j}.png") for j in range(8)]
    if all(os.path.exists(p) for p in poses):
        char_pose_map[ck] = poses
host_poses = char_pose_map.get("host", [scene_img] * 8)
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

print(f"\nDONE: {final_path}")
print(f"Size: {os.path.getsize(final_path) / (1024*1024):.1f}MB")
