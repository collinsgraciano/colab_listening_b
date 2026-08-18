"""Fix concat paths + finish compose."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from media_utils import concat_segments, burn_subtitles, apply_final_loudnorm, get_duration

base = Path(__file__).parent.resolve()
test_dir = base / "test_output"
tmp_dir = test_dir / "tmp_segments"
vid_dir = test_dir / "videos"
vid_dir.mkdir(exist_ok=True)

# Gather segments
segs = sorted(str(tmp_dir / f) for f in os.listdir(tmp_dir) if f.endswith(".mp4") and f.startswith("seg_"))
print(f"Segments: {len(segs)}")

# Concat with absolute paths
no_sub = str(vid_dir / "final_no_sub.mp4")
print("Concatenating...")
concat_segments(segs, no_sub, tmp_dir=tmp_dir)
print(f"Concat done: {os.path.getsize(no_sub) // 1024}KB")

# Load script
script = json.loads((base / "test_quest_script.json").read_text(encoding="utf-8"))

# Rebuild timeline
from tts_engine import TTSEngine
tts = TTSEngine()
audio_dir = test_dir / "audio"
normal_paths = [str(audio_dir / f"dialogue_{i}.mp3") for i in range(10)]
dialogue_durations = [get_duration(p) for p in normal_paths]
narration = {}
for name in ("welcome", "hook", "outro"):
    p = str(audio_dir / f"{name}.mp3")
    if os.path.exists(p):
        narration[name] = p

from quest.timeline_quest import build_quest_timeline
from timeline_enrich import enrich_timeline

pad = 0.4
timeline = build_quest_timeline(script, dialogue_durations, pad=pad)
enrich_timeline(timeline, tts, pad, dialogue_durations, [], narration)

# Burn subtitles with absolute paths
print("Burning subtitles...")
final_path = burn_subtitles(no_sub, timeline, script, str(test_dir), str(test_dir), pad,
                            lambda pct, msg: print(f"  [{pct}%] {msg}") if pct % 50 == 0 else None)

# Loudnorm
print("Loudnorm...")
apply_final_loudnorm(final_path, str(vid_dir))

size_mb = os.path.getsize(final_path) / (1024 * 1024)
print(f"\nDONE: {final_path} ({size_mb:.1f}MB)")
