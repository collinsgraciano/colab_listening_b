"""Tests for the quest (--structure quest) mode.

Pure-function tests always run. The compose smoke test is gated on
ffmpeg + Pillow availability.

Run:  python test_quest.py
"""
import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from quest.llm_client_quest import split_phase_lines
from quest.timeline_quest import (build_quest_timeline,
                                  build_srt_from_timeline_quest)
from pipeline import _validate_script
from tts_engine import build_voice_map

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def make_script(n=48, n_buildup=26, n_core=9, n_review=13):
    """Build a minimal valid quest script fixture."""
    dialogue = []
    for i in range(n_buildup):
        dialogue.append({"speaker": "char_a" if i % 2 == 0 else "char_b",
                         "phase": "buildup", "text": f"Buildup line {i}",
                         "phonetic": "/x/", "zh": f"鋪墊 {i}",
                         "image_prompt": "img"})
    for i in range(n_core):
        dialogue.append({"speaker": "char_a" if i % 2 == 0 else "char_c",
                         "phase": "core", "text": f"Core line {i}",
                         "phonetic": "/x/", "zh": f"核心 {i}",
                         "image_prompt": "img"})
    for i in range(n_review):
        dialogue.append({"speaker": "char_b" if i % 2 == 0 else "char_a",
                         "phase": "review", "text": f"Review line {i}",
                         "phonetic": "/x/", "zh": f"復盤 {i}",
                         "image_prompt": "img"})
    return {
        "title": "AT THE COFFEE SHOP",
        "title_zh": "在咖啡廳",
        "scene_zh": "咖啡廳 · 點餐",
        "hook_intro_en": "Hello friends. I will speak very slowly today.",
        "hook_intro_zh": "哈囉，我今天會說得很慢。",
        "listening_question_en": "What coffee does David order?",
        "listening_question_zh": "大衛點了什麼咖啡？",
        "char_a_description": "a young man",
        "char_b_description": "a young woman",
        "char_c_description": "a female barista",
        "char_a_gender": "male",
        "char_b_gender": "female",
        "char_c_gender": "female",
        "outro": "Write your answer in the comments. See you next time!",
        "outro_zh": "在評論區寫下你的答案。",
        "youtube_title": "【慢速英文聽力】TEST",
        "dialogue": dialogue,
    }


def test_split_phase_lines():
    print("[1] split_phase_lines")
    b, c, r = split_phase_lines(48)
    check("48 -> (26, 9, 13)", (b, c, r) == (26, 9, 13), f"got {(b, c, r)}")
    check("48 sums to 48", b + c + r == 48)
    b, c, r = split_phase_lines(24)
    check("24 sums to 24", b + c + r == 24)
    check("24 every phase >= 6", min(b, c, r) >= 6, f"got {(b, c, r)}")
    b, c, r = split_phase_lines(12)
    check("12 (small N) proportional, every phase >= 1",
          b + c + r == 12 and min(b, c, r) >= 1, f"got {(b, c, r)}")
    b, c, r = split_phase_lines(3)
    check("3 (tiny N) every phase >= 1", b + c + r == 3 and min(b, c, r) >= 1,
          f"got {(b, c, r)}")


def test_timeline():
    print("[2] build_quest_timeline + SRT")
    script = make_script()
    durations = [4.0] * 48
    tl = build_quest_timeline(script, durations, pad=5.0)

    check("segment count = 51 (1 title + 1 hook + 48 dialogue + 1 outro)",
          len(tl) == 51, f"got {len(tl)}")
    types = [s["type"] for s in tl]
    check("order: title_card, hook_intro, dialogue..., outro",
          types[0] == "title_card" and types[1] == "hook_intro"
          and types[-1] == "outro" and all(t == "dialogue" for t in types[2:-1]))
    phases = [s.get("phase") for s in tl if s["type"] == "dialogue"]
    check("phase carried: 26 buildup / 9 core / 13 review",
          (phases.count("buildup"), phases.count("core"), phases.count("review")) == (26, 9, 13))
    check("phase order contiguous",
          phases == sorted(phases, key=lambda p: {"buildup": 0, "core": 1, "review": 2}[p]))

    # Enrich like _enrich_timeline does (audio_dur + duration with pad)
    for s in tl:
        if s["type"] == "dialogue":
            s["audio_dur"] = 4.0
            s["duration"] = 4.0 + 5.0
        elif s["type"] in ("hook_intro", "outro"):
            s["audio_dur"] = 60.0
            s["duration"] = 60.0 + 5.0

    srt = build_srt_from_timeline_quest(tl, gap=0.0)
    entries = [b for b in srt.split("\n\n") if b.strip()]
    check("SRT has 48 dialogue entries", len(entries) == 48, f"got {len(entries)}")
    check("SRT contains no hook/outro text",
          "speak very slowly" not in srt and "comments" not in srt)

    # Monotonic timestamps
    import re
    times = re.findall(r"(\d+):(\d+):(\d+),(\d+) -->", srt)
    secs = [int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000 for h, m, s, ms in times]
    check("SRT timestamps monotonic", all(a <= b for a, b in zip(secs, secs[1:])))


def test_validate():
    print("[3] _validate_script quest")
    script = make_script()
    ok, msg = _validate_script(script, 48, quest=True)
    check("valid fixture passes", ok, msg)

    s2 = make_script()
    s2["dialogue"][30]["phase"] = "buildup"  # core speaker char_c now in buildup
    ok, msg = _validate_script(s2, 48, quest=True)
    check("char_c outside core rejected", not ok, msg)

    s3 = make_script()
    for line in s3["dialogue"]:
        if line["phase"] == "core":
            line["speaker"] = "char_b"
    ok, msg = _validate_script(s3, 48, quest=True)
    check("char_b in core rejected", not ok, msg)

    s4 = make_script()
    s4["listening_question_en"] = ""
    ok, msg = _validate_script(s4, 48, quest=True)
    check("empty listening_question rejected", not ok, msg)

    s5 = make_script()
    s5["dialogue"] = s5["dialogue"][:20]  # missing review phase entirely
    ok, msg = _validate_script(s5, 20, quest=True)
    check("missing review phase rejected", not ok, msg)

    s6 = make_script()
    s6["hook_intro_en"] = ""
    ok, msg = _validate_script(s6, 48, quest=True)
    check("empty hook_intro rejected", not ok, msg)

    # Backward compat: non-quest scripts unaffected
    plain = {"dialogue": [{"speaker": "char_a", "text": "Hi", "phonetic": "/x/", "zh": "嗨"}]}
    ok, msg = _validate_script(plain, 1)
    check("plain script still valid without quest", ok, msg)


def test_voice_map():
    print("[4] build_voice_map with char_c")
    script = make_script()
    vm = build_voice_map(script)
    check("char_a male -> am_adam", vm.get("char_a") == "am_adam")
    check("char_b female -> af_sarah", vm.get("char_b") == "af_sarah")
    check("char_c female -> af_sarah", vm.get("char_c") == "af_sarah")

    plain = {"char_a_gender": "male", "char_b_gender": "female"}
    vm2 = build_voice_map(plain)
    check("no char_c key without char_c_gender", "char_c" not in vm2)


def _ffmpeg_ok():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_silent_mp3(path, dur):
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=stereo:44100",
         "-t", f"{dur}", "-c:a", "libmp3lame", "-b:a", "128k", str(path)],
        capture_output=True, timeout=60)
    return r.returncode == 0


def test_compose_smoke():
    print("[5] compose_quest smoke (ffmpeg-gated)")
    if not _ffmpeg_ok():
        print("  SKIP ffmpeg not found")
        return
    try:
        from PIL import Image
    except ImportError:
        print("  SKIP Pillow not installed")
        return
    from quest.video_compose_quest import compose_quest

    tmp = Path(tempfile.mkdtemp(prefix="quest_smoke_"))
    try:
        img_dir = tmp / "images"
        audio_dir = tmp / "audio"
        img_dir.mkdir(); audio_dir.mkdir()

        # Solid-color images
        scene = str(img_dir / "scene.png")
        Image.new("RGB", (1280, 720), (60, 120, 160)).save(scene)
        dlg_imgs = []
        for i, color in enumerate([(120, 60, 60), (60, 120, 60), (60, 60, 120)]):
            p = str(img_dir / f"dialogue_img_{i}.png")
            Image.new("RGB", (1600, 900), color).save(p)  # native size != canvas on purpose
            dlg_imgs.append(p)

        # Silent audio with known durations
        hook_dur, outro_dur, line_dur = 2.5, 2.5, 2.0
        assert _make_silent_mp3(audio_dir / "hook.mp3", hook_dur)
        assert _make_silent_mp3(audio_dir / "outro.mp3", outro_dur)
        normal_paths = []
        for i in range(3):
            p = str(audio_dir / f"dialogue_{i}.mp3")
            assert _make_silent_mp3(p, line_dur)
            normal_paths.append(p)

        pad = 0.5
        script = make_script(n=3, n_buildup=2, n_core=1, n_review=0)
        # fix fixture: 2 buildup + 1 core
        script["dialogue"] = [
            {"speaker": "char_a", "phase": "buildup", "text": "I am so tired.",
             "phonetic": "/x/", "zh": "我好累。", "image_prompt": "img"},
            {"speaker": "char_b", "phase": "buildup", "text": "Me too. Coffee?",
             "phonetic": "/x/", "zh": "我也是。喝咖啡？", "image_prompt": "img"},
            {"speaker": "char_c", "phase": "core", "text": "Hi! What would you like?",
             "phonetic": "/x/", "zh": "嗨！您想要什麼？", "image_prompt": "img"},
        ]
        script["youtube_title"] = "QUEST_SMOKE_TEST"

        timeline = [
            {"type": "title_card", "duration": 5.0, "subtitle_en": "QUEST SMOKE",
             "subtitle_zh": "冒煙測試", "scene_zh": "測試", "speaker": "",
             "audio_index": 0, "image_idx": 0, "dialogue_idx": -1},
            {"type": "hook_intro", "duration": hook_dur + pad, "audio_dur": hook_dur,
             "subtitle_en": "hook", "subtitle_zh": "鉤子", "speaker": "",
             "audio_index": -1, "image_idx": 0, "dialogue_idx": -1},
        ]
        for i, line in enumerate(script["dialogue"]):
            timeline.append({
                "type": "dialogue", "duration": line_dur + pad, "audio_dur": line_dur,
                "subtitle_en": line["text"], "subtitle_zh": line["zh"],
                "speaker": line["speaker"], "phase": line["phase"],
                "audio_index": i, "image_idx": i + 1, "dialogue_idx": i})
        timeline.append({
            "type": "outro", "duration": outro_dur + pad, "audio_dur": outro_dur,
            "subtitle_en": "bye", "subtitle_zh": "再見", "speaker": "",
            "audio_index": 0, "image_idx": 0, "dialogue_idx": -1})

        work = tmp / "work"
        work.mkdir()
        out = compose_quest(
            work_dir=str(work),
            dialogue_images=dlg_imgs,
            timeline=timeline,
            script=script,
            narration={"hook": str(audio_dir / "hook.mp3"),
                       "outro": str(audio_dir / "outro.mp3")},
            normal_paths=normal_paths,
            scene_img=scene,
            srt_dir=str(work),
            pad=pad,
            progress_cb=None,
        )
        check("final video exists", os.path.exists(out) and os.path.getsize(out) > 10000, out or "missing")

        expected = 5.0 + (hook_dur + pad) + 3 * (line_dur + pad) + (outro_dur + pad)
        try:
            actual = float(subprocess.check_output(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", out], text=True).strip())
        except Exception:
            actual = 0.0
        check(f"duration ~{expected:.1f}s (got {actual:.1f}s)",
              abs(actual - expected) < 1.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_split_phase_lines()
    test_timeline()
    test_validate()
    test_voice_map()
    test_compose_smoke()
    print()
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
