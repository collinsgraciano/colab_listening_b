"""Minimal live test: generate 10-line quest script."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["SENSENOVA_API_KEY"] = "sk-8Tr86c17YvA5jBEoem2uYYAQGXGzmpDU"

from quest.llm_client_quest import generate_quest_script
from pipeline import _validate_script

print("Generating 10-line quest script...")
script = generate_quest_script("At the Ice Cream Shop", cefr="A1", num_lines=10)

# Save
Path("test_quest_script.json").write_text(
    json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved test_quest_script.json")

# Validate
ok, msg = _validate_script(script, 10, quest=True)
print(f"Validation: {ok} ({msg})")

# Summary
dl = script.get("dialogue", [])
print(f"\nTitle: {script.get('title','')} / {script.get('title_zh','')}")
print(f"Welcome: {script.get('welcome_en','')}")
print(f"Host: {script.get('host_description','')}")
print(f"Char A: {script.get('char_a_description','')}")
print(f"Char B: {script.get('char_b_description','')}")
print(f"Char C: {script.get('char_c_description','')}")
print(f"Question: {script.get('listening_question_en','')}")
print(f"Lines: {len(dl)}")

phases = {}
os_counts = {}
for line in dl:
    ph = line.get("phase", "?")
    phases[ph] = phases.get(ph, 0) + 1
    osk = str(line.get("on_screen", []))
    os_counts[osk] = os_counts.get(osk, 0) + 1
print(f"Phases: {phases}")
print(f"on_screen: {os_counts}")

for i, line in enumerate(dl):
    os_str = str(line.get("on_screen", []))
    print(f"  [{i:2d}] {line.get('phase',''):8s} {line.get('speaker',''):6s} {os_str:30s} | {line.get('text','')}")
