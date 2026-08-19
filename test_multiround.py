"""Test multi-round quest script generation."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["SENSENOVA_API_KEY"] = "sk-8Tr86c17YvA5jBEoem2uYYAQGXGzmpDU"

from quest.llm_client_quest import generate_quest_script
from pipeline import _validate_script

print("Generating quest script (multi-round, 10 lines)...")
script = generate_quest_script("At the Bubble Tea Shop", cefr="A1", num_lines=10)

# Save
Path("test_multiround_script.json").write_text(
    json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved test_multiround_script.json")

# Validate
ok, msg = _validate_script(script, 10, quest=True)
print(f"Validation: {ok} ({msg})")

# Summary
dl = script.get("dialogue", [])
print(f"\nTitle: {script.get('title','')}")
print(f"Question: {script.get('listening_question_en','')}")
print(f"Answer (from outline): see script")
print(f"Host: {script.get('host_description','')}")
print(f"Lines: {len(dl)}")

for i, line in enumerate(dl):
    os_str = str(line.get("on_screen", []))
    print(f"  [{i}] {line.get('phase',''):8s} {line.get('speaker',''):6s} {os_str:30s} | {line.get('text','')}")
    if line.get("zh"):
        print(f"       ZH: {line.get('zh','')}")
