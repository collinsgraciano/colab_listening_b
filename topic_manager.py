"""Topic manager for listening video pipeline — random selection + anti-duplicate.

Reads topics.json (categorized topic pool), randomly picks an unused topic,
and records it in used_topics.json to prevent repeats.

topics.json format:
    {"日常Life": ["Making Breakfast", "Doing Laundry", ...], "旅行": [...]}

used_topics.json format:
    {"Making Breakfast": {"used_at": "2026-08-10 18:00:00"}, ...}
"""
import json
import random
import sys
from datetime import datetime
from pathlib import Path


def load_topics(topics_file: str) -> dict:
    """Load topics.json — returns {category: [topic, ...]}."""
    p = Path(topics_file)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_used_topics(used_file: str) -> dict:
    """Load used_topics.json — returns {topic: {used_at: ...}}."""
    p = Path(used_file)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def get_all_topics(topics: dict) -> list[str]:
    """Flatten all topics from all categories into a single list."""
    all_topics = []
    for cat_topics in topics.values():
        all_topics.extend(cat_topics)
    return all_topics


def pick_random_topic(topics_file: str, used_file: str) -> str | None:
    """Pick a random topic that hasn't been used yet.

    If all topics are used, resets used_topics.json and picks from the full pool.
    Returns the topic string, or None if topics_file is empty/missing.
    """
    topics = load_topics(topics_file)
    if not topics:
        return None

    all_topics = get_all_topics(topics)
    if not all_topics:
        return None

    used = load_used_topics(used_file)
    available = [t for t in all_topics if t not in used]

    if not available:
        # All topics used — reset and start over
        print(f"  [Topic] All {len(all_topics)} topics have been used. Resetting used_topics.json.")
        Path(used_file).write_text("{}", encoding="utf-8")
        available = all_topics

    chosen = random.choice(available)
    print(f"  [Topic] Randomly selected: '{chosen}' (from {len(available)} available)")

    # Record in used_topics.json
    used[chosen] = {"used_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Path(used_file).parent.mkdir(parents=True, exist_ok=True)
    Path(used_file).write_text(json.dumps(used, ensure_ascii=False, indent=2), encoding="utf-8")

    return chosen


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pick a random topic")
    parser.add_argument("--topics-file", default="topics.json", help="Path to topics.json")
    parser.add_argument("--used-file", default="used_topics.json", help="Path to used_topics.json")
    args = parser.parse_args()

    topic = pick_random_topic(args.topics_file, args.used_file)
    if topic:
        print(f"Selected: {topic}")
    else:
        print("No topics found.")
