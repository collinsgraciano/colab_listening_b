#!/usr/bin/env python3
"""Test script: generate YouTube thumbnail with text baked into the AI prompt.

Uses thumbnail_gen.py's _build_thumbnail_prompt for the prompt construction.
Supports both single test and batch test modes.

Usage:
    python test_thumbnail.py --topic "At the Pharmacy" --cefr A2 --structure enhanced
    python test_thumbnail.py --batch
"""
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

from mcp_client import initialize, call_tool, parse_task_id, poll_task, download_file
from thumbnail_gen import _build_thumbnail_prompt


def test_single_thumbnail(topic: str, cefr: str, structure: str,
                           output_dir: str, mcp_token: str = None):
    """Generate a single test thumbnail."""
    print("=" * 60)
    print(f"Test Thumbnail Generator")
    print(f"  Topic: {topic}")
    print(f"  CEFR: {cefr}")
    print(f"  Structure: {structure}")
    print("=" * 60)

    # Initialize MCP
    print("\n[1] Initializing MCP...")
    initialize(token=mcp_token)
    print("    MCP connected.")

    # Build a mock script dict for the prompt builder
    script = {
        "title": topic.upper(),
        "title_zh": topic,
        "intro_zh": "",
        "cefr": cefr,
        "char_a_description": "friendly young person",
        "char_b_description": "friendly young person",
        "scene_zh": topic,
        "scene": topic.lower(),
        "thumbnail_expression": "surprised and excited",
        "thumbnail_action": "looking toward the camera and gesturing naturally",
        "thumbnail_subtitle": "18句聽力練習",
        "thumbnail_icons": [
            {"en": "Dialogue", "zh": "會話"},
            {"en": "Listening", "zh": "聽力"},
            {"en": "Shadowing", "zh": "跟讀"},
            {"en": "Practice", "zh": "練習"},
        ],
        "title": topic.upper(),
    }

    # Build prompt
    prompt = _build_thumbnail_prompt(script, structure)
    print(f"\n[2] Prompt ({len(prompt)} chars):")
    print(f"    {prompt[:300]}...")

    # Generate image
    print(f"\n[3] Generating thumbnail via generate_image (frontier)...")
    result = call_tool("generate_image", {
        "prompt": prompt,
        "provider": "frontier",
        "quality": "high",
        "image_size": '{"width": 1280, "height": 720}',
        "output_format": "jpeg",
        "num_images": 1,
    })

    task_id = parse_task_id(result)
    if not task_id:
        print("    ERROR: No task_id returned!")
        if result and "result" in result:
            content = result["result"].get("content", [])
            print(f"    Response: {json.dumps(content, ensure_ascii=False)[:500]}")
        return None

    print(f"    Task ID: {task_id}")

    # Poll
    print(f"\n[4] Polling for completion...")
    data = poll_task(task_id, interval=10)
    url = data.get("url", "")

    if not url:
        print(f"    ERROR: No image URL. Status={data.get('status')}")
        print(f"    Raw: {data.get('raw_json', '')[:300]}")
        return None

    # Download
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_topic = topic.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(output_dir, f"thumbnail_{safe_topic}_{cefr}_{structure}.jpg")

    print(f"\n[5] Downloading to: {out_path}")
    if download_file(url, out_path):
        size_kb = os.path.getsize(out_path) // 1024
        print(f"    SUCCESS: {out_path} ({size_kb}KB)")
        return out_path
    else:
        print(f"    ERROR: Download failed")
        return None


def test_multiple_thumbnails(mcp_token: str = None, output_dir: str = "./test_thumbnails"):
    """Generate multiple test thumbnails for comparison."""
    test_cases = [
        ("At the Pharmacy", "A2", "enhanced"),
        ("At the Pharmacy", "A2", "original"),
        ("Ordering Coffee", "A1", "enhanced"),
        ("Job Interview", "B1", "enhanced"),
        ("Apartment Hunting", "A2", "enhanced"),
        ("At the Airport", "A2", "original"),
    ]

    print("=" * 60)
    print(f"Batch Test: {len(test_cases)} thumbnails")
    print("=" * 60)

    initialize(token=mcp_token)

    results = []
    for i, (topic, cefr, structure) in enumerate(test_cases):
        print(f"\n[{i+1}/{len(test_cases)}] {topic} ({cefr}, {structure})")
        try:
            path = test_single_thumbnail(topic, cefr, structure, output_dir)
            results.append((topic, cefr, structure, path, "OK" if path else "FAILED"))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((topic, cefr, structure, None, str(e)[:80]))

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"{'Topic':<25} {'CEFR':<5} {'Structure':<10} {'Status':<10}")
    print("-" * 60)
    for topic, cefr, structure, path, status in results:
        print(f"{topic:<25} {cefr:<5} {structure:<10} {status:<10}")

    ok = sum(1 for _, _, _, p, _ in results if p)
    print(f"\nSuccess: {ok}/{len(results)}")
    print(f"Output dir: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(description="Test YouTube thumbnail generation with baked-in text")
    parser.add_argument("--topic", default=None, help="Topic (e.g. 'At the Pharmacy')")
    parser.add_argument("--cefr", default="A2", help="CEFR level (default A2)")
    parser.add_argument("--structure", default="enhanced", choices=["original", "enhanced"],
                        help="Video structure (affects top center text)")
    parser.add_argument("--output", default="./test_thumbnails", help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Run batch test with 6 presets")
    parser.add_argument("--mcp-token", default=None, help="MCP OAuth token")
    args = parser.parse_args()

    if args.batch:
        test_multiple_thumbnails(mcp_token=args.mcp_token, output_dir=args.output)
    elif args.topic:
        test_single_thumbnail(args.topic, args.cefr, args.structure, args.output, args.mcp_token)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python test_thumbnail.py --topic 'At the Pharmacy' --cefr A2 --structure enhanced")
        print("  python test_thumbnail.py --batch")
        print("  python test_thumbnail.py --topic 'Ordering Coffee' --cefr A1 --structure original")


if __name__ == "__main__":
    main()