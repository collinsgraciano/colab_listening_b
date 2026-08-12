#!/usr/bin/env python3
"""Test script: generate YouTube thumbnail with text baked into the AI prompt.

Instead of generating a background image then overlaying text via Pillow,
this script puts the title text, level badge, and Chinese subtitle directly
into the generate_image prompt so the AI renders everything in one step.

Usage:
    python test_thumbnail.py --topic "At the Pharmacy" --cefr A2 --structure enhanced
    python test_thumbnail.py --topic "Ordering Coffee" --cefr A2 --structure original

Requires: MCP token (from ~/.codely-cli/mcp-oauth-tokens.json or --mcp-token)
"""
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

from mcp_client import initialize, call_tool, parse_task_id, poll_task, download_file


def build_thumbnail_prompt(topic: str, cefr: str, structure: str,
                           title_en: str = "", title_zh: str = "") -> str:
    """Build a prompt that generates a complete YouTube thumbnail with text.

    The AI generates the entire thumbnail in one shot: character + scene + text + badge.
    """
    if not title_en:
        title_en = topic.upper()
    if not title_zh:
        title_zh = ""

    if structure == "enhanced":
        bottom_text = "Vocabulary + Quiz + Slow Speed + Shadowing"
    else:
        bottom_text = "Listen + Repeat + Shadowing"

    return f"""A high-quality YouTube thumbnail image, 1280x720 pixels, 16:9 aspect ratio.

LEFT SIDE: A 3D cartoon style character in a {topic} scene, with a surprised and excited facial expression, looking toward the camera. The character should be modern, colorful, and eye-catching. The scene background should show elements of {topic} (e.g. counter, shelves, products).

RIGHT SIDE: Large bold text that says "{title_en}" in bright yellow color with thick black outline/stroke. The text must be clearly readable and centered on the right half of the image.

BELOW THE ENGLISH TITLE: Chinese text "{title_zh}" in gold color, slightly smaller than the English title, with black outline.

TOP-RIGHT CORNER: A red circular badge with white text "{cefr}" inside it, like a level indicator.

BOTTOM BAR: A dark semi-transparent bar across the bottom with white text "{bottom_text}".

The overall style should be bright, colorful, and attention-grabbing like a popular YouTube educational video thumbnail. High contrast, vibrant colors, professional design. The text must be sharp, clear, and correctly spelled.

IMPORTANT: All text must be rendered as part of the image, clearly legible, with proper spelling. Do not use placeholder text or gibberish."""


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

    # Build prompt
    prompt = build_thumbnail_prompt(topic, cefr, structure)
    print(f"\n[2] Prompt ({len(prompt)} chars):")
    print(f"    {prompt[:200]}...")

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
                        help="Video structure (affects bottom bar text)")
    parser.add_argument("--output", default="./test_thumbnails", help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Run batch test with 5 presets")
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
