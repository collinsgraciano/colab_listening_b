#!/usr/bin/env python3
"""Test script: generate YouTube thumbnail with text baked into the AI prompt.

Calls LLM to generate a full script (including title_zh, thumbnail_* fields),
then uses that script to build the thumbnail prompt and generate the image.

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
from llm_client import generate_listening_script


def _generate_script_for_thumbnail(topic: str, cefr: str) -> dict:
    """Call LLM to generate a full script with thumbnail fields."""
    print(f"\n[1] Generating LLM script for topic: '{topic}'...")
    script = generate_listening_script(topic, cefr, num_lines=4)
    print(f"    Title: {script.get('title', '')}")
    print(f"    Title ZH: {script.get('title_zh', '')}")
    print(f"    Scene: {script.get('scene', '')}")
    print(f"    Expression: {script.get('thumbnail_expression', '')}")
    print(f"    Action: {script.get('thumbnail_action', '')}")
    print(f"    Subtitle: {script.get('thumbnail_subtitle', '')}")
    print(f"    Icons: {len(script.get('thumbnail_icons', []))} items")
    return script


def test_single_thumbnail(topic: str, cefr: str, structure: str,
                           output_dir: str, mcp_token: str = None,
                           api_key: str = None):
    """Generate a single test thumbnail via LLM script + MCP image gen."""
    print("=" * 60)
    print(f"Test Thumbnail Generator")
    print(f"  Topic: {topic}")
    print(f"  CEFR: {cefr}")
    print(f"  Structure: {structure}")
    print("=" * 60)

    # Set API key
    if api_key:
        os.environ["SENSENOVA_API_KEY"] = api_key
    if not os.environ.get("SENSENOVA_API_KEY"):
        print("ERROR: SENSENOVA_API_KEY not set. Pass --api-key or set env var.")
        return None

    # Step 1: Generate LLM script (provides title_zh, thumbnail_* fields)
    script = _generate_script_for_thumbnail(topic, cefr)

    # Step 2: Initialize MCP
    print("\n[2] Initializing MCP...")
    initialize(token=mcp_token)
    print("    MCP connected.")

    # Step 3: Build prompt from LLM-generated script
    prompt = _build_thumbnail_prompt(script, structure)
    print(f"\n[3] Prompt ({len(prompt)} chars):")
    # Write prompt to file for inspection
    prompt_path = os.path.join(output_dir, f"prompt_{topic.replace(' ', '_')}.txt")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"    Prompt saved: {prompt_path}")

    # Step 4: Generate image — first generate char_scene, then use as reference
    print(f"\n[4] Generating char_scene reference image...")
    char_a_desc = script.get("char_a_description", "friendly young person")
    char_b_desc = script.get("char_b_description", "friendly young person")
    char_scene_prompt = f"Character design sheet, {char_a_desc} on the left, {char_b_desc} on the right, plain white background, full body, front view, 3D cartoon style, no text, no background, 16:9"
    char_result = call_tool("generate_image", {
        "prompt": char_scene_prompt,
        "provider": "frontier",
        "quality": "high",
        "image_size": "landscape_16_9",
        "output_format": "png",
    })
    char_task_id = parse_task_id(char_result)
    char_scene_url = ""
    if char_task_id:
        char_data = poll_task(char_task_id, interval=10)
        char_scene_url = char_data.get("url", "")
        if char_scene_url:
            print(f"    Char scene URL: {char_scene_url[:60]}...")

    # Step 5: Generate thumbnail with char_scene as reference
    print(f"\n[5] Generating thumbnail via generate_image (frontier)...")
    if char_scene_url:
        prompt += "\n\nIMPORTANT: The characters' appearance, clothing, and hair MUST closely match the uploaded reference image."
    gen_args = {
        "prompt": prompt,
        "provider": "frontier",
        "quality": "high",
        "image_size": '{"width": 1280, "height": 720}',
        "output_format": "jpeg",
    }
    if char_scene_url:
        gen_args["image_urls"] = char_scene_url
    result = call_tool("generate_image", gen_args)

    task_id = parse_task_id(result)
    if not task_id:
        print("    ERROR: No task_id returned!")
        if result and "result" in result:
            content = result["result"].get("content", [])
            print(f"    Response: {json.dumps(content, ensure_ascii=False)[:500]}")
        return None
    print(f"    Task ID: {task_id}")

    # Step 6: Poll
    print(f"\n[6] Polling for completion...")
    data = poll_task(task_id, interval=10)
    url = data.get("url", "")
    if not url:
        print(f"    ERROR: No image URL. Status={data.get('status')}")
        print(f"    Raw: {data.get('raw_json', '')[:300]}")
        return None

    # Step 7: Download
    safe_topic = topic.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(output_dir, f"thumbnail_{safe_topic}_{cefr}_{structure}.jpg")
    print(f"\n[7] Downloading to: {out_path}")
    if download_file(url, out_path):
        size_kb = os.path.getsize(out_path) // 1024
        print(f"    SUCCESS: {out_path} ({size_kb}KB)")
        return out_path
    else:
        print(f"    ERROR: Download failed")
        return None


def test_multiple_thumbnails(mcp_token: str = None, api_key: str = None,
                              output_dir: str = "./test_thumbnails"):
    """Generate multiple test thumbnails for comparison."""
    test_cases = [
        ("At the Pharmacy", "A2", "enhanced"),
        ("Ordering Coffee", "A1", "enhanced"),
        ("Job Interview", "B1", "enhanced"),
        ("At the Airport", "A2", "original"),
    ]

    print("=" * 60)
    print(f"Batch Test: {len(test_cases)} thumbnails")
    print("=" * 60)

    if api_key:
        os.environ["SENSENOVA_API_KEY"] = api_key
    if not os.environ.get("SENSENOVA_API_KEY"):
        print("ERROR: SENSENOVA_API_KEY not set. Pass --api-key or set env var.")
        return

    initialize(token=mcp_token)

    results = []
    for i, (topic, cefr, structure) in enumerate(test_cases):
        print(f"\n{'='*40}")
        print(f"[{i+1}/{len(test_cases)}] {topic} ({cefr}, {structure})")
        print(f"{'='*40}")
        try:
            path = test_single_thumbnail(topic, cefr, structure, output_dir,
                                          mcp_token, api_key)
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
    parser = argparse.ArgumentParser(
        description="Test YouTube thumbnail generation with LLM-generated script")
    parser.add_argument("--topic", default=None,
                        help="Topic (e.g. 'At the Pharmacy')")
    parser.add_argument("--cefr", default="A2", help="CEFR level (default A2)")
    parser.add_argument("--structure", default="enhanced",
                        choices=["original", "enhanced"],
                        help="Video structure (affects top center text)")
    parser.add_argument("--output", default="./test_thumbnails",
                        help="Output directory")
    parser.add_argument("--batch", action="store_true",
                        help="Run batch test with 4 presets")
    parser.add_argument("--mcp-token", default=None, help="MCP OAuth token")
    parser.add_argument("--api-key", default=None,
                        help="SenseNova API key (or set SENSENOVA_API_KEY env var)")
    args = parser.parse_args()

    if args.batch:
        test_multiple_thumbnails(mcp_token=args.mcp_token, api_key=args.api_key,
                                  output_dir=args.output)
    elif args.topic:
        test_single_thumbnail(args.topic, args.cefr, args.structure, args.output,
                              args.mcp_token, args.api_key)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python test_thumbnail.py --topic 'At the Pharmacy' --cefr A2 --structure enhanced --api-key YOUR_KEY --mcp-token YOUR_TOKEN")
        print("  python test_thumbnail.py --batch --api-key YOUR_KEY --mcp-token YOUR_TOKEN")


if __name__ == "__main__":
    main()
