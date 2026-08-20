"""Diagnostic test: generate one image via MCP and dump ALL response fields.

Run on Colab:
    cd /content/listening_b && python test_mcp_image.py --mcp-tokens TOKEN1,TOKEN2
"""
import sys
import os
import json
import time

# Ensure UTF-8 output on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from mcp_client import initialize, call_tool, parse_task_id


def dump_response(label, result):
    """Print full MCP response structure for debugging."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    if not result:
        print("  (empty response)")
        return
    if "result" not in result:
        print(f"  No 'result' key. Top-level keys: {list(result.keys())}")
        print(f"  Full: {json.dumps(result, ensure_ascii=False)[:2000]}")
        return
    r = result["result"]
    print(f"  result keys: {list(r.keys())}")
    content = r.get("content", [])
    print(f"  content blocks: {len(content)}")
    for i, item in enumerate(content):
        ctype = item.get("type", "?")
        print(f"\n  --- content[{i}] type={ctype} ---")
        if ctype == "text":
            text = item.get("text", "")
            print(f"  text (len={len(text)}):")
            print(f"  {text[:3000]}")
            # Try parsing as JSON
            try:
                tj = json.loads(text)
                print(f"\n  Parsed JSON keys: {list(tj.keys())}")
                print(f"  Parsed JSON: {json.dumps(tj, ensure_ascii=False, indent=2)[:3000]}")
            except json.JSONDecodeError:
                print("  (not valid JSON)")
                # Try regex URL extraction
                import re
                urls = re.findall(r"(https?://[^\s`'\")]+)", text)
                if urls:
                    print(f"  URLs found in text: {urls}")
        elif ctype == "resource":
            res = item.get("resource", {})
            res_text = res.get("text", "")
            print(f"  resource.text (len={len(res_text)}):")
            try:
                rj = json.loads(res_text)
                print(f"  Parsed JSON keys: {list(rj.keys())}")
                print(f"  Full parsed JSON:")
                print(f"  {json.dumps(rj, ensure_ascii=False, indent=2)[:4000]}")
                # Deep search for any URL-like field
                _deep_find_urls(rj, prefix="  ")
            except json.JSONDecodeError:
                print(f"  (not valid JSON) raw: {res_text[:2000]}")


def _deep_find_urls(obj, prefix="", path=""):
    """Recursively find all URL-like strings in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if isinstance(v, str) and ("http" in v or "url" in k.lower()):
                print(f"  [URL FIELD] {p} = {v[:200]}")
            elif isinstance(v, (dict, list)):
                _deep_find_urls(v, prefix, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            if isinstance(v, str) and "http" in v:
                print(f"  [URL FIELD] {p} = {v[:200]}")
            elif isinstance(v, (dict, list)):
                _deep_find_urls(v, prefix, p)


def main():
    # Parse tokens from CLI args
    tokens = []
    for i, arg in enumerate(sys.argv):
        if arg == "--mcp-tokens" and i + 1 < len(sys.argv):
            tokens = [t.strip() for t in sys.argv[i + 1].split(",") if t.strip()]
            break
    if not tokens:
        # Try Windows local token file
        token_file = os.path.join(os.environ.get("USERPROFILE", ""), ".codely-cli", "mcp-oauth-tokens.json")
        if os.path.exists(token_file):
            with open(token_file, encoding="utf-8") as f:
                td = json.load(f)
            tokens = [t["token"]["accessToken"] for t in td if t["serverName"] == "TJGenerators"]
            print(f"Auto-detected {len(tokens)} token(s) from {token_file}")

    if not tokens:
        print("Usage: python test_mcp_image.py --mcp-tokens TOKEN1,TOKEN2")
        sys.exit(1)

    print(f"\nInitializing MCP with {len(tokens)} token(s)...")
    initialize(tokens=tokens)

    # --- Step 1: Create an image task ---
    prompt = "A cute cartoon cat sitting on a table, 3D cartoon style, 16:9"
    print(f"\n>>> Calling generate_image: {prompt}")

    gen_args = {
        "prompt": prompt,
        "provider": "seedream",
        "image_size": "landscape_16_9",
        "output_format": "png",
    }
    print(f">>> Arguments: {json.dumps(gen_args, ensure_ascii=False)}")

    result = call_tool("generate_image", gen_args)
    dump_response("TASK CREATION RESPONSE", result)

    task_id = parse_task_id(result)
    print(f"\n>>> Parsed task_id: '{task_id}'")
    if not task_id:
        print("FATAL: Could not parse task_id from response!")
        sys.exit(1)

    # --- Step 2: Poll with FULL raw dump ---
    print(f"\n>>> Polling task {task_id} (raw check_task dumps below)...")
    max_wait = 300
    interval = 10
    elapsed = 0
    final_url = None

    while elapsed < max_wait:
        print(f"\n--- check_task call at {elapsed}s ---")
        check_result = call_tool("check_task", {"task_id": task_id})
        dump_response(f"CHECK_TASK @ {elapsed}s", check_result)

        # Manual status + URL extraction
        status = ""
        url = ""
        if "result" in check_result:
            content = check_result["result"].get("content", [])
            for item in content:
                if item.get("type") == "resource":
                    try:
                        rj = json.loads(item["resource"]["text"])
                        status = rj.get("status", "")
                        out = rj.get("output") or {}
                        data = out.get("data") or {}
                        result_obj = data.get("result") or {}
                        # Try ALL possible URL fields
                        url = (result_obj.get("video_url") or result_obj.get("image_url") or
                               result_obj.get("url") or "")
                        if not url:
                            img_urls = data.get("imageUrls") or data.get("image_urls") or []
                            if img_urls and isinstance(img_urls, list) and img_urls[0]:
                                url = img_urls[0]
                        if not url:
                            url = data.get("url") or data.get("imageUrl") or ""
                        if not url:
                            # Deep search
                            _found = []
                            def _search(o):
                                nonlocal _found
                                if isinstance(o, dict):
                                    for k, v in o.items():
                                        if isinstance(v, str) and v.startswith("http"):
                                            _found.append(f"{k}={v}")
                                        elif isinstance(v, (dict, list)):
                                            _search(v)
                                elif isinstance(o, list):
                                    for v in o:
                                        if isinstance(v, str) and v.startswith("http"):
                                            _found.append(f"[item]={v}")
                                        elif isinstance(v, (dict, list)):
                                            _search(v)
                            _search(rj)
                            if _found:
                                print(f"\n  DEEP SEARCH found URLs:")
                                for u in _found:
                                    print(f"    {u}")
                                # Take first
                                url = _found[0].split("=", 1)[-1]
                    except Exception as e:
                        print(f"  Resource parse error: {e}")

                if item.get("type") == "text":
                    text = item.get("text", "")
                    if not status:
                        if "completed" in text.lower() or "已完成" in text:
                            status = "completed"
                        elif "running" in text.lower() or "queued" in text.lower() or "进行中" in text:
                            status = "running"
                        elif "failed" in text.lower() or "失败" in text:
                            status = "failed"
                    if not url:
                        import re
                        m = re.search(r"!\[.*?\]\((https?://[^\s)]+)\)", text)
                        if m:
                            url = m.group(1)
                        if not url:
                            m = re.search(r"(https?://[^\s`'\")]+)", text)
                            if m:
                                url = m.group(1)

        print(f"\n  STATUS: {status or '(unknown)'}")
        print(f"  URL:    {url or '(empty)'}")

        if status == "completed":
            if url:
                print(f"\n✅ SUCCESS: Task completed with URL")
                final_url = url
            else:
                print(f"\n❌ FAIL: Task completed but NO URL found in any field!")
            break
        if status == "failed":
            print(f"\n❌ FAIL: Task failed")
            break

        elapsed += interval
        time.sleep(interval)

    if not final_url:
        print("\n" + "="*70)
        print("  DIAGNOSIS: Task completed but URL extraction FAILED")
        print("  Check the raw dumps above to find the actual URL field")
        print("="*70)
    else:
        # Try downloading
        print(f"\n>>> Attempting download from: {final_url[:100]}")
        from mcp_client import download_file
        dest = "/tmp/test_image.png" if os.path.exists("/tmp") else "test_image.png"
        ok = download_file(final_url, dest)
        if ok:
            size = os.path.getsize(dest)
            print(f"✅ Downloaded: {dest} ({size} bytes)")
        else:
            print(f"❌ Download failed!")


if __name__ == "__main__":
    main()
