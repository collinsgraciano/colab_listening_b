"""TJGenerators MCP client — calls MCP server directly via HTTP JSON-RPC.

Usage:
    from mcp_client import initialize, call_tool, poll_task
    initialize()
    result = call_tool("generate_image", {"prompt": "...", "provider": "frontier"})
    task_id = parse_task_id(result)
    data = poll_task(task_id, interval=10)
    url = data.get("url")
"""
import sys
import os
import json
import time
import re
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

MCP_URL = "https://ai-generator.tuanjie.cn/mcp"
# Token from C:\Users\Administrator\.codely-cli\mcp-oauth-tokens.json
import os
_TOKEN_FILE = os.path.join(os.environ.get("USERPROFILE", ""), ".codely-cli", "mcp-oauth-tokens.json")
try:
    with open(_TOKEN_FILE, encoding="utf-8") as f:
        _tokens = json.load(f)
    TOKEN = next(t["token"]["accessToken"] for t in _tokens if t["serverName"] == "TJGenerators")
except Exception:
    TOKEN = ""  # Fallback: user must set manually

_session_id = None
_msg_id = 0


def _next_id():
    global _msg_id
    _msg_id += 1
    return _msg_id


def mcp_call(method, params=None):
    global _session_id
    msg_id = _next_id()
    payload = {"jsonrpc": "2.0", "method": method, "id": msg_id}
    if params is not None:
        payload["params"] = params
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}
    if _session_id:
        headers["Mcp-Session-Id"] = _session_id
    req = urllib.request.Request(MCP_URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
            if sid:
                _session_id = sid
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sid = e.headers.get("Mcp-Session-Id") or e.headers.get("mcp-session-id")
        if sid:
            _session_id = sid
        body = e.read().decode("utf-8")
        print(f"HTTP {e.code}: {body[:500]}")
        raise


def mcp_notify(method, params=None):
    global _session_id
    payload = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        payload["params"] = params
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}
    if _session_id:
        headers["Mcp-Session-Id"] = _session_id
    req = urllib.request.Request(MCP_URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except Exception:
        pass


def initialize(token=None):
    """Initialize MCP session.

    On Colab (or when token is passed), use the provided token.
    On Windows local, token auto-detected from ~/.codely-cli/mcp-oauth-tokens.json.
    """
    global TOKEN
    if token:
        TOKEN = token
    if not TOKEN:
        raise RuntimeError("No MCP token. Pass token to initialize(token='...') or set MCP_TOKEN env var.")
    result = mcp_call("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "codely-cli", "version": "1.0"},
    })
    mcp_notify("notifications/initialized")
    return result


def call_tool(name, arguments):
    return mcp_call("tools/call", {"name": name, "arguments": arguments})


def parse_task_id(result):
    """Extract task_id from MCP tool call response (Markdown or JSON)."""
    if "result" not in result:
        return ""
    content = result["result"].get("content", [])
    for item in content:
        if item.get("type") == "text":
            text = item["text"]
            try:
                return json.loads(text).get("task_id", "")
            except json.JSONDecodeError:
                pass
            m = re.search(r"Task\s+ID[\*\s'\"]*:\s*[`'\"]*([a-f0-9]+)", text)
            if m:
                return m.group(1)
    return ""


def poll_task(task_id, interval=40, max_wait=600):
    """Poll check_task until completed or failed. Returns dict containing:
    - status, url (success)
    - status, raw (full raw check_task response + error message on failure)
    """
    elapsed = 0
    while elapsed < max_wait:
        try:
            result = call_tool("check_task", {"task_id": task_id})
        except Exception as e:
            print(f"  [{elapsed}s] Task {task_id[:16]}...: HTTP error ({type(e).__name__}: {e}), retrying in {interval}s...")
            elapsed += interval
            time.sleep(interval)
            continue
        if "result" not in result:
            elapsed += interval
            time.sleep(interval)
            continue

        content = result["result"].get("content", [])
        task_data = {}
        # Keep the FULL raw response for debugging
        task_data["raw_response"] = json.dumps(result, ensure_ascii=False)

        # 1. Try resource block (contains JSON with output URLs + error info)
        for item in content:
            if item.get("type") == "resource":
                res_text = item.get("resource", {}).get("text", "")
                try:
                    res_json = json.loads(res_text)
                    task_data["status"] = res_json.get("status", "")
                    task_data["raw_json"] = json.dumps(res_json, ensure_ascii=False)
                    out = res_json.get("output") or {}
                    data = out.get("data") or {}
                    # Preserve error message if present
                    if res_json.get("error"):
                        task_data["error"] = res_json["error"]
                    elif data.get("error"):
                        task_data["error"] = data["error"]
                    elif data.get("message"):
                        task_data["error"] = data["message"]
                    # Video URL: output.data.result.video_url
                    result_obj = data.get("result") or {}
                    url = result_obj.get("video_url", "") or result_obj.get("image_url", "")
                    if url:
                        task_data["url"] = url
                    # Image URL: output.data.imageUrls[0] (array)
                    if "url" not in task_data:
                        img_urls = data.get("imageUrls") or []
                        if img_urls and isinstance(img_urls, list) and img_urls[0]:
                            task_data["url"] = img_urls[0]
                    break
                except json.JSONDecodeError:
                    pass

        # 2. Also check text block for URL (Markdown ![](URL)) — even if resource block was parsed
        if "url" not in task_data:
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    task_data["raw_text"] = text
                    if not task_data.get("status"):
                        if "completed" in text.lower() or "已完成" in text:
                            task_data["status"] = "completed"
                        elif "running" in text.lower() or "queued" in text.lower() or "进行中" in text:
                            task_data["status"] = "running"
                        elif "failed" in text.lower() or "失败" in text:
                            task_data["status"] = "failed"
                    # Extract URL from Markdown
                    m = re.search(r"!\[.*?\]\((https?://[^\s)]+)\)", text)
                    if m:
                        task_data["url"] = m.group(1)
                        break
                    if "url" not in task_data:
                        m = re.search(r"(https?://[^\s`'\")]+)", text)
                        if m:
                            task_data["url"] = m.group(1)
                            break

        status = task_data.get("status", "")
        if status:
            print(f"  [{elapsed}s] Task {task_id[:16]}...: {status}")
        if status == "completed":
            return task_data
        if status == "failed":
            print(f"  FAILED: {json.dumps(task_data, ensure_ascii=False)[:1000]}")
            return task_data

        elapsed += interval
        time.sleep(interval)

    print(f"  TIMEOUT after {max_wait}s for task {task_id}")
    return {}


def download_file(url, dest):
    """Download a file from URL to local path."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                print(f"  Download failed: got HTML page (not a file) from {url[:80]}")
                return False
            with open(dest, "wb") as f:
                f.write(resp.read())
        return os.path.getsize(dest) > 1000
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


if __name__ == "__main__":
    initialize()
    print(f"Session: {_session_id}")
    tools = call_tool("tools/list", {})
    for t in tools.get("result", {}).get("tools", []):
        print(f"  - {t['name']}")
