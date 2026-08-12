#!/usr/bin/env python3
"""TJGenerators MCP client — Colab-compatible version.

Changes from Windows version:
- TOKEN is passed as argument, not read from ~/.codely-cli/mcp-oauth-tokens.json
- No sys.stdout.reconfigure (Linux doesn't need it)
- Uses /content/ paths on Colab

Usage:
    from mcp_client import initialize, call_tool, parse_task_id, poll_task, download_file
    initialize(token="your_oauth_token")
"""
import sys
import json
import time
import re
import urllib.request
import urllib.error

MCP_URL = "https://ai-generator.tuanjie.cn/mcp"
TOKEN = ""  # Set via initialize(token=...) or set_mcp_token()
_session_id = None
_msg_id = 0


def set_mcp_token(token: str):
    """Set the MCP OAuth token manually (required on Colab)."""
    global TOKEN
    TOKEN = token


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
        with urllib.request.urlopen(req, timeout=120) as resp:
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


def initialize(token: str = None):
    """Initialize MCP session. Pass token on Colab (no local token file)."""
    if token:
        set_mcp_token(token)
    if not TOKEN:
        raise RuntimeError("No MCP token. Call initialize(token='...') or set_mcp_token('...')")
    result = mcp_call("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "colab-listening", "version": "1.0"},
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
    """Poll check_task until completed or failed. Returns dict with 'status' and 'url'."""
    elapsed = 0
    while elapsed < max_wait:
        result = call_tool("check_task", {"task_id": task_id})
        if "result" not in result:
            elapsed += interval
            time.sleep(interval)
            continue

        content = result["result"].get("content", [])
        task_data = {}

        # 1. Try resource block
        for item in content:
            if item.get("type") == "resource":
                res_text = item.get("resource", {}).get("text", "")
                try:
                    res_json = json.loads(res_text)
                    task_data["status"] = res_json.get("status", "")
                    out = res_json.get("output") or {}
                    data = out.get("data") or {}
                    result_obj = data.get("result") or {}
                    url = result_obj.get("video_url", "") or result_obj.get("image_url", "")
                    if url:
                        task_data["url"] = url
                    if "url" not in task_data:
                        img_urls = data.get("imageUrls") or []
                        if img_urls and isinstance(img_urls, list) and img_urls[0]:
                            task_data["url"] = img_urls[0]
                    break
                except json.JSONDecodeError:
                    pass

        # 2. Fall back to text block
        if "url" not in task_data:
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if not task_data.get("status"):
                        if "completed" in text.lower() or "已完成" in text:
                            task_data["status"] = "completed"
                        elif "running" in text.lower() or "queued" in text.lower() or "进行中" in text:
                            task_data["status"] = "running"
                        elif "failed" in text.lower() or "失败" in text:
                            task_data["status"] = "failed"
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
            print(f"  FAILED: {json.dumps(task_data, ensure_ascii=False)[:500]}")
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
            with open(dest, "wb") as f:
                f.write(resp.read())
        import os
        return os.path.getsize(dest) > 1000
    except Exception as e:
        print(f"  Download failed: {e}")
        return False
