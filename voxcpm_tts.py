"""VoxCPM TTS engine via Cloudflare Worker.

Generates English speech using VoxCPM model hosted on a Cloudflare Worker.
Voice characteristics are controlled via natural language descriptions
(the ``control`` parameter), so the LLM can design per-character voices.

No reference audio required — the ``control`` string describes the desired voice.
Chinese TTS and loudnorm are inherited from :class:`TTSEngine`.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Add parent to path so we can import tts_engine
_PARENT = str(Path(__file__).parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tts_engine import TTSEngine


class VoxCPMEngine(TTSEngine):
    """VoxCPM TTS engine via Cloudflare Worker.

    Inherits ``synth_chinese``, ``_loudnorm``, ``get_duration`` from TTSEngine.
    Overrides ``synth_english`` to call the VoxCPM Worker API.

    Flow per synthesis:
        submit /generate -> poll SSE /events/<id> -> download WAV -> ffmpeg MP3 -> loudnorm
    """

    def __init__(self, worker_url: str, api_key: str = ""):
        self.worker_url = worker_url.rstrip("/")
        self.api_key = api_key
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        h = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def test_connection(self) -> bool:
        """Quick health check against the Worker."""
        try:
            r = self._session.get(
                f"{self.worker_url}/health",
                headers=self._headers(),
                timeout=30,
            )
            r.raise_for_status()
            print(f"  [VoxCPM] Worker OK: {r.text[:80]}")
            return True
        except Exception as e:
            print(f"  [VoxCPM] Health check failed: {e}")
            return False

    def _submit(self, text: str, control: str,
                cfg_value: int = 2, do_normalize: bool = False,
                denoise: bool = False) -> str:
        """Submit a generation task, return event_id."""
        data = [
            text,
            control,
            None,            # reference_wav_path — none (text-described voice)
            False,           # use_prompt_text
            "",              # prompt_text
            cfg_value,
            do_normalize,
            denoise,
        ]
        r = self._session.post(
            f"{self.worker_url}/generate",
            json={"data": data},
            headers={**self._headers(), "Content-Type": "application/json"},
            timeout=120,
        )
        r.raise_for_status()
        result = r.json()
        if "event_id" not in result:
            raise RuntimeError(f"No event_id in VoxCPM response: {result}")
        return result["event_id"]

    @staticmethod
    def _parse_sse(event_text: str):
        """Parse a single SSE event block -> (event_type, data)."""
        event_type = None
        data_lines = []
        for line in event_text.splitlines():
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        data_text = "\n".join(data_lines)
        data = None
        if data_text:
            try:
                data = json.loads(data_text)
            except Exception:
                data = data_text
        return event_type, data

    def _wait(self, event_id: str, timeout: int = 600):
        """Block on SSE stream until 'complete' or 'error'."""
        url = f"{self.worker_url}/events/{event_id}"
        start = time.time()
        with self._session.get(
            url,
            headers={
                **self._headers(),
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
            },
            stream=True,
            timeout=(30, timeout),
        ) as r:
            r.raise_for_status()
            buf = ""
            for raw in r.iter_lines(decode_unicode=True):
                if time.time() - start > timeout:
                    raise TimeoutError("VoxCPM generation timed out")
                if raw is None:
                    continue
                if raw == "":
                    if buf.strip():
                        etype, edata = self._parse_sse(buf)
                        if etype == "complete":
                            return edata
                        if etype == "error":
                            raise RuntimeError(f"VoxCPM error: {edata}")
                        buf = ""
                    continue
                buf += raw + "\n"
        raise RuntimeError("SSE ended without 'complete' event")

    @staticmethod
    def _find_wav_path(obj):
        """Recursively search Gradio result for a .wav file path."""
        if isinstance(obj, str):
            if obj.endswith(".wav") or "/tmp/gradio/" in obj or "\\tmp\\gradio\\" in obj:
                return obj
            return None
        if isinstance(obj, dict):
            for k in ("path", "filepath", "file_path"):
                if k in obj and isinstance(obj[k], str):
                    return obj[k]
            for v in obj.values():
                found = VoxCPMEngine._find_wav_path(v)
                if found:
                    return found
            return None
        if isinstance(obj, (list, tuple)):
            for item in obj:
                found = VoxCPMEngine._find_wav_path(item)
                if found:
                    return found
        return None

    def _download(self, result, wav_path: str):
        """Download generated WAV from the Worker."""
        remote = self._find_wav_path(result)
        if not remote:
            raise RuntimeError(f"No WAV path in VoxCPM result: {result}")
        r = self._session.get(
            f"{self.worker_url}/download",
            params={"path": remote},
            headers=self._headers(),
            stream=True,
            timeout=300,
        )
        r.raise_for_status()
        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
        with open(wav_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # ------------------------------------------------------------------
    # Public API (same signature as TTSEngine.synth_english)
    # ------------------------------------------------------------------

    def synth_english(self, text: str, voice: str, out_path: str,
                      rate: str = "+0%", max_retries: int = 3) -> float:
        """Synthesize English text via VoxCPM.

        Args:
            text: English text to speak.
            voice: VoxCPM voice description (the ``control`` parameter).
            out_path: output MP3 path.
            rate: speed adjustment ('-15%' = slow; applied via FFmpeg atempo).
            max_retries: max retry attempts on failure.

        Returns:
            Audio duration in seconds.
        """
        # Convert rate string to atempo factor
        speed = 1.0
        if rate and rate != "+0%":
            try:
                pct = int(rate.replace("%", "").replace("+", ""))
                speed = 1.0 + pct / 100.0
            except (ValueError, TypeError):
                speed = 1.0

        wav_path = out_path.replace(".mp3", "_voxcpm.wav")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        last_err = None
        for attempt in range(max_retries):
            try:
                eid = self._submit(text, voice)
                result = self._wait(eid, timeout=600)
                self._download(result, wav_path)

                # Build audio filter chain (atempo for speed, then standardize)
                af_parts = []
                if 0.5 <= speed <= 2.0 and speed != 1.0:
                    af_parts.append(f"atempo={speed:.4f}")

                af = ",".join(af_parts) if af_parts else "anull"

                subprocess.run(
                    ["ffmpeg", "-y", "-i", wav_path,
                     "-af", af,
                     "-c:a", "libmp3lame", "-b:a", "128k",
                     "-ar", "24000", "-ac", "1", out_path],
                    check=True, capture_output=True,
                )
                if os.path.exists(wav_path):
                    os.remove(wav_path)

                # loudnorm (inherited from TTSEngine)
                self._loudnorm(out_path)
                return self.get_duration(out_path)

            except Exception as e:
                last_err = e
                print(f"    [VoxCPM retry {attempt+1}/{max_retries}] {str(e)[:120]}")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                if attempt < max_retries - 1:
                    time.sleep(5)

        raise RuntimeError(
            f"VoxCPM failed after {max_retries} retries: {last_err}"
        )
