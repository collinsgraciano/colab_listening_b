"""Standalone LLM client for listening video script generation.

Uses SenseNova DeepSeek V4 Flash (OpenAI-compatible API).
No external project imports.
"""
import json
import re
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

# SenseNova API (default and only LLM backend)
SENSENOVA_BASE = os.environ.get("SENSENOVA_BASE", "https://token.sensenova.cn/v1")
SENSENOVA_API_KEY = os.environ.get("SENSENOVA_API_KEY", "")
SENSENOVA_MODEL = os.environ.get("SENSENOVA_MODEL", "deepseek-v4-flash")

# Rate limiting: enforce minimum interval between LLM API calls to avoid HTTP 429.
# glm-5.2 is especially aggressive about "request rate increased too quickly".
_LAST_CALL_TIME = 0.0
_MIN_CALL_INTERVAL = float(os.environ.get("LLM_MIN_INTERVAL", "3.0"))


def _enforce_rate_limit():
    """Sleep if the previous LLM call was too recent."""
    import time as _time
    global _LAST_CALL_TIME
    elapsed = _time.time() - _LAST_CALL_TIME
    if elapsed < _MIN_CALL_INTERVAL:
        wait = _MIN_CALL_INTERVAL - elapsed
        _time.sleep(wait)
    _LAST_CALL_TIME = _time.time()


def _chat(messages: list[dict], temperature: float = 0.8, timeout: int = 180,
          max_tokens: int = 8192, reasoning_effort: str = "low") -> str:
    """Call LLM chat completion (SenseNova or OpenAI-compatible), return content string.

    Dispatches based on LLM_PROVIDER env var:
    - "sensenova" (default): SenseNova DeepSeek V4 Flash / glm-5.2
    - "openai": any OpenAI-compatible endpoint (x666.me, etc.)

    Retries on HTTP 429 (rate limit) with exponential backoff.
    Enforces a minimum interval between calls to avoid triggering rate limits.
    """
    import time as _time

    provider = os.environ.get("LLM_PROVIDER", "sensenova")

    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL", "grok-4.6")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://x666.me/v1")
    else:
        model = os.environ.get("SENSENOVA_MODEL", "deepseek-v4-flash")
        api_key = os.environ.get("SENSENOVA_API_KEY", "")
        base_url = os.environ.get("SENSENOVA_BASE", "https://token.sensenova.cn/v1")

    # Retry on 429 (rate limit) and 524 (Cloudflare gateway timeout)
    _RETRY_CODES = [429, 524]
    _RETRY_BACKOFFS = [15, 30, 60, 90, 120]

    for _retry_attempt in range(len(_RETRY_BACKOFFS) + 1):
        _enforce_rate_limit()
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # reasoning_effort is SenseNova-specific; OpenAI-compatible APIs don't support it
        if provider != "openai":
            body["reasoning_effort"] = reasoning_effort
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=data,
            method="POST",
        )
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")
        # Cloudflare-protected endpoints (e.g. x666.me) block default Python User-Agent with 403
        req.add_header("User-Agent", "CodelyLLM/1.0")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                if not raw.strip():
                    raise RuntimeError("LLM returned empty response body (HTTP 200, 0 bytes)")
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    raise RuntimeError(f"LLM returned non-JSON response (first 500 chars): {raw[:500]}") from None
                if "choices" not in result or not result["choices"]:
                    raise RuntimeError(f"LLM response has no 'choices' field: {raw[:500]}")
                content = result["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise RuntimeError(
                        f"LLM returned empty content (HTTP 200, model={model}). "
                        f"Raw response (first 500 chars): {raw[:500]}"
                    )
                return content
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            if e.code in _RETRY_CODES and _retry_attempt < len(_RETRY_BACKOFFS):
                wait = _RETRY_BACKOFFS[_retry_attempt]
                print(f"  [LLM] HTTP {e.code} ({'rate limited' if e.code == 429 else 'gateway timeout'}), "
                      f"waiting {wait}s before retry "
                      f"({_retry_attempt+1}/{len(_RETRY_BACKOFFS)})... "
                      f"Model: {model}")
                _time.sleep(wait)
                continue
            raise RuntimeError(f"LLM HTTP {e.code}: {err}") from e


def _repair_truncated_json(text: str) -> str:
    """Attempt to repair truncated JSON by closing open strings, arrays, and objects."""
    # Count unmatched braces/brackets
    in_string = False
    escape = False
    stack = []
    i = 0
    while i < len(text):
        c = text[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == '\\' and in_string:
            escape = True
            i += 1
            continue
        if c == '"' and not escape:
            in_string = not in_string
        elif not in_string:
            if c == '{':
                stack.append('}')
            elif c == '[':
                stack.append(']')
            elif c in ('}', ']'):
                if stack and stack[-1] == c:
                    stack.pop()
        i += 1
    # If we're in an unterminated string, close it
    if in_string:
        text += '"'
    # Remove trailing comma if present
    text = re.sub(r',\s*$', '', text.strip())
    # Close all open structures
    while stack:
        text += stack.pop()
    return text


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response (handles markdown fences + truncated JSON)."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try repairing truncated JSON
        repaired = _repair_truncated_json(text)
        return json.loads(repaired)


def _build_listening_prompt(topic: str, cefr: str, used_dialogues: list[str] = None,
                            num_lines: int = 18) -> str:
    """Build prompt for listening-practice lesson (num_lines + IPA + 繁中)."""
    used_hint = ""
    if used_dialogues:
        used_hint = f"""
IMPORTANT — AVOID DUPLICATES: The following dialogue scenarios have already been generated.
Do NOT create dialogue that is too similar to these. Use a DIFFERENT situation, different speakers, different story:
{chr(10).join(f"  - {d}" for d in used_dialogues[:20])}
"""
    return f"""You are an expert ESL teacher creating ENGLISH LISTENING PRACTICE content for overseas Chinese learners.

CORE MISSION: 帮助海外华人用最地道最日常的英语，搞定真实生活中的每一个场景。

Topic: {topic}
CEFR Level: {cefr}
Output: a JSON object ONLY (no markdown, no explanation).

CONTENT REQUIREMENTS — This is for a LISTENING PRACTICE video targeting overseas Chinese:
- The dialogue must be about REAL-LIFE situations that overseas Chinese people actually face in English-speaking countries — practical, relatable, and immediately useful.
- Topics should be things people encounter in daily life: ordering food, asking for directions, making small talk, dealing with a problem at a store, calling customer service, visiting a doctor, renting an apartment, banking, school registration, etc.
- The conversation must feel 100% NATURAL and REALISTIC — like something you'd overhear in real life, NOT a textbook. Use filler words (like "um", "well", "so"), natural pauses, back-channeling ("oh really?", "that makes sense"), and conversational flow.
- Characters should speak the way REAL Americans do in everyday life: contractions (don't, I'll, can't), casual phrasal verbs (pick up, figure out, run out of), common idioms and slang appropriate for the CEFR level, and natural sentence fragments.
- The dialogue must tell a COMPLETE story with a clear beginning, problem/development, and resolution — but keep it grounded in reality, not exaggerated or melodramatic.
- Include realistic communication patterns: clarifying questions, polite hedging ("I was wondering if...", "Would it be possible to..."), thanking, apologizing, expressing mild frustration or satisfaction naturally.
- Every line should teach something useful — a phrase, expression, or communication strategy that the viewer can immediately apply in their own life.
{used_hint}
TECHNICAL REQUIREMENTS:
- Exactly 2 speakers with natural American English
- Each speaker MUST have a clearly defined ROLE in the story (e.g. "customer" vs "waiter", "passenger" vs "check-in agent"). The role must be appropriate for the topic.
- Exactly {num_lines} dialogue lines (each under 15 words)
- The dialogue must flow as a continuous, coherent story (not disconnected Q&A)
- Every dialogue line MUST include:
  - "text": the English sentence
  - "phonetic": IPA phonetic transcription in /slashes/ (use proper IPA symbols)
  - "zh": Traditional Chinese (繁體中文) translation
  - "image_prompt": a detailed prompt describing what this character looks like AND what they are doing. MUST include: (1) the character's EXACT physical description (same every time for same speaker), (2) their role (e.g. "a waitress", "a customer"), (3) the scene location, (4) the action matching the dialogue text.
  - "video_prompt": a detailed prompt for AI video generation. MUST include the SAME character description and action as image_prompt. MUST also include the dialogue text so the character appears to be speaking those words naturally (e.g. "The character says: 'Hi, I'd like a latte, please.' while gesturing toward the menu"). CRITICAL: the video MUST closely reference the uploaded reference image — the character's appearance, clothing, and the scene must match the reference image exactly.
  - "poses": an array of exactly 2 pose descriptions for stop-motion animation. Each entry is a short prompt fragment describing ONLY the character's expression and gesture — NO props, NO objects, NO scene elements. MUST include the character's physical description (same as char_a/char_b_description). MUST describe a DIFFERENT facial expression or hand gesture for each pose. The first pose should show the character actively speaking (mouth open, expressive gesture). The second pose should show the character listening or reacting (e.g. head tilted, slight smile, relaxed hands). Example: ["a young woman with brown hair in a green apron, speaking with mouth open, raising right hand expressively", "a young woman with brown hair in a green apron, listening with a slight smile, relaxed posture"]. CRITICAL: the character's physical appearance (hair, clothing, accessories) MUST be IDENTICAL in both poses — only the gesture/expression changes. Do NOT mention any objects, props, or scene background in the poses.
- "char_a_description": a detailed physical description of speaker 1 (gender, hair color, hairstyle, clothing). This MUST be used identically in ALL of speaker 1's image_prompt and video_prompt entries.
- "char_b_description": a detailed physical description of speaker 2 (gender, hair color, hairstyle, clothing). This MUST be used identically in ALL of speaker 2's image_prompt and video_prompt entries.
- "char_a_gender": "male" or "female" — the gender of speaker 1
- "char_b_gender": "male" or "female" — the gender of speaker 2
- "char_a_role": the role of speaker 1 in the story (e.g. "waitress", "customer")
- "char_b_role": the role of speaker 2 in the story (e.g. "customer", "waitress")
- "youtube_title": a high-CTR YouTube title for overseas Chinese learners. ALL Chinese text in Traditional Chinese (繁體中文). Use the following format patterns (pick one, vary between videos):
  Pattern A: 【沉浸式英文動畫】{{hook phrase}} {{emoji}} {{topic description in 繁中}}：{{specific skills listed}}，聽完就能說！｜{{English topic}}
  Pattern B: 【每天50句英文】{{emoji}}{{topic 繁中}}情境對話｜🎬沉浸式英文動畫｜旅行必備英文｜不用背多聽就會用｜英文聽力訓練｜情境英文對話｜高頻口語句型｜英文口說跟讀練習｜英文高效學習法
  Pattern C: 【🎬沉浸式英文動畫】{{emoji}}{{topic 繁中}}英文｜✅{{specific skill 繁中}}｜🗣️small talk・{{sub-topic}}｜每天50句英文｜真實情境完整呈現｜💡不用背多聽就會用｜{{CEFR}} 初學者必學｜旅行必備英文｜英文聽力口說
  Rules: Start with 【】bracket tag. Use ｜ as separator. Include 3-8 topic-relevant emoji. Include catchy power phrases like "不用背多聽就會用", "聽完就能說", "超實用". End with ｜{{English topic name}}. Title length 80-150 chars.
  Examples: "【沉浸式英文動畫】出國怕開口？✈️ 超實用機場英文：訂票、報到、托運行李一次搞定，聽完就能說！｜Airport English"
  "【每天50句英文】🧳機場失物招領情境對話｜🎬沉浸式英文動畫｜旅行必備英文｜不用背多聽就會用｜英文聽力訓練｜情境英文對話｜高頻口語句型｜英文口說跟讀練習｜英文高效學習法"
- "youtube_description": a full YouTube video description (max 3000 chars). First line must be a hook with the main keyword. Include a "⏱️ Chapters:" section with timestamps for: 00:00 Title, 00:05 Dialogue, 00:xx Shadowing Practice, 00:xx Outro. End with 3 hashtags (#EnglishListening #ESL #LearnEnglish) and a subscribe CTA. ALL Chinese text in Traditional Chinese (繁體中文).
- "youtube_tags": an array of 15-20 SEO tags (mix of short and long-tail keywords, include both English and Traditional Chinese tags)
- "scene": the English name of the scene/location (e.g. "pharmacy", "coffee shop", "hotel lobby"). Used for thumbnail and prompts.
- "thumbnail_expression": the facial expression of the main character on the thumbnail (e.g. "surprised and excited", "confused and thinking", "cheerful and smiling", "friendly and confident")
- "thumbnail_action": a short description of what the main character is doing on the thumbnail (e.g. "pointing to a menu", "holding a shopping bag", "waving hello", "gesturing toward the counter")
- "thumbnail_subtitle": a short Traditional Chinese subtitle shown below the title on the thumbnail (e.g. "18句聽力練習", "每天50句", "實用日常英語")
- "thumbnail_icons": an array of 4-5 objects with "en" and "zh" string keys, describing scene-related keywords shown as circular icons at the bottom of the thumbnail. Each has an English label and a Traditional Chinese label. Example for pharmacy: [{{"en": "Prescription", "zh": "處方"}}, {{"en": "Refill", "zh": "補充"}}, {{"en": "Cough Syrup", "zh": "止咳糖漿"}}, {{"en": "Side Effects", "zh": "副作用"}}]
- "thumbnail_prompt": a detailed prompt for generating a YouTube thumbnail background image. Must describe: a 3D Pixar-style character with an expressive face, the scene location, bright colors, reference-style layout.
- "title": English title (e.g. "AT THE AIRPORT")
- "cefr": the CEFR level of this lesson, exactly "{cefr}" (used for thumbnail level badge)
- "title_zh": Traditional Chinese short title (max 6 characters, e.g. "在機場")
- "scene_zh": Traditional Chinese scene description (e.g. "餐廳 · 點餐")
- "story_hook": a compelling 1-sentence intro that sets the scene
- "intro_zh": Traditional Chinese translation of the intro
- "outro": a short closing line
- "outro_zh": Traditional Chinese translation of the outro
- "practice_intro_en": English instruction before the 跟讀 section
- "practice_intro_zh": Traditional Chinese translation of the practice intro
- ALL Chinese text MUST be in Traditional Chinese (繁體中文)
- CRITICAL: the gender in char_a_description and char_b_description MUST be consistent. If speaker 1 is female, her description MUST say "a young woman" and ALL her image_prompt/video_prompt entries MUST say "a young woman". NEVER mix up genders.
- CRITICAL: each speaker's description (hair, clothing, etc.) MUST be IDENTICAL across ALL their image_prompt and video_prompt entries. Do not change hair color, clothing, or any physical trait between lines.
- CRITICAL: image_prompt and video_prompt MUST match the dialogue context exactly. If at a restaurant, prompts must say "restaurant", NOT "airport" or "supermarket".
- CRITICAL: the scene location in image_prompt and video_prompt MUST be consistent throughout ALL lines.
- CRITICAL: the speaker field in dialogue MUST use "char_a" or "char_b" (not the actual name).

JSON schema:
{{
  "title": string,
  "cefr": string,
  "title_zh": string,
  "scene_zh": string,
  "lesson_type": "listening",
  "story_hook": string,
  "intro_zh": string,
  "outro": string,
  "outro_zh": string,
  "practice_intro_en": string,
  "practice_intro_zh": string,
  "char_a_description": string,
  "char_b_description": string,
  "char_a_gender": string,
  "char_b_gender": string,
  "char_a_role": string,
  "char_b_role": string,
  "youtube_title": string,
  "youtube_description": string,
  "youtube_tags": [string],
  "thumbnail_prompt": string,
  "scene": string,
  "thumbnail_expression": string,
  "thumbnail_action": string,
  "thumbnail_subtitle": string,
  "thumbnail_icons": [{{"en": string, "zh": string}}],
  "dialogue": [{{"speaker": string, "text": string, "phonetic": string, "zh": string, "image_prompt": string, "video_prompt": string, "poses": [string, string]}}]
}}

Topic: {topic}"""


def _load_used_listening_summaries(lessons_dir: str = None) -> list[str]:
    """Load summaries of previously generated listening dialogues for anti-duplicate.

    Scans a lessons/ directory for JSON files with lesson_type="listening".
    If lessons_dir is None or doesn't exist, returns empty list.
    """
    if not lessons_dir:
        return []
    lessons_path = Path(lessons_dir)
    if not lessons_path.exists():
        return []
    summaries = []
    for f in lessons_path.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            script = data.get("script", data)
            if script.get("lesson_type") != "listening":
                continue
            title = script.get("title", "")
            story = script.get("story_hook", "")
            first_line = script.get("dialogue", [{}])[0].get("text", "")
            summaries.append(f"{title}: {story} (starts: {first_line[:60]})")
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    return summaries


def generate_listening_script(topic: str, cefr: str = "A2",
                              lessons_dir: str = None,
                              num_lines: int = 18) -> dict:
    """Generate a listening-practice lesson script via SenseNova DeepSeek V4 Flash.

    Args:
        topic: e.g. "At the Pharmacy"
        cefr: CEFR level (A1, A2, B1, B2, C1, C2)
        lessons_dir: optional path to lessons/ directory for anti-duplicate check
        num_lines: number of dialogue lines to generate (default 18)

    Returns:
        Script dict with dialogue[], char descriptions, title, etc.
    """
    used_summaries = _load_used_listening_summaries(lessons_dir)
    prompt = _build_listening_prompt(topic, cefr, used_dialogues=used_summaries,
                                     num_lines=num_lines)

    # Retry up to 3 times on JSON parse errors (LLM may truncate or produce invalid JSON)
    last_error = None
    for attempt in range(3):
        try:
            content = _chat(
                [{"role": "user", "content": prompt}],
                temperature=0.8 if attempt == 0 else 0.7,
                max_tokens=8192,
            )
            script = _extract_json(content)
            break
        except (json.JSONDecodeError, RuntimeError) as e:
            last_error = e
            err_str = str(e)
            # Log raw content on JSON parse errors for debugging
            if isinstance(e, json.JSONDecodeError):
                print(f"  [LLM retry {attempt+1}/3] JSONDecodeError: {err_str[:200]}")
                print(f"  [LLM] Raw content (first 300 chars): {content[:300] if 'content' in dir() else 'N/A'}")
            else:
                print(f"  [LLM retry {attempt+1}/3] {type(e).__name__}: {err_str[:200]}")
            if attempt < 2:
                import time
                time.sleep(5)
    else:
        raise RuntimeError(f"LLM script generation failed after 3 retries: {last_error}")

    # Ensure lesson_type marker
    script["lesson_type"] = "listening"

    # Ensure all required fields exist
    script.setdefault("story_hook", "")
    script.setdefault("intro_zh", "")
    script.setdefault("outro", "That's all for today. Keep practicing!")
    script.setdefault("outro_zh", "")
    script.setdefault("title", "")
    script["cefr"] = script.get("cefr") or cefr  # used by thumbnail level badge
    script.setdefault("title_zh", script.get("intro_zh", ""))
    script.setdefault("practice_intro_en", "Now let's practice. Listen and repeat each sentence.")
    script.setdefault("practice_intro_zh", "現在來練習。請跟著朗讀每一句。")
    script.setdefault("char_a_description", "")
    script.setdefault("char_b_description", "")
    script.setdefault("char_a_gender", "male")
    script.setdefault("char_b_gender", "female")
    script.setdefault("char_a_role", "")
    script.setdefault("char_b_role", "")
    script.setdefault("youtube_title", "")
    script.setdefault("youtube_description", "")
    script.setdefault("youtube_tags", [])
    script.setdefault("thumbnail_prompt", "")
    script.setdefault("scene", "")
    script.setdefault("thumbnail_expression", "surprised and excited")
    script.setdefault("thumbnail_action", "looking toward the camera and gesturing naturally")
    script.setdefault("thumbnail_subtitle", "18句聽力練習")
    script.setdefault("thumbnail_icons", [])

    # Ensure dialogue has all required fields
    for line in script.get("dialogue", []):
        line.setdefault("phonetic", "")
        line.setdefault("zh", "")
        line.setdefault("image_prompt", "")
        line.setdefault("video_prompt", "")
        line.setdefault("poses", [])

    return script


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate listening script")
    parser.add_argument("--topic", required=True, help="Topic")
    parser.add_argument("--cefr", default="A2", choices=["A1", "A2", "B1", "B2", "C1", "C2"], help="CEFR level (default A2)")
    parser.add_argument("--output", default="script.json", help="Output JSON path")
    args = parser.parse_args()

    script = generate_listening_script(args.topic, args.cefr)
    Path(args.output).write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Script saved: {args.output}")
    print(f"Title: {script.get('title', '')}")
    print(f"Dialogue lines: {len(script.get('dialogue', []))}")
