"""Enhanced LLM client — 7-chapter structure with vocabulary + comprehension questions.

Extends the original llm_client prompt with two new JSON fields:
- vocabulary: 6 keywords from the dialogue with IPA + 繁中 + example sentence
- comprehension_questions: 3 multiple-choice questions about the dialogue

Reuses _chat, _extract_json, _repair_truncated_json from parent llm_client.
"""
import sys
import os
from pathlib import Path

# Add parent dir to path so we can import llm_client
_PARENT = str(Path(__file__).parent.parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from llm_client import _chat, _extract_json, _load_used_listening_summaries


def _build_enhanced_prompt(topic: str, cefr: str, used_dialogues: list[str] = None,
                            num_lines: int = 18) -> str:
    """Build prompt for enhanced listening-practice lesson with vocabulary + quiz."""
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
  - "video_prompt": a detailed prompt for AI video generation. MUST include the SAME character description and action as image_prompt. MUST also include the dialogue text so the character appears to be speaking those words naturally. CRITICAL: the video MUST closely reference the uploaded reference image.
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
- "youtube_description": a full YouTube video description (max 3000 chars). First line must be a hook with the main keyword. Include a "⏱️ Chapters:" section with timestamps for: 00:00 Vocabulary, 01:30 Title, 01:35 Dialogue, 04:30 Slow Speed, 08:00 Quiz, 09:30 Shadowing, 15:30 Outro. End with 3 hashtags (#EnglishListening #ESL #LearnEnglish) and a subscribe CTA. ALL Chinese text in Traditional Chinese (繁體中文).
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
- "vocabulary": an array of exactly 6 key vocabulary words from the dialogue. Each entry MUST have:
  - "word": the English word
  - "phonetic": IPA phonetic transcription in /slashes/
  - "zh": Traditional Chinese (繁體中文) meaning
  - "example": a short example sentence using the word in context
- "comprehension_questions": an array of exactly 3 multiple-choice questions about the dialogue content. Each entry MUST have:
  - "question": the question in English
  - "options": array of 4 options (include letter prefix, e.g. "A) ...", "B) ...", "C) ...", "D) ...")
  - "answer": the letter of the correct answer (e.g. "B")
- ALL Chinese text MUST be in Traditional Chinese (繁體中文)
- CRITICAL: the gender in char_a_description and char_b_description MUST be consistent.
- CRITICAL: each speaker's description MUST be IDENTICAL across ALL image_prompt and video_prompt entries.
- CRITICAL: image_prompt and video_prompt MUST match the dialogue context exactly.
- CRITICAL: the scene location MUST be consistent throughout ALL lines.
- CRITICAL: the speaker field in dialogue MUST use "char_a" or "char_b".

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
  "vocabulary": [{{"word": string, "phonetic": string, "zh": string, "example": string}}],
  "comprehension_questions": [{{"question": string, "options": [string, string, string, string], "answer": string}}],
  "dialogue": [{{"speaker": string, "text": string, "phonetic": string, "zh": string, "image_prompt": string, "video_prompt": string}}]
}}

Topic: {topic}"""


def generate_listening_script_enhanced(topic: str, cefr: str = "A2",
                                        lessons_dir: str = None,
                                        num_lines: int = 18) -> dict:
    """Generate an enhanced listening-practice lesson script with vocabulary + quiz.

    Returns script dict with all original fields PLUS:
    - vocabulary: list of {word, phonetic, zh, example}
    - comprehension_questions: list of {question, options, answer}
    """
    used_summaries = _load_used_listening_summaries(lessons_dir)
    prompt = _build_enhanced_prompt(topic, cefr, used_dialogues=used_summaries,
                                     num_lines=num_lines)

    import time
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
            if "quota exceeded" in err_str:
                print(f"  [LLM] FATAL: {err_str}")
                raise RuntimeError(err_str) from e
            if isinstance(e, json.JSONDecodeError):
                print(f"  [LLM enhanced retry {attempt+1}/3] JSONDecodeError: {err_str[:200]}")
                print(f"  [LLM] Raw content (first 300 chars): {content[:300] if 'content' in dir() else 'N/A'}")
            else:
                print(f"  [LLM enhanced retry {attempt+1}/3] {type(e).__name__}: {err_str[:200]}")
            if attempt < 2:
                time.sleep(5)
    else:
        raise RuntimeError(f"LLM script generation failed after 3 retries: {last_error}")

    import json
    script["lesson_type"] = "listening"

    # Ensure original required fields
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

    # Ensure enhanced fields
    script.setdefault("vocabulary", [])
    script.setdefault("comprehension_questions", [])

    for line in script.get("dialogue", []):
        line.setdefault("phonetic", "")
        line.setdefault("zh", "")
        line.setdefault("image_prompt", "")
        line.setdefault("video_prompt", "")

    for v in script.get("vocabulary", []):
        v.setdefault("phonetic", "")
        v.setdefault("zh", "")
        v.setdefault("example", "")

    for q in script.get("comprehension_questions", []):
        q.setdefault("options", ["", "", "", ""])
        q.setdefault("answer", "")

    return script
