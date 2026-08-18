"""Quest LLM client — task-hook slow-listening structure with 4-phase dialogue.

Structure mirrors a reference slow-listening video (welcome -> hook -> 4-phase
dialogue -> loop-closing CTA), scene-agnostic so it works for ANY topic.

4-phase dialogue: buildup -> core -> reveal -> review.
- buildup: question arises naturally between two friends
- core: mixed speakers (char_a/b/c) — education + transaction
- reveal: answer exposed inside dialogue
- review: friends confirm answer, evaluate experience

Includes a HOST character (节目主) who appears on-screen in Welcome, Hook,
and Outro segments via stop-motion animation.

Extends the original llm_client prompt with:
- char_c: a third character (service staff) for the core phase
- host: a fourth character (narrator/host) for narration segments
- dialogue lines carry "phase": "buildup" | "core" | "reveal" | "review"
- listening_question_*: the bait question whose answer is revealed in dialogue
- hook_intro_*: narrator opening (slow speech + assign listening task)
- welcome_*: short channel welcome line
- key_words: 5-8 target vocabulary items repeated 3+ times in the dialogue

Reuses _chat, _extract_json, _load_used_listening_summaries from parent llm_client.
"""
import sys
import os
from pathlib import Path

# Add parent dir to path so we can import llm_client
_PARENT = str(Path(__file__).parent.parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from llm_client import _chat, _extract_json, _load_used_listening_summaries


def split_phase_lines(num_lines: int) -> tuple[int, int, int, int]:
    """Split total lines into (buildup, core, reveal, review) by reference ratio ~10/54/15/21.

    Mirrors the bubble-tea reference video: short buildup where the listening
    question arises naturally, long core with mixed speakers (education +
    transactions), a reveal phase where the answer is exposed, and a review
    where the answer is confirmed.

    48 -> (5, 26, 7, 10). Each phase is at least 1; excess/deficit is
    absorbed by core (the largest phase).
    """
    n_buildup = round(num_lines * 0.10)
    n_core = round(num_lines * 0.54)
    n_reveal = round(num_lines * 0.15)
    n_review = num_lines - n_buildup - n_core - n_reveal

    n = [n_buildup, n_core, n_reveal, n_review]

    # Ensure each phase >= 1 (take from largest)
    for i in range(4):
        if n[i] < 1:
            max_idx = n.index(max(n))
            if max_idx != i and n[max_idx] > 1:
                n[max_idx] -= 1
                n[i] = 1

    # Ensure sum == num_lines (adjust core)
    total = sum(n)
    if total > num_lines:
        while sum(n) > num_lines:
            max_idx = n.index(max(n))
            if n[max_idx] > 1:
                n[max_idx] -= 1
            else:
                break
    elif total < num_lines:
        n[1] += (num_lines - total)

    return tuple(n)


def _build_quest_prompt(topic: str, cefr: str, used_dialogues: list[str] = None,
                        num_lines: int = 48,
                        n_buildup: int = 5, n_core: int = 26,
                        n_reveal: int = 7, n_review: int = 10) -> str:
    """Build prompt for the quest (task-hook slow listening) lesson.

    4-phase structure: buildup → core → reveal → review.
    Mirrors the bubble-tea reference video's narrative skeleton.
    """
    used_hint = ""
    if used_dialogues:
        used_hint = f"""
IMPORTANT — AVOID DUPLICATES: The following dialogue scenarios have already been generated.
Do NOT create dialogue that is too similar to these. Use a DIFFERENT situation, different speakers, different story:
{chr(10).join(f"  - {d}" for d in used_dialogues[:20])}
"""
    return f"""You are an expert ESL video director creating a SLOW LISTENING video for BEGINNER learners (overseas Chinese).

CORE MISSION: 幫助海外华人用最地道最日常的英语，搞定真实生活中的每一个场景。

Topic: {topic}
CEFR Level: {cefr}
Output: a JSON object ONLY (no markdown, no explanation).

THIS STRUCTURE WORKS FOR ANY SCENARIO — the topic can be airport check-in, hotel check-in, a job interview, grocery shopping, a pharmacy visit, opening a bank account, etc. Keep the same narrative skeleton regardless of the topic:
- BUILDUP: the two main characters feel a need and discuss options (e.g. tired -> want coffee; travelling -> need to check in; job hunting -> prepare for an interview). The LISTENING QUESTION arises NATURALLY during this conversation — one character asks a curious question, the other defers ("I'll tell you later", "you'll see").
- CORE: the characters arrive at the destination. This is the longest phase and includes BOTH friend-to-friend education AND transactions with the staff member (barista, gate agent, receptionist, interviewer, cashier...). The staff member explains options, the characters make choices, and the key transaction happens.
- REVEAL: after the characters try / experience the thing, one friend EXPLAINS THE ANSWER to the listening question. The other friend is surprised — they had a different guess. The true answer is revealed here, INSIDE the dialogue.
- REVIEW: the two main characters reunite and evaluate the experience. They explicitly CONFIRM the answer to the listening question (one asks again, the other answers correctly). They reuse the target vocabulary with positive, relaxed emotions, and mention coming back again.

{used_hint}
CONTENT REQUIREMENTS — SLOW LISTENING FOR ABSOLUTE BEGINNERS:
- Vocabulary MUST stay at {cefr} level (A1-A2 core words). Every sentence is AT MOST 10 WORDS.
- Pick 5-8 TARGET WORDS for the topic (nouns like "latte", "boarding pass", adjectives like "hot", "fresh", "expensive"). Each target word MUST appear naturally at least 3 times across the dialogue, in different sentences.
- Use simple high-frequency patterns that repeat naturally: "I want...", "Do you like...", "Can I have...", "Would you like...", "How much is it?".
- The conversation must feel warm, friendly, realistic — small daily life, not a textbook. Short lines, natural back-channeling ("Really?", "Sounds good!").
- The dialogue must tell a COMPLETE mini story: the need (buildup) -> the transaction (core) -> the answer (reveal) -> the happy reflection (review).

TECHNICAL REQUIREMENTS:
- THREE characters:
  - char_a and char_b: the two main characters (friends / colleagues / family). They talk in the buildup, reveal, and review phases. char_b ALSO appears in the core phase for friend-to-friend education scenes.
  - char_c: the staff member at the destination (barista, gate agent, receptionist, interviewer, ...). char_c ONLY appears in the core phase.
- The dialogue is ONE array of exactly {num_lines} lines, IN THIS EXACT ORDER, each line carrying a "phase" field:
  - lines 1-{n_buildup}: "phase": "buildup" — ONLY char_a and char_b speak. They express the need, discuss what each wants, and THE LISTENING QUESTION ARISES NATURALLY — one character asks a curious question related to the topic, the other says something like "I'll tell you later" or "you'll find out". Weave in the target vocabulary.
  - lines {n_buildup + 1}-{n_buildup + n_core}: "phase": "core" — char_a, char_b, AND char_c all speak. This is the practical heart: friend-to-friend education (explaining menu/options/how things work) AND transactions with the staff (greetings, polite requests, price/payment/confirmation). Extremely short, colloquial sentences. char_c MUST appear at least once in this phase.
  - lines {n_buildup + n_core + 1}-{n_buildup + n_core + n_reveal}: "phase": "reveal" — ONLY char_a and char_b speak. After trying/experiencing the thing, one friend EXPLAINS THE ANSWER to the listening question. The other friend is surprised ("Really? I thought..."). Clarify the common misunderstanding.
  - lines {n_buildup + n_core + n_reveal + 1}-{num_lines}: "phase": "review" — ONLY char_a and char_b speak. They evaluate the experience, CONFIRM THE ANSWER (one asks the question again, the other answers correctly), reuse target vocabulary in NEW sentence patterns, and mention coming back.
- CRITICAL: buildup, reveal, and review lines MUST use speaker "char_a" or "char_b" ONLY. Core lines MAY use "char_a", "char_b", or "char_c". Never let char_c appear outside core lines.
- Every dialogue line MUST include:
  - "speaker": "char_a" | "char_b" | "char_c"
  - "text": the English sentence (max 10 words)
  - "phase": "buildup" | "core" | "reveal" | "review"
  - "phonetic": IPA phonetic transcription in /slashes/ (use proper IPA symbols)
  - "zh": Traditional Chinese (繁體中文) translation
  - "image_prompt": a detailed prompt for ONE scene image: (1) the speaker's EXACT physical description (identical every time for the same character), (2) their role, (3) the scene location, (4) the action matching the dialogue text, (5) if two characters appear, describe both consistently. 3D cartoon style, 16:9.
- "welcome_en": a very short channel welcome line (max 8 words, e.g. "Welcome to English listening channel.")
- "welcome_zh": Traditional Chinese translation of the welcome line
- "char_a_description": detailed physical description of char_a (gender, hair, clothing). MUST be used identically in ALL char_a image_prompt entries.
- "char_b_description": detailed physical description of char_b. MUST be identical across ALL char_b entries.
- "char_c_description": detailed physical description of char_c (the staff member, including uniform or work attire). MUST be identical across ALL char_c entries.
- "host_description": detailed physical description of the HOST/NARRATOR character who appears on-screen in the Welcome, Hook Intro, and Outro segments (gender, hair, clothing — should look like a friendly, warm TV show host, NOT one of the 3 dialogue characters). This is a SEPARATE fourth character who never appears in the dialogue.
- "host_gender": "male" or "female"
- "char_a_gender", "char_b_gender", "char_c_gender": "male" or "female"
- "char_a_role", "char_b_role", "char_c_role": their roles (e.g. "office worker", "colleague", "barista")
- "listening_question_en": ONE specific, simple listening question whose ANSWER is revealed inside the REVEAL phase of the dialogue (e.g. "Why is it called bubble tea?"). The viewer must listen for it. Max 12 words.
- "listening_question_zh": Traditional Chinese translation of the question
- "hook_intro_en": the narrator's opening (60-90 seconds when read slowly, roughly 70-110 words, short simple sentences): (1) warm greeting and topic introduction, (2) reassure that you will speak VERY slowly and clearly, (3) give the listening task — repeat the listening question, (4) tell viewers the answer is INSIDE the video, (5) encourage them to write the answer in English in the comments at the end, (6) "Okay, let's begin." A1-level English.
- "hook_intro_zh": Traditional Chinese translation of the hook intro
- "key_words": an array of exactly 5-8 objects {{"en": string, "zh": string}} — the target vocabulary of this lesson (used for video description)
- "outro": the narrator's closing (60-90 seconds, ~80-110 words, short sentences): (1) a brief transition ("How was it?"), (2) warmly repeat the listening question, (3) encourage viewers to answer in English in the comments — even one simple sentence is perfect, (4) briefly describe the channel ("slow and easy English with small but useful tips"), (5) ask to subscribe, like, and mention channel memberships, (6) "See you next time. Bye." A1-level English.
- "outro_zh": Traditional Chinese translation of the outro
- "youtube_title": a high-CTR YouTube title for overseas Chinese beginners. ALL Chinese in Traditional Chinese (繁體中文). Prefer Pattern D for this series (vary a little between videos):
  Pattern D: 【慢速英文聽力】{{emoji}}{{topic 繁中}}情境對話｜超慢速清晰發音｜初學者必聽｜帶著問題聽，答案就在影片裡！｜留言寫下你的答案｜{{English topic}}
  You may also use Pattern A/B/C:
  Pattern A: 【沉浸式英文動畫】{{hook phrase}} {{emoji}} {{topic description in 繁中}}：{{specific skills listed}}，聽完就能說！｜{{English topic}}
  Pattern B: 【每天50句英文】{{emoji}}{{topic 繁中}}情境對話｜🎬沉浸式英文動畫｜旅行必備英文｜不用背多聽就會用｜英文聽力訓練｜情境英文對話｜高頻口語句型｜英文口說跟讀練習｜英文高效學習法
  Rules: Start with 【】bracket tag. Use ｜ as separator. Include 3-8 topic-relevant emoji. End with ｜{{English topic name}}. Title length 80-150 chars.
- "youtube_description": a full YouTube description (max 3000 chars). First line hook with the main keyword. Include the listening question early, a "⏱️ Chapters:" section with timestamps for: 00:00 Intro · Listening Task, 00:xx Slow Dialogue, 00:xx Outro · Answer & CTA. Also list the key_words with 繁中 meanings. End with 3 hashtags (#EnglishListening #SlowEnglish #LearnEnglish) and a subscribe CTA. ALL Chinese in Traditional Chinese.
- "youtube_tags": an array of 15-20 SEO tags (mix short and long-tail, English + Traditional Chinese, include "slow English", "English for beginners")
- "scene": the English name of the scene/location (e.g. "coffee shop", "airport check-in counter", "hotel lobby")
- "thumbnail_expression", "thumbnail_action": as usual (main character on thumbnail)
- "thumbnail_subtitle": short Traditional Chinese subtitle (e.g. "慢速聽力", "初學者必聽", "帶著問題聽")
- "thumbnail_icons": an array of 4-5 objects {{"en": string, "zh": string}} — target words shown as circular icons
- "thumbnail_prompt": detailed prompt for the thumbnail background image (3D Pixar style, expressive face, the scene, bright colors)
- "title": English title (e.g. "AT THE COFFEE SHOP")
- "cefr": exactly "{cefr}"
- "title_zh": Traditional Chinese short title (max 6 characters)
- "scene_zh": Traditional Chinese scene description (e.g. "咖啡廳 · 點餐")
- "story_hook": a compelling 1-sentence intro that sets the scene
- "intro_zh": Traditional Chinese translation of the story hook
- "practice_intro_en" / "practice_intro_zh": short generic placeholders (kept for schema compatibility)
- ALL Chinese text MUST be in Traditional Chinese (繁體中文)
- CRITICAL: the gender in each char description MUST be consistent with ALL of that character's image_prompt entries.
- CRITICAL: each character's physical description MUST be IDENTICAL across ALL their image_prompt entries. Do not change hair, clothing, or any trait between lines.
- CRITICAL: the scene location MUST be consistent throughout ALL lines (one main location for the whole topic).
- CRITICAL: the speaker field MUST use "char_a", "char_b", or "char_c" — never actual names.

JSON schema:
{{
  "title": string,
  "cefr": string,
  "title_zh": string,
  "scene_zh": string,
  "lesson_type": "listening",
  "story_hook": string,
  "intro_zh": string,
  "welcome_en": string,
  "welcome_zh": string,
  "hook_intro_en": string,
  "hook_intro_zh": string,
  "listening_question_en": string,
  "listening_question_zh": string,
  "key_words": [{{"en": string, "zh": string}}],
  "outro": string,
  "outro_zh": string,
  "practice_intro_en": string,
  "practice_intro_zh": string,
  "char_a_description": string,
  "char_b_description": string,
  "char_c_description": string,
  "char_a_gender": string,
  "char_b_gender": string,
  "char_c_gender": string,
  "host_description": string,
  "host_gender": string,
  "char_a_role": string,
  "char_b_role": string,
  "char_c_role": string,
  "youtube_title": string,
  "youtube_description": string,
  "youtube_tags": [string],
  "thumbnail_prompt": string,
  "scene": string,
  "thumbnail_expression": string,
  "thumbnail_action": string,
  "thumbnail_subtitle": string,
  "thumbnail_icons": [{{"en": string, "zh": string}}],
  "dialogue": [{{"speaker": string, "phase": string, "text": string, "phonetic": string, "zh": string, "image_prompt": string}}]
}}

Topic: {topic}"""


def generate_quest_script(topic: str, cefr: str = "A1",
                          lessons_dir: str = None,
                          num_lines: int = 48) -> dict:
    """Generate a quest (task-hook slow listening) lesson script.

    Returns script dict with all original fields PLUS quest fields:
    - dialogue[].phase (4 phases: buildup/core/reveal/review), char_c_*,
      listening_question_*, hook_intro_*, welcome_*, key_words, extended outro
    """
    import json
    import time

    n_buildup, n_core, n_reveal, n_review = split_phase_lines(num_lines)
    used_summaries = _load_used_listening_summaries(lessons_dir)
    prompt = _build_quest_prompt(topic, cefr, used_dialogues=used_summaries,
                                 num_lines=num_lines,
                                 n_buildup=n_buildup, n_core=n_core,
                                 n_reveal=n_reveal, n_review=n_review)

    last_error = None
    content = ""
    for attempt in range(3):
        try:
            content = _chat(
                [{"role": "user", "content": prompt}],
                temperature=0.8 if attempt == 0 else 0.7,
                max_tokens=16384,
            )
            script = _extract_json(content)
            break
        except (json.JSONDecodeError, RuntimeError) as e:
            last_error = e
            err_str = str(e)
            if isinstance(e, json.JSONDecodeError):
                print(f"  [LLM quest retry {attempt+1}/3] JSONDecodeError: {err_str[:200]}")
                print(f"  [LLM] Raw content (first 300 chars): {content[:300] if content else 'N/A'}")
            else:
                print(f"  [LLM quest retry {attempt+1}/3] {type(e).__name__}: {err_str[:200]}")
            if attempt < 2:
                time.sleep(5)
    else:
        raise RuntimeError(f"LLM script generation failed after 3 retries: {last_error}")

    script["lesson_type"] = "listening"

    # Original required fields
    script.setdefault("story_hook", "")
    script.setdefault("intro_zh", "")
    script.setdefault("outro", "That's all for today. Keep practicing!")
    script.setdefault("outro_zh", "")
    script.setdefault("title", "")
    script["cefr"] = script.get("cefr") or cefr
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
    script.setdefault("thumbnail_subtitle", "慢速聽力")
    script.setdefault("thumbnail_icons", [])

    # Quest-specific fields
    script.setdefault("welcome_en", "Welcome to English listening channel.")
    script.setdefault("welcome_zh", "歡迎來到英文聽力頻道。")
    script.setdefault("host_description", "a friendly young woman with short brown hair, wearing a smart blue blazer, warm smile, professional TV host appearance")
    script.setdefault("host_gender", "female")
    script.setdefault("char_c_description", "")
    script.setdefault("char_c_gender", "female")
    script.setdefault("char_c_role", "staff")
    script.setdefault("listening_question_en", "")
    script.setdefault("listening_question_zh", "")
    script.setdefault("hook_intro_en", "")
    script.setdefault("hook_intro_zh", "")
    script.setdefault("key_words", [])

    # Per-line defaults; repair missing "phase" by position (buildup->core->reveal->review)
    dialogue = script.get("dialogue", [])
    for i, line in enumerate(dialogue):
        line.setdefault("phonetic", "")
        line.setdefault("zh", "")
        line.setdefault("image_prompt", "")
        if not line.get("phase"):
            if i < n_buildup:
                line["phase"] = "buildup"
            elif i < n_buildup + n_core:
                line["phase"] = "core"
            elif i < n_buildup + n_core + n_reveal:
                line["phase"] = "reveal"
            else:
                line["phase"] = "review"

    return script
