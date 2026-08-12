"""Dialogue grouping for 方案 B — merge consecutive dialogue lines regardless
of speaker into a single video clip (≤ clip_duration seconds each).

Unlike Plan A (which only merges same-speaker consecutive lines), Plan B merges
ALL consecutive lines whose combined TTS audio fits within the clip_duration,
even if speakers alternate (A→B→A→B...).

Multi-character reference: the group's video clip is generated with BOTH
character reference images passed as image_urls (comma-separated), so Seedance2
knows what both characters look like.

No external dependencies. Pure Python.
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_dialogue_groups(dialogue: list[dict], dialogue_durations: list[float],
                          clip_duration: float = 15.0) -> list[dict]:
    """Group consecutive dialogue lines by combined audio <= clip_duration.

    Unlike Plan A, speaker identity is NOT a grouping criterion — consecutive
    lines are merged as long as total audio fits within clip_duration.

    Args:
        dialogue: list of dialogue line dicts (each has 'speaker').
        dialogue_durations: per-line TTS audio durations in seconds.
        clip_duration: max combined audio duration per group (Seedance2 limit).

    Returns:
        list of dicts: [{"lines": [0,1,2], "total_audio": 8.5, "speakers": ["char_a","char_b","char_a"]}, ...]
    """
    groups = []
    cur_lines: list[int] = []
    cur_total = 0.0
    cur_speakers: list[str] = []

    for i, line in enumerate(dialogue):
        speaker = line.get("speaker", "char_a")
        dur = dialogue_durations[i] if i < len(dialogue_durations) else 3.0

        if cur_total + dur <= clip_duration:
            cur_lines.append(i)
            cur_total += dur
            cur_speakers.append(speaker)
        else:
            if cur_lines:
                groups.append({
                    "lines": list(cur_lines),
                    "total_audio": cur_total,
                    "speakers": list(cur_speakers),
                })
            cur_lines = [i]
            cur_total = dur
            cur_speakers = [speaker]

    if cur_lines:
        groups.append({
            "lines": list(cur_lines),
            "total_audio": cur_total,
            "speakers": list(cur_speakers),
        })

    return groups


def compute_line_slices(line_durations: list[float], group_total: float,
                        clip_duration: float, lines: list[int]) -> dict:
    """Compute per-line clip slice offsets within a group's clip.

    Args:
        line_durations: TTS audio durations indexed by absolute dialogue line index.
        group_total: sum of TTS audio durations for this group.
        clip_duration: original clip length (e.g. 15s).
        lines: absolute dialogue line indices in this group.

    Returns:
        dict mapping line_idx -> {"clip_start": float, "clip_segment_dur": float}
    """
    slices = {}
    cumulative = 0.0
    total = group_total if group_total > 0 else 1.0
    for line_idx in lines:
        line_dur = line_durations[line_idx] if line_idx < len(line_durations) else 0.0
        clip_start = (cumulative / total) * clip_duration
        clip_segment_dur = (line_dur / total) * clip_duration
        slices[line_idx] = {
            "clip_start": clip_start,
            "clip_segment_dur": clip_segment_dur,
        }
        cumulative += line_dur
    return slices


def merge_group_prompt(group: dict, dialogue: list[dict]) -> str:
    """Merge all video_prompts from a group's lines into one continuous prompt.

    Args:
        group: a group dict from build_dialogue_groups (has 'lines').
        dialogue: full dialogue list.

    Returns:
        Combined single prompt describing the whole multi-line conversation.
    """
    parts = []
    for i in group["lines"]:
        line = dialogue[i]
        p = line.get("video_prompt", "") or line.get("image_prompt", "")
        if p:
            parts.append(p)
    return " ".join(parts)


def get_group_image_urls(group: dict, char_a_url: str, char_b_url: str,
                         scene_url: str = "") -> str:
    """Build comma-separated image_urls for a group, including both characters.

    Seedance2 supports up to 9 reference images. We pass both character images
    so the video generation knows what both look like.

    Args:
        group: group dict (has 'speakers' list).
        char_a_url: CDN URL for character A design image.
        char_b_url: CDN URL for character B design image.
        scene_url: optional scene URL (can be prepended for context).

    Returns:
        Comma-separated string of image URLs.
    """
    urls = []
    has_a = "char_a" in group.get("speakers", [])
    has_b = "char_b" in group.get("speakers", [])

    if has_a and char_a_url:
        urls.append(char_a_url)
    if has_b and char_b_url:
        urls.append(char_b_url)
    # If neither speaker found, fall back to both
    if not urls:
        if char_a_url:
            urls.append(char_a_url)
        if char_b_url:
            urls.append(char_b_url)

    return ",".join(urls)


if __name__ == "__main__":
    # Quick self-test: alternating speakers
    dlg = [
        {"speaker": "char_a"}, {"speaker": "char_b"}, {"speaker": "char_a"},
        {"speaker": "char_b"}, {"speaker": "char_a"}, {"speaker": "char_b"},
    ]
    durs = [3.1, 2.8, 2.5, 2.2, 3.0, 3.4]
    groups = build_dialogue_groups(dlg, durs, 15.0)
    print("Plan B groups (alternating speakers):")
    for g in groups:
        print(f"  lines={g['lines']} speakers={g['speakers']} total={g['total_audio']:.1f}s")
    print(f"Total groups: {len(groups)} (Plan A would give 6, Plan B gives fewer)")
