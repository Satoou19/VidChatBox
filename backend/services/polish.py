"""Background task for AI-polishing transcripts.

Extracted from the old monolithic main.py.
"""

import os
import re
import traceback

from backend.pipeline.project_manager import format_timestamp
from backend.services.llm_client import call_llm_for_polishing


def format_chunk_raw(chunk_segments, base_url):
    """Formats raw transcript segments into markdown bullet points with timestamp links."""
    lines = []
    for seg in chunk_segments:
        start_time = seg.get("start", 0.0)
        time_str = format_timestamp(start_time)
        text = seg.get("text", "").strip()
        if base_url:
            if "youtube.com" in base_url or "youtu.be" in base_url:
                link = f"{base_url}&t={int(start_time)}" if "?" in base_url else f"{base_url}?t={int(start_time)}"
            else:
                link = f"{base_url}#t={int(start_time)}"
            lines.append(f"- **[{time_str}]({link})**: {text}")
        else:
            lines.append(f"- **[{time_str}]**: {text}")
    return "\n".join(lines)


def polish_transcript_background(
    task_id,
    project_dir,
    safe_video_id,
    metadata,
    segments,
    provider,
    model,
    keys,
    polishing_tasks,
):
    """Runs AI polishing in background, updating task status in polishing_tasks store.

    Args:
        polishing_tasks: A TaskStore instance for tracking polishing progress.
    """
    try:
        # 1. Group segments into 15-minute chunks
        chunk_duration = 900.0
        chunks = []
        current_chunk = []
        current_start = 0.0

        for seg in segments:
            seg_start = seg.get("start", 0.0)
            if not current_chunk:
                current_start = seg_start
                current_chunk.append(seg)
            elif seg_start - current_start < chunk_duration:
                current_chunk.append(seg)
            else:
                chunks.append(current_chunk)
                current_chunk = [seg]
                current_start = seg_start

        if current_chunk:
            chunks.append(current_chunk)

        if not chunks:
            raise ValueError("No transcript segments found to polish.")

        active_provider = provider or "gemini"
        active_model = model or ""

        # Setup keys
        gemini_key = keys.get("gemini_key") or os.getenv("GEMINI_API_KEY")
        openai_key = keys.get("openai_key") or os.getenv("OPENAI_API_KEY")
        groq_key = keys.get("groq_key") or os.getenv("GROQ_API_KEY")
        deepseek_key = keys.get("deepseek_key") or os.getenv("DEEPSEEK_API_KEY")
        openrouter_key = keys.get("openrouter_key") or os.getenv("OPENROUTER_API_KEY")
        llm7_key = keys.get("llm7_key") or os.getenv("LLM7_API_KEY")

        # System prompt for formatting
        system_prompt = (
            "You are an expert technical editor and content summarizer. "
            "Your task is to take a raw timestamped video transcript section in Markdown and format it into a premium, readable, structured document.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST preserve the exact chronological order of the statements.\n"
            "2. You MUST keep the exact markdown timestamp citation links, e.g., `[HH:MM:SS](URL_WITH_TIMESTAMP)`. Never change the URLs or the link texts.\n"
            "3. Correct grammar, spelling, and sentence flow, removing filler words (like 'um', 'uh', stuttering) while maintaining the original meaning and tone.\n"
            "4. Group segments into logical chapters/sections with descriptive H2 (`##`) headings.\n"
            "5. Under each heading, provide a brief 1-2 sentence bulleted summary of what was discussed in that section, followed by the polished chronological timestamped bullet points.\n"
            "6. Return ONLY the polished Markdown content. Do not add conversational intro/outro text or markdown block wrappers (e.g. ```markdown)."
        )

        polished_chunks = []
        base_url = metadata.get("url") or metadata.get("webpage_url") or ""

        for idx, chunk_segs in enumerate(chunks):
            polishing_tasks.set(task_id, {
                "status": "processing",
                "percent": float(idx) / len(chunks) * 100,
                "current_chunk": idx + 1,
                "total_chunks": len(chunks),
                "error": None,
            })

            raw_chunk_text = format_chunk_raw(chunk_segs, base_url)

            polished_text = call_llm_for_polishing(
                text=raw_chunk_text,
                system_prompt=system_prompt,
                provider=active_provider,
                model=active_model,
                gemini_key=gemini_key,
                openai_key=openai_key,
                groq_key=groq_key,
                deepseek_key=deepseek_key,
                openrouter_key=openrouter_key,
                llm7_key=llm7_key,
            )
            polished_chunks.append(polished_text)

        # Assemble polished document
        title = metadata.get("title", "Unknown Video")
        duration = metadata.get("duration")
        uploader = metadata.get("uploader", "Unknown")
        duration_str = format_timestamp(duration) if duration else "Unknown"

        header_lines = [
            f"# {title} (Polished Summary & Transcript)",
            "",
            f"- **Uploader**: {uploader}",
        ]
        if base_url:
            header_lines.append(f"- **Original VID**: [Watch on YouTube]({base_url})")
        header_lines.append(f"- **Duration**: {duration_str}")
        header_lines.append("")
        header_lines.append("---")
        header_lines.append("")

        full_polished_md = "\n".join(header_lines) + "\n\n" + "\n\n".join(polished_chunks)

        # Save to disk
        md_dir = os.path.join(project_dir, "markdown")
        os.makedirs(md_dir, exist_ok=True)
        md_filepath = os.path.join(md_dir, f"video_{safe_video_id}_polished.md")
        with open(md_filepath, "w", encoding="utf-8") as md_f:
            md_f.write(full_polished_md)

        polishing_tasks.set(task_id, {
            "status": "completed",
            "percent": 100.0,
            "error": None,
        })

    except Exception as e:
        traceback.print_exc()
        polishing_tasks.set(task_id, {
            "status": "failed",
            "percent": 0.0,
            "error": str(e),
        })
