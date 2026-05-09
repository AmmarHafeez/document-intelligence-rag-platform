from __future__ import annotations


def preview_text(text: str, max_chars: int = 160) -> str:
    preview = " ".join(text.split())
    if len(preview) <= max_chars:
        return preview
    return f"{preview[: max_chars - 3]}..."
