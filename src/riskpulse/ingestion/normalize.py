"""Text normalisation shared by all source adapters."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")
_HANDLE = re.compile(r"@\w+")


def clean_text(text: str | None, strip_urls: bool = True) -> str:
    """Unescape HTML, normalise unicode, optionally drop URLs, collapse whitespace."""
    if not text:
        return ""
    t = html.unescape(str(text))
    t = unicodedata.normalize("NFKC", t)
    if strip_urls:
        t = _URL.sub(" ", t)
    return _WS.sub(" ", t).strip()


def dedupe_key(text: str) -> str:
    """Hash of a canonical form used for exact-duplicate detection.

    Lowercased, URLs and @handles removed, punctuation and whitespace collapsed, so
    retweets and syndicated copies of the same headline collide.
    """
    t = _HANDLE.sub(" ", _URL.sub(" ", text.lower()))
    t = re.sub(r"^rt\b", " ", t.strip())
    t = re.sub(r"[^0-9a-z$%]+", " ", t)
    return hashlib.sha1(_WS.sub(" ", t).strip().encode()).hexdigest()
