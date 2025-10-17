# YouTube → Whisper → Timestamped Summary (Railway + Shortcut)

POST /summarize
{
  "youtube_url": "https://youtu.be/VIDEOID",
  "title": "Optional title",
  "max_chunk_seconds": 240
}

Env:
- OPENAI_API_KEY = ...
- SHORTCUT_BEARER = ... (optional shared secret for Shortcut Authorization header)
