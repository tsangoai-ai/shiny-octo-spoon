import os, tempfile, math, re, json, subprocess
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SHORTCUT_BEARER = os.getenv("SHORTCUT_BEARER")  # optional shared secret
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

class SummarizeIn(BaseModel):
    youtube_url: str
    title: str | None = None
    max_chunk_seconds: int | None = 240   # ~4 min per chunk
    model_summary: str | None = "gpt-5-turbo"
    model_whisper: str | None = "whisper-1"

def hhmmss(seconds: float) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def youtube_timestamp_link(url: str, seconds: int) -> str:
    # Works for both watch?v=... and youtu.be/...
    if "watch?v=" in url:
        return f"{url}&t={seconds}s"
    return f"{url}?t={seconds}s"

def download_audio_m4a(url: str) -> str:
    """
    Download best audio (m4a) without re-encoding to avoid ffmpeg dependency.
    Returns local file path.
    """
    tmpdir = tempfile.mkdtemp()
    out = os.path.join(tmpdir, "%(id)s.%(ext)s")
    # Prefer m4a; fall back to bestaudio
    cmd = [
        "yt-dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "-o", out,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]
    cmd.append(url)
    subprocess.check_call(cmd)
    # Find the downloaded file
    import glob
    files = glob.glob(os.path.join(tmpdir, "*.*"))
    if not files:
        raise RuntimeError("Audio download failed.")
    return files[0]

def whisper_transcribe(file_path: str, model: str = "whisper-1") -> Dict[str, Any]:
    """
    Calls OpenAI Whisper with verbose JSON to get segments (with timestamps).
    """
    with open(file_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="verbose_json",  # includes segments with start/end
            temperature=0.0,
        )
    # The SDK returns a pydantic-ish object; coerce to dict
    if hasattr(tr, "model_dump"):
        return tr.model_dump()
    return json.loads(json.dumps(tr, default=lambda o: getattr(o, "__dict__", str(o))))

def chunk_segments(segments: List[Dict[str, Any]], max_secs: int) -> List[List[Dict[str, Any]]]:
    """
    Group Whisper segments into ~max_secs buckets by start time.
    """
    if not segments:
        return []
    buckets, cur, start0 = [], [], segments[0].get("start", 0)
    for seg in segments:
        if not cur:
            start0 = seg.get("start", 0)
        cur.append(seg)
        if (seg.get("start", 0) - start0) >= max_secs:
            buckets.append(cur); cur = []
    if cur:
        buckets.append(cur)
    return buckets

def summarize_chunk(text: str, start_t: int, end_t: int, model_summary: str) -> str:
    prompt = f"""You are an expert note-taker.
Summarize this transcript segment into a concise, skimmable section with:
- A short heading (<=8 words)
- 3–6 bullets with concrete facts (no fluff)
- Quote key phrases exactly in quotation marks, if impactful
- Include the segment's start time {hhmmss(start_t)} and end time {hhmmss(end_t)} on the first line in (brackets)

Transcript segment:
{text}"""
    resp = client.chat.completions.create(
        model=model_summary,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def synthesize_overall(sections_markdown: str, title: str, model_summary: str) -> str:
    prompt = f"""Create an executive summary (5–8 bullets) of the video titled “{title}” based on these section notes.
End with 3 short 'Apply It Tomorrow' action steps (imperative). Avoid repetition; preserve technical terms.

Section notes:
{sections_markdown}
"""
    resp = client.chat.completions.create(
        model=model_summary,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/summarize")
async def summarize(req: Request, body: SummarizeIn):
    # Optional bearer check to keep your endpoint private for your Shortcut
    if SHORTCUT_BEARER:
        auth = req.headers.get("Authorization", "")
        if auth != f"Bearer {SHORTCUT_BEARER}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    url = body.youtube_url.strip()
    title = body.title or "YouTube Video"
    max_secs = int(body.max_chunk_seconds or 240)
    model_summary = body.model_summary or "gpt-5-turbo"
    model_whisper = body.model_whisper or "whisper-1"

    try:
        audio_path = download_audio_m4a(url)
        tr = whisper_transcribe(audio_path, model=model_whisper)
        segments = tr.get("segments") or []
        if not segments:
            raise RuntimeError("No segments returned by Whisper.")

        # Build chunks
        chunks = chunk_segments(segments, max_secs=max_secs)

        # Summarize each chunk
        section_blocks = []
        for idx, segs in enumerate(chunks, start=1):
            start_t = int(segs[0].get("start", 0))
            end_t = int(segs[-1].get("end", segs[-1].get("start", 0)))
            text = " ".join(s.get("text", "") for s in segs).strip()
            block = summarize_chunk(text, start_t, end_t, model_summary=model_summary)
            # Add a timestamped anchor header that links back to YouTube
            ts_link = youtube_timestamp_link(url, start_t)
            header = f"### [{hhmmss(start_t)}]({ts_link})"
            section_blocks.append(f"{header}\n\n{block}")

        # Synthesize overall
        joined_sections = "\n\n".join(section_blocks)
        overall = synthesize_overall(joined_sections, title=title, model_summary=model_summary)

        # Final Markdown
        md = f"# Summary — {title}\n\n## Executive Summary\n{overall}\n\n## Timestamped Sections\n\n{joined_sections}\n"
        return JSONResponse({"markdown": md})

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"yt-dlp failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
