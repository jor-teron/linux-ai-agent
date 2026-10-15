"""
Streaming chat replies (v0.48), stdlib only.

stream_chat() is a generator of small event dicts the web server turns into
Server-Sent Events:

  {"type": "thinking", "text": "..."}   live reasoning text (folded box)
  {"type": "content",  "text": "..."}   live reply text
  {"type": "final", ...run_chat-shaped result..., "streamed": bool}

What streams: OpenAI-compatible providers (local Ollama / llama.cpp and
online OpenAI-compat like Groq, OpenRouter, DeepSeek ...) when Tools are OFF.
What does not: Gemini, Anthropic, and any turn with Tools ON. Those run the
normal brain.run_chat path and come back as one content event + final,
so the UI code is the same either way. (Assembling streamed tool_calls is
fragile across providers; the tool loop stays on the proven non-stream path.)

Imports from: ai_agent.config, brain, ratelimit, tools.
Used by: server (POST /api/chat/stream).
"""
import json
import time
import urllib.error
import urllib.request

from ai_agent.brain import _wants_no_think_tag, run_chat
from ai_agent.config import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    allowed,
    default_provider,
    get_provider_meta,
    history_turns,
    provider_key,
    provider_openai_base,
    resolve_use_tools,
)
from ai_agent.ratelimit import before_online_call, limits_override
from ai_agent.tools import build_system


# ---------------------------------------------------------------------------
# Live <think> splitter
# ---------------------------------------------------------------------------


class _ThinkSplitter:
    """Split streamed text into (thinking, content) as chunks arrive.

    Handles <think> / </think> tags that are cut across two chunks by
    holding back a few trailing characters that could be the start of a tag.
    """

    _OPEN, _CLOSE = "<think>", "</think>"

    def __init__(self):
        self.in_think = False
        self.buf = ""

    def feed(self, text):
        """Add a chunk; return a list of ("thinking"|"content", text) pieces."""
        self.buf += text or ""
        out = []
        while True:
            tag = self._CLOSE if self.in_think else self._OPEN
            i = self.buf.find(tag)
            if i >= 0:
                if i:
                    out.append(("thinking" if self.in_think else "content", self.buf[:i]))
                self.buf = self.buf[i + len(tag):]
                self.in_think = not self.in_think
                continue
            # Hold back a possible partial tag at the end of the buffer.
            keep = 0
            for n in range(1, len(tag)):
                if self.buf.endswith(tag[:n]):
                    keep = n
            emit = self.buf[: len(self.buf) - keep]
            if emit:
                out.append(("thinking" if self.in_think else "content", emit))
            self.buf = self.buf[len(self.buf) - keep:]
            return out

    def flush(self):
        """Return whatever is still held back at the end of the stream."""
        rest, self.buf = self.buf, ""
        return [("thinking" if self.in_think else "content", rest)] if rest else []


# ---------------------------------------------------------------------------
# SSE reader for an OpenAI-style /chat/completions stream
# ---------------------------------------------------------------------------


def _sse_lines(resp):
    """Yield parsed JSON objects from "data: {...}" lines until [DONE]."""
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _open_stream(url, body, headers, api_limit):
    """Rate-limit (unless local / Limits off), then open the streaming POST."""
    payload = json.dumps(body)
    before_online_call(payload, url=url, limit=True if api_limit else False)
    req = urllib.request.Request(url, data=payload.encode("utf-8"), headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=120)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def stream_chat(message, provider=None, model=None, history=None, use_tools=None, think=None, limits=None):
    """Generator of chat events (see module doc). Same flags as brain.run_chat."""
    provider = (provider or default_provider() or DEFAULT_PROVIDER).lower().strip()
    meta = PROVIDERS.get(provider) or {}
    want_tools = resolve_use_tools(message, None if use_tools is None else bool(use_tools))

    # Not streamable → normal path, delivered as one chunk + final.
    if meta.get("kind") != "openai" or want_tools:
        out = run_chat(message, provider=provider, model=model, history=history,
                       use_tools=use_tools, think=think, limits=limits)
        if out.get("thinking"):
            yield {"type": "thinking", "text": out["thinking"]}
        if out.get("ok") and out.get("reply"):
            yield {"type": "content", "text": out["reply"]}
        out["streamed"] = False
        yield dict(out, type="final")
        return

    with limits_override(limits):
        yield from _stream_openai(provider, message, model, history, think)


def _stream_openai(provider, message, model, history, think):
    """Stream one plain (no tools) OpenAI-compat completion as events."""
    meta = get_provider_meta(provider) or PROVIDERS[provider]
    key = provider_key(provider)
    model = allowed(provider, model)
    fail = {"type": "final", "ok": False, "reply": "", "tools": [], "model": model,
            "provider": provider, "streamed": True}
    if meta.get("needs_key", True) and not key:
        yield dict(fail, error="%s missing. Add it to .env." % (meta.get("env_key") or "API key"))
        return

    # Same message layout as brain.openai_compat_chat (tools off).
    messages = [{"role": "system", "content": build_system(use_tools=False)}]
    for turn in (history or [])[-history_turns():]:
        role = "user" if turn.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": turn.get("content", "")})
    user_msg = message + (" /no_think" if think is False and _wants_no_think_tag(model) else "")
    messages.append({"role": "user", "content": user_msg})

    base = meta.get("base") or provider_openai_base(provider)
    url = base.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer %s" % (key or "local")}
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/jor-teron/linux-ai-agent"
        headers["X-Title"] = "AI-Agent"
    api_limit = (meta.get("mode") or "online") != "local"

    body = {"model": model, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}}
    # Think toggle: same body fields as the non-stream path (Ollama only).
    if provider == "ollama" and think is not None:
        body["think"] = bool(think)
        if think is False:
            body["reasoning_effort"] = "none"

    # Open; on HTTP 400 drop the optional fields once and retry.
    t0 = time.perf_counter()
    try:
        try:
            resp = _open_stream(url, body, headers, api_limit)
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            for k in ("stream_options", "think", "reasoning_effort"):
                body.pop(k, None)
            t0 = time.perf_counter()
            resp = _open_stream(url, body, headers, api_limit)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:800]
        yield dict(fail, error="%s HTTP %s: %s" % (provider, e.code, err))
        return
    except Exception as e:  # noqa: BLE001 (includes RateLimited)
        yield dict(fail, error=str(e))
        return

    split = _ThinkSplitter()
    reply, thinking = [], []
    chunks, usage_tokens = 0, 0
    try:
        with resp:
            for obj in _sse_lines(resp):
                usage = obj.get("usage") or {}
                if usage.get("completion_tokens"):
                    usage_tokens = int(usage["completion_tokens"])
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    r = d.get("reasoning") or d.get("reasoning_content") or ""
                    if r:
                        chunks += 1
                        thinking.append(r)
                        yield {"type": "thinking", "text": r}
                    c = d.get("content") or ""
                    if c:
                        chunks += 1
                        for kind, text in split.feed(c):
                            (thinking if kind == "thinking" else reply).append(text)
                            yield {"type": kind, "text": text}
            for kind, text in split.flush():
                (thinking if kind == "thinking" else reply).append(text)
                yield {"type": kind, "text": text}
    except Exception as e:  # noqa: BLE001 (connection dropped mid-stream)
        yield dict(fail, error="stream interrupted: %s" % e, reply="".join(reply).strip())
        return

    secs = time.perf_counter() - t0
    # Real usage when sent; otherwise one chunk ≈ one token, marked estimate.
    tokens = usage_tokens or chunks
    stats = None
    if tokens and secs > 0:
        stats = {"tokens": tokens, "seconds": round(secs, 2), "tps": round(tokens / secs, 1),
                 "estimate": not usage_tokens}
    yield {"type": "final", "ok": True, "reply": "".join(reply).strip() or "(No text)",
           "thinking": "".join(thinking).strip(), "tools": [], "model": model,
           "provider": provider, "stats": stats, "streamed": True}
