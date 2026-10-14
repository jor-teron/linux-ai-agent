"""
LLM providers: Gemini, OpenAI-compat, Anthropic, and chat dispatch.

Talks to cloud APIs and local OpenAI-compatible runtimes (Ollama,
llama.cpp) via stdlib urllib, and runs tool-call loops. Provider
catalogs live in providers.py (re-exported through config).

Online HTTP goes through ai_agent.ratelimit first (API_DELAY_SEC / RPM /
RPD / TPM). Local ollama and llamacpp skip that limiter.

Imports from: ai_agent.config, ai_agent.tools.
Used by: server (run_chat), tools (plain chat for jobs, _http_json).
"""
import json
import re
import time
import urllib.error
import urllib.request
from ai_agent.config import (
    DEFAULT_PROVIDER,
    GEMINI_API_BASE,
    PROVIDERS,
    allowed,
    default_model,
    default_provider,
    get_provider_meta,
    history_turns,
    provider_key,
    provider_openai_base,
    resolve_use_tools,
)
from ai_agent.ratelimit import before_online_call, is_local_api_url
from ai_agent.tools import TOOL_DECLS, anthropic_tools, build_system, dispatch, openai_tools


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------


def _http_json(url, body, headers, timeout=90, limit=None):
    """POST JSON and parse the JSON reply. Raises on HTTP/network/rate errors.

    Before the socket opens, online calls go through the shared limiter
    (API_DELAY_SEC, RPM, RPD, TPM). limit=False skips it (local ollama /
    llama.cpp — not a billed API). limit=None skips loopback and the
    configured local base URLs. A RateLimited error starts with
    "rate limited"; callers return {ok: False, error: ...} and do not retry.
    """
    try:
        payload = body if isinstance(body, str) else json.dumps(body)
    except (TypeError, ValueError):
        payload = str(body)
    apply_limit = limit
    if apply_limit is None:
        apply_limit = not is_local_api_url(url)
    if apply_limit:
        # payload chars feed the TPM estimate (real token counts not available).
        before_online_call(payload, url=url, limit=True)
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    req = urllib.request.Request(
        url,
        data=raw,
        headers=headers,
        method="POST",
    )
    # Time only the network/model part (after any rate-limit wait) so the
    # tokens/sec line under each reply reflects real model speed.
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if isinstance(data, dict):
        data["_elapsed"] = time.perf_counter() - t0
    return data


# ---------------------------------------------------------------------------
# Thinking split (v0.45): reasoning models like Qwen3
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.S | re.I)


def _split_think(text):
    """Split "<think>...</think>" blocks out of a reply.

    Returns (reply_without_thinking, thinking_text). An unclosed <think>
    at the end counts as thinking too. Plain replies come back unchanged.
    """
    text = text or ""
    if "<think" not in text.lower():
        return text, ""
    parts = [m.group(1).strip() for m in _THINK_RE.finditer(text)]
    clean = _THINK_RE.sub("", text).strip()
    return clean, "\n\n".join(x for x in parts if x)


def _http_err_think_field(err_text):
    """True if an HTTP 400 body looks like the server rejected the "think" field."""
    t = (err_text or "").lower()
    return "think" in t or "reasoning" in t or "unknown" in t or "unrecognized" in t or "invalid" in t


# ---------------------------------------------------------------------------
# Speed stats (tokens / second under each reply, v0.44)
# ---------------------------------------------------------------------------


def _out_tokens(data):
    """Output (generated) token count from any provider reply, or 0.

    OpenAI-compat (incl. Ollama / llama.cpp): usage.completion_tokens
    Gemini: usageMetadata.candidatesTokenCount
    Anthropic: usage.output_tokens
    """
    if not isinstance(data, dict):
        return 0
    usage = data.get("usage") or {}
    meta = data.get("usageMetadata") or {}
    n = usage.get("completion_tokens") or usage.get("output_tokens") or meta.get("candidatesTokenCount") or 0
    try:
        return int(n)
    except (TypeError, ValueError):
        return 0


class _Speed:
    """Adds up tokens and model seconds across tool rounds of one reply."""

    def __init__(self):
        self.tokens = 0
        self.secs = 0.0

    def add(self, data):
        self.tokens += _out_tokens(data)
        if isinstance(data, dict):
            self.secs += float(data.get("_elapsed") or 0)

    def stats(self):
        """{tokens, seconds, tps} or None when the provider sent no count."""
        if self.tokens <= 0 or self.secs <= 0:
            return None
        return {
            "tokens": self.tokens,
            "seconds": round(self.secs, 2),
            "tps": round(self.tokens / self.secs, 1),
        }


# ---------------------------------------------------------------------------
# Plain chat (no tools) — used by scheduled jobs in tools.py
# ---------------------------------------------------------------------------


def _plain_chat(prompt, provider=None, model=None):
    """Short call without tools (for scheduled jobs)."""
    provider = provider or default_provider()
    if provider not in PROVIDERS:
        provider = DEFAULT_PROVIDER
    model = allowed(provider, model or default_model(provider))
    meta = get_provider_meta(provider) or PROVIDERS[provider]
    kind = meta["kind"]
    needs_key = bool(meta.get("needs_key", True))
    key = provider_key(provider)
    if needs_key and not key:
        return {"ok": False, "error": "No API key for %s" % provider, "reply": ""}
    # Local runtimes are not billed — do not consume RPM/RPD/TPM/delay.
    api_limit = (meta.get("mode") or "online") != "local"
    try:
        if kind == "gemini":
            url = "%s/models/%s:generateContent?key=%s" % (GEMINI_API_BASE, model, key)
            data = _http_json(
                url,
                {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 1024},
                },
                {"Content-Type": "application/json"},
                timeout=60, limit=api_limit,
            )
            parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
            texts = [p.get("text", "") for p in parts if "text" in p]
            reply = "\n".join(texts).strip() or "(No text)"
            return {"ok": True, "reply": reply, "model": model, "provider": provider}
        if kind == "openai":
            # Cloud base from catalog, or local Ollama/llama.cpp OpenAI path.
            base = meta.get("base") or provider_openai_base(provider)
            url = base.rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            # Local runtimes need no real key; send a dummy bearer some servers expect.
            headers["Authorization"] = "Bearer %s" % (key or "local")
            data = _http_json(
                url,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                },
                headers,
                timeout=60, limit=api_limit,
            )
            msg = ((data.get("choices") or [{}])[0].get("message") or {})
            reply = (msg.get("content") or "").strip() or "(No text)"
            return {"ok": True, "reply": reply, "model": model, "provider": provider}
        if kind == "anthropic":
            url = meta["base"].rstrip("/") + "/messages"
            data = _http_json(
                url,
                {
                    "model": model,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
                {
                    "Content-Type": "application/json",
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=60, limit=api_limit,
            )
            texts = [
                b.get("text", "")
                for b in (data.get("content") or [])
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            reply = "\n".join(texts).strip() or "(No text)"
            return {"ok": True, "reply": reply, "model": model, "provider": provider}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "reply": ""}
    return {"ok": False, "error": "Unknown provider kind", "reply": ""}


# ---------------------------------------------------------------------------
# Gemini (generateContent + function calling)
# ---------------------------------------------------------------------------


def gemini_chat(message, model, history, use_tools=True):
    """Chat with Gemini; tools optional (token trim when use_tools=False)."""
    key = provider_key("gemini")
    if not key:
        return {
            "ok": False,
            "error": "GEMINI_API_KEY missing. Copy .env.example to .env and add your key.",
            "reply": "",
            "tools": [],
            "provider": "gemini",
        }

    model = allowed("gemini", model)
    contents = []
    n_hist = history_turns()
    for turn in (history or [])[-n_hist:]:
        role = "user" if turn.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [{"text": turn.get("content", "")}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    url = "%s/models/%s:generateContent?key=%s" % (GEMINI_API_BASE, model, key)
    tool_trace = []
    reply = ""
    system = build_system(use_tools=use_tools)
    rounds = 6 if use_tools else 1
    speed = _Speed()

    for _ in range(rounds):
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
        }
        if use_tools:
            body["tools"] = [{"function_declarations": TOOL_DECLS}]
        try:
            data = _http_json(url, body, {"Content-Type": "application/json"}, timeout=90)
            speed.add(data)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")[:800]
            return {
                "ok": False,
                "error": "Gemini HTTP %s: %s" % (e.code, err),
                "reply": "",
                "tools": tool_trace,
                "model": model,
                "provider": "gemini",
            }
        except Exception as e:  # noqa: BLE001
            return {
                "ok": False,
                "error": str(e),
                "reply": "",
                "tools": tool_trace,
                "model": model,
                "provider": "gemini",
            }

        parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
        fn_calls, texts = [], []
        for p in parts:
            if "functionCall" in p:
                fn_calls.append(p["functionCall"])
            elif "text" in p:
                texts.append(p["text"])

        if (not use_tools) or (not fn_calls):
            reply = "\n".join(texts).strip() or "(No text)"
            break

        contents.append({"role": "model", "parts": parts})
        response_parts = []
        for fc in fn_calls:
            name = fc.get("name") or ""
            args = fc.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = dispatch(name, args)
            tool_trace.append({"name": name, "args": args, "result": result})
            response_parts.append({"functionResponse": {"name": name, "response": result}})
        contents.append({"role": "user", "parts": response_parts})
    else:
        reply = reply or "Stopped after too many tool steps."

    # Move any <think> blocks out of the visible reply (v0.45).
    reply, thinking = _split_think(reply)
    return {"ok": True, "reply": reply or "(No text)", "thinking": thinking, "tools": tool_trace, "model": model, "provider": "gemini", "stats": speed.stats()}


# ---------------------------------------------------------------------------
# OpenAI-compatible Chat Completions (OpenAI, xAI, DeepSeek, Ollama, llama.cpp)
# ---------------------------------------------------------------------------

# Curated tiny local models that reject OpenAI tool schemas (Ollama 400:
# "does not support tools"). Match flexibly: ignore registry path and :tag.
_NO_TOOLS_MODEL_FRAGMENTS = ("tinydolphin", "tinyllama")


def _model_lacks_tool_support(model, meta=None):
    """True when we should skip tools up front (known tiny / supports_tools=False)."""
    meta = meta or {}
    if meta.get("supports_tools") is False:
        return True
    mid = (model or "").lower()
    # e.g. registry.ollama.ai/library/tinydolphin:latest → tinydolphin
    leaf = mid.rsplit("/", 1)[-1]
    name = leaf.split(":", 1)[0]
    for frag in _NO_TOOLS_MODEL_FRAGMENTS:
        if frag in mid or frag == name:
            return True
    return False


def _http_err_no_tools(err_text):
    """True if the server rejected the request because the model has no tools."""
    return "does not support tools" in (err_text or "").lower()


def openai_compat_chat(provider, message, model, history, use_tools=True, think=None):
    """Chat with an OpenAI-style API, running workspace tools as tool_calls.

    think (v0.45): None = leave the request alone. False = ask local Ollama
    to skip thinking ("think": false in the body, plus " /no_think" on the
    last user message for qwen3 models). Thinking text from `reasoning` /
    `reasoning_content` / <think> tags is returned as "thinking", never in reply.

    Used for cloud OpenAI-compat providers and local Ollama / llama.cpp
    (local base URL from providers.provider_openai_base; no API key required).

    Small local models (tinydolphin, tinyllama, …) often reject tool schemas.
    We skip tools for known no-tool names / supports_tools=False, and if any
    model returns "does not support tools", retry once without tools/tool_choice
    (plain messages + system prompt, empty tools trace, no tool loop).
    """
    meta = get_provider_meta(provider) or PROVIDERS[provider]
    needs_key = bool(meta.get("needs_key", True))
    key = provider_key(provider)
    if needs_key and not key:
        return {
            "ok": False,
            "error": "%s missing. Add it to .env." % (meta.get("env_key") or "API key"),
            "reply": "",
            "tools": [],
            "provider": provider,
        }

    model = allowed(provider, model)
    messages = [{"role": "system", "content": build_system(use_tools=use_tools)}]
    n_hist = history_turns()
    for turn in (history or [])[-n_hist:]:
        role = "user" if turn.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": turn.get("content", "")})
    # Qwen3 understands a "/no_think" soft switch inside the user message.
    is_ollama = provider == "ollama"
    user_msg = message
    if think is False and "qwen3" in (model or "").lower():
        user_msg = message + " /no_think"
    messages.append({"role": "user", "content": user_msg})
    # Only local Ollama gets the "think" body field; dropped once if rejected.
    send_think_field = is_ollama and think is not None
    thinking_parts = []

    # Resolved base includes /v1 for local Ollama and llama.cpp servers.
    base = meta.get("base") or provider_openai_base(provider)
    url = base.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        # Local: dummy bearer; online: real key from .env.
        "Authorization": "Bearer %s" % (key or "local"),
    }
    # OpenRouter asks for these optional attribution headers on free/paid routes.
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/jor-teron/linux-ai-agent"
        headers["X-Title"] = "AI-Agent"
    # Local ollama/llamacpp skip the limiter even if the base is not loopback.
    api_limit = (meta.get("mode") or "online") != "local"
    tool_trace = []
    reply = ""

    # Skip tools when caller asked off, or for known tiny local models.
    use_tools = bool(use_tools) and (not _model_lacks_tool_support(model, meta))
    no_tools_retry_done = False
    speed = _Speed()
    # Without tools: single completion only (no tool loop).
    rounds = 1 if not use_tools else 6
    i = 0
    while i < rounds:
        i += 1
        body = {
            "model": model,
            "messages": messages,
        }
        if use_tools:
            body["tools"] = openai_tools()
            body["tool_choice"] = "auto"
        if send_think_field:
            body["think"] = bool(think)
            # Ollama's /v1 endpoint ignores "think" but honours reasoning_effort.
            if think is False:
                body["reasoning_effort"] = "none"
        try:
            data = _http_json(url, body, headers, timeout=90, limit=api_limit)
            speed.add(data)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")[:800]
            # Server rejected the "think" field → one retry without it.
            if send_think_field and e.code == 400 and _http_err_think_field(err):
                send_think_field = False
                rounds += 1
                continue
            # Auto-fallback: model rejected tools → one plain retry (no tools).
            if use_tools and not no_tools_retry_done and _http_err_no_tools(err):
                use_tools = False
                no_tools_retry_done = True
                rounds = i + 1  # allow exactly one more attempt without tools
                continue
            return {
                "ok": False,
                "error": "%s HTTP %s: %s" % (provider, e.code, err),
                "reply": "",
                "tools": tool_trace,
                "model": model,
                "provider": provider,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "ok": False,
                "error": str(e),
                "reply": "",
                "tools": tool_trace,
                "model": model,
                "provider": provider,
            }

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        # Thinking can arrive as a separate field or inline <think> tags.
        r = msg.get("reasoning") or msg.get("reasoning_content") or ""
        if r and r.strip():
            thinking_parts.append(r.strip())
        content, inline_think = _split_think(content)
        if inline_think:
            thinking_parts.append(inline_think)

        if not use_tools or not tool_calls:
            reply = (content or "").strip() or "(No text)"
            break

        # Append assistant turn (with tool_calls)
        messages.append(
            {
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": tool_calls,
            }
        )
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}
            result = dispatch(name, args)
            tool_trace.append({"name": name, "args": args, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id") or name,
                    "content": json.dumps(result),
                }
            )
    else:
        reply = reply or "Stopped after too many tool steps."

    return {
        "ok": True,
        "reply": reply,
        "thinking": "\n\n".join(thinking_parts),
        "tools": tool_trace,
        "model": model,
        "provider": provider,
        "stats": speed.stats(),
    }


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------


def anthropic_chat(message, model, history, use_tools=True):
    """Chat with Claude; tools optional (token trim when use_tools=False)."""
    meta = PROVIDERS["anthropic"]
    key = provider_key("anthropic")
    if not key:
        return {
            "ok": False,
            "error": "ANTHROPIC_API_KEY missing. Add it to .env.",
            "reply": "",
            "tools": [],
            "provider": "anthropic",
        }

    model = allowed("anthropic", model)
    messages = []
    n_hist = history_turns()
    for turn in (history or [])[-n_hist:]:
        role = "user" if turn.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": turn.get("content", "")})
    messages.append({"role": "user", "content": message})

    url = meta["base"].rstrip("/") + "/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    tool_trace = []
    reply = ""
    system = build_system(use_tools=use_tools)
    rounds = 6 if use_tools else 1
    speed = _Speed()

    for _ in range(rounds):
        body = {
            "model": model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
        }
        if use_tools:
            body["tools"] = anthropic_tools()
        try:
            data = _http_json(url, body, headers, timeout=90)
            speed.add(data)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")[:800]
            return {
                "ok": False,
                "error": "Anthropic HTTP %s: %s" % (e.code, err),
                "reply": "",
                "tools": tool_trace,
                "model": model,
                "provider": "anthropic",
            }
        except Exception as e:  # noqa: BLE001
            return {
                "ok": False,
                "error": str(e),
                "reply": "",
                "tools": tool_trace,
                "model": model,
                "provider": "anthropic",
            }

        content_blocks = data.get("content") or []
        tool_uses = [b for b in content_blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        texts = [
            b.get("text", "")
            for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ]

        if (not use_tools) or (not tool_uses):
            reply = "\n".join(texts).strip() or "(No text)"
            break

        messages.append({"role": "assistant", "content": content_blocks})
        result_blocks = []
        for tu in tool_uses:
            name = tu.get("name") or ""
            args = tu.get("input") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = dispatch(name, args)
            tool_trace.append({"name": name, "args": args, "result": result})
            result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.get("id") or name,
                    "content": json.dumps(result),
                }
            )
        messages.append({"role": "user", "content": result_blocks})
    else:
        reply = reply or "Stopped after too many tool steps."

    # Move any <think> blocks out of the visible reply (v0.45).
    reply, thinking = _split_think(reply)
    return {
        "ok": True,
        "reply": reply or "(No text)",
        "thinking": thinking,
        "tools": tool_trace,
        "model": model,
        "provider": "anthropic",
        "stats": speed.stats(),
    }


# ---------------------------------------------------------------------------
# Dispatch by provider kind
# ---------------------------------------------------------------------------


def run_chat(message, provider=None, model=None, history=None, use_tools=None, think=None):
    """Pick the right chat function from PROVIDERS[id].kind (online + local).

    think: None = untouched; True/False = Think toggle (OpenAI-compat path only).

    use_tools: None → resolve from TOOLS_DEFAULT / request / keywords;
    True/False → explicit. When False, no tool schemas are sent.
    """
    provider = (provider or default_provider() or DEFAULT_PROVIDER).lower().strip()
    if provider not in PROVIDERS:
        return {
            "ok": False,
            "error": "Unknown provider %r. Use one of: %s" % (provider, ", ".join(PROVIDERS)),
            "reply": "",
            "tools": [],
        }
    if use_tools is None:
        want_tools = resolve_use_tools(message, None)
    else:
        # Explicit flag still OR'd with default/keywords via resolve when True-ish;
        # False means only default+keywords can re-enable.
        want_tools = resolve_use_tools(message, bool(use_tools))
    meta = PROVIDERS[provider]
    kind = meta["kind"]
    if kind == "gemini":
        return gemini_chat(message, model, history, use_tools=want_tools)
    if kind == "openai":
        return openai_compat_chat(provider, message, model, history, use_tools=want_tools, think=think)
    if kind == "anthropic":
        return anthropic_chat(message, model, history, use_tools=want_tools)
    return {"ok": False, "error": "Unsupported provider kind", "reply": "", "tools": []}
