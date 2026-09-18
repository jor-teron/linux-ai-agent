"""
Task runners: chat | image | video | vision (Online + Local).

Chat delegates to brain.run_chat. Image/video generate media via provider
APIs when supported; vision sends an image + prompt to a multimodal model.
Unsupported combinations return a friendly error (Local is never grayed out
in the UI — we just explain when the runtime cannot do the task).

Online provider HTTP uses brain._http_json, which applies the rate limiter.
Local calls pass limit=False. This module does not talk to Telegram.

Imports from: ai_agent.config, providers, brain, media, tools.
Used by: server (/api/chat), telegram (optional task routing).
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from ai_agent.config import (
    DEFAULT_PROVIDER,
    GEMINI_API_BASE,
    PROVIDERS,
    allowed,
    default_model,
    default_provider,
    provider_key,
    provider_openai_base,
    get_provider_meta,
)
from ai_agent.media import detect_mime, save_generated_bytes
from ai_agent.brain import _http_json, run_chat
from ai_agent.ratelimit import RateLimited, before_online_call

# Canonical task ids (UI order: Chat | Image | Video | Vision).
TASKS = ("chat", "image", "video", "vision")


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def run_task(
    task="chat",
    message="",
    provider=None,
    model=None,
    history=None,
    use_tools=None,
    image=None,
    image_name="",
):
    """Run one task. Returns the same shape as brain.run_chat plus media fields.

    Extra keys when media is produced:
      media: {name, path, mime, bytes, download, rel}
      media_list: [...]
    """
    task = (task or "chat").strip().lower()
    if task not in TASKS:
        return {
            "ok": False,
            "error": "Unknown task %r. Use: %s" % (task, ", ".join(TASKS)),
            "reply": "",
            "tools": [],
            "task": task,
        }
    provider = (provider or default_provider() or DEFAULT_PROVIDER).lower().strip()
    if provider not in PROVIDERS:
        return {
            "ok": False,
            "error": "Unknown provider %r" % provider,
            "reply": "",
            "tools": [],
            "task": task,
        }
    model = allowed(provider, model or default_model(provider))
    if task == "chat":
        out = run_chat(
            message,
            provider=provider,
            model=model,
            history=history,
            use_tools=use_tools,
        )
        out["task"] = "chat"
        return out
    if task == "image":
        return _task_image(message, provider, model)
    if task == "video":
        return _task_video(message, provider, model)
    if task == "vision":
        return _task_vision(message, provider, model, history, image=image, image_name=image_name)
    return {"ok": False, "error": "Unhandled task", "reply": "", "tools": [], "task": task}


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


def _task_image(prompt, provider, model):
    """Generate an image from a text prompt; save under workspace/generated/."""
    prompt = (prompt or "").strip()
    if not prompt:
        return {"ok": False, "error": "prompt required", "reply": "", "tools": [], "task": "image"}
    meta = get_provider_meta(provider) or PROVIDERS[provider]
    mode = meta.get("mode") or "online"
    kind = meta.get("kind")
    key = provider_key(provider)

    if mode == "online":
        if kind == "gemini":
            return _gemini_image(prompt, model, key)
        if provider == "openai" or (kind == "openai" and provider == "openai"):
            return _openai_image(prompt, model, key)
        # Other OpenAI-compat clouds: best-effort images endpoint, else clear msg.
        if kind == "openai":
            return _openai_compat_image(provider, prompt, model, key, meta)
        return _unsupported("image", provider, "Online image generation is not wired for this provider.")

    # Local: try Ollama image-ish models; else clear error.
    if provider == "ollama":
        return _ollama_image(prompt, model, meta)
    return _unsupported(
        "image",
        provider,
        "Local image generation is not available for %s. "
        "Use Online (Gemini / OpenAI) or an Ollama image model if installed."
        % (meta.get("label") or provider),
    )


def _gemini_image(prompt, model, key):
    """Gemini native image via generateContent + responseModalities IMAGE."""
    if not key:
        return {"ok": False, "error": "GEMINI_API_KEY missing", "reply": "", "tools": [], "task": "image"}
    # Prefer an image-capable model id when the UI still has a chat default.
    img_model = _prefer_gemini_image_model(model)
    url = "%s/models/%s:generateContent?key=%s" % (GEMINI_API_BASE, img_model, key)
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            # Request image (and optional text caption) back.
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }
    try:
        data = _http_json(url, body, {"Content-Type": "application/json"}, timeout=120)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:800]
        # Fallback: try imagen-style predict if generateContent rejects.
        if e.code in (400, 404):
            alt = _gemini_imagen_predict(prompt, key)
            if alt.get("ok"):
                return alt
        return {
            "ok": False,
            "error": "Gemini image HTTP %s: %s" % (e.code, err),
            "reply": "",
            "tools": [],
            "task": "image",
            "model": img_model,
            "provider": "gemini",
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "reply": "", "tools": [], "task": "image"}

    texts, images = _parse_gemini_parts(data)
    if not images:
        msg = "\n".join(texts).strip() or "Gemini returned no image."
        return {
            "ok": False,
            "error": msg,
            "reply": msg,
            "tools": [],
            "task": "image",
            "model": img_model,
            "provider": "gemini",
        }
    media_list = []
    for i, (mime, raw) in enumerate(images):
        stem = "image" if i == 0 else ("image_%d" % (i + 1))
        saved = save_generated_bytes(raw, stem=stem, mime=mime)
        if saved.get("ok"):
            media_list.append(saved)
    if not media_list:
        return {"ok": False, "error": "failed to save image", "reply": "", "tools": [], "task": "image"}
    caption = "\n".join(texts).strip()
    reply = caption or ("Generated image: %s" % media_list[0]["name"])
    reply += "\n[[download:%s]]" % media_list[0]["name"]
    return {
        "ok": True,
        "reply": reply,
        "tools": [],
        "task": "image",
        "model": img_model,
        "provider": "gemini",
        "media": media_list[0],
        "media_list": media_list,
    }


def _prefer_gemini_image_model(model):
    """Swap chat-only Gemini ids for a known image-capable default."""
    m = (model or "").lower()
    if "image" in m or "imagen" in m or "banana" in m:
        return model
    # Curated image model (Nano Banana / Flash Image family).
    return "gemini-2.5-flash-image"


def _gemini_imagen_predict(prompt, key):
    """Best-effort Imagen predict endpoint (may be unavailable on free tier)."""
    model = "imagen-3.0-generate-002"
    url = "%s/models/%s:predict?key=%s" % (GEMINI_API_BASE, model, key)
    body = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1},
    }
    try:
        data = _http_json(url, body, {"Content-Type": "application/json"}, timeout=120)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": "Imagen fallback failed: %s" % e, "reply": "", "tools": [], "task": "image"}
    preds = data.get("predictions") or []
    images = []
    for p in preds:
        b64 = p.get("bytesBase64Encoded") or p.get("image", {}).get("bytesBase64Encoded")
        if b64:
            try:
                images.append(("image/png", base64.b64decode(b64)))
            except Exception:  # noqa: BLE001
                pass
    if not images:
        return {"ok": False, "error": "Imagen returned no image", "reply": "", "tools": [], "task": "image"}
    saved = save_generated_bytes(images[0][1], stem="image", mime=images[0][0])
    if not saved.get("ok"):
        return saved
    reply = "Generated image: %s\n[[download:%s]]" % (saved["name"], saved["name"])
    return {
        "ok": True,
        "reply": reply,
        "tools": [],
        "task": "image",
        "model": model,
        "provider": "gemini",
        "media": saved,
        "media_list": [saved],
    }


def _parse_gemini_parts(data):
    """Extract (texts, [(mime, bytes), ...]) from a generateContent response."""
    texts = []
    images = []
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    for p in parts:
        if not isinstance(p, dict):
            continue
        if "text" in p and p.get("text"):
            texts.append(p["text"])
        inline = p.get("inlineData") or p.get("inline_data")
        if isinstance(inline, dict) and inline.get("data"):
            mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            try:
                raw = base64.b64decode(inline["data"])
                images.append((mime, raw))
            except Exception:  # noqa: BLE001
                pass
    return texts, images


def _openai_image(prompt, model, key):
    """OpenAI Images API (generations)."""
    if not key:
        return {"ok": False, "error": "OPENAI_API_KEY missing", "reply": "", "tools": [], "task": "image"}
    # Map chat model names to an images model when needed.
    img_model = model if model in ("dall-e-2", "dall-e-3", "gpt-image-1") else "dall-e-3"
    url = "https://api.openai.com/v1/images/generations"
    body = {"model": img_model, "prompt": prompt, "n": 1, "size": "1024x1024"}
    # gpt-image-1 returns b64 by default on some tiers; request b64 for all.
    body["response_format"] = "b64_json"
    try:
        data = _http_json(
            url,
            body,
            {"Content-Type": "application/json", "Authorization": "Bearer %s" % key},
            timeout=120,
        )
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:800]
        # Retry without response_format for gpt-image-1 quirks.
        if "response_format" in err.lower():
            body.pop("response_format", None)
            try:
                data = _http_json(
                    url,
                    body,
                    {"Content-Type": "application/json", "Authorization": "Bearer %s" % key},
                    timeout=120,
                )
            except Exception as e2:  # noqa: BLE001
                return {"ok": False, "error": "OpenAI image: %s" % e2, "reply": "", "tools": [], "task": "image"}
        else:
            return {
                "ok": False,
                "error": "OpenAI image HTTP %s: %s" % (e.code, err),
                "reply": "",
                "tools": [],
                "task": "image",
            }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "reply": "", "tools": [], "task": "image"}

    items = data.get("data") or []
    if not items:
        return {"ok": False, "error": "OpenAI returned no image", "reply": "", "tools": [], "task": "image"}
    item = items[0]
    raw = None
    if item.get("b64_json"):
        raw = base64.b64decode(item["b64_json"])
    elif item.get("url"):
        raw = _fetch_url_bytes(item["url"])
    if not raw:
        return {"ok": False, "error": "OpenAI image payload empty", "reply": "", "tools": [], "task": "image"}
    saved = save_generated_bytes(raw, stem="image", mime="image/png")
    if not saved.get("ok"):
        return {**saved, "task": "image", "tools": [], "reply": ""}
    reply = "Generated image: %s\n[[download:%s]]" % (saved["name"], saved["name"])
    return {
        "ok": True,
        "reply": reply,
        "tools": [],
        "task": "image",
        "model": img_model,
        "provider": "openai",
        "media": saved,
        "media_list": [saved],
    }


def _openai_compat_image(provider, prompt, model, key, meta):
    """Best-effort /images/generations on OpenAI-compat bases; else unsupported."""
    base = (meta.get("base") or provider_openai_base(provider) or "").rstrip("/")
    # Strip trailing /v1 if present then re-add — images live next to chat.
    if base.endswith("/v1"):
        root = base
    else:
        root = base + "/v1" if base else ""
    if not root:
        return _unsupported("image", provider, "No images endpoint for this provider.")
    if meta.get("needs_key", True) and not key:
        return {
            "ok": False,
            "error": "%s missing" % (meta.get("env_key") or "API key"),
            "reply": "",
            "tools": [],
            "task": "image",
        }
    url = root + "/images/generations"
    body = {"model": model, "prompt": prompt, "n": 1, "response_format": "b64_json"}
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % (key or "local"),
    }
    try:
        data = _http_json(url, body, headers, timeout=120)
    except Exception as e:  # noqa: BLE001
        return _unsupported(
            "image",
            provider,
            "Image generation not supported by %s (%s)." % (meta.get("label") or provider, e),
        )
    items = data.get("data") or []
    if not items or not items[0].get("b64_json"):
        return _unsupported("image", provider, "No image data from provider.")
    raw = base64.b64decode(items[0]["b64_json"])
    saved = save_generated_bytes(raw, stem="image", mime="image/png")
    if not saved.get("ok"):
        return {**saved, "task": "image", "tools": [], "reply": ""}
    reply = "Generated image: %s\n[[download:%s]]" % (saved["name"], saved["name"])
    return {
        "ok": True,
        "reply": reply,
        "tools": [],
        "task": "image",
        "model": model,
        "provider": provider,
        "media": saved,
        "media_list": [saved],
    }


def _ollama_image(prompt, model, meta):
    """Try Ollama /api/generate for image models; else clear unsupported."""
    # Heuristic: model names that often imply image gen.
    mid = (model or "").lower()
    image_ish = any(x in mid for x in ("flux", "sdxl", "stable-diffusion", "image", "draw"))
    if not image_ish:
        return _unsupported(
            "image",
            "ollama",
            "Local Ollama image generation needs an image model "
            "(e.g. flux). Chat models cannot generate images here. "
            "Pull an image model or switch Mode to Online (Gemini/OpenAI).",
        )
    root = provider_openai_base("ollama")
    # Ollama native generate (some image models use this).
    from ai_agent.providers import ollama_base_url

    url = ollama_base_url().rstrip("/") + "/api/generate"
    body = {"model": model, "prompt": prompt, "stream": False}
    try:
        # Local daemon — not a billed API (skip TPM/RPM/RPD/delay).
        data = _http_json(url, body, {"Content-Type": "application/json"}, timeout=180, limit=False)
    except Exception as e:  # noqa: BLE001
        return _unsupported("image", "ollama", "Ollama image attempt failed: %s" % e)
    # Some forks return images: [b64...]
    imgs = data.get("images") or []
    if imgs:
        raw = base64.b64decode(imgs[0])
        saved = save_generated_bytes(raw, stem="image", mime="image/png")
        if saved.get("ok"):
            reply = "Generated image: %s\n[[download:%s]]" % (saved["name"], saved["name"])
            return {
                "ok": True,
                "reply": reply,
                "tools": [],
                "task": "image",
                "model": model,
                "provider": "ollama",
                "media": saved,
                "media_list": [saved],
            }
    return _unsupported(
        "image",
        "ollama",
        "Ollama model %r did not return image bytes. "
        "Use Online Gemini/OpenAI, or an Ollama build that supports image output."
        % model,
    )


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------


def _task_video(prompt, provider, model):
    """Generate a short video when the provider supports it; else friendly msg."""
    prompt = (prompt or "").strip()
    if not prompt:
        return {"ok": False, "error": "prompt required", "reply": "", "tools": [], "task": "video"}
    meta = get_provider_meta(provider) or PROVIDERS[provider]
    mode = meta.get("mode") or "online"
    kind = meta.get("kind")
    key = provider_key(provider)

    if mode == "local":
        return _unsupported(
            "video",
            provider,
            "Local video generation is not available yet. "
            "Switch Mode to Online and try Gemini (Veo) if your key has access.",
        )
    if kind == "gemini":
        return _gemini_video(prompt, model, key)
    return _unsupported(
        "video",
        provider,
        "Video generation is not supported for %s in this agent. "
        "Try Gemini Online (Veo) when available on your API key."
        % (meta.get("label") or provider),
    )


def _gemini_video(prompt, model, key):
    """Pragmatic Veo-style predictLongRunning + poll (when the API allows)."""
    if not key:
        return {"ok": False, "error": "GEMINI_API_KEY missing", "reply": "", "tools": [], "task": "video"}
    veo = model if "veo" in (model or "").lower() else "veo-2.0-generate-001"
    # Long-running predict (Vertex-style on AI Studio may differ — keep pragmatic).
    url = "%s/models/%s:predictLongRunning?key=%s" % (GEMINI_API_BASE, veo, key)
    body = {
        "instances": [{"prompt": prompt}],
        "parameters": {"aspectRatio": "16:9", "sampleCount": 1},
    }
    try:
        data = _http_json(url, body, {"Content-Type": "application/json"}, timeout=60)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:600]
        return _unsupported(
            "video",
            "gemini",
            "Gemini video (Veo) not available for this key/model (%s: %s). "
            "Video generation stays optional — try again when Veo is enabled."
            % (e.code, err[:200]),
        )
    except Exception as e:  # noqa: BLE001
        return _unsupported("video", "gemini", "Gemini video failed: %s" % e)

    op_name = data.get("name") or (data.get("operation") or {}).get("name")
    if not op_name:
        # Some responses may inline a video — try to extract.
        extracted = _extract_video_bytes(data)
        if extracted:
            return _save_video_result(extracted, veo)
        return _unsupported("video", "gemini", "Gemini video returned no operation handle.")

    # Poll operation (cap ~2 minutes).
    op_url = "%s/%s?key=%s" % (GEMINI_API_BASE.rstrip("/").rsplit("/v1beta", 1)[0], op_name, key)
    # Prefer absolute if name is already a full path under v1beta.
    if op_name.startswith("operations/") or op_name.startswith("models/"):
        op_url = "%s/%s?key=%s" % (GEMINI_API_BASE, op_name, key)
    deadline = time.time() + 120
    last = data
    while time.time() < deadline:
        if last.get("done"):
            break
        time.sleep(4)
        try:
            # Poll counts as an online Gemini call (same caps, no retry storm).
            before_online_call(url=op_url)
            req = urllib.request.Request(op_url, method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:
                last = json.loads(resp.read().decode("utf-8"))
        except RateLimited as e:
            return {
                "ok": False,
                "error": str(e),
                "reply": "",
                "tools": [],
                "task": "video",
                "model": veo,
                "provider": "gemini",
            }
        except Exception:  # noqa: BLE001
            continue
    if not last.get("done"):
        return {
            "ok": False,
            "error": "Video generation timed out (still running on Google). Try again later.",
            "reply": "",
            "tools": [],
            "task": "video",
            "model": veo,
            "provider": "gemini",
        }
    if last.get("error"):
        return {
            "ok": False,
            "error": "Video op error: %s" % last.get("error"),
            "reply": "",
            "tools": [],
            "task": "video",
        }
    raw = _extract_video_bytes(last.get("response") or last)
    if not raw:
        return _unsupported("video", "gemini", "Video finished but no bytes were returned.")
    return _save_video_result(raw, veo)


def _extract_video_bytes(blob):
    """Best-effort pull of mp4/webm bytes from nested Gemini/Veo JSON."""
    if not isinstance(blob, dict):
        return None
    # Common shapes: generateVideoResponse.generatedSamples[].video.bytesBase64Encoded
    for key in ("generateVideoResponse", "response", "predictions"):
        node = blob.get(key) if key != "predictions" else blob
        if key == "predictions" and isinstance(blob.get("predictions"), list):
            for p in blob["predictions"]:
                b64 = _find_b64_video(p)
                if b64:
                    try:
                        return base64.b64decode(b64)
                    except Exception:  # noqa: BLE001
                        pass
        if isinstance(node, dict):
            b64 = _find_b64_video(node)
            if b64:
                try:
                    return base64.b64decode(b64)
                except Exception:  # noqa: BLE001
                    pass
    b64 = _find_b64_video(blob)
    if b64:
        try:
            return base64.b64decode(b64)
        except Exception:  # noqa: BLE001
            return None
    return None


def _find_b64_video(node, depth=0):
    """Walk a small JSON tree for a base64 video field."""
    if depth > 6 or not isinstance(node, dict):
        return None
    for k, v in node.items():
        lk = k.lower()
        if isinstance(v, str) and v and ("video" in lk or "bytes" in lk or lk.endswith("b64")):
            if len(v) > 200:  # likely base64 payload
                return v
        if isinstance(v, dict):
            found = _find_b64_video(v, depth + 1)
            if found:
                return found
        if isinstance(v, list):
            for item in v[:5]:
                if isinstance(item, dict):
                    found = _find_b64_video(item, depth + 1)
                    if found:
                        return found
    return None


def _save_video_result(raw, model):
    saved = save_generated_bytes(raw, stem="video", ext="mp4", mime="video/mp4")
    if not saved.get("ok"):
        return {**saved, "task": "video", "tools": [], "reply": ""}
    reply = "Generated video: %s\n[[download:%s]]" % (saved["name"], saved["name"])
    return {
        "ok": True,
        "reply": reply,
        "tools": [],
        "task": "video",
        "model": model,
        "provider": "gemini",
        "media": saved,
        "media_list": [saved],
    }


# ---------------------------------------------------------------------------
# Vision (image + prompt → text)
# ---------------------------------------------------------------------------


def _task_vision(prompt, provider, model, history=None, image=None, image_name=""):
    """Describe / answer about a user image with a vision-capable model."""
    from ai_agent.media import load_image_bytes

    prompt = (prompt or "").strip() or "Describe this image."
    raw, mime, err = load_image_bytes(image=image, image_name=image_name)
    if err or not raw:
        return {
            "ok": False,
            "error": err or "image required for Vision task (upload or Telegram photo)",
            "reply": "",
            "tools": [],
            "task": "vision",
        }
    # Cap vision upload size (~4MB) to keep requests sane.
    if len(raw) > 4_000_000:
        return {
            "ok": False,
            "error": "Image too large for vision (max ~4MB)",
            "reply": "",
            "tools": [],
            "task": "vision",
        }
    meta = get_provider_meta(provider) or PROVIDERS[provider]
    kind = meta.get("kind")
    key = provider_key(provider)
    mime = mime or detect_mime(data=raw) or "image/jpeg"

    if kind == "gemini":
        return _gemini_vision(prompt, model, key, raw, mime, history)
    if kind == "openai":
        return _openai_vision(provider, prompt, model, key, raw, mime, history, meta)
    if kind == "anthropic":
        return _anthropic_vision(prompt, model, key, raw, mime, history)
    return _unsupported("vision", provider, "Vision is not wired for this provider.")


def _gemini_vision(prompt, model, key, raw, mime, history):
    if not key:
        return {"ok": False, "error": "GEMINI_API_KEY missing", "reply": "", "tools": [], "task": "vision"}
    model = allowed("gemini", model)
    b64 = base64.b64encode(raw).decode("ascii")
    contents = []
    # Light history (text only) — keep short.
    for turn in (history or [])[-4:]:
        role = "user" if turn.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [{"text": turn.get("content", "")}]})
    contents.append(
        {
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": mime, "data": b64}},
                {"text": prompt},
            ],
        }
    )
    url = "%s/models/%s:generateContent?key=%s" % (GEMINI_API_BASE, model, key)
    try:
        data = _http_json(
            url,
            {"contents": contents, "generationConfig": {"maxOutputTokens": 2048}},
            {"Content-Type": "application/json"},
            timeout=90,
        )
    except urllib.error.HTTPError as e:
        # Retry with camelCase inlineData (some API revisions).
        contents[-1]["parts"][0] = {"inlineData": {"mimeType": mime, "data": b64}}
        try:
            data = _http_json(
                url,
                {"contents": contents, "generationConfig": {"maxOutputTokens": 2048}},
                {"Content-Type": "application/json"},
                timeout=90,
            )
        except Exception as e2:  # noqa: BLE001
            err = ""
            try:
                err = e.read().decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001
                err = str(e2)
            return {
                "ok": False,
                "error": "Gemini vision HTTP %s: %s" % (e.code, err),
                "reply": "",
                "tools": [],
                "task": "vision",
            }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "reply": "", "tools": [], "task": "vision"}
    texts, _ = _parse_gemini_parts(data)
    reply = "\n".join(texts).strip() or "(No text)"
    return {
        "ok": True,
        "reply": reply,
        "tools": [],
        "task": "vision",
        "model": model,
        "provider": "gemini",
    }


def _openai_vision(provider, prompt, model, key, raw, mime, history, meta):
    needs_key = bool(meta.get("needs_key", True))
    if needs_key and not key:
        return {
            "ok": False,
            "error": "%s missing" % (meta.get("env_key") or "API key"),
            "reply": "",
            "tools": [],
            "task": "vision",
        }
    model = allowed(provider, model)
    b64 = base64.b64encode(raw).decode("ascii")
    data_url = "data:%s;base64,%s" % (mime, b64)
    messages = []
    for turn in (history or [])[-4:]:
        role = "user" if turn.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": turn.get("content", "")})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    )
    base = meta.get("base") or provider_openai_base(provider)
    url = base.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % (key or "local"),
    }
    # Local vision (ollama/llamacpp) is not billed; online vision is.
    api_limit = (meta.get("mode") or "online") != "local"
    try:
        data = _http_json(
            url,
            {"model": model, "messages": messages, "max_tokens": 2048},
            headers,
            timeout=90,
            limit=api_limit,
        )
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": "Vision failed for %s: %s" % (provider, e),
            "reply": "",
            "tools": [],
            "task": "vision",
        }
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    reply = (msg.get("content") or "").strip() or "(No text)"
    return {
        "ok": True,
        "reply": reply,
        "tools": [],
        "task": "vision",
        "model": model,
        "provider": provider,
    }


def _anthropic_vision(prompt, model, key, raw, mime, history):
    if not key:
        return {"ok": False, "error": "ANTHROPIC_API_KEY missing", "reply": "", "tools": [], "task": "vision"}
    model = allowed("anthropic", model)
    # Anthropic wants media types image/jpeg|png|gif|webp.
    if mime not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
        mime = "image/jpeg"
    b64 = base64.b64encode(raw).decode("ascii")
    messages = []
    for turn in (history or [])[-4:]:
        role = "user" if turn.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": turn.get("content", "")})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                },
                {"type": "text", "text": prompt},
            ],
        }
    )
    meta = PROVIDERS["anthropic"]
    url = meta["base"].rstrip("/") + "/messages"
    try:
        data = _http_json(
            url,
            {"model": model, "max_tokens": 2048, "messages": messages},
            {
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            timeout=90,
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": "Anthropic vision: %s" % e, "reply": "", "tools": [], "task": "vision"}
    texts = [
        b.get("text", "")
        for b in (data.get("content") or [])
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    reply = "\n".join(texts).strip() or "(No text)"
    return {
        "ok": True,
        "reply": reply,
        "tools": [],
        "task": "vision",
        "model": model,
        "provider": "anthropic",
    }


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def _unsupported(task, provider, detail):
    """Friendly unsupported result (ok=False, clear reply text)."""
    msg = detail or ("%s task is not supported for %s." % (task, provider))
    return {
        "ok": False,
        "error": msg,
        "reply": msg,
        "tools": [],
        "task": task,
        "provider": provider,
        "unsupported": True,
    }


def _fetch_url_bytes(url, timeout=60):
    """GET raw bytes from a short-lived image URL (OpenAI images).

    Not gated: the generations POST already counted against the limiter.
    This is a CDN download of the result, not another billed API call.
    """
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()
