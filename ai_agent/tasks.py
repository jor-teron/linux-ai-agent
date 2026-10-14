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
import urllib.parse
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


def _run_task(
    task="chat",
    message="",
    provider=None,
    model=None,
    history=None,
    use_tools=None,
    image=None,
    image_name="",
    think=None,
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
            think=think,  # Think toggle (v0.45); None = untouched
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
        if kind == "pollinations":
            return _pollinations_image(prompt, model, meta)
        if kind == "segmind":
            return _segmind_image(prompt, model, key, meta)
        if provider == "openai" or (kind == "openai" and provider == "openai"):
            return _openai_image(prompt, model, key)
        # FreeTheAi + other OpenAI-compat clouds: /images/generations.
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
        "Use Online (Gemini / OpenAI / Pollinations / Segmind) or an Ollama image model if installed."
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
    # FreeTheAi catalog id gpt-image-2 → live alias eve/gpt-image-2.
    call_model = model
    if provider == "freetheai":
        mid = (model or "").strip().lower()
        if mid in ("gpt-image-2", "gpt-image-2-low", "gpt-image-2-medium"):
            call_model = "eve/" + mid
        elif mid and not mid.startswith(("eve/", "ever/")):
            call_model = "eve/gpt-image-2"
    url = root + "/images/generations"
    body = {"model": call_model, "prompt": prompt, "n": 1, "response_format": "b64_json"}
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % (key or "local"),
    }
    try:
        data = _http_json(url, body, headers, timeout=180)
    except Exception as e:  # noqa: BLE001
        # Retry without response_format (some hosts reject it).
        body.pop("response_format", None)
        try:
            data = _http_json(url, body, headers, timeout=180)
        except Exception as e2:  # noqa: BLE001
            return _unsupported(
                "image",
                provider,
                "Image generation not supported by %s (%s)." % (meta.get("label") or provider, e2),
            )
    items = data.get("data") or []
    if not items:
        return _unsupported("image", provider, "No image data from provider.")
    item = items[0]
    raw = None
    if item.get("b64_json"):
        raw = base64.b64decode(item["b64_json"])
    elif item.get("url"):
        try:
            raw = _fetch_url_bytes(item["url"])
        except Exception as e:  # noqa: BLE001
            return _unsupported("image", provider, "Failed to download image URL: %s" % e)
    if not raw:
        return _unsupported("image", provider, "No image data from provider.")
    saved = save_generated_bytes(raw, stem="image", mime="image/png")
    if not saved.get("ok"):
        return {**saved, "task": "image", "tools": [], "reply": ""}
    reply = "Generated image: %s\n[[download:%s]]" % (saved["name"], saved["name"])
    return {
        "ok": True,
        "reply": reply,
        "tools": [],
        "task": "image",
        "model": call_model,
        "provider": provider,
        "media": saved,
        "media_list": [saved],
    }


# Segmind catalog id → API slug (https://api.segmind.com/v1/<slug>).
_SEGMIND_SLUGS = {
    "sdxl": "sdxl1.0-txt2img",
    "flux": "flux-schnell",
    "realistic-vision": "sd1.5-realisticvision",
}


def _pollinations_image(prompt, model, meta):
    """Keyless GET image.pollinations.ai/prompt/<quote>?model=…"""
    mid = (model or "flux").strip() or "flux"
    if mid.lower() in ("pollinations", "default"):
        mid = "flux"
    base = (meta.get("base") or "https://image.pollinations.ai").rstrip("/")
    # Public CDN-style endpoint — not a billed API; still gate lightly via before_online_call
    # only if we go through _http_json. Use direct GET for raw bytes.
    q = urllib.parse.quote(prompt, safe="")
    url = "%s/prompt/%s?model=%s" % (base, q, urllib.parse.quote(mid, safe=""))
    try:
        before_online_call(url=url)
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            ctype = (resp.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip()
    except RateLimited as e:
        return {"ok": False, "error": str(e), "reply": "", "tools": [], "task": "image", "provider": "pollinations"}
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": "Pollinations image failed: %s" % e,
            "reply": "",
            "tools": [],
            "task": "image",
            "provider": "pollinations",
        }
    if not raw or len(raw) < 100:
        return _unsupported("image", "pollinations", "Pollinations returned empty/short payload.")
    mime = ctype if ctype.startswith("image/") else "image/jpeg"
    saved = save_generated_bytes(raw, stem="image", mime=mime)
    if not saved.get("ok"):
        return {**saved, "task": "image", "tools": [], "reply": ""}
    reply = "Generated image: %s\n[[download:%s]]" % (saved["name"], saved["name"])
    return {
        "ok": True,
        "reply": reply,
        "tools": [],
        "task": "image",
        "model": mid,
        "provider": "pollinations",
        "media": saved,
        "media_list": [saved],
    }


def _segmind_image(prompt, model, key, meta):
    """POST api.segmind.com/v1/<slug> with x-api-key; raw image or JSON."""
    if not key:
        return {
            "ok": False,
            "error": "SEGMIND_API_KEY missing",
            "reply": "",
            "tools": [],
            "task": "image",
        }
    mid = (model or "flux").strip().lower()
    slug = _SEGMIND_SLUGS.get(mid, mid)
    base = (meta.get("base") or "https://api.segmind.com/v1").rstrip("/")
    url = "%s/%s" % (base, slug)
    body = {"prompt": prompt}
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
    }
    try:
        before_online_call(url=url)
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
            ctype = (resp.headers.get("Content-Type") or "").lower()
    except RateLimited as e:
        return {"ok": False, "error": str(e), "reply": "", "tools": [], "task": "image", "provider": "segmind"}
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:600]
        return {
            "ok": False,
            "error": "Segmind HTTP %s: %s" % (e.code, err),
            "reply": "",
            "tools": [],
            "task": "image",
            "provider": "segmind",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": "Segmind image failed: %s" % e,
            "reply": "",
            "tools": [],
            "task": "image",
            "provider": "segmind",
        }
    img_bytes = None
    mime = "image/png"
    if "application/json" in ctype or (raw[:1] in (b"{", b"[")):
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            data = None
        if isinstance(data, dict):
            b64 = (
                data.get("image")
                or data.get("b64_json")
                or (data.get("data") or [{}])[0].get("b64_json")
                if isinstance(data.get("data"), list) and data.get("data")
                else None
            )
            # Some Segmind JSON wraps under output / images.
            if not b64 and isinstance(data.get("output"), str):
                b64 = data["output"]
            if not b64 and isinstance(data.get("images"), list) and data["images"]:
                first = data["images"][0]
                b64 = first if isinstance(first, str) else (first or {}).get("image")
            if b64:
                # Strip data-URL prefix if present.
                if isinstance(b64, str) and "," in b64 and b64.strip().startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                try:
                    img_bytes = base64.b64decode(b64)
                except Exception:  # noqa: BLE001
                    img_bytes = None
            elif data.get("url") or data.get("image_url"):
                try:
                    img_bytes = _fetch_url_bytes(data.get("url") or data.get("image_url"))
                except Exception as e:  # noqa: BLE001
                    return {
                        "ok": False,
                        "error": "Segmind URL download failed: %s" % e,
                        "reply": "",
                        "tools": [],
                        "task": "image",
                    }
    else:
        img_bytes = raw
        if "jpeg" in ctype or "jpg" in ctype:
            mime = "image/jpeg"
        elif "webp" in ctype:
            mime = "image/webp"
    if not img_bytes or len(img_bytes) < 100:
        return _unsupported("image", "segmind", "Segmind returned no image bytes.")
    saved = save_generated_bytes(img_bytes, stem="image", mime=mime)
    if not saved.get("ok"):
        return {**saved, "task": "image", "tools": [], "reply": ""}
    reply = "Generated image: %s\n[[download:%s]]" % (saved["name"], saved["name"])
    return {
        "ok": True,
        "reply": reply,
        "tools": [],
        "task": "image",
        "model": mid,
        "provider": "segmind",
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
            "Switch Mode to Online and try Gemini (Veo) or Agnes if your key has access.",
        )
    if kind == "gemini":
        return _gemini_video(prompt, model, key)
    if kind == "agnes":
        return _agnes_video(prompt, model, key, meta)
    return _unsupported(
        "video",
        provider,
        "Video generation is not supported for %s in this agent. "
        "Try Gemini Online (Veo) or Agnes when available on your API key."
        % (meta.get("label") or provider),
    )


def _agnes_video(prompt, model, key, meta):
    """Agnes AI: POST /v1/videos then poll GET /agnesapi?video_id=… (best-effort)."""
    if not key:
        return {
            "ok": False,
            "error": "AGNES_API_KEY missing",
            "reply": "",
            "tools": [],
            "task": "video",
        }
    base = (meta.get("base") or "https://apihub.agnes-ai.com/v1").rstrip("/")
    # Hub root without /v1 for /agnesapi poll.
    hub = base
    if hub.endswith("/v1"):
        hub = hub[:-3].rstrip("/") or "https://apihub.agnes-ai.com"
    mid = (model or "agnes-video-v2.0").strip()
    create_url = base + "/videos"
    body = {
        "model": mid,
        "mode": "text",
        "prompt": prompt,
        "size": "720P",
        "aspect_ratio": "16:9",
        "seconds": "5",
        "n": 1,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % key,
    }
    try:
        data = _http_json(create_url, body, headers, timeout=60)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:600]
        return _unsupported(
            "video",
            "agnes",
            "Agnes video create failed (HTTP %s: %s)." % (e.code, err[:200]),
        )
    except Exception as e:  # noqa: BLE001
        return _unsupported("video", "agnes", "Agnes video create failed: %s" % e)

    video_id = data.get("video_id") or data.get("id") or data.get("task_id")
    if not video_id:
        return _unsupported("video", "agnes", "Agnes returned no video_id.")

    # Poll until completed/failed (cap ~3 minutes).
    poll_q = urllib.parse.urlencode({"video_id": video_id, "model_name": mid})
    poll_url = "%s/agnesapi?%s" % (hub, poll_q)
    deadline = time.time() + 180
    last = data
    while time.time() < deadline:
        status = (last.get("status") or "").lower()
        if status in ("completed", "succeeded", "success", "done"):
            break
        if status in ("failed", "error", "cancelled", "canceled"):
            return {
                "ok": False,
                "error": "Agnes video failed: %s" % (last.get("error") or last.get("message") or status),
                "reply": "",
                "tools": [],
                "task": "video",
                "model": mid,
                "provider": "agnes",
            }
        time.sleep(2)
        try:
            before_online_call(url=poll_url)
            req = urllib.request.Request(
                poll_url,
                headers={"Authorization": "Bearer %s" % key},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                last = json.loads(resp.read().decode("utf-8"))
        except RateLimited as e:
            return {
                "ok": False,
                "error": str(e),
                "reply": "",
                "tools": [],
                "task": "video",
                "model": mid,
                "provider": "agnes",
            }
        except Exception:  # noqa: BLE001
            continue
    status = (last.get("status") or "").lower()
    if status not in ("completed", "succeeded", "success", "done"):
        return {
            "ok": False,
            "error": "Agnes video timed out (still running). Try again later.",
            "reply": "",
            "tools": [],
            "task": "video",
            "model": mid,
            "provider": "agnes",
        }
    # Prefer metadata.url / url / download_url
    meta_blob = last.get("metadata") if isinstance(last.get("metadata"), dict) else {}
    vurl = (
        meta_blob.get("url")
        or last.get("url")
        or last.get("download_url")
        or last.get("video_url")
    )
    raw = None
    if vurl:
        try:
            raw = _fetch_url_bytes(vurl, timeout=120)
        except Exception as e:  # noqa: BLE001
            return {
                "ok": False,
                "error": "Agnes video download failed: %s" % e,
                "reply": "",
                "tools": [],
                "task": "video",
                "provider": "agnes",
            }
    if not raw:
        # Nested base64 fallback
        raw = _extract_video_bytes(last)
    if not raw:
        return _unsupported("video", "agnes", "Agnes finished but no video bytes/URL were returned.")
    # Save with provider tag (reuse helper but fix provider field after).
    out = _save_video_result(raw, mid)
    out["provider"] = "agnes"
    return out


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
    # OpenAI-compat vision (OpenAI, xAI, DeepSeek, OpenRouter, DeepInfra, Zhipu, Groq, …).
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


def run_task(*args, limits=None, **kwargs):
    """Run one task with the UI Limits switch applied (v0.46).

    limits: None = .env limits; False = skip waits/caps for every online
    call in this task (chat and media), still counted; True = .env limits.
    Same arguments and result as _run_task.
    """
    from ai_agent.ratelimit import limits_override

    with limits_override(limits):
        return _run_task(*args, **kwargs)
