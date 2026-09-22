"""
Provider catalogs and readiness helpers (online cloud + local runtimes).

Owns the Provider/Model lists used by the UI and brain dispatch. Online
providers need API keys; local providers (Ollama, llama.cpp) talk to a
local OpenAI-compatible HTTP server and need no key.

Mode: online | local (UI may show Online / Local).

Imports from: ai_agent.config lazily (load_env) to avoid circular imports.
Used by: config (re-exports), brain, server, ui (via API).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

# Canonical mode ids (UI labels are capitalized Online / Local).
MODES: List[str] = ["online", "local"]

# Default mode when .env PROVIDER points at an online catalog entry.
DEFAULT_MODE = "online"


# ---------------------------------------------------------------------------
# Online provider catalog (cloud APIs — same behavior as pre-0.1.29)
# ---------------------------------------------------------------------------

# Gemini REST root. The API key is passed as a query parameter.
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Each online provider: display label, env var for the key, API "kind"
# (gemini / openai / anthropic), models list (string ids), and a default.
# kind "openai" means Chat Completions (OpenAI, xAI, DeepSeek, OpenRouter,
# DeepInfra). Local runtimes also use kind "openai".
ONLINE_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "gemini": {
        "label": "Gemini",
        "mode": "online",
        "needs_key": True,
        "env_key": "GEMINI_API_KEY",
        "kind": "gemini",
        "models": [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.6-flash",
            "gemini-3.6-pro",
            "gemini-3.5-pro",
            "gemini-2.5-pro",
            # Image / video task models (also usable when Task=Chat for multimodal)
            "gemini-2.5-flash-image",
            "gemini-2.0-flash-preview-image-generation",
            "veo-2.0-generate-001",
        ],
        "default": "gemini-3.5-flash-lite",
        # Task capability hints (used by catalog_for_api filter).
        "tasks": ["chat", "image", "video", "vision"],
    },
    "openai": {
        "label": "OpenAI (ChatGPT)",
        "mode": "online",
        "needs_key": True,
        "env_key": "OPENAI_API_KEY",
        "kind": "openai",
        "base": "https://api.openai.com/v1",
        "models": [
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4.1",
            "dall-e-3",
            "gpt-image-1",
        ],
        "default": "gpt-4o-mini",
        "tasks": ["chat", "image", "vision"],
    },
    "xai": {
        "label": "xAI (Grok)",
        "mode": "online",
        "needs_key": True,
        "env_key": "XAI_API_KEY",
        "kind": "openai",
        "base": "https://api.x.ai/v1",
        "models": ["grok-4.3", "grok-3-mini", "grok-4.6", "grok-4.5"],
        "default": "grok-4.3",
        "tasks": ["chat"],
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "mode": "online",
        "needs_key": True,
        "env_key": "ANTHROPIC_API_KEY",
        "kind": "anthropic",
        "base": "https://api.anthropic.com/v1",
        "models": ["claude-haiku-4-5", "claude-sonnet-5", "claude-sonnet-4-6"],
        "default": "claude-haiku-4-5",
        "tasks": ["chat"],
    },
    "deepseek": {
        "label": "DeepSeek",
        "mode": "online",
        "needs_key": True,
        "env_key": "DEEPSEEK_API_KEY",
        "kind": "openai",
        "base": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default": "deepseek-chat",
        "tasks": ["chat"],
    },
    "openrouter": {
        "label": "OpenRouter",
        "mode": "online",
        "needs_key": True,
        "env_key": "OPENROUTER_API_KEY",
        "kind": "openai",
        "base": "https://openrouter.ai/api/v1",
        "models": [
            "openrouter/auto",
            "openai/gpt-4o-mini",
            "google/gemini-2.0-flash-001",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        "default": "openrouter/auto",
        "tasks": ["chat"],
    },
    "deepinfra": {
        "label": "DeepInfra",
        "mode": "online",
        "needs_key": True,
        "env_key": "DEEPINFRA_API_KEY",
        "kind": "openai",
        "base": "https://api.deepinfra.com/v1/openai",
        "models": [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "google/gemma-2-9b-it",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        "default": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "tasks": ["chat"],
    },
    # --- Image specialists (v0.42) ---
    # Segmind: POST https://api.segmind.com/v1/<slug> with x-api-key; body JSON;
    # response is raw image bytes (or JSON with image). Slug map lives in tasks.py.
    "segmind": {
        "label": "Segmind",
        "mode": "online",
        "needs_key": True,
        "env_key": "SEGMIND_API_KEY",
        "kind": "segmind",
        "base": "https://api.segmind.com/v1",
        "models": [
            {"id": "sdxl", "label": "SDXL"},
            {"id": "flux", "label": "FLUX"},
            {"id": "realistic-vision", "label": "Realistic Vision"},
        ],
        "default": "flux",
        "tasks": ["image"],
    },
    # FreeTheAi: OpenAI-compat /images/generations (Bearer FREETHEAI_API_KEY).
    # Catalog id gpt-image-2; tasks.py maps to eve/gpt-image-2 for the live call.
    "freetheai": {
        "label": "FreeTheAi",
        "mode": "online",
        "needs_key": True,
        "env_key": "FREETHEAI_API_KEY",
        "kind": "openai",
        "base": "https://api.freetheai.xyz/v1",
        "models": ["gpt-image-2"],
        "default": "gpt-image-2",
        "tasks": ["image"],
    },
    # Pollinations: keyless GET https://image.pollinations.ai/prompt/<prompt>
    # Unlimited public image endpoint; always ready (needs_key False).
    "pollinations": {
        "label": "Pollinations.ai",
        "mode": "online",
        "needs_key": False,
        "env_key": "",
        "kind": "pollinations",
        "base": "https://image.pollinations.ai",
        "models": [
            {"id": "flux", "label": "FLUX (Pollinations)"},
            {"id": "pollinations", "label": "Pollinations (default)"},
        ],
        "default": "flux",
        "tasks": ["image"],
    },
    # --- Video specialists (v0.42) ---
    # Agnes: POST /v1/videos then poll GET /agnesapi?video_id=… (Bearer AGNES_API_KEY).
    # InferencePort localhost:8080 deferred — do not add in this release.
    "agnes": {
        "label": "Agnes AI",
        "mode": "online",
        "needs_key": True,
        "env_key": "AGNES_API_KEY",
        "kind": "agnes",
        "base": "https://apihub.agnes-ai.com/v1",
        "models": ["agnes-video-v2.0"],
        "default": "agnes-video-v2.0",
        "tasks": ["video"],
    },
    # --- Vision specialists (v0.42) ---
    # Zhipu GLM-4V: OpenAI-compat chat + image_url (ZHIPU_API_KEY).
    "zhipu": {
        "label": "Zhipu (GLM)",
        "mode": "online",
        "needs_key": True,
        "env_key": "ZHIPU_API_KEY",
        "kind": "openai",
        "base": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4v-flash"],
        "default": "glm-4v-flash",
        # Vision-only so Chat list stays clean.
        "tasks": ["vision"],
    },
    # Groq vision preview: OpenAI-compat (GROQ_API_KEY).
    "groq": {
        "label": "Groq",
        "mode": "online",
        "needs_key": True,
        "env_key": "GROQ_API_KEY",
        "kind": "openai",
        "base": "https://api.groq.com/openai/v1",
        "models": ["llama-3.2-11b-vision-preview"],
        "default": "llama-3.2-11b-vision-preview",
        "tasks": ["vision"],
    },
}


# ---------------------------------------------------------------------------
# Local provider catalog (OpenAI-compatible runtimes)
# ---------------------------------------------------------------------------

# Built-in curated lists (used when models_ollama / models_llamacpp missing).
_OLLAMA_MODELS_FALLBACK: List[Dict[str, str]] = [
    {"id": "tinydolphin", "label": "tinydolphin (~1B)"},
    {"id": "tinyllama", "label": "tinyllama (~1B)"},
    {"id": "llama3.2:1b", "label": "llama3.2:1b"},
    {"id": "qwen2.5:0.5b", "label": "qwen2.5:0.5b"},
    {"id": "qwen2.5:1.5b", "label": "qwen2.5:1.5b"},
    {"id": "gemma2:2b", "label": "gemma2:2b"},
    {"id": "phi3:mini", "label": "phi3:mini (~3.8B)"},
    {"id": "llama3.2:3b", "label": "llama3.2:3b"},
    {"id": "mistral:7b", "label": "mistral:7b"},
    {"id": "llama3.1:8b", "label": "llama3.1:8b"},
    {"id": "llama3.1:70b", "label": "llama3.1:70b (needs lots of RAM)"},
]

_LLAMACPP_MODELS_FALLBACK: List[Dict[str, str]] = [
    {"id": "tinydolphin", "label": "TinyDolphin (~1B)"},
    {"id": "tinyllama", "label": "TinyLlama (~1B)"},
    {"id": "llama-3.2-1b", "label": "Llama 3.2 1B"},
    {"id": "qwen2.5-0.5b", "label": "Qwen2.5 0.5B"},
    {"id": "qwen2.5-1.5b", "label": "Qwen2.5 1.5B"},
    {"id": "gemma-2-2b", "label": "Gemma 2 2B"},
    {"id": "phi-3-mini", "label": "Phi-3 Mini (~3.8B)"},
    {"id": "llama-3.2-3b", "label": "Llama 3.2 3B"},
    {"id": "mistral-7b", "label": "Mistral 7B"},
    {"id": "llama-3.1-8b", "label": "Llama 3.1 8B"},
    {"id": "llama-3.1-70b", "label": "Llama 3.1 70B (needs lots of RAM)"},
]


def _load_model_list_file(filename, fallback):
    """Load curated models from a package text file (one id per line).

    Format: `id` or `id|label`. Blank lines and # comments skipped.
    Missing/empty file → fallback list.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent / filename
    out: List[Dict[str, str]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                mid, label = line.split("|", 1)
                mid, label = mid.strip(), label.strip()
            else:
                mid, label = line, line
            if mid:
                out.append({"id": mid, "label": label or mid})
    except OSError:
        out = []
    return out if out else list(fallback)


# Prefer ai_agent/models_ollama and models_llamacpp when present.
_OLLAMA_MODELS: List[Dict[str, str]] = _load_model_list_file(
    "models_ollama", _OLLAMA_MODELS_FALLBACK
)
_LLAMACPP_MODELS: List[Dict[str, str]] = _load_model_list_file(
    "models_llamacpp", _LLAMACPP_MODELS_FALLBACK
)

# Default local OpenAI-compat base URLs (overridable via .env).
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LLAMACPP_BASE_URL = "http://127.0.0.1:8080"

LOCAL_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "ollama": {
        "label": "Ollama",
        "mode": "local",
        "needs_key": False,
        "env_key": "",  # no API key
        "kind": "openai",  # OpenAI-compatible /v1/chat/completions
        "base_env": "OLLAMA_BASE_URL",
        "base_default": DEFAULT_OLLAMA_BASE_URL,
        # Ollama OpenAI path is {base}/v1 — base is the daemon root without /v1.
        "openai_path": "/v1",
        "models": _OLLAMA_MODELS,
        "default": "llama3.2:3b",
        "tags_note": "Install models with: ollama pull <tag>",
        # Chat-focused curated list (no image/video generators in the stock list).
        "tasks": ["chat"],
    },
    "llamacpp": {
        "label": "llama.cpp",
        "mode": "local",
        "needs_key": False,
        "env_key": "",
        "kind": "openai",
        "base_env": "LLAMACPP_BASE_URL",
        "base_default": DEFAULT_LLAMACPP_BASE_URL,
        # llama.cpp server usually serves OpenAI routes at the root already
        # (e.g. http://127.0.0.1:8080/v1/chat/completions). base_default
        # is the server root; openai_path empty means base is already /v1
        # OR we append /v1 when calling — see provider_openai_base().
        "openai_path": "/v1",
        "models": _LLAMACPP_MODELS,
        "default": "llama-3.2-3b",
        "tags_note": "Load a GGUF in llama-server; curated ids are hints for the UI",
        "tasks": ["chat"],
    },
}


# ---------------------------------------------------------------------------
# Combined catalog (backward-compatible PROVIDERS dict)
# ---------------------------------------------------------------------------

# Merged map used by brain/server/tools. Prefer list_providers(mode) in new code.
PROVIDERS: Dict[str, Dict[str, Any]] = {}
PROVIDERS.update(ONLINE_PROVIDERS)
PROVIDERS.update(LOCAL_PROVIDERS)

# Fallback if PROVIDER in .env is missing or not in PROVIDERS.
DEFAULT_PROVIDER = "gemini"

# Model entry type: plain id string (online) or {id, label} (local curated).
ModelEntry = Union[str, Dict[str, str]]


# ---------------------------------------------------------------------------
# Env helpers (lazy import of config to avoid circular import)
# ---------------------------------------------------------------------------


def _env_get(name: str) -> str:
    """One env value: .env first, then process environment, else empty.

    Imports config.load_env lazily so providers.py can be imported from
    config.py without a circular import at module load time.
    """
    import os

    from ai_agent.config import load_env

    return (load_env().get(name) or os.environ.get(name) or "").strip()


def status_blink_ms() -> int:
    """STATUS_BLINK_MS from .env (default 2000). UI red LED blink period."""
    raw = _env_get("STATUS_BLINK_MS") or "2000"
    try:
        ms = int(raw)
    except ValueError:
        return 2000
    # Keep a sane floor so a typo does not strobe the page.
    return ms if ms >= 200 else 2000


def ollama_base_url() -> str:
    """OLLAMA_BASE_URL from .env, else http://127.0.0.1:11434."""
    return _env_get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL


def llamacpp_base_url() -> str:
    """LLAMACPP_BASE_URL from .env, else http://127.0.0.1:8080."""
    return _env_get("LLAMACPP_BASE_URL") or DEFAULT_LLAMACPP_BASE_URL


# ---------------------------------------------------------------------------
# Model id / label helpers
# ---------------------------------------------------------------------------


def model_id(entry: ModelEntry) -> str:
    """Extract the model id from a string or {id, label} dict."""
    if isinstance(entry, dict):
        return (entry.get("id") or entry.get("label") or "").strip()
    return str(entry).strip()


def model_label(entry: ModelEntry) -> str:
    """Display label for a model entry (falls back to id)."""
    if isinstance(entry, dict):
        return (entry.get("label") or entry.get("id") or "").strip()
    return str(entry).strip()


def model_ids(meta: Dict[str, Any]) -> List[str]:
    """List of model ids for a provider meta dict."""
    out: List[str] = []
    for entry in meta.get("models") or []:
        mid = model_id(entry)
        if mid:
            out.append(mid)
    return out


def models_for_api(meta: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize models to [{id, label}, ...] for /api/models JSON."""
    out: List[Dict[str, str]] = []
    for entry in meta.get("models") or []:
        mid = model_id(entry)
        if not mid:
            continue
        out.append({"id": mid, "label": model_label(entry)})
    return out


# ---------------------------------------------------------------------------
# Catalog query helpers (modes / providers / models / meta)
# ---------------------------------------------------------------------------


def list_modes() -> List[Dict[str, str]]:
    """Return [{id, label}, ...] for the Mode dropdown."""
    return [
        {"id": "online", "label": "Online"},
        {"id": "local", "label": "Local"},
    ]


def list_providers(mode: Optional[str] = None) -> List[str]:
    """Provider ids, optionally filtered by mode (online|local)."""
    mode_l = (mode or "").strip().lower()
    out: List[str] = []
    for pid, meta in PROVIDERS.items():
        if mode_l and (meta.get("mode") or "online") != mode_l:
            continue
        out.append(pid)
    return out


def list_models(provider: str) -> List[Dict[str, str]]:
    """Curated models for a provider as [{id, label}, ...]."""
    meta = PROVIDERS.get(provider) or {}
    return models_for_api(meta)


def provider_openai_base(provider: str) -> str:
    """Resolved OpenAI-compat base URL including /v1 for chat/completions.

    Online: meta['base'] as stored (already .../v1).
    Local: {OLLAMA|LLAMACPP}_BASE_URL + openai_path (default /v1).
    """
    meta = PROVIDERS.get(provider) or {}
    if meta.get("base"):
        return str(meta["base"]).rstrip("/")
    # Local: resolve from env
    if provider == "ollama":
        root = ollama_base_url().rstrip("/")
    elif provider == "llamacpp":
        root = llamacpp_base_url().rstrip("/")
    else:
        root = (meta.get("base_default") or "").rstrip("/")
        env_name = meta.get("base_env") or ""
        if env_name:
            root = (_env_get(env_name) or root).rstrip("/")
    path = meta.get("openai_path")
    if path is None:
        path = "/v1"
    path = str(path).rstrip("/")
    if not path:
        return root
    # Avoid double /v1 if the user already put it in the env URL.
    if root.endswith(path):
        return root
    return root + path


def get_provider_meta(provider: str) -> Dict[str, Any]:
    """Provider meta for UI/brain: label, needs_key, base, default, mode, kind.

    Resolves local base URLs from .env. Returns a shallow copy so callers
    can mutate without touching the catalog.
    """
    raw = PROVIDERS.get(provider)
    if not raw:
        return {}
    meta = dict(raw)
    meta["id"] = provider
    meta["mode"] = meta.get("mode") or "online"
    meta["needs_key"] = bool(meta.get("needs_key", True))
    meta["label"] = meta.get("label") or provider
    # Resolved OpenAI-compat base (online already has base; local computed).
    if meta.get("kind") == "openai":
        meta["base"] = provider_openai_base(provider)
    elif meta.get("base_env") and not meta.get("base"):
        meta["base"] = provider_openai_base(provider)
    meta["default"] = meta.get("default") or (
        model_ids(meta)[0] if model_ids(meta) else ""
    )
    meta["models_api"] = models_for_api(meta)
    return meta


# ---------------------------------------------------------------------------
# Defaults and allow-list (moved from config for one catalog home)
# ---------------------------------------------------------------------------


def provider_key(provider: str) -> str:
    """API key for a provider id, or empty if unset / local (no key)."""
    meta = PROVIDERS.get(provider) or {}
    if not meta.get("needs_key", True):
        return ""
    ek = meta.get("env_key") or ""
    return _env_get(ek) if ek else ""


def keys_status() -> Dict[str, bool]:
    """Map of provider id → True if that key is set (online providers only).

    Local providers are False here (they do not use keys); use
    provider_ready() / readiness_status() for LED state.
    """
    out: Dict[str, bool] = {}
    for pid, meta in PROVIDERS.items():
        if meta.get("needs_key", True):
            out[pid] = bool(provider_key(pid))
        else:
            # Keyless online (Pollinations) or local — treat as "keyed" for UI.
            out[pid] = True
    return out


def default_provider() -> str:
    """PROVIDER from .env if valid, otherwise DEFAULT_PROVIDER."""
    p = (_env_get("PROVIDER") or DEFAULT_PROVIDER).lower()
    return p if p in PROVIDERS else DEFAULT_PROVIDER


def default_mode() -> str:
    """Mode for the default provider (online|local)."""
    meta = PROVIDERS.get(default_provider()) or {}
    return meta.get("mode") or DEFAULT_MODE


def default_model(provider: Optional[str] = None) -> str:
    """Default model for a provider (optional GEMINI_MODEL / OPENAI_MODEL / …)."""
    provider = provider or default_provider()
    meta = PROVIDERS.get(provider) or PROVIDERS[DEFAULT_PROVIDER]
    # Optional per-provider model env: GEMINI_MODEL, OPENAI_MODEL, etc.
    # Local providers have empty env_key — skip the replace path.
    ek = meta.get("env_key") or ""
    m = ""
    if ek and ek.endswith("_API_KEY"):
        env_name = ek.replace("_API_KEY", "_MODEL")
        m = _env_get(env_name)
    if not m:
        # Optional OLLAMA_MODEL / LLAMACPP_MODEL
        if provider == "ollama":
            m = _env_get("OLLAMA_MODEL")
        elif provider == "llamacpp":
            m = _env_get("LLAMACPP_MODEL")
    m = m or str(meta.get("default") or "")
    return allowed(provider, m)


def allowed(provider: str, model: Optional[str]) -> str:
    """Return model if in that provider's list; else the provider default."""
    meta = PROVIDERS.get(provider)
    if not meta:
        provider = default_provider()
        meta = PROVIDERS[provider]
    all_m = set(model_ids(meta))
    if model and model in all_m:
        return model
    d = str(meta.get("default") or "")
    if d in all_m:
        return d
    ids = model_ids(meta)
    return ids[0] if ids else d


# ---------------------------------------------------------------------------
# Reachability / readiness (for status LED)
# ---------------------------------------------------------------------------


def _http_get_json(url: str, timeout: float = 1.5) -> Tuple[bool, Any]:
    """GET url; return (ok, parsed_json_or_None). Short timeout for UI polls."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return True, json.loads(raw) if raw.strip() else None
            except json.JSONDecodeError:
                return True, None
    except Exception:  # noqa: BLE001 — unreachable / timeout / refused
        return False, None


def local_runtime_reachable(provider: str) -> bool:
    """True if the local daemon answers a cheap probe (Ollama tags / models)."""
    if provider == "ollama":
        root = ollama_base_url().rstrip("/")
        ok, _ = _http_get_json(root + "/api/tags")
        return ok
    if provider == "llamacpp":
        base = provider_openai_base("llamacpp")
        ok, _ = _http_get_json(base + "/models")
        if ok:
            return True
        # Some builds expose /health on the server root (no /v1).
        root = llamacpp_base_url().rstrip("/")
        ok2, _ = _http_get_json(root + "/health")
        return ok2
    return False


def local_installed_models(provider: str) -> Optional[List[str]]:
    """Installed/loaded model ids when detectable; None if unknown.

    Ollama: names from /api/tags.
    llama.cpp: ids from /v1/models when the server lists them; else None
    (single-model servers often do not expose a useful list — treat as
    'reachable ⇒ ready' at provider level, model check skipped).
    """
    if provider == "ollama":
        root = ollama_base_url().rstrip("/")
        ok, data = _http_get_json(root + "/api/tags")
        if not ok or not isinstance(data, dict):
            return None
        names: List[str] = []
        for item in data.get("models") or []:
            if isinstance(item, dict):
                n = (item.get("name") or item.get("model") or "").strip()
                if n:
                    names.append(n)
        return names
    if provider == "llamacpp":
        base = provider_openai_base("llamacpp")
        ok, data = _http_get_json(base + "/models")
        if not ok or not isinstance(data, dict):
            return None
        names = []
        for item in data.get("data") or []:
            if isinstance(item, dict):
                n = (item.get("id") or "").strip()
                if n:
                    names.append(n)
        return names if names else None
    return None


# Thin aliases (mode id renamed offline → local in 0.2.0).
offline_runtime_reachable = local_runtime_reachable
offline_installed_models = local_installed_models
OFFLINE_PROVIDERS = LOCAL_PROVIDERS  # noqa: N816 — legacy name


def model_looks_installed(installed: List[str], model: str) -> bool:
    """True if model matches an installed tag (exact or prefix before ':').

    Ollama tags can be 'llama3.2:3b' or 'llama3.2:3b-instruct-q4_0'; we
    accept exact match or installed name starting with model + ':' / model.
    """
    model = (model or "").strip()
    if not model:
        return False
    for name in installed:
        if name == model:
            return True
        if name.startswith(model + ":") or name.startswith(model + "-"):
            return True
        # Also accept curated id that is a prefix of a longer installed tag.
        if model.startswith(name.split(":")[0]) and name.split(":")[0] == model.split(":")[0]:
            if name == model or name.startswith(model):
                return True
    return False


def provider_ready(
    provider: str, model: Optional[str] = None
) -> Dict[str, Any]:
    """Readiness dict for the status LED / health API.

    Online: ready if API key present.
    Local: ready if runtime reachable; if model given and install list is
    detectable, also require the model to appear installed.

    Returns keys: ready (bool), mode, reason (short), has_key, reachable,
    model_installed (bool|None), label.
    """
    meta = get_provider_meta(provider)
    if not meta:
        return {
            "ready": False,
            "mode": "",
            "reason": "unknown provider",
            "has_key": False,
            "reachable": False,
            "model_installed": None,
            "label": provider,
        }
    label = meta["label"]
    mode = meta["mode"]
    if mode == "online":
        # Keyless online providers (e.g. Pollinations) are always ready.
        if not meta.get("needs_key", True):
            return {
                "ready": True,
                "mode": mode,
                "reason": "ok",
                "has_key": True,
                "reachable": None,
                "model_installed": None,
                "label": label,
            }
        has = bool(provider_key(provider))
        return {
            "ready": has,
            "mode": mode,
            "reason": "ok" if has else "missing API key",
            "has_key": has,
            "reachable": None,
            "model_installed": None,
            "label": label,
        }
    # Local path
    reachable = local_runtime_reachable(provider)
    model_installed: Optional[bool] = None
    reason = "ok"
    ready = reachable
    if not reachable:
        reason = "runtime unreachable"
        ready = False
    else:
        installed = local_installed_models(provider)
        if model and installed is not None:
            model_installed = model_looks_installed(installed, model)
            if not model_installed:
                ready = False
                reason = "model not installed"
        elif model and installed is None and provider == "llamacpp":
            # Single-model llama.cpp: reachable is enough for green LED.
            model_installed = None
            reason = "ok"
        elif not model:
            reason = "ok"
    return {
        "ready": ready,
        "mode": mode,
        "reason": reason,
        "has_key": False,
        "reachable": reachable,
        "model_installed": model_installed,
        "label": label,
    }


def readiness_status(selected_provider: Optional[str] = None, selected_model: Optional[str] = None) -> Dict[str, Any]:
    """Compact readiness map for /api/health (all providers + optional selected)."""
    all_ready: Dict[str, bool] = {}
    for pid in PROVIDERS:
        # Cheap: online = key check only; local = reachability (no model).
        st = provider_ready(pid, model=None)
        all_ready[pid] = bool(st.get("ready"))
    selected = None
    if selected_provider:
        selected = provider_ready(selected_provider, selected_model)
    return {
        "providers_ready": all_ready,
        "selected": selected,
        "status_blink_ms": status_blink_ms(),
    }



# ---------------------------------------------------------------------------
# Task → model filtering (Chat | Image | Video | Vision)
# ---------------------------------------------------------------------------
# Chat soft-filters generators. Image/Video/Vision return the filtered list
# even when empty (no fallback to full catalog). catalog_for_api omits
# providers whose tasks list does not include the requested task.


TASK_IDS = ("chat", "image", "video", "vision")


def _model_matches_task(mid: str, task: str) -> bool:
    """Heuristic: does this model id look suitable for task?"""
    m = (mid or "").lower()
    task = (task or "chat").lower()
    if task == "chat":
        # Hide dedicated image/video generators from Chat.
        hide = (
            "dall-e",
            "gpt-image",
            "imagen",
            "veo-",
            "flux",
            "sdxl",
            "stable-diffusion",
            "realistic-vision",
            "pollinations",
            "agnes-video",
            "-image",  # e.g. gemini-*-image, flash-image
        )
        if any(x in m for x in hide):
            return False
        return True
    if task == "image":
        return any(
            x in m
            for x in (
                "image",
                "imagen",
                "dall-e",
                "gpt-image",
                "flux",
                "sdxl",
                "stable-diffusion",
                "realistic-vision",
                "pollinations",
            )
        )
    if task == "video":
        return any(
            x in m
            for x in (
                "veo",
                "video",
                "agnes-video",
                "kling",
                "wan",
                "vidu",
                "seedance",
            )
        )
    if task == "vision":
        # Explicit vision markers, or known multimodal chat ids — NOT everything.
        if any(x in m for x in ("vision", "vl-", "4v", "llava", "moondream")):
            return True
        if "gpt-4o" in m:
            return True
        # Gemini flash/pro chat multimodal — exclude pure image/veo generators.
        if "gemini" in m and ("flash" in m or "pro" in m):
            if any(x in m for x in ("image", "veo", "imagen")):
                return False
            return True
        return False
    return True


def filter_models_for_task(models, task: str):
    """Filter [{id,label}|str] list for task.

    Chat: soft-filter (drop generators; if somehow empty, keep original).
    Image/Video/Vision: return filtered even when empty — NO full-list fallback.
    """
    task = (task or "chat").lower()
    filtered = []
    for entry in models or []:
        mid = model_id(entry)
        if _model_matches_task(mid, task):
            filtered.append(entry)
    if task == "chat" or not task:
        return filtered if filtered else list(models or [])
    # Media tasks: empty is intentional (provider may be omitted by catalog).
    return filtered


def catalog_for_api(task: Optional[str] = None) -> Dict[str, Any]:
    """Structured catalog for GET /api/models (modes + providers + models).

    When task is set:
      - omit providers whose `tasks` does not include that task
      - missing `tasks` defaults to ["chat"] only (not all TASK_IDS)
      - model lists are filtered (media: no empty→full fallback)
    """
    task_l = (task or "").strip().lower() or None
    providers_out: Dict[str, Any] = {}
    for pid, meta in PROVIDERS.items():
        # Default missing tasks → chat only (not every TASK_ID).
        ptasks = meta.get("tasks")
        if not ptasks:
            ptasks = ["chat"]
        else:
            ptasks = list(ptasks)
        if task_l and task_l not in ptasks:
            continue
        models = models_for_api(meta)
        if task_l:
            models = [
                {"id": model_id(e), "label": model_label(e)}
                for e in filter_models_for_task(models, task_l)
            ]
        providers_out[pid] = {
            "label": meta.get("label") or pid,
            "mode": meta.get("mode") or "online",
            "needs_key": bool(meta.get("needs_key", True)),
            "models": models,
            "default": meta.get("default") or "",
            "tasks": ptasks,
        }
    dp = default_provider()
    return {
        "modes": list_modes(),
        "tasks": [
            {"id": "chat", "label": "Chat"},
            {"id": "image", "label": "Image"},
            {"id": "video", "label": "Video"},
            {"id": "vision", "label": "Vision"},
        ],
        "providers": providers_out,
        "default_provider": dp,
        "default_model": default_model(dp),
        "default_mode": default_mode(),
        "default_task": "chat",
        "status_blink_ms": status_blink_ms(),
        "task": task_l or "chat",
    }
