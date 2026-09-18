"""
Config, env, paths, and constants (stdlib only).

Single place for workspace location, listen address, prompts/blocklist
loaders, .env helpers, TOOLS_DEFAULT / HISTORY_TURNS, PUBLIC_BASE_URL /
download_url, and optional UI_LIGHT_* theme overrides. Provider catalogs live in providers.py; this
module re-exports names brain/server/tools already import.

Imports from: stdlib; ai_agent.providers (catalog).
Used by: tools, brain, server, run, telegram, providers (lazy load_env).
"""
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

# ---------------------------------------------------------------------------
# Project root and version
# ---------------------------------------------------------------------------

# Package dir (this folder) and install root (parent: run.sh, VERSION, .env).
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent


def app_version():
    """Read VERSION for UI /api/health. Returned as text — not parsed.

    Current format is two-number sub-version + change counter, e.g. "0.39 (39)".
    Older three-part strings ("0.4.2 (38)") still pass through unchanged.
    Two parts or three parts are both OK; nothing here splits on X.Y.Z.
    The counter in brackets +1 every release and never resets.
    """
    vp = ROOT / "VERSION"
    try:
        return vp.read_text(encoding="utf-8").strip() or "0.0"
    except OSError:
        return "0.0"



# User secrets and overrides (API keys, HOST, PORT, WORKSPACE, PROVIDER,
# ALLOW_SUDO, SHELL_NET).
ENV_PATH = ROOT / ".env"


# ---------------------------------------------------------------------------
# Load .env (simple KEY=value — no pip/dotenv library)
# ---------------------------------------------------------------------------


def _read_dotenv():
    """Parse .env into a dict. Missing file → empty dict. Reloaded each call."""
    env = {}
    if not ENV_PATH.exists():
        return env
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _workspace_path():
    """Workspace folder: WORKSPACE from .env, else ~/ai-workspace."""
    raw = (_read_dotenv().get("WORKSPACE") or "").strip()
    if raw:
        raw = os.path.expandvars(raw)  # supports $USER, $HOME
        return Path(raw).expanduser()
    return Path.home() / "ai-workspace"


def _host_port():
    """Listen address and port from .env (defaults: 127.0.0.1:9191).

    Set HOST=0.0.0.0 in .env for LAN / Tailscale access (phone uses Tailscale IP).
    Localhost on the PC still works with HOST=0.0.0.0. Pair with PUBLIC_BASE_URL
    so Telegram gets absolute /api/download links. Default stays localhost-only.
    """
    env = _read_dotenv()
    host = (env.get("HOST") or os.environ.get("HOST") or "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int((env.get("PORT") or os.environ.get("PORT") or "9191").strip() or "9191")
    except ValueError:
        port = 9191
    return host, port


# ---------------------------------------------------------------------------
# Workspace files (created on demand when tools write them)
# ---------------------------------------------------------------------------

# Jail for file/shell tools = whole tree (default ~/ai-workspace).
# Subdirs created by ensure_ws: chats/, memory/, workspace/, workspace/generated/,
# test/, trash/, user/ plus memory/{date,topics}/ and session/user/assistant.md.
# Prefer workspace/ for new agent work; user/ is agent read-only.
WORKSPACE = _workspace_path()
# Agent must not write/delete under this folder (list/read OK).
USER_DIR = WORKSPACE / "user"
# Legacy single-file memory (migrated once into memory/date/YYYY_MM.md).
MEMORY_FILE = WORKSPACE / "memory.md"
# Split memory tree under workspace/memory/.
MEMORY_DIR = WORKSPACE / "memory"
MEMORY_SESSION = MEMORY_DIR / "session.md"
MEMORY_USER = MEMORY_DIR / "user.md"
MEMORY_ASSISTANT = MEMORY_DIR / "assistant.md"
MEMORY_DATE_DIR = MEMORY_DIR / "date"
# Topic notes live under memory/topics/ (plural). Old memory/topic/ migrates once.
MEMORY_TOPIC_DIR = MEMORY_DIR / "topics"
# Chat history Markdown: chats/YYYY_MM/YYYY_MM_DD.md under WORKSPACE (disk archive only).
# Old path was memory/chats/ (pre-0.4.2); user moves files manually — no auto-migrate.
CHATS_DIR = WORKSPACE / "chats"
# Legacy singular folder name (pre-0.4.1); files moved into topics/ on ensure.
MEMORY_TOPIC_DIR_LEGACY = MEMORY_DIR / "topic"
# JSON blob for a shell command waiting for the UI Confirm button.
# Shape: {"command": "...", "sudo": true|false}
PENDING_SHELL = WORKSPACE / ".pending_shell.json"
# Reminder list (id, text, due, notified).
REMINDERS_FILE = WORKSPACE / "reminders.json"
# Recurring jobs (every_minutes + prompt).
JOBS_FILE = WORKSPACE / "jobs.json"
# Append-only log of job results.
JOBS_LOG = WORKSPACE / "jobs_log.md"
# Bind address printed at startup and returned by /api/health.
HOST, PORT = _host_port()


def memory_date_path(when=None):
    """Path to memory/date/YYYY_MM.md for the given (or current) local month."""
    when = when or datetime.now()
    return MEMORY_DATE_DIR / ("%04d_%02d.md" % (when.year, when.month))


def memory_topic_path(name):
    """Safe path under memory/topics/<name>.md (alphanumeric, dash, underscore)."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    if not safe:
        raise ValueError("Empty or invalid topic name")
    return MEMORY_TOPIC_DIR / (safe + ".md")


def migrate_legacy_topic_dir():
    """Best-effort: move files from old memory/topic/ into memory/topics/ once.

    Only migrates when the legacy singular folder exists. Existing targets in
    topics/ are left alone (no overwrite). Empty legacy dir is removed.
    """
    legacy = MEMORY_TOPIC_DIR_LEGACY
    if not legacy.exists() or not legacy.is_dir():
        return False
    MEMORY_TOPIC_DIR.mkdir(parents=True, exist_ok=True)
    moved = False
    try:
        for p in list(legacy.iterdir()):
            if not p.is_file():
                continue
            dest = MEMORY_TOPIC_DIR / p.name
            if dest.exists():
                continue
            p.replace(dest)
            moved = True
        # Remove empty legacy dir (ignore leftover non-empty).
        try:
            next(legacy.iterdir())
        except StopIteration:
            legacy.rmdir()
    except OSError:
        return moved
    return moved


# ---------------------------------------------------------------------------
# Provider catalog — owned by providers.py; re-export for existing imports
# ---------------------------------------------------------------------------

# catalogs, defaults, key/status helpers, blink period, local base URLs
from ai_agent.providers import (  # noqa: E402
    DEFAULT_PROVIDER,
    DEFAULT_LLAMACPP_BASE_URL,
    DEFAULT_OLLAMA_BASE_URL,
    GEMINI_API_BASE,
    PROVIDERS,
    allowed,
    catalog_for_api,
    default_mode,
    default_model,
    default_provider,
    get_provider_meta,
    keys_status,
    list_modes,
    list_models,
    list_providers,
    llamacpp_base_url,
    ollama_base_url,
    provider_key,
    provider_openai_base,
    provider_ready,
    readiness_status,
    status_blink_ms,
)


# ---------------------------------------------------------------------------
# Status LED / local runtime env (also documented in .env.example)
# ---------------------------------------------------------------------------


def status_blink_ms_config():
    """Alias kept for clarity — STATUS_BLINK_MS from .env (default 2000)."""
    return status_blink_ms()


# ---------------------------------------------------------------------------
# Safety and the model's system prompt
# ---------------------------------------------------------------------------

# Fallback blocklist fragments if shell_blocklist file is missing.
_BLOCKED_FALLBACK = (
    r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/(?:\s|$|\*)"
    r"|rm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+~"
    r"|rm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+\$\{?HOME\}?"
    r"|chmod\s+-R\b[^\n]*\s+/(?:\s|$)"
    r"|chown\s+-R\b[^\n]*\s+/(?:\s|$)"
    r"|\bmkfs\b"
    r"|\bdd\b[^\n]*\bof=/dev/"
    r"|\b(shutdown|reboot|poweroff)\b"
    r"|crontab\s+-r\b"
    r"|\b(useradd|userdel|passwd)\b"
    r"|\bwipefs\b"
    r"|\blosetup\b"
    r"|\bsystemctl\s+(start|stop|restart|reload|enable|disable|mask|unmask)\b"
    r"|(?:curl|wget)\b[^\n]*\|\s*(?:ba)?sh\b"
    r"|bash\s+<\(\s*curl\b"
)


def _load_shell_blocklist():
    """Compile BLOCKED from ai_agent/shell_blocklist (one pattern per line).

    Blank lines and # comments are skipped. Missing/unreadable file → safe
    fallback list baked into this module.
    """
    path = PKG_DIR / "shell_blocklist"
    parts = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts.append(line)
    except OSError:
        parts = []
    body = "|".join(parts) if parts else _BLOCKED_FALLBACK
    return re.compile(r"(?ix)(" + body + r")")


# Commands we refuse even if the user clicks Confirm (destructive / privilege).
BLOCKED = _load_shell_blocklist()


# Tiny built-in prompts if prompt_chat / prompt_tools files are missing.
_FALLBACK_CHAT = (
    "Helpful Linux PC agent. Jail: %s. Prefer workspace/; user/ read-only. "
    "Short answers. Name downloadable files; include download links when files are written."
)
_FALLBACK_TOOLS = (
    "Tools: files, memory, run_shell, web_search, reminders, jobs. "
    "sudo needs Confirm / Telegram YES. Shell net on unless SHELL_NET=0. After write_file include download URL."
)


def _read_pkg_text(name):
    """Read a UTF-8 text file from the package dir, or None if missing."""
    try:
        return (PKG_DIR / name).read_text(encoding="utf-8")
    except OSError:
        return None


def load_prompt_chat():
    """Short system prompt (prompt_chat). Substitutes %%WORKSPACE%% / %WORKSPACE%."""
    raw = _read_pkg_text("prompt_chat")
    if raw is None or not raw.strip():
        text = _FALLBACK_CHAT % (WORKSPACE,)
    else:
        text = raw.strip()
        text = text.replace("%WORKSPACE%", str(WORKSPACE))
        text = text.replace("%%WORKSPACE%%", str(WORKSPACE))
    return text


def load_prompt_tools():
    """Extra system text when tools are enabled (prompt_tools file)."""
    raw = _read_pkg_text("prompt_tools")
    if raw is None or not raw.strip():
        return _FALLBACK_TOOLS
    return raw.strip()


def system_prompt(use_tools=False):
    """Compose system text: prompt_chat, plus prompt_tools when tools are on."""
    base = load_prompt_chat()
    if use_tools:
        extra = load_prompt_tools()
        if extra:
            return base + "\n\n" + extra
    return base


# ---------------------------------------------------------------------------
# Env helpers (provider defaults live in providers.py; re-exported above)
# ---------------------------------------------------------------------------


def load_env():
    """Public alias for _read_dotenv (reload each call so .env edits apply)."""
    return _read_dotenv()


def _env_get(name):
    """One env value: .env first, then the process environment, else empty."""
    return (load_env().get(name) or os.environ.get(name) or "").strip()


# ---------------------------------------------------------------------------
# Optional light-theme color overrides (UI_LIGHT_* in .env)
# ---------------------------------------------------------------------------
# Empty / unset → built-in soft-gray light defaults in ui.py.
# Served to the browser via /api/health and /api/models as "ui_light".

# Map: API/JSON key → env var name. CSS vars: --bg, --panel, …
_UI_LIGHT_ENV = (
    ("bg", "UI_LIGHT_BG"),
    ("panel", "UI_LIGHT_PANEL"),
    ("border", "UI_LIGHT_BORDER"),
    ("text", "UI_LIGHT_TEXT"),
    ("muted", "UI_LIGHT_MUTED"),
    ("user", "UI_LIGHT_USER"),
    ("bot", "UI_LIGHT_BOT"),
    ("bot_border", "UI_LIGHT_BOT_BORDER"),
    ("input", "UI_LIGHT_INPUT"),
)

# Loose hex: #rgb or #rrggbb (case-insensitive). Invalid values are ignored.
_HEX_COLOR_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _valid_hex_color(raw):
    """Return normalized hex string if valid #rgb/#rrggbb, else None."""
    s = (raw or "").strip()
    if not s or not _HEX_COLOR_RE.match(s):
        return None
    return s


def ui_light_theme():
    """Optional UI_LIGHT_* hex overrides from .env for the soft light theme.

    Returns a dict of only valid keys (e.g. {"bg": "#d2d7e0"}). Empty dict
    when nothing is set or all values are invalid. Dark theme is never
    affected — the UI applies these only while data-theme=light.
    """
    out = {}
    for key, env_name in _UI_LIGHT_ENV:
        val = _valid_hex_color(_env_get(env_name))
        if val:
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# Session memory reset (SESSION_RESET_HOURS / SESSION_RESET_AFTER)
# ---------------------------------------------------------------------------
# If neither env is set, default to clearing session.md when older than 24h.
# If both are set, either condition clears it.
# SESSION_RESET_AFTER=HH:MM — clear when local clock has passed that time since
# the session file's last modification (daily reset-at-clock pattern).


def session_reset_hours():
    """Optional SESSION_RESET_HOURS as float, or None if unset/invalid."""
    raw = _env_get("SESSION_RESET_HOURS")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def session_reset_after():
    """Optional SESSION_RESET_AFTER as (hour, minute), or None if unset/invalid."""
    raw = _env_get("SESSION_RESET_AFTER")
    if not raw or ":" not in raw:
        return None
    parts = raw.split(":", 1)
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return h, m


def session_should_reset(mtime_ts, now=None):
    """True if session.md should be cleared given its mtime and current local time.

    Defaults: if neither SESSION_RESET_HOURS nor SESSION_RESET_AFTER is set,
    treat as HOURS=24. If both are set, either trigger clears the session.
    """
    now = now or datetime.now()
    hours = session_reset_hours()
    after = session_reset_after()
    if hours is None and after is None:
        hours = 24.0
    if hours is not None:
        age_s = now.timestamp() - float(mtime_ts)
        if age_s >= float(hours) * 3600.0:
            return True
    if after is not None:
        h, m = after
        boundary = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if now < boundary:
            boundary = boundary - timedelta(days=1)
        # Reset if file was last touched before the most recent HH:MM boundary.
        if float(mtime_ts) < boundary.timestamp():
            return True
    return False


# ---------------------------------------------------------------------------
# Optional Telegram bridge (remote DMs; UI stays on localhost)
# ---------------------------------------------------------------------------
# HOST stays 127.0.0.1. Telegram is the remote path: set TELEGRAM_BOT_TOKEN
# (and TELEGRAM_ALLOWED_CHAT_ID) in .env; run.py starts a long-poll thread.


def telegram_bot_token():
    """BotFather token from .env, or empty if Telegram is off."""
    return _env_get("TELEGRAM_BOT_TOKEN")


def telegram_allowed_chat_id():
    """DM allow-list chat id (string). Empty → bot replies with the caller's id only."""
    return _env_get("TELEGRAM_ALLOWED_CHAT_ID")


def telegram_provider():
    """Optional TELEGRAM_PROVIDER, else config default_provider()."""
    p = (_env_get("TELEGRAM_PROVIDER") or "").lower()
    if p and p in PROVIDERS:
        return p
    return default_provider()


def telegram_model():
    """Optional TELEGRAM_MODEL for the Telegram provider, else that provider's default."""
    provider = telegram_provider()
    m = _env_get("TELEGRAM_MODEL")
    if m:
        return allowed(provider, m)
    return default_model(provider)


def public_base_url():
    """Absolute origin for download links (PUBLIC_BASE_URL), or '' if unset.

    Example Tailscale: http://100.100.50.XXX:9191 — no trailing slash required.
    Empty → relative /api/download links only (browser same-origin OK).
    """
    return _env_get("PUBLIC_BASE_URL").rstrip("/")


def download_url(name):
    """Build a download link for a workspace basename.

    If PUBLIC_BASE_URL is set → {base}/api/download?name={urlencoded}.
    Else → relative /api/download?name=… (browser / same host).
    """
    # Basename only — never put path separators into the query.
    base_name = Path(str(name or "")).name
    q = quote(base_name, safe="")
    rel = "/api/download?name=%s" % q
    base = public_base_url()
    if base:
        return "%s%s" % (base, rel)
    return rel


def ensure_memory_dirs():
    """Create memory/, date/, topics/ and touch core md files if missing.

    Also best-effort migrates old memory/topic/ → memory/topics/.
    Does not create memory/chats/ (chat archives live under WORKSPACE/chats/).
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DATE_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_TOPIC_DIR.mkdir(parents=True, exist_ok=True)
    # Empty placeholders so the tree is visible on disk after install/start.
    for md in (MEMORY_SESSION, MEMORY_USER, MEMORY_ASSISTANT):
        if not md.exists():
            try:
                md.touch()
            except OSError:
                pass
    migrate_legacy_topic_dir()



# ---------------------------------------------------------------------------
# Token trim: tools default off, shorter history
# ---------------------------------------------------------------------------


def tools_default():
    """True when TOOLS_DEFAULT=1/true/yes/on (default: off — chat without tools)."""
    raw = (_env_get("TOOLS_DEFAULT") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def history_turns():
    """Max prior chat turns sent to the model (HISTORY_TURNS, default 10)."""
    raw = _env_get("HISTORY_TURNS") or "10"
    try:
        n = int(raw.strip())
    except ValueError:
        n = 10
    return max(1, min(n, 50))


# Keywords that optionally boost tools on for a single turn (UI may still set flag).
_TOOL_KEYWORDS = re.compile(
    r"(?i)\b("
    r"run|shell|bash|cmd|command|terminal|"
    r"file|write|read|search|list\s+dir|directory|folder|"
    r"remind|reminder|job|jobs|memory|memor|"
    r"download|upload|sudo|workspace"
    r")\b"
)


def message_wants_tools(message):
    """True if the user message looks like it needs workspace tools."""
    return bool(_TOOL_KEYWORDS.search(message or ""))


def resolve_use_tools(message, request_flag=None):
    """Decide whether to send tool schemas for this turn.

    Enable if: TOOLS_DEFAULT=1, OR explicit request_flag True from UI/API,
    OR message matches tool keywords (optional boost). Explicit False from
    the client still allows keyword boost when default is off.
    """
    if tools_default():
        return True
    if request_flag is True:
        return True
    if message_wants_tools(message):
        return True
    return False


def allow_sudo():
    """True unless ALLOW_SUDO is 0/false/no/off (default: enabled)."""
    raw = (_env_get("ALLOW_SUDO") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def shell_net_env_off():
    """True when SHELL_NET=0/false/no/off (opt out of shell network under bwrap).

    Default is network ON. Unset or SHELL_NET=1/true/yes → keep host net.
    """
    raw = (_env_get("SHELL_NET") or "1").strip().lower()
    return raw in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Online API rate limits (TPM / RPM / RPD / API_DELAY_SEC)
# ---------------------------------------------------------------------------
# 0 means that knob is off. Defaults match .env.example.
# Read on every call so an .env edit applies without a code change.
# The limiter itself lives in ai_agent.ratelimit (used before online HTTP).


def _env_nonneg(name, default, integer=False):
    """Non-negative number from .env / process env. Bad or negative → default.

    Empty/unset uses default. "0" is kept (it means the limit is off).
    """
    raw = _env_get(name)
    if raw == "":
        n = float(default)
    else:
        try:
            n = float(raw)
        except ValueError:
            n = float(default)
    if n < 0:
        n = float(default)
    if integer:
        return int(n)
    return n


def api_tpm():
    """Max estimated tokens per rolling minute (TPM). Default 0 = off."""
    return _env_nonneg("TPM", 0, integer=True)


def api_rpm():
    """Max online API requests per rolling 60s (RPM). Default 14. 0 = off."""
    return _env_nonneg("RPM", 14, integer=True)


def api_rpd():
    """Max online API requests per local calendar day (RPD). Default 500. 0 = off."""
    return _env_nonneg("RPD", 500, integer=True)


def api_delay_sec():
    """Minimum seconds between online API calls (API_DELAY_SEC). Default 5. 0 = off."""
    return _env_nonneg("API_DELAY_SEC", 5)


def ensure_ws():
    """Create WORKSPACE tree: chats/, memory/, workspace/, user/, trash/, test/.

    Creates chats/ plus current-month chats/YYYY_MM/. Memory gets date/,
    topics/ plus empty user.md / assistant.md / session.md (touch). Also
    ensures workspace/generated/ for Image/Video. Does not create
    memory/chats/ (old path pre-0.4.2). Does not delete existing content.
    """
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for name in ("chats", "memory", "workspace", "test", "trash", "user"):
        (WORKSPACE / name).mkdir(parents=True, exist_ok=True)
    # Current month folder for daily chat archives (optional but convenient).
    now = datetime.now()
    (CHATS_DIR / ("%04d_%02d" % (now.year, now.month))).mkdir(parents=True, exist_ok=True)
    # Generated media from Image / Video tasks (Telegram attach + web download).
    (WORKSPACE / "workspace" / "generated").mkdir(parents=True, exist_ok=True)
    ensure_memory_dirs()
