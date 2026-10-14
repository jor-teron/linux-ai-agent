"""
Online API rate limiter (stdlib only, Python 3.8+).

Knobs live in .env and are read by ai_agent.config on every call (0 = off):

  API_DELAY_SEC  minimum seconds between calls (default 5)
  RPM            max requests in a rolling 60s window (default 14)
  RPD            max requests per local calendar day (default 500)
  TPM            max estimated tokens in a rolling 60s window (default 0)

Token estimate when the provider does not return a count:
  max(1, len(request payload chars) // 4)

Local Ollama / llama.cpp are not billed APIs: callers pass limit=False, or
a loopback host / configured local base URL is skipped automatically.

Behaviour:
  - Short waits (<= 60s) for API_DELAY_SEC and the RPM/TPM window. Sleep is
    done while holding one lock so Telegram and the web UI cannot both slip
    past the cap.
  - If the needed wait is longer than ~60s, raise RateLimited (no retry loop).
  - If RPD is already used up for this calendar day, raise immediately.
    Do not wait until tomorrow.

RPD day is the machine's local date (the user TZ when the box clock is set
that way). If local time is unavailable, the UTC date is used. Counts are
in-memory and reset when the process restarts.
"""
import threading
import time
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urlparse

# Rolling window for RPM and TPM (seconds).
_WINDOW = 60.0
# Wait this long at most; anything bigger is a hard error.
_MAX_WAIT = 60.0

# Hosts that are never a billed cloud API.
_LOOPBACK = ("localhost", "127.0.0.1", "::1", "0.0.0.0")

_lock = threading.Lock()
# Monotonic time of the last counted online call (0 = none yet).
_last_mono = 0.0
# Recent call timestamps (monotonic) inside the RPM window.
_req_times = deque()
# Recent (monotonic, estimated tokens) inside the TPM window.
_tok_times = deque()
# Local calendar day key "YYYY-MM-DD" and how many calls landed that day.
_day_key = ""
_day_count = 0


# Per-thread "Limits" switch from the web UI (v0.46). None = follow .env.
# Each HTTP request runs on its own server thread, so one user's switch
# never leaks into Telegram or another request.
_override = threading.local()


class limits_override:
    """Context manager: run a block with the UI Limits switch applied.

    enabled=None  → normal .env behaviour (nothing changes).
    enabled=False → online calls skip API_DELAY_SEC / RPM / RPD / TPM waits
                    and caps, but are still counted so RPD stays accurate.
    enabled=True  → normal .env behaviour.
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self.prev = None

    def __enter__(self):
        self.prev = getattr(_override, "enabled", None)
        _override.enabled = self.enabled
        return self

    def __exit__(self, *exc):
        _override.enabled = self.prev
        return False


def limits_disabled():
    """True when the current thread asked to turn limits off (UI switch)."""
    return getattr(_override, "enabled", None) is False


def _record_only(payload):
    """Count one online call (RPM/RPD/TPM history) without waiting or capping.

    Used when Limits is off, so later calls with Limits on still see real usage.
    """
    global _day_key, _day_count, _last_mono
    tokens = estimate_tokens(payload)
    with _lock:
        now = time.monotonic()
        _prune(now)
        day = _calendar_day()
        if day != _day_key:
            _day_key = day
            _day_count = 0
        _last_mono = now
        _req_times.append(now)
        _tok_times.append((now, tokens))
        _day_count += 1


class RateLimited(Exception):
    """Call must not proceed. str(exc) always starts with 'rate limited'."""


def estimate_tokens(payload):
    """Estimate tokens from a request body when real usage is unavailable.

    max(1, len(chars) // 4). None counts as 1 (a poll GET still costs a request).
    """
    if payload is None:
        n = 0
    elif isinstance(payload, bytes):
        n = len(payload)
    elif isinstance(payload, str):
        n = len(payload)
    else:
        n = len(str(payload))
    return max(1, n // 4)


def _calendar_day():
    """Local calendar day, else UTC. Used only for the RPD counter."""
    try:
        return datetime.now().strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _same_endpoint(url, base):
    """True if url's scheme/host/port match a configured local base URL.

    Prefix checks would treat http://127.0.0.1:11434 as a prefix of :114349.
    Comparing host and port avoids that.
    """
    if not url or not base:
        return False
    try:
        u = urlparse(url)
        b = urlparse(base)
    except Exception:  # noqa: BLE001
        return False
    if not b.hostname or not u.hostname:
        return False
    if (u.hostname or "").lower() != (b.hostname or "").lower():
        return False
    # Default ports so http://host and http://host:80 compare equal.
    def _port(parsed):
        if parsed.port:
            return parsed.port
        if (parsed.scheme or "").lower() == "https":
            return 443
        if (parsed.scheme or "").lower() == "http":
            return 80
        return None
    return _port(u) == _port(b)


def is_local_api_url(url):
    """True for loopback or the configured Ollama / llama.cpp base.

    Those calls are not billed cloud APIs and must not consume RPM/RPD/TPM.
    """
    if not url:
        return False
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        host = ""
    if host in _LOOPBACK:
        return True
    # Lazy import: providers pulls config; safe once modules have finished loading.
    try:
        from ai_agent.providers import llamacpp_base_url, ollama_base_url
        bases = (ollama_base_url(), llamacpp_base_url())
    except Exception:  # noqa: BLE001
        bases = ()
    for base in bases:
        if _same_endpoint(url, base):
            return True
    return False


def _prune(now):
    """Drop timestamps that have left the rolling 60s window."""
    cutoff = now - _WINDOW
    while _req_times and _req_times[0] <= cutoff:
        _req_times.popleft()
    while _tok_times and _tok_times[0][0] <= cutoff:
        _tok_times.popleft()


def _tpm_wait(now, tokens, cap):
    """Seconds until enough old tokens fall out so this call fits, or None.

    None means the call itself is larger than TPM — waiting cannot help.
    0 means it fits now.
    """
    used = 0
    for _ts, tok in _tok_times:
        used += tok
    if used + tokens <= cap:
        return 0.0
    if tokens > cap:
        return None
    need = used + tokens - cap
    dropped = 0
    wait = 0.0
    for ts, tok in _tok_times:
        dropped += tok
        wait = (ts + _WINDOW) - now
        if dropped >= need:
            break
    if wait < 0:
        wait = 0.0
    return wait


def before_online_call(payload=None, url=None, limit=None):
    """Gate one online provider HTTP call. Return None or raise RateLimited.

    limit=False  — skip (local runtime).
    limit=None   — skip when url is local/loopback, else enforce.
    limit=True   — always enforce (caller already decided this is online).

    Holds the module lock while waiting so two threads cannot both pass.
    """
    if limit is False:
        return
    if limit is None and is_local_api_url(url):
        return
    # UI Limits switch off (v0.46): count the call, but never wait or refuse.
    if limits_disabled():
        _record_only(payload)
        return

    # Read knobs outside the lock (they hit disk). 0 means that knob is off.
    from ai_agent.config import api_delay_sec, api_rpd, api_rpm, api_tpm

    delay = float(api_delay_sec() or 0)
    rpm = int(api_rpm() or 0)
    rpd = int(api_rpd() or 0)
    tpm = int(api_tpm() or 0)
    if delay <= 0 and rpm <= 0 and rpd <= 0 and tpm <= 0:
        return

    tokens = estimate_tokens(payload)

    # A single request bigger than TPM can never fit the window.
    if tpm > 0 and tokens > tpm:
        raise RateLimited(
            "rate limited: estimated %d tokens exceeds TPM=%d" % (tokens, tpm)
        )

    with _lock:
        # Bound the loop so a stalled clock cannot spin forever.
        for _attempt in range(8):
            now = time.monotonic()
            _prune(now)

            global _day_key, _day_count, _last_mono
            day = _calendar_day()
            if day != _day_key:
                _day_key = day
                _day_count = 0
            # Daily cap is a hard stop — do not sleep until tomorrow.
            if rpd > 0 and _day_count >= rpd:
                raise RateLimited(
                    "rate limited: daily request cap reached (RPD=%d for %s). "
                    "No more online API calls until the next local calendar day."
                    % (rpd, day)
                )

            waits = []  # (knob name, seconds)
            if delay > 0 and _last_mono > 0:
                gap = delay - (now - _last_mono)
                if gap > 0:
                    waits.append(("API_DELAY_SEC", gap))
            if rpm > 0 and len(_req_times) >= rpm:
                gap = (_req_times[0] + _WINDOW) - now
                if gap > 0:
                    waits.append(("RPM", gap))
            if tpm > 0:
                gap = _tpm_wait(now, tokens, tpm)
                if gap is None:
                    raise RateLimited(
                        "rate limited: estimated %d tokens exceeds TPM=%d" % (tokens, tpm)
                    )
                if gap > 0:
                    waits.append(("TPM", gap))

            if not waits:
                _last_mono = now
                _req_times.append(now)
                _tok_times.append((now, tokens))
                _day_count += 1
                return

            # Longest wait is what we would actually sleep.
            knob, gap = max(waits, key=lambda item: item[1])
            if gap > _MAX_WAIT:
                raise RateLimited(
                    "rate limited: %s needs a %.0fs wait (over 60s); not retrying. "
                    "RPM=%d RPD=%d TPM=%d API_DELAY_SEC=%s"
                    % (knob, gap, rpm, rpd, tpm, delay)
                )
            # Hold the lock across the sleep so a second caller cannot sneak in.
            time.sleep(gap)
        raise RateLimited(
            "rate limited: still over cap after waiting (RPM=%d TPM=%d)" % (rpm, tpm)
        )
