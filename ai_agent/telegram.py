"""
Optional Telegram DM bridge (long polling, no webhook).

When TELEGRAM_BOT_TOKEN is set in .env, a background thread polls getUpdates
and routes private text DMs through brain.run_chat (same chat path as the
web UI). v0.39 is text-only: photos get a short text reply and are not
downloaded into Vision. /image /video /vision are ordinary chat text, not
task routes. Shell confirm stays YES/NO (case-insensitive); sudo may be
YES <password>.

sendPhoto / sendVideo / sendDocument and send_result_media / send_tool_media
remain defined so other code can import them, but the message handler does
not call them. Text replies do not depend on media.py. Chat turns still
append to chats/YYYY_MM/YYYY_MM_DD.md (sibling of memory/).

Imports from: ai_agent.config, brain, chat_history, tools (media is lazy).
Used by: run (start_telegram_thread). Respects TOOLS_DEFAULT via run_chat.
"""
import json
import threading
import time
import urllib.error
import urllib.request

from ai_agent.config import (
    PENDING_SHELL,
    default_model,
    default_provider,
    telegram_allowed_chat_id,
    telegram_bot_token,
    telegram_model,
    telegram_provider,
)
from ai_agent.brain import run_chat
from ai_agent.chat_history import append_exchange
from ai_agent.tools import cancel_pending_shell, confirm_pending_shell


# Max characters per Telegram sendMessage (Bot API hard limit).
_TG_MAX = 4096
# Keep last N history entries (user+assistant pairs → ~10 turns).
_HISTORY_MAX = 20
# Long-poll timeout (seconds) for getUpdates.
_POLL_TIMEOUT = 25

# Per-chat short history for run_chat (in-memory only).
_histories = {}
# Avoid re-prompting YES/NO for the same pending command string.
_last_shell_prompt = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# API helpers (urllib only — no pip)
# ---------------------------------------------------------------------------


def api_call(method, params=None, token=None):
    """Call api.telegram.org/bot<token>/<method>. Never logs the token.

    params: dict of query/JSON fields. Returns parsed JSON or {"ok": False, ...}.
    """
    tok = token or telegram_bot_token()
    if not tok:
        return {"ok": False, "description": "no token"}
    url = "https://api.telegram.org/bot%s/%s" % (tok, method)
    data = None
    headers = {}
    if params:
        # POST application/json is fine for Bot API methods we use.
        data = json.dumps(params).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=_POLL_TIMEOUT + 15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "description": "HTTP %s %s" % (e.code, body)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "description": str(e)}


def send_text(chat_id, text, token=None):
    """Send text via sendMessage, splitting into <=4096 chunks if needed."""
    text = text if text is not None else ""
    if not text:
        text = "(empty)"
    chunks = []
    s = str(text)
    while s:
        chunks.append(s[:_TG_MAX])
        s = s[_TG_MAX:]
    ok_all = True
    for chunk in chunks:
        res = api_call(
            "sendMessage",
            {"chat_id": chat_id, "text": chunk},
            token=token,
        )
        if not res.get("ok"):
            ok_all = False
    return ok_all


# ---------------------------------------------------------------------------
# Multipart file upload (sendPhoto / sendVideo / sendDocument)
# ---------------------------------------------------------------------------
# Bot API accepts application/json for text methods, but file uploads need
# multipart/form-data. Built with stdlib only (no requests).


def _multipart_body(fields, files):
    """Build multipart body + content-type for Bot API file methods.

    fields: dict of str→str form fields (chat_id, caption, …).
    files: dict of field_name → (filename, bytes, content_type).
    Returns (body_bytes, content_type_header).
    """
    import uuid

    boundary = "----TgBoundary%s" % uuid.uuid4().hex
    chunks = []
    for name, value in (fields or {}).items():
        chunks.append(("--%s\r\n" % boundary).encode("ascii"))
        chunks.append(
            ('Content-Disposition: form-data; name="%s"\r\n\r\n' % name).encode("utf-8")
        )
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    for name, triple in (files or {}).items():
        filename, data, ctype = triple
        ctype = ctype or "application/octet-stream"
        chunks.append(("--%s\r\n" % boundary).encode("ascii"))
        chunks.append(
            (
                'Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
                % (name, filename.replace('"', "_"))
            ).encode("utf-8")
        )
        chunks.append(("Content-Type: %s\r\n\r\n" % ctype).encode("utf-8"))
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(("--%s--\r\n" % boundary).encode("ascii"))
    body = b"".join(chunks)
    ctype_hdr = "multipart/form-data; boundary=%s" % boundary
    return body, ctype_hdr


def api_call_multipart(method, fields=None, files=None, token=None):
    """POST multipart to api.telegram.org/bot<token>/<method>. Never logs token."""
    tok = token or telegram_bot_token()
    if not tok:
        return {"ok": False, "description": "no token"}
    url = "https://api.telegram.org/bot%s/%s" % (tok, method)
    body, ctype = _multipart_body(fields or {}, files or {})
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:  # noqa: BLE001
            pass
        return {"ok": False, "description": "HTTP %s %s" % (e.code, err_body)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "description": str(e)}


def send_photo(chat_id, data, filename="photo.jpg", caption="", mime="", token=None):
    """sendPhoto multipart. data = image bytes."""
    mime = mime or "image/jpeg"
    fields = {"chat_id": str(chat_id)}
    if caption:
        fields["caption"] = str(caption)[:1024]
    return api_call_multipart(
        "sendPhoto",
        fields=fields,
        files={"photo": (filename, data, mime)},
        token=token,
    )


def send_video(chat_id, data, filename="video.mp4", caption="", mime="", token=None):
    """sendVideo multipart. data = video bytes."""
    mime = mime or "video/mp4"
    fields = {"chat_id": str(chat_id)}
    if caption:
        fields["caption"] = str(caption)[:1024]
    return api_call_multipart(
        "sendVideo",
        fields=fields,
        files={"video": (filename, data, mime)},
        token=token,
    )


def send_document(chat_id, data, filename="file.bin", caption="", mime="", token=None):
    """sendDocument multipart. data = file bytes."""
    mime = mime or "application/octet-stream"
    fields = {"chat_id": str(chat_id)}
    if caption:
        fields["caption"] = str(caption)[:1024]
    return api_call_multipart(
        "sendDocument",
        fields=fields,
        files={"document": (filename, data, mime)},
        token=token,
    )


def send_media_file(chat_id, path_or_bytes, filename=None, caption="", mime="", token=None):
    """Pick sendPhoto/sendVideo/sendDocument; enforce 50MB. Returns result dict.

    path_or_bytes: filesystem path (str/Path) or raw bytes.
    On oversize: {"ok": False, "description": "…50 MB…", "skipped": True}.

    media.py is imported here (not at module load) so a text DM still works
    if the media helpers fail. The Telegram message handler does not call this.
    """
    from pathlib import Path as _P
    from ai_agent.media import check_telegram_size, detect_mime, telegram_send_selector

    data = None
    name = filename or "file.bin"
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    else:
        p = _P(path_or_bytes)
        if not p.is_file():
            return {"ok": False, "description": "file not found: %s" % p}
        name = filename or p.name
        data = p.read_bytes()
        if not mime:
            mime = detect_mime(p, data)
    err = check_telegram_size(len(data))
    if err:
        return {"ok": False, "description": err, "skipped": True, "error": err}
    if not mime:
        mime = detect_mime(_P(name), data)
    method = telegram_send_selector(mime, _P(name))
    cap = (caption or "")[:1024]
    if method == "sendPhoto":
        return send_photo(chat_id, data, filename=name, caption=cap, mime=mime, token=token)
    if method == "sendVideo":
        return send_video(chat_id, data, filename=name, caption=cap, mime=mime, token=token)
    return send_document(chat_id, data, filename=name, caption=cap, mime=mime, token=token)


# ---------------------------------------------------------------------------
# Allow-list helpers
# ---------------------------------------------------------------------------


def _allowed_id_str():
    """TELEGRAM_ALLOWED_CHAT_ID as a stripped string, or '' if unset."""
    return (telegram_allowed_chat_id() or "").strip()


def _is_allowed(chat_id):
    """True if chat_id matches the configured allow-list id."""
    want = _allowed_id_str()
    if not want:
        return False
    return str(chat_id) == want


# ---------------------------------------------------------------------------
# Shell YES/NO confirm over Telegram
# ---------------------------------------------------------------------------

def _pending_shell():
    """Return pending dict {command, sudo} or None."""
    if not PENDING_SHELL.exists():
        return None
    try:
        data = json.loads(PENDING_SHELL.read_text(encoding="utf-8"))
        cmd = (data.get("command") or "").strip()
        if not cmd:
            return None
        return {"command": cmd, "sudo": bool(data.get("sudo"))}
    except Exception:  # noqa: BLE001
        return None


def _try_delete_message(chat_id, message_id, token=None):
    """Best-effort deleteMessage (e.g. after reading a sudo password). Never raises."""
    if not message_id:
        return
    try:
        api_call(
            "deleteMessage",
            {"chat_id": chat_id, "message_id": message_id},
            token=token,
        )
    except Exception:  # noqa: BLE001
        pass


def maybe_prompt_shell_confirm(chat_id, token=None):
    """If a pending shell exists, ask the user to reply YES or NO (once per cmd)."""
    pend = _pending_shell()
    if not pend:
        _last_shell_prompt.pop(chat_id, None)
        return False
    cmd = pend["command"]
    prev = _last_shell_prompt.get(chat_id)
    if prev == cmd:
        return True  # already prompted for this command
    if pend.get("sudo"):
        msg = (
            "Run sudo shell?\n`%s`\n"
            "Reply YES <password> or Y <password> (or YES alone to try passwordless). "
            "NO to cancel."
        ) % cmd
    else:
        msg = "Run shell?\n`%s`\nReply YES or NO." % cmd
    send_text(chat_id, msg, token=token)
    _last_shell_prompt[chat_id] = cmd
    return True


def _parse_yes_no(text):
    """Parse confirm reply. Returns ('yes', password_or_None) | ('no', None) | (None, None).

    Accepts yes/YES/Yes/Y/y and no/NO/etc case-insensitive.
    For sudo: YES <password> / Y <password> — password is the rest of the line.
    """
    raw = (text or "").strip()
    if not raw:
        return None, None
    parts = raw.split(None, 1)
    word = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else None
    if word in ("yes", "y"):
        return "yes", rest  # rest may be sudo password; None if plain YES
    if word in ("no", "n"):
        return "no", None
    return None, None


def _format_confirm_result(result):
    """Build Telegram reply text from confirm_pending_shell result."""
    if result.get("ok"):
        stdout = (result.get("stdout") or "").strip()
        stderr = (result.get("stderr") or "").strip()
        body = []
        if stdout:
            body.append(stdout)
        if stderr:
            body.append(stderr)
        if body:
            return "Done.\n" + "\n".join(body)
        return "Done."
    return result.get("error") or "Command failed"


def _handle_shell_reply(chat_id, text, token=None, message_id=None):
    """If pending shell exists, handle YES/NO (or remind). Returns True if consumed.

    For sudo pending: YES <password> feeds sudo -S. Plain YES tries sudo -n.
    Password is never stored in memory files; best-effort delete of the TG message.
    """
    pend = _pending_shell()
    if not pend:
        return False
    is_sudo = bool(pend.get("sudo"))
    kind, secret = _parse_yes_no(text)
    if kind == "yes":
        password = None
        had_password = False
        if is_sudo:
            if secret is not None and str(secret) != "":
                password = str(secret)
                had_password = True
                # Best-effort: remove the message that contained the password.
                _try_delete_message(chat_id, message_id, token=token)
            # else: try sudo -n (password=None)
        try:
            result = confirm_pending_shell(password=password)
        finally:
            password = None
            secret = None
        _last_shell_prompt.pop(chat_id, None)
        send_text(chat_id, _format_confirm_result(result), token=token)
        # If sudo failed without a password, hint the YES <password> form.
        if is_sudo and not result.get("ok") and not had_password:
            err_blob = (
                (result.get("stderr") or "")
                + " "
                + (result.get("error") or "")
                + " "
                + (result.get("stdout") or "")
            ).lower()
            if "password" in err_blob or result.get("exit_code") not in (None, 0):
                send_text(
                    chat_id,
                    "Sudo needs a password — reply: YES yourpassword",
                    token=token,
                )
        return True
    if kind == "no":
        cancel_pending_shell()
        _last_shell_prompt.pop(chat_id, None)
        send_text(chat_id, "Shell cancelled.", token=token)
        return True
    # Pending but not YES/NO — require a clear answer first.
    if is_sudo:
        send_text(chat_id, "Pending sudo — reply YES <password> or NO", token=token)
    else:
        send_text(chat_id, "Pending shell — reply YES or NO", token=token)
    maybe_prompt_shell_confirm(chat_id, token=token)
    return True


# Reply enrich + media attach (no Download: http links on Telegram)
# ---------------------------------------------------------------------------


def _tool_file_paths(tools):
    """(basename, path_str) from write_file / write results in a tool trace."""
    out = []
    seen = set()
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        r = t.get("result") or {}
        if not isinstance(r, dict) or not r.get("ok"):
            continue
        n = r.get("name")
        p = r.get("path")
        if not n and p:
            n = str(p).rstrip("/").split("/")[-1]
        if not n:
            continue
        if t.get("name") in ("write_file", "write_bytes") or r.get("download") or p:
            if n not in seen:
                seen.add(n)
                out.append((n, p))
    return out


def enrich_telegram_text(reply, tools=None):
    """Clean reply for Telegram: strip download-link injection / markers.

    Web UI still uses /api/download and [[download:name]] — Telegram sends
    the file itself via sendPhoto/sendVideo/sendDocument instead.
    Does NOT append Download: http links (PUBLIC_BASE_URL unused here).
    """
    import re

    text = "" if reply is None else str(reply)
    # Drop [[download:name]] markers (files are attached separately).
    text = re.sub(r"\[\[download:[^\]]+\]\]", "", text)
    # Drop bare /api/download?name=… and absolute …/api/download?name=… lines.
    text = re.sub(
        r"https?://[^\s]*?/api/download\?name=[^\s\)\]\"\']+",
        "",
        text,
    )
    text = re.sub(r"/api/download\?name=[^\s\)\]\"\']+", "", text)
    # Drop leftover "Download: …" lines (legacy enrich / model habit),
    # including empty "Download:" after the URL was stripped above.
    text = re.sub(r"(?m)^\s*Download:\s*\S*\s*$", "", text)
    text = re.sub(r"(?i)\bDownload:\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def send_tool_media(chat_id, tools=None, caption="", token=None):
    """After write_file / generated media: upload each file to the allow-listed chat.

    Skips files over 50MB with a short error text to the chat.
    """
    from pathlib import Path as _P
    from ai_agent.tools import safe_path

    for name, path_str in _tool_file_paths(tools):
        target = None
        if path_str:
            try:
                target = _P(path_str)
            except Exception:  # noqa: BLE001
                target = None
        if target is None or not target.is_file():
            try:
                target = safe_path(name)
            except Exception:  # noqa: BLE001
                target = None
        if target is None or not target.is_file():
            send_text(chat_id, "Could not attach file: %s" % name, token=token)
            continue
        cap = caption or name
        res = send_media_file(chat_id, target, filename=name, caption=cap, token=token)
        if not res.get("ok"):
            desc = res.get("description") or res.get("error") or "send failed"
            send_text(chat_id, "Telegram attach skipped (%s): %s" % (name, desc), token=token)


def send_result_media(chat_id, result, token=None):
    """Send media from a tasks.run_task result (media / media_list)."""
    items = []
    if result.get("media_list"):
        items = list(result["media_list"])
    elif result.get("media"):
        items = [result["media"]]
    for m in items:
        if not isinstance(m, dict) or not m.get("ok"):
            # media dict from save_generated_bytes always has ok when saved
            if not isinstance(m, dict) or not (m.get("path") or m.get("name")):
                continue
        path = m.get("path")
        name = m.get("name") or "media.bin"
        mime = m.get("mime") or ""
        cap = name
        if path:
            res = send_media_file(
                chat_id, path, filename=name, caption=cap, mime=mime, token=token
            )
        else:
            continue
        if not res.get("ok"):
            desc = res.get("description") or res.get("error") or "send failed"
            send_text(chat_id, "Telegram attach skipped (%s): %s" % (name, desc), token=token)


# ---------------------------------------------------------------------------
# Message handling + short history
# ---------------------------------------------------------------------------


def _history_for(chat_id):
    """Return (and keep) the in-memory history list for this chat."""
    with _lock:
        if chat_id not in _histories:
            _histories[chat_id] = []
        return _histories[chat_id]


def _append_history(chat_id, role, content):
    """Append one turn and trim to _HISTORY_MAX entries."""
    hist = _history_for(chat_id)
    hist.append({"role": role, "content": content})
    if len(hist) > _HISTORY_MAX:
        del hist[: len(hist) - _HISTORY_MAX]


def _download_telegram_file(file_id, token=None):
    """Download a Telegram file by file_id → (bytes, path_hint) or (None, err)."""
    meta = api_call("getFile", {"file_id": file_id}, token=token)
    if not meta.get("ok"):
        return None, meta.get("description") or "getFile failed"
    fpath = ((meta.get("result") or {}).get("file_path") or "").lstrip("/")
    if not fpath:
        return None, "no file_path"
    tok = token or telegram_bot_token()
    url = "https://api.telegram.org/file/bot%s/%s" % (tok, fpath)
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read(), fpath
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def handle_message(update, token=None):
    """Process one Telegram update (private text only)."""
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    chat = msg.get("chat") or {}
    # v1: private DMs only
    if (chat.get("type") or "") != "private":
        return
    chat_id = chat.get("id")
    if chat_id is None:
        return

    # Allow-list: if unset, tell them their chat id and do not run the agent.
    if not _allowed_id_str():
        send_text(
            chat_id,
            "Your Telegram chat id is: %s\n"
            "Add this to .env and restart:\n"
            "TELEGRAM_ALLOWED_CHAT_ID=%s" % (chat_id, chat_id),
            token=token,
        )
        return

    if not _is_allowed(chat_id):
        # Quietly ignore unknown chats (no info leak).
        return

    # Photos do not enter Vision and are not downloaded. Text chat only.
    if msg.get("photo"):
        send_text(chat_id, "Telegram is text chat only for now.", token=token)
        return

    # Text messages only past this point (stickers / video / documents ignored).
    text = msg.get("text")
    if text is None:
        return
    text = str(text).strip()
    if not text:
        return

    # Shell confirm takes priority over normal chat.
    if _handle_shell_reply(chat_id, text, token=token, message_id=msg.get("message_id")):
        return

    provider = telegram_provider()
    model = telegram_model()
    hist = list(_history_for(chat_id))
    # Every text DM goes to run_chat. /image /video /vision stay chat text.
    # Do not start image/video/vision tasks, and do not attach files.
    result = run_chat(text, provider=provider, model=model, history=hist)
    reply = (result.get("reply") or "").strip()
    if not result.get("ok"):
        err = result.get("error") or "chat failed"
        reply = reply or ("Error: %s" % err)
    if not reply:
        reply = "(No reply)"
    # Drop download-link markers. Reply is text only (no file upload).
    reply = enrich_telegram_text(reply, result.get("tools") or [])
    send_text(chat_id, reply, token=token)
    _append_history(chat_id, "user", text)
    _append_history(chat_id, "assistant", reply)
    # Persist to chats/YYYY_MM/YYYY_MM_DD.md (same store as the web UI).
    try:
        append_exchange(text, reply)
    except Exception:  # noqa: BLE001
        pass

    # After the chat reply, prompt for shell confirm if tools queued one.
    maybe_prompt_shell_confirm(chat_id, token=token)


# ---------------------------------------------------------------------------
# Long-polling loop
# ---------------------------------------------------------------------------


def _poll_loop(token):
    """Long-poll getUpdates forever (daemon thread). Uses offset to ack updates."""
    offset = None
    # Verify token once (mask in any print — we only print a short status).
    me = api_call("getMe", token=token)
    if not me.get("ok"):
        print("Telegram: getMe failed — check TELEGRAM_BOT_TOKEN")
        return
    while True:
        params = {"timeout": _POLL_TIMEOUT}
        if offset is not None:
            params["offset"] = offset
        try:
            # getUpdates with long poll — pass timeout in params.
            res = api_call("getUpdates", params, token=token)
        except Exception:  # noqa: BLE001
            time.sleep(3)
            continue
        if not res.get("ok"):
            # Transient network / 429 — back off briefly.
            time.sleep(3)
            continue
        for upd in res.get("result") or []:
            uid = upd.get("update_id")
            if uid is not None:
                offset = uid + 1
            try:
                handle_message(upd, token=token)
            except Exception as e:  # noqa: BLE001
                # Never crash the poller on one bad message.
                print("Telegram: handle error: %s" % e)


def start_telegram_thread():
    """Start the Telegram poller as a daemon thread if a token is set.

    No token → no-op (safe to call always). Returns True if started.
    """
    token = telegram_bot_token()
    if not token:
        return False
    t = threading.Thread(
        target=_poll_loop,
        args=(token,),
        name="telegram-poll",
        daemon=True,
    )
    t.start()
    return True
