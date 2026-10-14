"""
HTTP UI and API handler (stdlib http.server).

Serves the chat page from ai_agent/ui/ (via thin ui.py) and JSON API routes
(online/local catalog, Task chat|image|video|vision, status LED readiness,
optional UI_LIGHT_* theme overrides, and Markdown chat history).
Chat/media dispatch goes through tasks.run_task → brain when needed.

Imports from: ai_agent.config, tools, brain, ui (static page loader).
Used by: run (ThreadingHTTPServer(..., Handler)).
"""
import base64
import json
import socket
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ai_agent.config import (
    HOST,
    PENDING_SHELL,
    PORT,
    WORKSPACE,
    allow_sudo,
    app_version,
    catalog_for_api,
    default_provider,
    keys_status,
    provider_ready,
    readiness_status,
    status_blink_ms,
    ui_light_theme,
)
from ai_agent.tools import (
    apply_update,
    cancel_pending_shell,
    confirm_pending_shell,
    safe_path,
    shell_freehand,
    shell_net_enabled,
    shell_sandbox_mode,
    tool_reminder_list,
    tool_write_bytes,
    tool_write_file,
)
from ai_agent.tasks import run_task
from ai_agent.chat_history import append_exchange, history_for_api
from ai_agent.ui import load_index, load_static


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    """One object per HTTP request. Routes GET/POST to the methods below."""

    def log_message(self, fmt, *args):  # quieter terminal
        """Print a one-line access log (default http.server is noisier)."""
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def _send(self, code, body, content_type):
        """Write a raw HTTP response (status, headers, body)."""
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        """Send a dict/list as JSON (UTF-8)."""
        raw = json.dumps(obj).encode("utf-8")
        self._send(code, raw, "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        """Handle GET: page, health, models, reminders, download, pending, chat history."""
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            # Chat page assembled from ai_agent/ui/ (thin loader in ui.py)
            self._send(200, load_index().encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/ui/style.css":
            body, ctype = load_static("style.css")
            self._send(200 if body is not None else 404, body or b"not found", ctype)
            return
        if path == "/ui/app.js":
            body, ctype = load_static("app.js")
            self._send(200 if body is not None else 404, body or b"not found", ctype)
            return
        # Alias /static/* → same files (in case HTML is edited to use /static/)
        if path in ("/static/style.css", "/static/app.js"):
            name = path.rsplit("/", 1)[-1]
            body, ctype = load_static(name)
            self._send(200 if body is not None else 404, body or b"not found", ctype)
            return
        if path == "/api/health":
            # Keys, readiness, version, workspace, due reminders (UI LED + banner)
            rem = tool_reminder_list()
            due = rem.get("due") or []
            keys = keys_status()
            # Optional ?provider=&model= so the LED can reflect the UI selection.
            qs = parse_qs(urlparse(self.path).query)
            sel_p = (qs.get("provider") or [""])[0].strip() or None
            sel_m = (qs.get("model") or [""])[0].strip() or None
            ready_blob = readiness_status(sel_p, sel_m)
            selected = ready_blob.get("selected")
            if selected is None and sel_p:
                selected = provider_ready(sel_p, sel_m)
            self._json(
                200,
                {
                    "ok": True,
                    "keys": keys,
                    "api_key_set": any(keys.values()),
                    "providers_ready": ready_blob.get("providers_ready") or {},
                    "selected": selected,
                    "status_blink_ms": status_blink_ms(),
                    "ui_light": ui_light_theme(),
                    "workspace": str(WORKSPACE),
                    "pending_shell": PENDING_SHELL.exists(),
                    "version": app_version(),
                    "host": HOST,
                    "port": PORT,
                    "default_provider": default_provider(),
                    "shell_sandbox": shell_sandbox_mode(),
                    "shell_freehand": shell_freehand(),
                    "shell_net": shell_net_enabled(),
                    "allow_sudo": allow_sudo(),
                    "due_reminders": [
                        {"id": r.get("id"), "text": r.get("text"), "due": r.get("due")} for r in due
                    ],
                },
            )
            return
        if path == "/api/models":
            # Modes + tasks + providers + models; optional ?task= filters models.
            # ui_light carries optional UI_LIGHT_* .env hex overrides for light theme.
            qs = parse_qs(urlparse(self.path).query)
            task = (qs.get("task") or [""])[0].strip() or None
            blob = catalog_for_api(task=task)
            blob["ui_light"] = ui_light_theme()
            self._json(200, blob)
            return
        if path == "/api/reminders":
            # Same list the model sees (due + upcoming)
            self._json(200, tool_reminder_list())
            return
        if path == "/api/download":
            # Serve a workspace file by basename (?name=foo.txt)
            qs = parse_qs(urlparse(self.path).query)
            name = Path((qs.get("name") or [""])[0]).name
            if not name:
                self._json(400, {"ok": False, "error": "name required"})
                return
            try:
                target = safe_path(name)
            except PermissionError as e:
                self._json(403, {"ok": False, "error": str(e)})
                return
            if not target.is_file():
                # Image/Video tasks save under workspace/generated/<basename>.
                alt = WORKSPACE / "workspace" / "generated" / name
                try:
                    alt_res = alt.resolve()
                    alt_res.relative_to(WORKSPACE.resolve())
                    if alt_res.is_file():
                        target = alt_res
                except (ValueError, OSError):
                    pass
            if not target.is_file():
                self._json(404, {"ok": False, "error": "file not found"})
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % name)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/pending":
            # Sticky confirm bar polls this: {command, sudo} or {command: null}
            if PENDING_SHELL.exists():
                self._json(200, json.loads(PENDING_SHELL.read_text(encoding="utf-8")))
            else:
                self._json(200, {"command": None})
            return
        if path == "/api/chat/history":
            # Parsed turns from today's chats/YYYY_MM/YYYY_MM_DD.md (UI refresh).
            self._json(200, history_for_api())
            return
        self._json(404, {"ok": False, "error": "Not found"})

    def do_POST(self):  # noqa: N802
        """Handle POST: confirm/cancel shell, upload, update, chat."""
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""

        if path == "/api/confirm":
            # User clicked Confirm — optional JSON {password} for sudo (never logged)
            password = None
            if raw:
                try:
                    body = json.loads(raw.decode("utf-8") or "{}")
                    if isinstance(body, dict) and body.get("password") is not None:
                        password = str(body.get("password"))
                except json.JSONDecodeError:
                    password = None
            try:
                self._json(200, confirm_pending_shell(password=password))
            finally:
                password = None
            return
        if path == "/api/cancel":
            # User clicked Cancel — drop the queued command
            self._json(200, cancel_pending_shell())
            return
        if path == "/api/update":
            # git pull --ff-only then schedule systemctl --user restart
            self._json(200, apply_update())
            return
        if path == "/api/upload":
            # Browser file picker: text as UTF-8, binaries as base64
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json(400, {"ok": False, "error": "Invalid JSON"})
                return
            name = Path(data.get("name") or "").name
            if not name:
                self._json(400, {"ok": False, "error": "name required"})
                return
            enc = (data.get("encoding") or "text").lower()
            content = data.get("content") or ""
            if enc == "base64":
                try:
                    blob = base64.b64decode(content)
                except Exception as e:  # noqa: BLE001
                    self._json(400, {"ok": False, "error": "bad base64: %s" % e})
                    return
                if len(blob) > 2_000_000:
                    self._json(400, {"ok": False, "error": "file too large (max ~2MB)"})
                    return
                result = tool_write_bytes(name, blob)
            else:
                result = tool_write_file(name, content)
            self._json(200 if result.get("ok") else 400, result)
            return
        if path != "/api/chat":
            self._json(404, {"ok": False, "error": "Not found"})
            return
        # Chat / Image / Video / Vision: message + optional task / image → tasks.run_task
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"ok": False, "error": "Invalid JSON"})
            return
        task = (data.get("task") or "chat").strip().lower() or "chat"
        message = (data.get("message") or "").strip()
        # Vision may send only an image + empty message (default prompt inside tasks).
        if not message and task not in ("vision",):
            self._json(400, {"ok": False, "error": "message required"})
            return
        # Optional tools flag from UI (bool). None → brain resolves default/keywords.
        tools_flag = data.get("tools", None)
        if tools_flag is not None:
            tools_flag = bool(tools_flag)
        # Optional Think toggle (v0.45). None → leave the model's default.
        think_flag = data.get("think", None)
        if think_flag is not None:
            think_flag = bool(think_flag)
        # Optional Limits toggle (v0.46). None → .env limits as before.
        limits_flag = data.get("limits", None)
        if limits_flag is not None:
            limits_flag = bool(limits_flag)
        result = run_task(
            task=task,
            message=message,
            provider=data.get("provider"),
            model=data.get("model"),
            history=data.get("history") or [],
            use_tools=tools_flag,
            image=data.get("image"),
            image_name=(data.get("image_name") or "").strip(),
            think=think_flag,
            limits=limits_flag,
        )
        # Persist completed exchange to chats/YYYY_MM/YYYY_MM_DD.md (source of truth).
        # Do this for both ok and error replies so refresh still shows the turn.
        reply = (result.get("reply") or "").strip()
        if not reply and not result.get("ok"):
            reply = (result.get("error") or "chat failed").strip()
        try:
            label = message or ("[%s]" % task)
            append_exchange(label, reply or "(No reply)")
        except Exception:  # noqa: BLE001
            pass
        self._json(200 if result.get("ok") else 400, result)



