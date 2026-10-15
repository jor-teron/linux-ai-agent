Versioning: `0.42 (42)` — two-number sub-version + change # (`0.XX (N)`, not three-part semver). Counter +1 every release (never resets).

- One-line install via `install.sh` (detects apt-get/dnf/yum/pacman/zypper/apk; most Linux distros)
- Bubblewrap optional in install (`INSTALL_BWRAP=0/1` or prompt); install continues if bwrap unavailable

# Features

## Now
- **Tasks** — header **Task** dropdown: Chat | Image | Video | Vision; **Mode** Online | Local (never grayed out); labels above Task/Mode/Provider/Model
- **Short LED** — status text `OK` / `No key` / `No model` (no provider/model name)
- **Image / Video / Vision** — Online APIs where available (Gemini image + Veo best-effort, OpenAI Images, multimodal vision); Local best-effort / clear unsupported; saves under `workspace/generated/`
- **API rate limits** — `.env` `TPM=0` (off), `RPM=14`, `RPD=500`, `API_DELAY_SEC=5`. `0` turns that knob off. Online LLM/media HTTP only (local Ollama / llama.cpp skipped). Waits up to ~60s for delay/RPM; daily RPD is a hard `rate limited` error (no retry storm). One lock shared by Telegram + web.
- **Telegram text-only** — DMs go to `run_chat` (YES/NO shell confirm kept). Photos: "Telegram is text chat only for now." No `sendPhoto` / `sendVideo` / `sendDocument` from the message handler. Web Image/Video still works.
- **PUBLIC_BASE_URL** — still for web/LAN absolute `/api/download`; Telegram does not rely on links
- **Chat history Markdown** — persist under `chats/YYYY_MM/YYYY_MM_DD.md` (sibling of `memory/` under AI_HOME; disk archive); `GET /api/chat/history`; web UI restores on refresh; `HISTORY_TURNS` caps model context (not the full day file)
- **.env header** — `.env.example` starts with `env_version` matching `VERSION` (no runtime outdated warning)
- **topics/** — memory notes in `memory/topics/` (was `topic/`); dest `topics:<name>`; legacy folder migrates once
- **ensure_ws tree** — creates `chats/` (+ current month), `workspace/`, `user/`, `trash/`, `test/`, `memory/{date,topics}/` and touches `user.md` / `assistant.md` / `session.md`
- **Memory date dedupe** — `_dated_line` strips a leading date the model may prepend
- **Package layout** — `ai_agent/` package; `media.py` + `tasks.py` helpers; `ai_agent/ui/` static files
- **Editable prompts** — `prompt_chat` / `prompt_tools` / `prompt_image` / `prompt_video` / `prompt_vision`
- **Token trim** — `TOOLS_DEFAULT=0`; UI **Tools** checkbox; keyword boost; `HISTORY_TURNS=10`
- **Online / Local modes** — Mode before Provider; catalogs for all tasks
- **Multi-provider chat (online)** — gemini, openai, xai, anthropic, deepseek, openrouter, deepinfra; media: segmind/freetheai/pollinations (image), agnes (video), zhipu/groq (vision)
- **Local providers** — Ollama + llama.cpp; OpenAI-compatible; no API key
- **Status LED** — green solid when ready; red blink when not; `STATUS_BLINK_MS`
- **Workspace folders** — jail = `~/ai-workspace` with `chats/`, `memory/` (date/topics + md), `workspace/`, `workspace/generated/`, `test/`, `trash/`, `user/`
- **Light mode default** — dark available; optional `UI_LIGHT_*` hex overrides
- **Update button** — `POST /api/update`
- Shell blocklist + sudo Confirm; bubblewrap freehand when available
- Sticky confirm bar; Enter sends; upload/download; web search / reminders / jobs
- **Thinking (v0.45)** — qwen3:1.7b in Local list; folded "Thinking" box; Think on/off toggle
- **Limits toggle (v0.46)** — auto off in Local, on in Online; off skips waits/caps but still counts
- **More Ollama models (v0.47)** — qwen3.5 0.8b/2b, deepseek-r1:1.5b, smallthinker:3b, phi4-mini-reasoning:3.8b (thinking); qwen2.5-coder:3b (coder)
- **Streaming replies (v0.48)** — Stream toggle; live text + thinking for OpenAI-compat (Ollama, llama.cpp, Groq, OpenRouter…) with Tools off; others fall back
- UI port **9191**; Telegram YES/NO / sudo password

## Later
7. Command allow-list
8. Optional UI install/pull helpers for Ollama models
9. **No Docker** — not pursuing; keep Bubblewrap.
