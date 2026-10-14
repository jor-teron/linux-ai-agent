# AI Agent

Version: see `VERSION` (e.g. `0.42 (42)`).

Tiny browser chat agent for Linux. Talk in the browser or Telegram. Use cloud APIs (Online) or models on your PC (Local).

Install page: https://jor-teron.github.io/linux-ai-agent/

## Task

In the header: **Task → Mode → Provider → Model**.

| Task | What it does |
|------|----------------|
| **Chat** | Normal chat. Optional tools (shell, files). |
| **Image** | Make an image from a prompt (when the provider supports it). |
| **Video** | Make a video when the API allows it. |
| **Vision** | Ask about an image you upload (web UI). |

**Mode:** Online (API key) or Local (Ollama / llama.cpp). Both stay available.

## Install

One line (Linux):

```bash
curl -fsSL https://raw.githubusercontent.com/jor-teron/linux-ai-agent/main/install.sh | bash
```

Then:

```bash
nano ~/linux-ai-agent/.env
systemctl --user restart linux-ai-agent
```

Open http://127.0.0.1:9191

Or clone the repo and run `./install.sh`. Update later with the **Update** button or the install line again (`.env` is kept).

## Telegram (optional)

Text chat from your phone (PC stays on).

1. In Telegram, open **@BotFather** → `/newbot` → copy the token.
2. Put it in `.env` as `TELEGRAM_BOT_TOKEN=…` then restart.
3. Message your bot once — it replies with your **chat id**.
4. Put that id in `.env` as `TELEGRAM_ALLOWED_CHAT_ID=…` and restart again.

Only your chat works. Shell confirm: reply **YES** or **NO**. For sudo: **YES yourpassword**.

Telegram is **text only** for now (no photo/file sends from the bot).

## Dependencies

- Python **3.8+** (`python3`)
- `git` (Linux install + Update button)
- `bubblewrap` (`bwrap`) — optional; installer asks **[Y/n]**
- At least one Online API key, or a Local runtime (Ollama / llama.cpp)

## API keys

| Provider | Env var | Get key |
|----------|---------|---------|
| gemini | `GEMINI_API_KEY` | https://aistudio.google.com/apikey |
| openai | `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| xai (Grok) | `XAI_API_KEY` | https://console.x.ai/ |
| anthropic | `ANTHROPIC_API_KEY` | https://console.anthropic.com/ |
| deepseek | `DEEPSEEK_API_KEY` | https://platform.deepseek.com/ |
| openrouter | `OPENROUTER_API_KEY` | https://openrouter.ai/keys |
| deepinfra | `DEEPINFRA_API_KEY` | https://deepinfra.com/dash/api_keys |
| segmind (image) | `SEGMIND_API_KEY` | https://www.segmind.com/ |
| freetheai (image) | `FREETHEAI_API_KEY` | https://freetheai.xyz / api.freetheai.xyz |
| pollinations (image) | *(none — keyless)* | https://pollinations.ai |
| agnes (video) | `AGNES_API_KEY` | https://agnes-ai.com |
| zhipu (vision) | `ZHIPU_API_KEY` | https://open.bigmodel.cn |
| groq (vision) | `GROQ_API_KEY` | https://console.groq.com/keys |

Same keys cover Chat / Image / Vision / Video when that provider allows it. Never commit `.env`.

## Notes

- Browser default: `127.0.0.1:9191`. For LAN/Tailscale: `HOST=0.0.0.0` in `.env`.
- Jail: `~/ai-workspace` (`workspace/`, `user/`, `memory/`, `chats/`, …).
- Chat logs: `~/ai-workspace/chats/YYYY_MM/YYYY_MM_DD.md` (disk only; API uses `HISTORY_TURNS`).
- API limits (optional in `.env`): `TPM`, `RPM`, `RPD`, `API_DELAY_SEC` (`0` = no limit).
- Stop: `systemctl --user stop linux-ai-agent` · Manual: `./run.sh`

## About this project

### How it started
It began as a small personal agent called debian-ai-agent, running on Jor's Debian 11 PC. It grew step by step, was renamed linux-ai-agent in 0.2.0, and was split into the `ai_agent/` package later on.

### What the user needed
A simple AI helper on their own Linux PC that a non-technical person can use. It needed a local web chat page, phone access through Telegram, a choice of free or paid cloud APIs, offline models when there's no key, and real safety around shell and sudo.

### What it does
It runs a small web server on the PC. You chat in the browser or on Telegram, and the agent can answer, make images and videos, read uploaded images, and (if you allow it) run commands and work with files inside `~/ai-workspace`.

### Features
- Task picker: Chat, Image, Video, Vision
- Online (cloud API) or Local (Ollama / llama.cpp) mode
- Many providers: Gemini, OpenAI, xAI, Anthropic, DeepSeek, OpenRouter, DeepInfra, plus media providers (Segmind, FreeTheAi, Pollinations, Agnes, Zhipu, Groq)
- Status light (OK / No key / No model)
- Shell and sudo always need a Confirm (web) or YES / NO (Telegram)
- Optional Bubblewrap sandbox
- Memory of facts, plus daily Markdown chat logs
- API rate limits set in `.env`, with a Limits switch (auto off for Local)
- Soft light and dark themes, plus an Update button
- Tokens/sec under each reply; folded "Thinking" box and a Think on/off switch (e.g. qwen3:1.7b)

### Methods and techniques
- One install line, and `.env` is kept on every update
- Tool schemas are only sent when needed, and short history (`HISTORY_TURNS`) keeps token use low
- Each Task only lists models suited to it, so chat models don't show up for Image, Video or Vision
- A workspace jail keeps file access inside `~/ai-workspace`
- Prompts are plain text files, so they're easy to edit

### Technology
Python 3 (standard library), a plain HTML/CSS/JS web page, systemd user service, bash installer, git, optional Bubblewrap, the Telegram Bot API, and OpenAI-compatible and Gemini HTTP APIs.

### Future plans
- Login protection when the server is open to the LAN
- Download links that work with files in subfolders
- Safer chat log formatting
- Telegram photos and files again (maybe as documents)
- Windows version (zip + `run.bat`)
- Clearer API-limit settings
- InferencePort local video proxy

### Ending notes
This is a personal project built in small, careful releases (version format `0.XX (N)`). Changes are discussed first and only built or pushed after Jor says go.
