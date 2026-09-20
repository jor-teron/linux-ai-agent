# AI Agent

Version: see `VERSION` (e.g. `0.40 (40)`).

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

Same keys cover Chat / Image / Vision / Video when that provider allows it. Never commit `.env`.

## Notes

- Browser default: `127.0.0.1:9191`. For LAN/Tailscale: `HOST=0.0.0.0` in `.env`.
- Jail: `~/ai-workspace` (`workspace/`, `user/`, `memory/`, `chats/`, …).
- Chat logs: `~/ai-workspace/chats/YYYY_MM/YYYY_MM_DD.md` (disk only; API uses `HISTORY_TURNS`).
- API limits (optional in `.env`): `TPM`, `RPM`, `RPD`, `API_DELAY_SEC` (`0` = no limit).
- Stop: `systemctl --user stop linux-ai-agent` · Manual: `./run.sh`
