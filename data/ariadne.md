# Ariadne POC

Interactive social-manipulation prototype. The **server owns objective truth**. NPC memory is local for now; **dialogue uses OpenAI**. Backboard is configured for a later swap.

## Clarifications

| Topic | Current choice |
|---|---|
| Dialogue LLM | OpenAI (`LLM_PROVIDER=openai`) |
| Memory store | Local `FakeMemoryAgent` (same adapter interface) |
| Backboard key | Saved in `.env` for later — not used yet |
| Client | Clear guided text UI in `game/` |

> Security: API keys live only in `.env` (gitignored). Because keys were pasted in chat, **rotate them** when you can.

## What you can play

**Full campaign (Acts I–VI)** in a visual scene client:

1. **THE TOUR** — a Backboard open house; win Vale's referral (social / evidence / dream)
2. **SUPERDAY** — Daniel's check-in + Arthur Bell's final round
3. **ONBOARDING** — collect your materials from the Archive; something is wrong
4. **THE JOB** — the Ambassador role is the game itself; Daniel admits he's a hacked debug backdoor
5. **WHO YOU ARE** — your own Anchor dream, the Transfer fragment, the false hospital waking
6. **THE CHOICE** — Reboot vs Transfer

UI: clickable room stage, character hotspots, dialogue overlay, dream layer, memory meter, ending screen.

Full route with spoilers: [`guides/GUIDE_v3.md`](guides/GUIDE_v3.md) (also linked in the in-game HUD).

---

## Setup

```powershell
cd c:\Users\J\Documents\GitHub\ariadne
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".\server[dev]"
copy .env.example .env
# Put OPENAI_API_KEY in .env (already done if you followed the agent setup)
```

## Run the game

```powershell
cd c:\Users\J\Documents\GitHub\ariadne
.\.venv\Scripts\Activate.ps1
cd server
python -m uvicorn ariadne.main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/**

Follow the on-screen **What to do next** checklist.

## Test

Tests force `LLM_PROVIDER=fake` (no network LLM calls):

```powershell
cd c:\Users\J\Documents\GitHub\ariadne
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

## Switch providers later

In `.env`:

```env
LLM_PROVIDER=openai   # current
# LLM_PROVIDER=backboard  # future
```
