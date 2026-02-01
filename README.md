# RIT Detection Baseline Purple Agent

This document lists **all environment variables read by `agent.py`** and how they affect provider selection, model choice, and request behavior.

---

## Provider auto-detection (priority order)

The agent checks API keys in this exact order and picks the **first** one it finds:

1. `ANTHROPIC_API_KEY` → Anthropic
2. `GOOGLE_API_KEY` → Google Gemini
3. `XAI_API_KEY` → xAI Grok
4. `OPENAI_API_KEY` **or** `LLM_API_KEY` → OpenAI-compatible (OpenAI / OpenRouter / other compatible endpoints)

If **no key is found**, the agent runs in **test mode** and returns the default label `"Error - no LLM configured"` for every request.

---

## Supported environment variables

### API keys (pick one provider)
- **`ANTHROPIC_API_KEY`**
  - What it does: Enables **Anthropic** provider.
  - Default: *(unset)*
  - Notes: If set, it overrides all other keys due to priority order.

- **`GOOGLE_API_KEY`**
  - What it does: Enables **Google Gemini** provider.
  - Default: *(unset)*

- **`XAI_API_KEY`**
  - What it does: Enables **xAI Grok** provider.
  - Default: *(unset)*

- **`OPENAI_API_KEY`**
  - What it does: Enables **OpenAI-compatible** provider.
  - Default: *(unset)*
  - Notes: Used only if no Anthropic/Google/xAI key is present.

- **`LLM_API_KEY`**
  - What it does: Fallback API key for OpenAI-compatible provider if `OPENAI_API_KEY` is not set.
  - Default: *(unset)*
  - Notes: Same behavior as `OPENAI_API_KEY` for this agent.

---

### Model selection
- **`LLM_MODEL`**
  - What it does: Overrides the model name used by the selected provider.
  - Default (depends on provider):
    - Anthropic: `claude-sonnet-4-20250514`
    - Google: `gemini-2.5-pro`
    - xAI: `grok-4`
    - OpenAI-compatible: `gpt-4o`
  - Example:
    - `LLM_MODEL="some-model-name"`

---

### Base URL / endpoint (OpenAI-compatible only)
These only matter when the selected provider is **OpenAI-compatible**.

- **`OPENAI_API_BASE`**
  - What it does: Sets a custom base URL for the OpenAI-compatible client.
  - Default: *(unset → library default base URL is used)*
  - Example:
    - `OPENAI_API_BASE=https://your-openai-compatible-host/v1`

- **`LLM_API_BASE`**
  - What it does: Fallback base URL if `OPENAI_API_BASE` is not set.
  - Default: *(unset)*
  - Example:
    - `LLM_API_BASE=https://your-openai-compatible-host/v1`

**Important:** For xAI, the base URL is hardcoded to `https://api.x.ai/v1` in the code; `OPENAI_API_BASE/LLM_API_BASE` do **not** override it.

---

### Timeouts
- **`LLM_TIMEOUT`**
  - What it does: Sets request timeout in **seconds** (float).
  - Default: `200.0`
  - Example:
    - `LLM_TIMEOUT=120`

---

## Example input in the benchmark scenario.toml

[[participants]]
agentbeats_id = "019c1788-faf4-7170-9953-a84321132e16"
name = "agent"
env = {LLM_API_KEY = "${LLM_API_KEY}", LLM_API_BASE = "https://api.tokenfactory.nebius.com/v1/", LLM_MODEL = "deepseek-ai/DeepSeek-V3.2"}

---
