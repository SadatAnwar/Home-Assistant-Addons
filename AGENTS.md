# Agent Safety Rules for this Repository

This file applies to ALL AI agents (Claude, Copilot, Cursor, etc.) working in this repo.

## ⚠️ CRITICAL: Home Assistant Supervisor API Exposes Credentials in Plaintext

The HA Supervisor API **does not redact secrets**. Querying addon details returns passwords, API keys, and tokens in plaintext — which then get included in AI conversation context and transmitted to the AI provider.

**NEVER call:**
- `supervisor/api` → `/addons/{slug}/info`
- `supervisor/api` → `/addons/{slug}/options`
- Any Supervisor endpoint that returns addon configuration

**To diagnose a failing addon:**
- Use `/addons/{slug}/logs` (returns log output only, no config)
- Or ask the user to paste logs manually from: Settings → Add-ons → [addon] → Log tab

This rule exists because an AI agent called `/addons/{slug}/info` to debug a failing addon, which returned the user's Govee email, password, and API key in plaintext. The user had to revoke and reset all credentials.

---

## General Secrets Rules

- Never read or print `.env` file contents
- Never log or output values of environment variables that may contain secrets
- Never include token/credential values in any output, even partially
- If you need to verify a credential is set, check that the variable exists (`HA_TOKEN` is set) — do not read its value
