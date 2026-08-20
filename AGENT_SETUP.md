# Agent setup

## Required Render settings

Set only model IDs which your Gemini project actually exposes:

```text
GEMINI_OCR_MODELS=gemini-3.6-flash
# Legacy fallback supported:
GEMINI_MODEL=gemini-3.6-flash
DAILY_QUOTA_COOLDOWN_SECONDS=21600
```

Do not use guessed model IDs. `GEMINI_OCR_MODELS` overrides model discovery.

## Natural-language Telegram use

Send the bot both links in one normal message, e.g.

`New book set karo: DRIVE_PDF_URL and new sheet: GOOGLE_SHEET_URL`

Then send `start karo`, `status batao`, or `pause karo`.

The sheet must be shared with the service account as Editor. The worker skips pages detected as Contents, Foreword, Acknowledgements, Dedication, Preface, copyright, etc. It does not skip an actual chapter merely because the word appears later on a normal page.

## Provider limits

Use API keys only under provider terms. Multiple keys from the same quota project do not increase a per-project quota. This worker uses configured providers as normal failover and observes provider cooldowns.
