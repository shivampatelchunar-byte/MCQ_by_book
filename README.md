# Secure MCQ Generator

Deploy this directory as the Render repository root. Add every secret in the Render dashboard, never in Git.

Required configuration: `MONGO_URI`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USER_IDS`, `WEBHOOK_URL`, `GCP_SERVICE_ACCOUNT_JSON`, one Gemini key, and at least one MCQ-provider key. Render generates `TELEGRAM_WEBHOOK_SECRET` from `render.yaml`.

`TELEGRAM_ALLOWED_USER_IDS` contains numeric Telegram user IDs separated by commas. The Google Sheet must be shared with the `client_email` inside the service-account JSON as **Editor**.

Provider selection is automatic: only keys present in Render are enabled. Defaults can be overridden with `CEREBRAS_MODEL`, `GROQ_MODEL`, `MISTRAL_MODEL`, `SAMBANOVA_MODEL`, or `OPENROUTER_MODEL`.

Telegram commands: `/set_pdf <Google Drive URL>`, `/set_sheet <Google Sheets URL>`, `/start`, `/pause`, `/reset <PDF page>`, `/status`, and `/clear_and_restart CONFIRM ALL`. The last command permanently clears the first Sheet tab, resets its jobs, and processes the whole PDF from page 1.

## Quota-aware OCR configuration

The worker extracts embedded text from digital PDFs with PyMuPDF first. Gemini vision is called only for scanned/image-only pages, reducing OCR quota use substantially.

Preferred Gemini configuration uses a comma-separated list of credentials and a configured list of models:

```text
GEMINI_API_KEYS=key_one,key_two
GEMINI_OCR_MODELS=gemini-3.6-flash
```

The legacy `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, and `GEMINI_MODEL` names also work. A 429 puts the affected credential/model in a MongoDB-backed cooldown; the worker waits for the provider's retry period rather than repeatedly consuming failed requests. API keys in the same Google project do not increase a quota that is measured per project and model.

See [DEPLOYMENT.md](DEPLOYMENT.md) for Render variables, supported MCQ provider model lists, and operational notes.
