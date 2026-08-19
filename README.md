# Secure MCQ Generator

Deploy this directory as the Render repository root. Add every secret in the Render dashboard, never in Git.

Required configuration: `MONGO_URI`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USER_IDS`, `WEBHOOK_URL`, `GCP_SERVICE_ACCOUNT_JSON`, one Gemini key, and at least one MCQ-provider key. Render generates `TELEGRAM_WEBHOOK_SECRET` from `render.yaml`.

`TELEGRAM_ALLOWED_USER_IDS` contains numeric Telegram user IDs separated by commas. The Google Sheet must be shared with the `client_email` inside the service-account JSON as **Editor**.

Provider selection is automatic: only keys present in Render are enabled. Defaults can be overridden with `CEREBRAS_MODEL`, `GROQ_MODEL`, `MISTRAL_MODEL`, `SAMBANOVA_MODEL`, or `OPENROUTER_MODEL`.

Telegram commands: `/set_pdf <Google Drive URL>`, `/set_sheet <Google Sheets URL>`, `/start`, `/pause`, `/reset <PDF page>`, `/status`, and `/clear_and_restart CONFIRM ALL`. The last command permanently clears the first Sheet tab, resets its jobs, and processes the whole PDF from page 1.
