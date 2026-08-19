# Render deployment and quota configuration

## Start command

Use exactly one Uvicorn worker because this service includes a background worker:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```

## Required Render environment variables

```text
MONGO_URI
TELEGRAM_BOT_TOKEN
TELEGRAM_WEBHOOK_SECRET
WEBHOOK_URL
TELEGRAM_ALLOWED_USER_IDS
GCP_SERVICE_ACCOUNT_JSON
GEMINI_API_KEYS
GEMINI_OCR_MODELS
```

`GEMINI_API_KEYS` is a comma-separated list of API keys you are authorised to use. Legacy names `GEMINI_API_KEY_1` and `GEMINI_API_KEY_2` are also supported.

Example:

```text
GEMINI_OCR_MODELS=gemini-3.6-flash,gemini-3.5-flash
OCR_DPI=120
WORKER_LEASE_SECONDS=300
MAX_PAGE_ATTEMPTS=3
```

Only place models in `GEMINI_OCR_MODELS` that are actually enabled for your Gemini account. The worker tries an available configured model/credential once, stores a shared cooldown after a 429, and waits instead of continuously retrying a quota-exhausted credential.

## MCQ provider configuration

At least one of these API keys is required:

```text
CEREBRAS_API_KEY
GROQ_API_KEY
MISTRAL_API_KEY
SAMBANOVA_API_KEY
OPENROUTER_API_KEY
```

Optional per-provider model lists (comma-separated) provide normal failover:

```text
CEREBRAS_MODELS
GROQ_MODELS
MISTRAL_MODELS
SAMBANOVA_MODELS
OPENROUTER_MODELS
```

The older singular names such as `GROQ_MODEL` remain supported.

## Important notes

- A Gemini 429 must be solved with a suitable paid plan/billing, higher approved quota, or by reducing Gemini requests. Multiple API keys in the same Google Cloud project do not increase a quota that is measured per project/model.
- Digital PDFs with embedded text now use PyMuPDF text extraction first and do **not** call Gemini vision. Gemini is used only for scanned/image-only pages. This is the biggest quota reduction.
- The Drive PDF must be accessible to the Render service. For the `gdown` approach, use an appropriate link-accessible Google Drive file.
- Share the Google Sheet with the `client_email` in your service-account JSON as **Editor**.
- PDF page images/text are sent to Gemini for scanned-page OCR, and extracted source text is sent to the selected MCQ provider. Do not process sensitive content without appropriate consent and provider-policy review.
