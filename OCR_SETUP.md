# OCR fallback setup

This release uses the following order by default:

1. embedded PDF text
2. local Tesseract (`eng`)
3. Mistral OCR (`mistral-ocr-latest`)
4. optional OpenRouter vision
5. Gemini Vision

## Render deployment

This repository now uses `runtime: docker` in `render.yaml`. Ensure Render deploys the new `Dockerfile`; it installs `tesseract-ocr` and the English language data. A Docker build log should contain the Tesseract apt installation before Python packages are installed.

## Environment variables

Do not commit secret values. Set/update only in Render:

```text
OCR_PROVIDER_ORDER=tesseract,mistral,openrouter,gemini
LOCAL_OCR_ENABLED=true
LOCAL_OCR_LANG=eng
LOCAL_OCR_TIMEOUT_SECONDS=90
LOCAL_OCR_MIN_TEXT_CHARS=120
MISTRAL_OCR_ENABLED=true
MISTRAL_OCR_MODEL=mistral-ocr-latest
OPENROUTER_OCR_ENABLED=false
GEMINI_OCR_ENABLED=true
```

`OPENROUTER_OCR_ENABLED` must remain false until a specific authorised image-capable model is entered in `OPENROUTER_OCR_MODEL`. Never let the code auto-select paid vision models.

## Expected logs

```text
OCR succeeded via local_tesseract for PDF page N
OCR succeeded via mistral_ocr for PDF page N
OCR succeeded via openrouter_vision for PDF page N
OCR succeeded via gemini_vision for PDF page N
```

## Operational notes

- Tesseract is best for ordinary scanned text. It is deliberately first because it has no API quota.
- Mistral OCR may require API billing/model access. Failure falls through to the next provider.
- Gemini remains a final fallback for richer visual descriptions, subject to its quota.
- Send `pause karo` before deploying. After a healthy deploy, use `start karo`; no new PDF/Sheet/TOC configuration is needed for the current run.
