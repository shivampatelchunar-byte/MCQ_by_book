# OCR and book-page operation

## OCR order

The default production order is:

1. Gemini 3.5 Flash vision OCR
2. Mistral OCR
3. local Tesseract
4. optional OpenRouter vision

Gemini receives the page with a structured request for printed header/footer page numbers plus visual/table/diagram information. If Gemini returns quota/rate-limit errors, the worker **continues** to Mistral and Tesseract instead of stopping the page.

## Book Page correctness

At the first real TOC chapter, the active profile is updated in memory and MongoDB with both the PDF start and printed TOC start. Thus the first question page is immediately resolved as printed page 1, not `Unreadable printed page`. Subsequent pages use validated footer/header labels when plausible and otherwise the TOC-calibrated sequence. Citation years are rejected.

## Clean deploy

1. `pause karo`
2. Deploy this Docker release.
3. Set Render variables:
   - `GEMINI_OCR_MODELS=gemini-3.5-flash`
   - `GEMINI_MODEL=gemini-3.5-flash`
   - `OCR_PROVIDER_ORDER=gemini,mistral,tesseract,openrouter`
4. `/clear_and_restart CONFIRM ALL`

The restart keeps PDF, Sheet and TOC profile but rewinds content-start calibration and clears invalid historical output.
