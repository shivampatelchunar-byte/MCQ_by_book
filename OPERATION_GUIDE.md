# Production operation guide

## Contiguous Sheet rows

`MCQS_PER_PAGE=5` forces exactly five MCQs for each valid content page. The worker now creates a logical page job first, but reserves five Sheet rows **only after** OCR/classification succeeds and five MCQs are ready. Skipped Contents/Foreword/Preface pages consume zero Sheet rows. This removes both kinds of gaps: variable-count gaps and skipped-page gaps.

After deployment, use `/clear_and_restart CONFIRM ALL` once. It clears older output and starts a clean run with the active PDF, Sheet, and TOC profile preserved.

## Book Page semantics

The `Book Page` column comes only from a printed header/footer label. OCR first reads the page, then performs a narrow footer/header-band Tesseract pass if the full page omitted the label. It never writes a physical PDF index as a Book Page. If a label is still unreadable, the Sheet explicitly says `Unreadable printed page`; no incorrect topic mapping is guessed.

## TOC start guard

The worker skips introductory pages until the first TOC topic/chapter heading is detected. Clean restart resets this state, so the first real chapter is rediscovered every run.

## Explicit MCQ model order

No arbitrary model discovery is used for question generation. Models from `render.yaml` are tried in this safe order: Groq, Mistral, SambaNova, OpenRouter, Cerebras. This avoids Prompt Guard and Safeguard moderation models.

## Deploy / restart

1. `pause karo`
2. Deploy this release to the Docker service only.
3. Confirm the old service is stopped.
4. `/clear_and_restart CONFIRM ALL`
5. `status batao`
