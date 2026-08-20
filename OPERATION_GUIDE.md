# Correct processing setup

## Fixed contiguous question rows

`MCQS_PER_PAGE=5` is deliberate. Every completed source PDF page generates exactly five MCQs and reserves exactly five Google Sheet rows. This removes the old 4–5 blank rows caused by a ten-row reservation combined with variable 5–8 question output.

After deploying this version, run `/clear_and_restart CONFIRM ALL` once to remove old rows/gaps. Existing page jobs from a prior run use old row reservations; a fresh run uses five-row reservations.

## Printed page metadata

The `Book Page` column is only populated from an OCR/header/footer number. The app never writes a physical PDF page index there. If OCR cannot read a printed page label, the Sheet says `Unreadable printed page` and the topic says `Unclassified (printed page unreadable)` rather than silently using an incorrect page number.

## Provider models

The app no longer discovers arbitrary provider models for MCQs. It uses configured chat-capable models in failover order: Groq, Mistral, SambaNova, OpenRouter, then Cerebras. This avoids trying Groq Prompt Guard and safeguard models as question generators.

## Clean restart sequence

1. `pause karo`
2. Deploy this release to the Docker service.
3. Verify the Docker service is the only active worker.
4. `/clear_and_restart CONFIRM ALL`
5. `status batao`

The current PDF, Sheet, and TOC profile remain in MongoDB. `clear_and_restart` starts a clean output sheet while retaining the TOC profile.
