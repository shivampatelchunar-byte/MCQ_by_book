# Production operation guide

## Contiguous Sheet rows

`MCQS_PER_PAGE=5` forces five MCQs for every valid source page. A page job is created before OCR, but five Google Sheet rows are allocated only after classification and MCQ generation succeeds. Skipped Contents/Foreword/Preface pages allocate no rows, so there are no blank page blocks.

## Printed Book Page validation

The app extracts header/footer candidates with full-page OCR plus a narrow footer/header band pass. It rejects implausible values above `MAX_REASONABLE_BOOK_PAGE` (default 1000), which prevents citation years such as `1927`, `1933`, and `1945` from being stored as Book Page numbers.

At the first TOC chapter heading, the app records both the physical PDF start page and the configured printed TOC start page. Later pages accept a header/footer label only when it agrees with the calibrated TOC progression. Otherwise the Book Page is the TOC-calibrated sequence, never a raw PDF page index and never an OCR citation year.

## Clean restart

1. `pause karo`
2. Deploy this Docker release only.
3. Ensure old services are stopped.
4. `/clear_and_restart CONFIRM ALL`
5. `status batao`

The current PDF, Sheet and TOC profile stay saved. A clean restart resets chapter detection so the first real TOC chapter is located again.
