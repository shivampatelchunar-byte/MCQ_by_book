# Telegram TOC agent guide

A new PDF **automatically deletes the old book profile**. The bot will not retain old hard-coded chapter/page ranges.

After setting a new PDF, paste its TOC in an ordinary Telegram message. Include `TOC profile` or `Table of Contents` and one topic per line:

```text
TOC profile set karo
Plant Breeding: 1 to 61
Plant Genetics: 62 to 93
Cell Biology: 94 to 109
Forestry: 375 to END
```

The bot confirms the number of saved topics. Then say `start karo`. Topic labels in the Google Sheet are resolved from this profile using printed footer/header page numbers. Pages whose beginning contains Contents, Foreword, Acknowledgement, Dedication, Preface, Copyright, Index, References, Bibliography, Appendix, Answer Key, or Glossary are skipped.

Use `status batao` to see whether the active PDF has a TOC profile. A `new PDF` resets it by design.
