# LibOCR — Claude Code Instructions

## Profile
- Project: LibOCR — offline OCR for library cards
- Location: `B:\Documents\LibOCR`
- Stack: Python 3.11, EasyOCR, Tesseract, TrOCR (HuggingFace), OpenCV, PowerShell

## Product Rules (NEVER violate)
1. **Offline only** — no cloud APIs, no online-only libs
2. **raw_text is sacred** — never edit, delete, or truncate content in `raw_text`
3. **Config-driven** — all settings from `config.json`, no hardcoded values in logic
4. **Portable** — paths relative via `pathlib`, project runs from flash drive
5. **Nothing deleted** — doubtful OCR items get marked with `possible_*` prefix and `— требует проверки`
6. **Three OCR engines** — EasyOCR + Tesseract + TrOCR all integrated
7. **Dictionary system** — LibraryLoader singleton, fuzzy Levenshtein ≤ 1

## Pre-Commit Checklist (MANDATORY)
Before every commit, verify ALL of:
- [ ] `python -m py_compile` passes on all `.py` files
- [ ] `git status --short` shows only expected changes
- [ ] No absolute paths hardcoded (`C:\`, etc. only in `config.json`)
- [ ] No credentials or API keys in any file
- [ ] `.gitignore` covers generated dirs (`output_texts/`, `logs/`, `debug/`, `libraries/`, `~/.cache/huggingface/`)
- [ ] Russian text stays in Russian (comments, messages, config)
- [ ] TrOCR model: `Kansallisarkisto/cyrillic-large-handwritten`, CER=4%, Apache 2.0
- [ ] All JSON output fields present: `reader_names[]`, `dates[]`, `possible_dates[]`, `inventory_numbers[]`, `possible_stamps[]`, `raw_text`, `alternatives{ easyocr/tesseract/trocr }`, `confidence{ overall/easyocr/tesseract/trocr }`, `warnings[]`, `library_terms[]`, `fuzzy_corrections[]`

## Workflow
1. Read `agents.md` and `claude.md` at session start to refresh context
2. Update `agents.md` / `claude.md` when requirements change
3. Keep commit messages short (1-2 lines)
4. Never push force to `main` without explicit user approval
5. Use PowerShell syntax on Windows (no Unix commands like `head`, `ls`)
6. Every new requirement → update agents.md, README.md, and relevant code files simultaneously

## Preferences
- CLI args via `argparse` — no `input()` in main logic
- Verbose logging to `logs/ocr.log`, concise output to console
- Batch state saved to `.batch_state.json` for resume support
- EasyOCR and TrOCR lazy-loaded (not on import, only on first use)
- TrOCR timeout is 2x other engines (it's heavier)

## OCR Engine Details
| Engine | Model | Purpose | Confidence Source |
|--------|-------|---------|-------------------|
| EasyOCR | easyocr.Reader | Printed ru/en | per-bbox confidence |
| Tesseract | tesseract.exe | Printed, flexible PSM/OEM | pytesseract image_to_data |
| TrOCR | Kansallisarkisto/cyrillic-large-handwritten | Handwritten ru | fixed 0.92 (no per-char conf) |

## Dictionary Categories
`library_terms`, `literary_terms`, `institutions`, `document_types`, `person_titles`, `date_formats`, `common_fillers`, `verification_markers`

## On Failure
- Log error with filename to `logs/ocr.log`
- Add to `warnings[]` in result JSON
- Retry up to `config.processing.retry` times with `retry_delay`
- Always call `memory_cleanup()` after each image
