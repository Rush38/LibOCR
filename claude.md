# LibOCR — Claude Code Instructions

## Profile
- Project: LibOCR — offline OCR for library cards
- Location: `B:\Documents\LibOCR`
- Stack: Python 3.11, EasyOCR, Tesseract, OpenCV, PowerShell

## Product Rules (NEVER violate)
1. **Offline only** — no cloud APIs, no online-only libs
2. **raw_text is sacred** — never edit, delete, or truncate content in `raw_text`
3. **Config-driven** — all settings from `config.json`, no hardcoded values in logic
4. **Portable** — paths relative via `pathlib`, project runs from flash drive
5. **Nothing deleted** — doubtful OCR items get marked with `possible_*` prefix and `— требует проверки` suffix

## Pre-Commit Checklist
Before committing, verify ALL of:
- [ ] `python -m py_compile` passes on all `.py` files
- [ ] `git status --short` shows only expected changes
- [ ] No absolute paths hardcoded (`C:\`, etc. only in `config.json`)
- [ ] No credentials or API keys in any file
- [ ] `.gitignore` covers generated dirs (`output_texts/`, `logs/`, `debug/`, `libraries/`)
- [ ] Russian text stays in Russian (comments, messages, config)

## Workflow
1. Read `agents.md` and `claude.md` at session start to refresh context
2. Update `agents.md` / `claude.md` when requirements change
3. Keep commit messages short (1-2 lines) in Russian or English
4. Never push force to `main` without explicit user approval
5. Use PowerShell syntax on Windows (no Unix commands like `head`, `ls`)

## Preferences
- CLI args via `argparse` — no `input()` in main logic
- Verbose logging to `logs/ocr.log`, concise output to console
- Batch state saved to `.batch_state.json` for resume support
- EasyOCR lazy-loaded (not on import, only on first use)

## On Failure
- Log error with filename to `logs/ocr.log`
- Add to `warnings[]` in result JSON
- Retry up to `config.processing.retry` times with `retry_delay`
- Always call `memory_cleanup()` after each image
