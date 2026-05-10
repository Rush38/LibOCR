# LibOCR — Product Requirements & Notes

## Core Goal
Автономный оффлайн OCR для распознавания текста с фотографий **библиотечных карточек**. JPG/PNG, печатный + рукописный текст, штампы, линии, углы поворота 0–270°. **Всё в raw_text** — ничего не удаляется, сомнительные элементы помечаются.

## Hard Constraints
- Offline mode only — open-source движки (EasyOCR + Tesseract)
- GPU capable with large models (2.5GB+)
- Portability (flash drive compatible)
- All paths relative via `pathlib`
- `config.json` — loads and used, NOT hardcoded
- Russian language support
- Python 3.11+ compatible

## Key Decisions
- Hybrid OCR: EasyOCR + Tesseract (parallel ThreadPoolExecutor)
- Postprocessor marks doubtful items but NEVER deletes from `raw_text`
- `run.bat` uses system Python if available, else prompts for `install.ps1`
- `install.ps1` handles full installation (Python + Tesseract + dependencies)
- `--resume` / `--reset` / `--single` flags for batch control
- `--debug` flag saves intermediate processed images

## File Structure
```
LibOCR/
├── main.py              — entry point, CLI args
├── ocr_engine.py        — EasyOCR + Tesseract hybrid engine
├── image_processor.py   — preprocessing pipeline
├── postprocessor.py    — text parsing (names, dates, numbers, stamps)
├── batch_state.py       — resume/reset state management
├── utils.py             — logging, I/O, memory cleanup
├── config.json          — all configuration
├── requirements.txt     — Python dependencies
├── run.bat              — Windows launcher
├── install.ps1          — PowerShell installer
├── README.md            — documentation
├── .gitignore           — no venv, logs, output_texts, debug/
└── agents.md / claude.md — agent instructions & product rules
```

## Verification Checklist (check before each commit)
- [ ] All paths use `pathlib` (no hardcoded absolute paths)
- [ ] `config.json` values are loaded, not hardcoded in code
- [ ] `raw_text` is never modified or deleted
- [ ] `possible_*` fields exist for all extractors
- [ ] Python files pass `python -m py_compile`
- [ ] Working tree is clean (`git status --short`)
- [ ] No secrets/keys in code
- [ ] `.gitignore` covers `libraries/`, `__pycache__/`, `output_texts/`, `logs/`, `debug/`

## Output Format
- JSON: `filename`, `reader_names[]`, `dates[]`, `possible_dates[]`, `inventory_numbers[]`, `possible_stamps[]`, `raw_text` (complete, unedited), `alternatives{ easyocr_text, tesseract_text }`, `confidence{ overall, easyocr, tesseract }`, `warnings[]`, `language`, `detected_angle`, `corrected`, `processing_time_sec`
- TXT: human-readable report
- CSV: `summary_report.csv` — one row per file
