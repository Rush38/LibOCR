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
- **Third OCR engine: TrOCR (HuggingFace)** для рукописного текста — загружается с `microsoft/troat-base-ru-handwriting` или аналога, offline via `transformers` + `torch`
- Postprocessor marks doubtful items but NEVER deletes from `raw_text`
- Dictionary-loaded: `library_terms[]` + `fuzzy_corrections[]` из `dictionaries/library_dictionary.json`
- `run.bat` uses system Python if available, else prompts for `install.ps1`
- `install.ps1` handles full installation (Python + Tesseract + dependencies)
- `--resume` / `--reset` / `--single` flags for batch control
- `--debug` flag saves intermediate processed images

## File Structure
```
LibOCR/
├── main.py              — entry point, CLI args
├── ocr_engine.py        — EasyOCR + Tesseract + TrOCR hybrid engine
├── image_processor.py   — preprocessing pipeline
├── postprocessor.py    — text parsing (names, dates, numbers, stamps)
├── batch_state.py       — resume/reset state management
├── utils.py             — logging, I/O, memory cleanup
├── config.json          — all configuration
├── requirements.txt     — Python dependencies
├── run.bat              — Windows launcher
├── install.ps1          — PowerShell installer
├── README.md            — documentation
├── dictionaries/        — library_terms + fuzzy_corrections
│   ├── library_dictionary.json  — библиотечные и литературные термины
│   └── dictionary_loader.py     — загрузчик, fuzzy-поиск, Levenshtein
├── .gitignore           — no venv, logs, output_texts, debug/, HuggingFace cache
└── agents.md / claude.md — agent instructions & product rules
```

## CLI Flags
- `--input`     — путь к папке изображений
- `--debug`     — сохранять промежуточные изображения в debug/
- `--resume`    — продолжить batch, пропустить обработанные
- `--reset`     — сбросить состояние batch
- `--single F`  — обработать один файл
- `--retry N`   — число повторов при ошибке

## Hard Constraints
- Offline mode only — open-source движки (EasyOCR + Tesseract + TrOCR)
- GPU capable with large models (2.5GB+)
- Portability (flash drive compatible)
- All paths relative via `pathlib`
- `config.json` — loads and used, NOT hardcoded
- Russian language support
- Python 3.11+ compatible

## Key Decisions
- Hybrid OCR: EasyOCR + Tesseract + TrOCR (parallel ThreadPoolExecutor, 3 workers)
- TrOCR: `microsoft/troat-base-ru-handwriting` from HuggingFace, lazy-loaded, GPU-auto
- Postprocessor marks doubtful items but NEVER deletes from `raw_text`
- Dictionary: `library_terms[]` + `fuzzy_corrections[]` from `dictionaries/library_dictionary.json`
- Fuzzy-поиск: расстояние Левенштейна ≤ 1
- `run.bat` uses system Python if available, else prompts for `install.ps1`
- `install.ps1` handles full installation (Python + Tesseract + dependencies + transformers/torch)
- `--resume` / `--reset` / `--single` / `--debug` flags for batch control

## Output Format
- JSON: `filename`, `reader_names[]`, `dates[]`, `possible_dates[]`, `inventory_numbers[]`, `possible_stamps[]`, `raw_text` (complete, unedited), `alternatives{ easyocr_text, tesseract_text, trocr_text }`, `confidence{ overall, easyocr, tesseract, trocr }`, `warnings[]`, `language`, `detected_angle`, `corrected`, `processing_time_sec`, `library_terms[]`, `fuzzy_corrections[]`
- TXT: human-readable report
- CSV: `summary_report.csv` — one row per file
