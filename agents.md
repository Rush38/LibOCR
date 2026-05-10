# LibOCR — Product Requirements & Notes

## Core Goal
Автономный оффлайн OCR для распознавания текста с фотографий **библиотечных карточек**. JPG/PNG, печатный + рукописный текст, штампы, линии, углы поворота 0–270°. **Всё в raw_text** — ничего не удаляется, сомнительные элементы помечаются.

## Hard Constraints
- **Offline mode only** — open-source движки (EasyOCR + Tesseract + TrOCR)
- **GPU capable** with large models (2.5GB+)
- **Portability** (flash drive compatible)
- **All paths relative** via `pathlib`
- **`config.json`** — loads and used, NOT hardcoded
- **Russian language** support
- **Python 3.11+** compatible

## OCR Engines (3 движка)

### EasyOCR
- Печатный текст, русский/английский
- GPU-auto (cuda если доступна)
- Lazy-loaded (при первом распознавании)

### Tesseract OCR
- Печатный текст, гибкая настройка PSM/OEM
- Путь к executable из `config.json`
- Языковая модель: `rus+eng`

### TrOCR (HuggingFace)
- **Рукописный текст** — основное назначение
- Модель: `Kansallisarkisto/cyrillic-large-handwritten`
- CER = 4.06% (character error rate)
- 0.6B параметров, Apache 2.0
- Lazy-loaded, GPU-auto (cuda если доступна)
- Timeout x2 относительно других движков
- Загружается через `VisionEncoderDecoderModel` + `TrOCRProcessor`

## Dictionary System

- **Файл**: `dictionaries/library_dictionary.json`
- **Загрузчик**: `dictionaries/dictionary_loader.py`
- **Singleton pattern** — одна инстанция на процесс
- **Fuzzy-поиск**: расстояние Левенштейна ≤ 1
- **Категории**: `library_terms`, `literary_terms`, `institutions`, `document_types`, `person_titles`, `date_formats`, `common_fillers`, `verification_markers`
- **Graceful fallback**: если словарь отсутствует — не падает

## CLI Flags
- `--input`     — путь к папке изображений
- `--debug`     — сохранять промежуточные изображения в debug/
- `--resume`    — продолжить batch, пропустить обработанные
- `--reset`     — сбросить состояние batch
- `--single F`  — обработать один файл
- `--retry N`   — число повторов при ошибке

## Error Tracking
- При ошибке файл изображения копируется в `error_reports/<filename>_<timestamp>_error.txt`
- Содержит: имя файла, путь, timestamp, текст ошибки
- Позволяет повторно обработать проблемные файлы

## Preprocessing Pipeline
1. Denoise (`fastNlMeansDenoisingColored`)
2. Angle correction (Hough lines, max deviation 45°)
3. CLAHE contrast enhancement
4. Line removal (morphological + inpaint)
5. Stamp suppression (HSV color ranges)
6. Column splitting (vertical projection)

## File Structure
```
LibOCR/
├── main.py                    — entry point, CLI args
├── ocr_engine.py              — EasyOCR + Tesseract + TrOCR hybrid engine
├── image_processor.py         — preprocessing pipeline
├── postprocessor.py           — text parsing (names, dates, numbers, stamps)
├── batch_state.py             — resume/reset state management
├── utils.py                   — logging, I/O, memory cleanup, error reports
├── dictionaries/
│   ├── library_dictionary.json — библиотечные и литературные термины
│   └── dictionary_loader.py    — загрузчик, fuzzy-поиск, Levenshtein
├── config.json               — all configuration
├── requirements.txt           — Python dependencies
├── run.bat                   — Windows launcher
├── install.ps1               — PowerShell installer
├── README.md                 — documentation
├── .gitignore                — no venv, logs, output_texts, debug/, error_reports/, HuggingFace cache
└── agents.md / claude.md      — agent instructions & product rules
```

## Output Format (JSON)
```json
{
  "filename": "photo.jpg",
  "reader_names": ["Иванов А.Б."],
  "dates": ["15.03.2024"],
  "possible_dates": ["12.06.19 — требует проверки"],
  "inventory_numbers": ["№ СП-12345"],
  "possible_stamps": ["ШАНС"],
  "possible_noise": [],
  "library_terms": ["библиотека"],
  "fuzzy_corrections": [{"original": "...", "corrected": "...", "distance": 1}],
  "raw_text": "полный текст без удалений и редактирования",
  "alternatives": {
    "easyocr_text": "...",
    "tesseract_text": "...",
    "trocr_text": "..."
  },
  "confidence": {
    "overall": 0.87,
    "easyocr": 0.82,
    "tesseract": 0.91,
    "trocr": 0.92
  },
  "warnings": [],
  "language": "ru",
  "detected_angle": 3.5,
  "corrected": true,
  "processing_time_sec": 12.5
}
```

Also produces:
- `<name>.txt` — human-readable report
- `summary_report.csv` — one row per file

## Verification Checklist (check before each commit)
- [ ] All paths use `pathlib` (no hardcoded absolute paths in logic)
- [ ] `config.json` values are loaded, not hardcoded in code
- [ ] `raw_text` is never modified or deleted
- [ ] `possible_*` fields exist for all extractors
- [ ] `fuzzy_corrections[]` and `library_terms[]` from dictionary present in results
- [ ] TrOCR results in `alternatives.trocr_text` and `confidence.trocr`
- [ ] All 3 OCR engines (EasyOCR, Tesseract, TrOCR) integrated
- [ ] Dictionary is loaded via `DictionaryLoader` singleton
- [ ] TrOCR uses `VisionEncoderDecoderModel` + `TrOCRProcessor` from `Kansallisarkisto/cyrillic-large-handwritten`
- [ ] Error tracking saves to `error_reports/`
- [ ] Python files pass `python -m py_compile`
- [ ] Working tree is clean (`git status --short`)
- [ ] No secrets/keys in code
- [ ] `.gitignore` covers `libraries/`, `__pycache__/`, `output_texts/`, `logs/`, `debug/`, `error_reports/`, `~/.cache/huggingface/`

## Key Decisions
- Hybrid OCR: EasyOCR + Tesseract + TrOCR (parallel ThreadPoolExecutor, 3 workers)
- TrOCR: `Kansallisarkisto/cyrillic-large-handwritten` — CER=4%, 0.6B, Apache 2.0, lazy-loaded, GPU-auto
- Postprocessor marks doubtful items but NEVER deletes from `raw_text`
- Dictionary: `library_terms[]` + `fuzzy_corrections[]` from `dictionaries/library_dictionary.json`
- Fuzzy-поиск: расстояние Левенштейна ≤ 1
- `run.bat` uses system Python if available, else prompts for `install.ps1`
- `install.ps1` handles full installation (Python + Tesseract + dependencies + transformers/torch)
- `--resume` / `--reset` / `--single` / `--debug` flags for batch control
- Batch state saved to `.batch_state.json` for resume support
- Retry up to `config.processing.retry` times with `retry_delay`
- Always call `memory_cleanup()` after each image
- Error reports saved to `error_reports/` on failure