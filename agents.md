# LibOCR — Product Requirements & Notes

## Core Goal
Автономный оффлайн OCR для распознавания текста с фотографий **библиотечных карточек**. JPG/PNG, печатный + рукописный текст, штампы, линии, углы поворота 0–270°. **Всё в raw_text** — ничего не удаляется, сомнительные элементы помечаются.

## Processing Strategy
1. **Определение типа текста** — анализ изображения для детекции рукописный/печатный
2. **Приоритетный движок** — на основе типа: TrOCR для рукописного, Tesseract/EasyOCR для печатного
3. **Все результаты в одном файле** — каждый движок отображается в raw_text с тегом
4. **primary_engine** — поле указывающее какой движок выбран как основной

## Hard Constraints
- **Offline mode only** — open-source движки (EasyOCR + Tesseract + TrOCR)
- **GPU capable** with large models (2.5GB+), требуется CUDA
- **Portability** (flash drive compatible)
- **All paths relative** via `pathlib`
- **`config.json`** — loads and used, NOT hardcoded
- **Russian language** support
- **Python 3.11-3.12** compatible (для GPU требуется Python 3.11 или 3.12)

## GPU Требования
- NVIDIA GPU с CUDA поддержкой (RTX 3080 тестировано, 8GB+ VRAM)
- NVIDIA Driver (любая последняя версия)
- CUDA Toolkit 12.x
- **Важно**: Python 3.11 или 3.12 — PyTorch с CUDA не поддерживает Python 3.13+

## OCR Engines (7 движков, все обязательные)

### EasyOCR
- Печатный текст, русский/английский
- GPU-auto (cuda если доступна)
- Lazy-loaded (при первом распознавании)

### Tesseract OCR
- Печатный текст, гибкая настройка PSM/OEM
- Поддержка `rus_old` (дореволюционная орфография)
- Путь к executable из `config.json`
- Языковая модель: `rus+eng` (или `rus_old+rus+eng`)

### TrOCR (HuggingFace)
- **Рукописный текст** — основное назначение
- Модель: `Kansallisarkisto/cyrillic-large-handwritten`
- CER = 4.06% (character error rate)
- 0.6B параметров, Apache 2.0
- Lazy-loaded, GPU-auto (cuda если доступна)
- Timeout x2 относительно других движков
- Загружается через `VisionEncoderDecoderModel` + `ViTImageProcessor` + `AutoTokenizer`
- **Важно:** требуется `transformers<4.50.0` из-за несовместимости TokenizersBackend
- **Local model support**: модель скачивается в `models/` через `snapshot_download`, используется локально без повторной загрузки

### PaddleOCR (обязательный)
- Печатный русский/английский
- Собственный DL-фреймворк (PaddlePaddle), GPU
- Устанавливается через `--install`

### Surya OCR (обязательный)
- Универсальный OCR на базе torch
- GPU-auto (cuda если доступна)
- Устанавливается через `--install`

### docTR (обязательный)
- Документный OCR от Mindee
- Распознавание слов и строк, русский язык
- GPU-auto (cuda если доступна)
- Устанавливается через `--install`

### GOT-OCR2 (обязательный)
- Современная transformer-модель
- Высокое качество на сложных документах
- GPU-auto (cuda если доступна)
- Устанавливается через `--install`

### Альтернативные движки (не встроены)
- (нет — все движки обязательны)

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

## Nested Images
- Обрабатываются изображения во всех подпапках `input_files/` (рекурсивно)
- Структура папок сохраняется: `input_files/subfolder/image.jpg` → `output_texts/subfolder/image.json`
- Иерархия директорий воспроизводится в `output_texts/`
- **Skip already processed**: если JSON-файл результата уже существует — изображение пропускается (даже без `--resume`)

## Error Tracking
- При ошибке файл изображения копируется в `error_reports/<filename>_<timestamp>_error.txt`
- Содержит: имя файла, путь, timestamp, текст ошибки
- Позволяет повторно обработать проблемные файлы

## Auto-open Results Folder
- После завершения обработки автоматически открывается папка `output_texts/` (Windows: `os.startfile`)
- Позволяет сразу просмотреть результаты без ручного поиска папки

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
- [ ] Nested image paths preserved: `input_files/sub/name.jpg` → `output_texts/sub/name.json`
- [ ] Python files pass `python -m py_compile`
- [ ] Working tree is clean (`git status --short`)
- [ ] No secrets/keys in code
- [ ] `.gitignore` covers `libraries/`, `__pycache__/`, `output_texts/`, `logs/`, `debug/`, `error_reports/`, `~/.cache/huggingface/`

## Key Decisions
- Hybrid OCR: EasyOCR + Tesseract + TrOCR (каскадный режим, все 3 движка всегда запускаются)
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