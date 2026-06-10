# LibOCR — OCR для библиотечных карточек

Автономный оффлайн OCR для распознавания текста с фотографий библиотечных карточек. Поддерживает печатный и рукописный текст, штампы, линии, углы поворота 0–270°. **Всё в raw_text** — ничего не удаляется, сомнительные элементы помечаются.

## Требования

- Windows 10/11 (x64)
- **Python 3.11 или 3.12** — версии 3.13+ не поддерживают CUDA
- Tesseract OCR 5.3+ (для `rus_old` нужен пакет `rus_old.traineddata`)
- GPU с CUDA (опционально, для EasyOCR и TrOCR)

## Быстрый старт

```batch
run.bat                — интерактивный режим
run.bat --auto         — обработать всё в input_files
run.bat --debug        — сохранять промежуточные изображения
run.bat --input path   — указать папку
run.bat --check        — проверка системы
run.bat --resume       — продолжить batch
run.bat --reset        — сбросить batch

python main.py --input photos --format md
python main.py --input photos --mode universal
python main.py --input photos --workers 4
python main.py --single photo.jpg --format doc
```

## Архитектура

Три OCR движка работают по **каскадной схеме** (сначала быстрые, потом точные):

| Engine | Назначение | Модель | Когда запускается |
|--------|-----------|--------|-------------------|
| **EasyOCR** | Печатный ru/en | easyocr Reader | Всегда (с fallback) |
| **Tesseract** | Печатный, адаптивный PSM | tesseract.exe | Если EasyOCR < 70% |
| **TrOCR** | Рукописный текст | `Kansallisarkisto/cyrillic-large-handwritten` (HuggingFace) | Если все < 60% |

### Альтернативные OCR движки (не встроены, устанавливаются вручную)

| Engine | Назначение | Установка | Особенности |
|--------|-----------|-----------|-------------|
| **PaddleOCR** | Печатный ru/en, высокая точность | `pip install paddleocr` | Свой DL-фреймворк (PaddlePaddle), GPU, работает отдельно |
| **Surya OCR** | Универсальный, много языков | `pip install surya-ocr` | На базе torch, современная архитектура |

PaddleOCR рекомендуется как дополнительный движок для сложных случаев — он имеет собственный пайплайн и не зависит от EasyOCR/Tesseract.

Каскадная схема даёт:
- **EasyOCR** (1-2с) → быстрое распознавание печатного текста
- **Tesseract** (3-8с) → если EasyOCR не уверен
- **TrOCR** (10-20с) → только для сложных/рукописных случаев

## Логика работы

### 1. Определение типа документа
Перед обработкой изображение анализируется:
- **photo** — высокая детализация, много текста → профиль `photo`
- **old_document** — низкий контраст, старая бумага → Sauvola-бинаризация + Tesseract с `rus_old`
- **blank** — почти пустой лист → пропуск
- **default** — стандартный CLAHE

### 2. Загрузка (JPG/PNG/PDF)
- **Изображения**: OpenCV (`np.fromfile` + `cv2.imdecode` — кириллические пути)
- **PDF**: PyMuPDF — извлечение страниц без внешних зависимостей
- **Кеш**: LRU-кеш (MD5 хеш) для повторяющихся изображений

### 3. Каскадная обработка (Cascade OCR)
```
Image → [EasyOCR] → если confidence < 70% → [Tesseract] → если всё < 60% → [TrOCR]
```

Каждый движок замеряет своё время (`engine_times`). Лучший результат выбирается по max confidence (`best_engine`).

### 4. EasyOCR fallback
Если EasyOCR с `paragraph=True` не дал текста — автоматический перезапуск с `paragraph=False`.

### 5. Постобработка
- Извлечение дат, инвентарных номеров, штампов
- Поиск библиотечных терминов через словарь
- Fuzzy-коррекция по словарю (Левенштейн ≤ 1)

### 6. Контекстная коррекция (LanguageModelCorrector)
- Загрузка `models/ru_bigrams.json` — частоты пар слов
- Исправление слов с расстоянием Левенштейна ≤ 2 с учётом соседних слов
- Пример: "читательский билет" → исправление "билет" в контексте "читательский"

### 7. Кросс-проверка движков
- Выравнивание строк из 3 движков
- Консенсус 2 из 3 → `consensus_validated: true`
- Расхождения → `possible_errors`

### 8. Детекция структуры текста
- Заголовки (короткие строки, ALL CAPS) → `## Заголовок`
- Абзацы (объединение строк по пустым) → параграфы
- Списки (строки с `-`, `*`, `1.`) → маркированные списки
- Колонтитулы и номера страниц → отдельные поля

### 9. Сохранение
- **JSON** — полная структура со всеми полями
- **TXT** / **DOC** (Word) / **MD** (Markdown) — по выбору (`--format`)

### 4. Выбор приоритетного движка
- Для `handwritten` → TrOCR (если доступен)
- Для `printed` → Tesseract или EasyOCR (по max confidence)
- Поле `primary_engine` в JSON указывает основной движок

### 5. Постобработка
- Извлечение дат, инвентарных номеров, штампов
- Поиск библиотечных терминов через словарь
- Fuzzy-коррекция (расстояние Левенштейна ≤ 1)
- Определение языка (ru/en/mixed)

### 6. Сохранение результатов
- **JSON**: полная структура с `raw_text`, `alternatives`, `confidence`, `library_terms`, `fuzzy_corrections`
- **TXT**: человекочитаемый отчёт
- **CSV**: сводная таблица

### GPU-ускорение
При наличии CUDA:
- EasyOCR автоматически использует GPU
- TrOCR загружается на GPU
- Обработка в 5-10x быстрее

## Словарь

Библиотечные и литературные термины загружаются из `dictionaries/library_dictionary.json`:

| Категория | Примеры |
|---|---|
| `library_terms` | библиотека, абонемент, каталог, инвентарный |
| `literary_terms` | проза, поэзия, роман, повесть |
| `institutions` | библиотека, архив, издательство |
| `document_types` | формуляр, читательский билет, каталог |
| `person_titles` | библиотекарь, библиограф, хранитель |
| `verification_markers` | требует проверки, неразборчиво |

Fuzzy-поиск: расстояние Левенштейна ≤ 1.

## Установка

### Автоматическая (рекомендуется)

1. Откройте PowerShell **от имени администратора**
2. Перейдите в папку проекта: `cd путь\к\LibOCR`
3. Запустите: `.\install.ps1`

Установщик скачает и установит Python, Tesseract OCR, языковую модель `rus.traineddata`, и все Python-библиотеки.

### Ручная

1. **Python 3.11 или 3.12** с [python.org](https://www.python.org/downloads/) — **важно: для GPU требуется 3.11-3.12, версии 3.13+ не поддерживают CUDA**
2. Tesseract OCR 5.3+ с [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/releases)
3. `rus.traineddata` → папка `tessdata` Tesseract
4. `pip install -r requirements.txt`
5. **Важно**: если устанавливали ранее — переустановите: `pip install --upgrade -r requirements.txt`
6. Обновите путь к Tesseract в `config.json` при необходимости

### GPU (опционально)

Для использования видеокарты:
- NVIDIA GPU с CUDA (RTX 3080+ рекомендуется, 8GB+ VRAM)
- NVIDIA Driver (любая последняя версия)
- CUDA Toolkit 12.x
- Python 3.11 или 3.12 (обязательно для PyTorch CUDA)

## Стратегия обработки

1. **Определение типа текста** — анализ распознанного текста для детекции `handwritten` / `printed`
2. **Приоритетный движок** — на основе типа: TrOCR для рукописного, Tesseract/EasyOCR для печатного
3. **Все результаты в одном файле** — каждый движок отображается в `raw_text` с тегом `[ОСНОВНОЙ: TROCR] (handwritten)`
4. **Поля `text_type`** и **`primary_engine`** — указывают тип текста и выбранный основной движок
5. **Адаптивный PSM** — Tesseract автоматически выбирает PSM режим (3/6/7/11) под изображение
6. **Сегментация строк для TrOCR** — горизонтальная проекция + адаптивная бинаризация для рукописного текста
7. **Кросс-проверка** — выравнивание строк из 3 движков, поиск консенсуса (2 из 3), пометка расхождений
8. **Детекция штампов** — HSV-фильтрация красного/синего, отдельный OCR на области штампа
9. **Контекстная коррекция** — биграммы библиотечных терминов, Levenshtein ≤ 2 с учётом соседних слов

## Конфигурация

Все настройки — в `config.json`. Никаких hardcoded значений в коде.

```json
{
  "paths": {
    "input_dir": "input_files",
    "output_dir": "output_texts",
    "log_dir": "logs",
    "debug_dir": "debug",
    "error_reports_dir": "error_reports"
  },
  "ocr": {
    "easyocr": {
      "enabled": true,
      "languages": ["ru", "en"],
      "gpu": true,
      "confidence_threshold": 0.3,
      "paragraph": true
    },
    "tesseract": {
      "enabled": true,
      "path": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
      "languages": "rus+eng",
      "oem": 3,
      "psm": 6,
      "dpi": 300,
      "adaptive_psm": true
    },
    "trocr": {
      "enabled": true,
      "model_name": "Kansallisarkisto/cyrillic-large-handwritten",
      "device": "auto",
      "stride": 128,
      "local_models_dir": "models",
      "line_segmentation": true
    },
    "glmocr": {
      "enabled": false,
      "model_name": "zai-org/GLM-OCR",
      "device": "auto",
      "local_models_dir": "models",
      "note": "model incompatible"
    }
  },
  "pdf": {
    "enabled": true,
    "dpi": 300,
    "first_page_only": false,
    "max_pages": 50,
    "color_conversion": "RGB",
    "auto_rotate": true,
    "poppler_path": "bin/poppler"
  },
  "preprocessing": {
    "clahe": { "clip_limit": 2.0, "tile_grid_size": [8, 8] },
    "line_removal": { "h_thresh": 0.4, "v_thresh": 0.3 },
    "stamp_suppression": {
      "hsv_ranges": [
        {"h": [100, 140], "s": [20, 255], "v": [20, 255]},
        {"h": [140, 170], "s": [20, 255], "v": [20, 255]}
      ],
      "sat_reduction": 0.0
    },
    "angle_correction": { "max_deviation": 45.0 },
    "cache": { "enabled": true, "maxsize": 100 },
    "stamp_detection": { "enabled": true, "min_area": 400 }
  },
  "processing": {
    "max_workers": 2,
    "timeout_engine": 60,
    "timeout_image": 180,
    "retry": 3,
    "retry_delay": 2
  },
  "output": {
    "encoding": "utf-8-sig",
    "csv_sep": ";",
    "max_log_mb": 10,
    "log_backups": 5
  }
}
```

## Использование

```batch
run.bat                — интерактивный режим
run.bat --auto         — обработать всё в input_files
run.bat --debug        — сохранять промежуточные изображения
run.bat --input path   — указать папку
run.bat --check        — проверка системы
run.bat --resume       — продолжить batch
run.bat --reset        — сбросить batch

python main.py --input photos --debug --resume
python main.py --single photo.jpg
```

### Все CLI-флаги

| Флаг | Описание |
|------|----------|
| `--input` | Путь к папке с изображениями |
| `--debug` | Сохранять промежуточные изображения в `debug/` |
| `--resume` | Пропустить уже обработанные файлы |
| `--reset` | Сбросить состояние batch |
| `--single F` | Обработать один файл |
| `--retry N` | Число повторов при ошибке |
| `--format` | Формат вывода: `txt` (по умолч.), `doc` (Word), `md` (Markdown) |
| `--mode` | Режим: `library` (библиотечные поля) или `universal` (только текст) |
| `--workers N` | Число параллельных потоков для batch (по умолч. 1) |

## Структура проекта

```
LibOCR/
├── main.py                    — точка входа, CLI
├── ocr_engine.py              — 3 engines: EasyOCR + Tesseract + TrOCR (каскад)
├── image_processor.py         — preprocessing + автоопределение типа документа
├── postprocessor.py           — разбор текста, структура, коррекция
├── binarization.py            — Sauvola/Niblack/Otsu адаптивная бинаризация
├── language_corrector.py      — контекстная коррекция (биграммы)
├── pdf_processor.py           — PDF → изображения (PyMuPDF)
├── preprocessing_cache.py     — LRU-кеш предобработки (MD5)
├── batch_state.py             — resume/reset state
├── utils.py                   — логирование, I/O, save_txt/doc/md
├── dictionaries/
│   ├── library_dictionary.json — библиотечные термины
│   └── dictionary_loader.py   — fuzzy-поиск, Levenshtein
├── models/
│   ├── ru_bigrams.json         — биграммы для контекстной коррекции
│   └── Kansallisarkisto_*     — TrOCR модель (скачивается при первом запуске)
├── config.json               — вся конфигурация
├── requirements.txt           — зависимости
├── run.bat                   — Windows launcher
├── install.ps1               — PowerShell installer
├── README.md                 — документация
├── .gitignore                — no venv, logs, output_texts, debug/, error_reports/, HuggingFace cache
└── agents.md / claude.md      — agent instructions & product rules
```
├── requirements.txt           — зависимости
├── run.bat                    — запуск (Windows)
├── install.ps1                — установщик
├── README.md                  — эта документация
├── agents.md                  — правила продукта
└── claude.md                  — инструкции для Claude Code
```

## Вывод в консоли

При обработке отображается:
- Имя файла
- Тип текста (`printed`/`handwritten`)
- Приоритетный движок (`primary_engine`)
- Найдено библиотечных терминов
- Confidence по каждому движку
- Время обработки
- Превью распознанного текста (первые 80 символов)

```
[OK] image00019.jpg | printed | tesseract | 3 терминов | T:91% Tr:0% | 5.8s
     Text: 427935 Автогр. В 48 Винников, Вадим...
[WARN] image00001.jpg | handwritten | tesseract | 0 терминов | T:0% Tr:0% | 12.1s
     Text: / sone ah | one м os ey м Saha yo...
```

## Результаты

После обработки в `output_texts/`:

- **`<имя>.json`** — полная структура
- **`<имя>.txt`** — человекочитаемый отчёт
- **`summary_report.csv`** — сводная таблица
- Папка `output_texts/` автоматически открывается после завершения (Windows)

При ошибке в `error_reports/`:
- **`<filename>_<timestamp>_error.txt`** — отчёт с именем файла, путём и текстом ошибки

### JSON-поля

```json
{
  "filename": "photo.jpg",
  "reader_names": ["Иванов А.Б."],
  "dates": ["15.03.2024"],
  "possible_dates": ["12.06.19 — требует проверки"],
  "inventory_numbers": ["№ СП-12345"],
  "possible_stamps": ["ШАНС"],
  "library_terms": ["библиотека", "читатель"],
  "fuzzy_corrections": [
    {"original": "боблиотека", "corrected": "библиотека", "distance": 1}
  ],
  "raw_text": "полный текст без удалений",
  "text_type": "printed",
  "primary_engine": "tesseract",
  "alternatives": {
    "easyocr_text": "...",
    "tesseract_text": "...",
    "trocr_text": "...",
    "glmocr_text": "..."
  },
  "confidence": {
    "overall": 0.87,
    "easyocr": 0.82,
    "tesseract": 0.91,
    "trocr": 0.92,
    "glmocr": 0.88
  },
  "warnings": [],
  "language": "ru",
  "detected_angle": 3.5,
  "corrected": true,
  "processing_time_sec": 12.5
}
```

### Важно: `raw_text` не редактируется

Всё распознанное сохраняется в `raw_text` без изменений. Сомнительные элементы → отдельные поля `possible_*` с пометкой `— требует проверки`.

## Решение проблем

| Проблема | Решение |
|----------|---------|
| Python не найден | `.\install.ps1` от администратора |
| Tesseract не найден | Установить вручную, обновить путь в `config.json` |
| CUDA OOM | `easyocr.gpu: false` в `config.json` |
| Низкая точность | `--debug` → смотреть `debug/` |
| Timeout | Увеличить `processing.timeout_engine` в `config.json` |

## Лицензия

MIT