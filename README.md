# LibOCR — OCR для библиотечных карточек

Автономный оффлайн OCR для распознавания текста с фотографий библиотечных карточек. Поддерживает печатный и рукописный текст, штампы, линии, углы поворота 0–270°.

## Требования

- Windows 10/11 (x64)
- Python 3.11+
- Tesseract OCR 5.3+
- Видеокарта с CUDA (опционально, для ускорения EasyOCR)

## Установка

### Автоматическая установка (рекомендуется)

1. Откройте PowerShell **от имени администратора**
2. Перейдите в папку проекта:
   ```
   cd путь\к\LibOCR
   ```
3. Запустите установщик:
   ```
   .\install.ps1
   ```
4. Следуйте инструкциям на экране. Установщик:
   - Скачает и установит Python 3.11 (в папку `libraries\`)
   - Установит все зависимости
   - Скачает и установит Tesseract OCR
   - Скачает языковую модель `rus.traineddata`
   - Проведёт тест OCR

### Ручная установка

1. Установите Python 3.11+ с [python.org](https://www.python.org/downloads/)
2. Установите Tesseract OCR 5.3+ с [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/releases)
3. Скачайте `rus.traineddata` в папку `tessdata`:
   ```
   https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata
   ```
4. Обновите путь к Tesseract в `config.json`:
   ```json
   "tesseract": {
     "path": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
     ...
   }
   ```
5. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```

## Использование

### Быстрый старт

Дважды кликните на **`run.bat`** — откроется интерактивный режим.

### Режимы запуска

```batch
run.bat              — интерактивный режим (спросит папку)
run.bat --auto       — обработать всё в input_images
run.bat --debug      — режим отладки (сохраняет промежуточные изображения)
run.bat --input path — указать папку вручную
run.bat --check      — проверка системы (Python, Tesseract)
run.bat --resume     — продолжить с места остановки
run.bat --reset      — сбросить состояние batch
```

### Python напрямую

```bash
python main.py --input input_images
python main.py --input photos --debug
python main.py --input photos --single my_photo.jpg
python main.py --input photos --resume
```

### Аргументы командной строки

| Аргумент       | Описание                                             |
|----------------|------------------------------------------------------|
| `--input`      | Путь к папке с изображениями                         |
| `--debug`      | Сохранять промежуточные изображения в `debug/`        |
| `--retry N`    | Число повторов при ошибке (по умолчанию из config)    |
| `--resume`     | Пропустить уже обработанные файлы                    |
| `--reset`      | Сбросить состояние batch                             |
| `--single F`   | Обработать один файл                                 |

## Структура проекта

```
LibOCR/
├── main.py              — точка входа
├── ocr_engine.py        — движок OCR (EasyOCR + Tesseract)
├── image_processor.py   — предобработка изображений
├── postprocessor.py     — разбор текста (имена, даты, номера)
├── batch_state.py       — управление состоянием batch
├── utils.py             — утилиты (логирование, ввод/вывод)
├── config.json          — конфигурация
├── requirements.txt     — зависимости Python
├── run.bat              — запуск (Windows)
├── install.ps1          — установщик
├── input_images/        — сюда положить изображения
├── output_texts/        — результаты (JSON, TXT, CSV)
├── logs/                — логи
└── debug/               — отладочные изображения
```

## Конфигурация

Отредактируйте `config.json`:

```json
{
  "paths": {
    "input_dir": "input_images",
    "output_dir": "output_texts"
  },
  "ocr": {
    "easyocr": {
      "languages": ["ru", "en"],
      "gpu": true
    },
    "tesseract": {
      "path": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
      "languages": "rus+eng",
      "psm": 6
    }
  },
  "preprocessing": {
    "clahe": {
      "clip_limit": 2.0,
      "tile_grid_size": [8, 8]
    },
    "line_removal": {
      "h_thresh": 0.4,
      "v_thresh": 0.3
    },
    "stamp_suppression": {
      "hsv_ranges": [
        {"h": [100, 140], "s": [20, 255], "v": [20, 255]},
        {"h": [140, 170], "s": [20, 255], "v": [20, 255]}
      ]
    }
  },
  "processing": {
    "timeout_engine": 60,
    "retry": 3,
    "retry_delay": 2
  }
}
```

### Ключевые параметры

| Параметр                      | По умолчанию | Описание                                |
|-------------------------------|--------------|----------------------------------------|
| `easyocr.gpu`                 | `true`       | Использовать GPU для EasyOCR            |
| `easyocr.languages`            | `["ru","en"]`| Языки EasyOCR                           |
| `tesseract.psm`                | `6`          | Режим сегментации Tesseract             |
| `preprocessing.clahe.clip_limit` | `2.0`      | Контрастность CLAHE                     |
| `preprocessing.line_removal.*` | разное       | Пороги удаления линий                    |
| `processing.timeout_engine`    | `60` сек     | Таймаут на один движок OCR              |
| `processing.retry`            | `3`          | Число повторов при ошибке               |

## Результаты

После обработки в `output_texts/` создаются файлы:

- **`<имя>.json`** — полная структура с `raw_text`, `confidence`, `warnings`
- **`<имя>.txt`** — человекочитаемый отчёт
- **`summary_report.csv`** — сводная таблица всех файлов

### Поля JSON

```json
{
  "filename": "...",
  "reader_names": ["Иванов А.Б."],
  "dates": ["15.03.2024"],
  "possible_dates": ["12.06.19 — требует проверки"],
  "inventory_numbers": ["№ СП-12345"],
  "possible_stamps": ["ШАНС"],
  "raw_text": "полный текст без удалений",
  "alternatives": {
    "easyocr_text": "...",
    "tesseract_text": "..."
  },
  "confidence": {
    "overall": 0.87,
    "easyocr": 0.82,
    "tesseract": 0.91
  },
  "warnings": ["EasyOCR: превышен таймаут"],
  "language": "ru"
}
```

### Важно: `raw_text` не редактируется

Всё, что распознано — сохраняется в `raw_text` без изменений и удалений. Сомнительные элементы помечаются в отдельных полях как `possible_*` и снабжаются пометкой `— требует проверки`.

## Решение проблем

### Python не найден

Запустите `install.ps1` от имени администратора.

### Tesseract не найден

Установите вручную: https://github.com/UB-Mannheim/tesseract/releases
Обновите путь в `config.json`.

### Низкая точность распознавания

1. Используйте `--debug` для сохранения промежуточных изображений
2. Улучшите фото: хорошее освещение, минимум тени
3. Настройте `preprocessing` в `config.json`
4. Для русской рукописной text попробуйте EasyOCR отдельно

### CUDA out of memory

Установите `easyocr.gpu: false` в `config.json`.

### Превышен таймаут

Увеличьте `processing.timeout_engine` в `config.json`.

## Лицензия

MIT
