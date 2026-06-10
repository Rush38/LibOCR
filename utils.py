import os
import sys
import gc
import json
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from colorama import init, Fore, Style

init(autoreset=True)


class Colors:
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED = Fore.RED
    BLUE = Fore.CYAN
    GRAY = Fore.LIGHTBLACK_EX
    MAGENTA = Fore.MAGENTA
    WHITE = Fore.WHITE


CONFIG = None
TRANSLATE_LANG = "ru"

_TRANSLATIONS = {
    "ru": {},
    "en": {
        "Установка зависимостей...": "Installing dependencies...",
        "Установка": "Installing",
        "установлен": "installed",
        "Дополнительные OCR движки:": "Additional OCR engines:",
        "Загрузка rus_old.traineddata...": "Downloading rus_old.traineddata...",
        "Установка завершена": "Installation complete",
        "Не удалось загрузить rus_old": "Failed to download rus_old",
        "уже есть": "already exists",
        "Не найдено": "Not found",
        "Обработка": "Processing",
        "файл": "file",
        "сек": "sec",
        "ошибок": "errors",
        "Готово": "Done",
        "обработано": "processed",
        "Ошибка": "Error",
        "терминов": "terms",
        "Лучший": "Best",
        "Всего": "Total",
        "CUDA доступен": "CUDA available",
        "CUDA не доступен": "CUDA not available",
        "все движки работают на CPU": "all engines running on CPU",
        "Состояние сброшено": "State reset",
        "Статистика по типам текста:": "Statistics by text type:",
        "Успешность по движкам:": "Success rate by engine:",
        "файлов": "files",
        "Результаты в": "Results in",
        "Нет изображений в": "No images in",
        "Пропущено": "Skipped",
        "с существующими результатами": "with existing results",
        "Найдено изображений": "Images found",
        "низкая уверенность": "low confidence",
        "проверьте результат": "check the result",
    },
}


def tr(text):
    if TRANSLATE_LANG == "en":
        return _TRANSLATIONS["en"].get(text, text)
    return text


def get_project_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def load_config():
    global CONFIG
    config_path = get_project_root() / "config.json"
    with open(config_path, "r", encoding="utf-8-sig") as f:
        CONFIG = json.load(f)
    _validate_config(CONFIG)
    return CONFIG


def _validate_config(cfg):
    required_paths = ["input_dir", "output_dir", "log_dir", "debug_dir"]
    for p in required_paths:
        if p not in cfg.get("paths", {}):
            raise ValueError(f"config.json: missing paths.{p}")

    if not cfg.get("ocr", {}).get("easyocr", {}).get("languages"):
        raise ValueError("config.json: missing ocr.easyocr.languages")
    if not cfg.get("ocr", {}).get("tesseract", {}).get("languages"):
        raise ValueError("config.json: missing ocr.tesseract.languages")


def setup_logging(config):
    log_dir = get_project_root() / config["paths"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "ocr.log"

    logger = logging.getLogger("LibraryOCR")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=config["output"]["max_log_mb"] * 1024 * 1024,
        backupCount=config["output"]["log_backups"],
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


def print_colored(msg, color=Colors.BLUE):
    try:
        print(f"{color}{msg}{Style.RESET_ALL}")
    except UnicodeEncodeError:
        safe = msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        print(f"{color}{safe}{Style.RESET_ALL}")


def get_image_files(directory):
    path = get_project_root() / directory
    if not path.exists():
        return []
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tiff", "*.tif", "*.pdf"]
    files = []
    for ext in extensions:
        files.extend(path.glob(ext))
        files.extend(path.glob(ext.upper()))
    return sorted(files)


def get_all_image_files(directory):
    path = get_project_root() / directory
    if not path.exists():
        return []
    extensions = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "pdf"]
    files = []
    for ext in extensions:
        for f in path.rglob(f"*.{ext}"):
            if f.is_file():
                files.append(f)
        for f in path.rglob(f"*.{ext.upper()}"):
            if f.is_file():
                files.append(f)
    return sorted(set(files))


def get_relative_path(file_path, base_path):
    return file_path.relative_to(base_path)


def make_output_subdir(file_path, input_dir, output_dir):
    rel = file_path.relative_to(input_dir)
    out_file = output_dir / rel
    out_file.parent.mkdir(parents=True, exist_ok=True)
    return out_file


def save_json_output(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def save_txt_output(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    sep = "=" * 60
    ssep = "-" * 40

    conf = data.get("confidence", {})
    engine_times = data.get("engine_times", {})
    alternatives = data.get("alternatives", {})
    best = data.get("best_engine", {})

    lines.append(sep)
    lines.append(f"Файл: {data.get('filename', 'unknown')}")
    lines.append(f"Тип текста: {data.get('text_type', 'unknown')}")
    lines.append(f"Тип документа: {data.get('document_type', 'unknown')}")
    lines.append(f"Язык: {data.get('language', 'unknown')}")
    lines.append(f"Общая уверенность: {conf.get('overall', 0):.0%}")
    lines.append(f"Общее время: {data.get('processing_time_sec', 0):.1f}с")
    lines.append(f"Угол: {data.get('detected_angle', 0):.1f}°, коррекция: {data.get('corrected', False)}")
    lines.append(sep)

    lines.append("ОЦЕНКА МОДЕЛЕЙ (0-100):")
    model_names = {"easyocr": "EasyOCR", "tesseract": "Tesseract", "trocr": "TrOCR",
                   "paddleocr": "PaddleOCR", "suryaocr": "Surya OCR", "glmocr": "GLM-OCR"}
    sorted_models = sorted(
        [(k, v) for k, v in conf.items() if k != "overall" and v > 0],
        key=lambda x: -x[1]
    )
    for i, (model_key, score) in enumerate(sorted_models, 1):
        name = model_names.get(model_key, model_key)
        et = engine_times.get(model_key, {})
        dur = et.get("duration", 0) if isinstance(et, dict) else 0
        marker = " ★" if best.get("name") == model_key else ""
        lines.append(f"  {i}. {name}: {score*100:.0f}/100 (время: {dur:.1f}с){marker}")
    lines.append(sep)

    lines.append("РЕЗУЛЬТАТЫ ВСЕХ МОДЕЛЕЙ:")
    lines.append("")
    for model_key in ["easyocr", "tesseract", "trocr", "paddleocr", "suryaocr", "glmocr"]:
        alt_text = alternatives.get(f"{model_key}_text", "")
        if not alt_text or not alt_text.strip():
            continue
        name = model_names.get(model_key, model_key)
        score = conf.get(model_key, 0)
        et = engine_times.get(model_key, {})
        start_t = et.get("start", "?") if isinstance(et, dict) else "?"
        end_t = et.get("end", "?") if isinstance(et, dict) else "?"
        dur = et.get("duration", 0) if isinstance(et, dict) else 0

        lines.append(ssep)
        lines.append(f"Модель: {name}")
        lines.append(f"Оценка: {score*100:.0f}/100")
        lines.append(f"Время начала: {start_t}")
        lines.append(f"Время завершения: {end_t}")
        lines.append(f"Длительность: {dur:.1f}с")
        lines.append(ssep)
        lines.append("Определенный текст:")
        lines.append(alt_text.strip()[:500])
        lines.append("")

    lines.append(sep)
    lines.append("ПОЛНЫЙ ТЕКСТ:")
    display_text = data.get("formatted_text", data.get("raw_text", ""))
    lines.append(display_text)
    lines.append(sep)
    if data.get("library_terms"):
        lines.append("LIBRARY TERMS:")
        for t in data["library_terms"]:
            lines.append(f"  - {t}")
        lines.append(sep)
    if data.get("possible_noise"):
        lines.append("POSSIBLE NOISE:")
        for n in data["possible_noise"]:
            lines.append(f"  - {n}")
        lines.append(sep)
    if data.get("possible_errors"):
        lines.append("POSSIBLE ERRORS:")
        for e in data["possible_errors"]:
            lines.append(f"  - {e}")
        lines.append(sep)
    if data.get("warnings"):
        lines.append("WARNINGS:")
        for warn in data["warnings"]:
            lines.append(f"  - {warn}")
        lines.append(sep)
    if data.get("fuzzy_corrections"):
        lines.append("FUZZY CORRECTIONS:")
        for fc in data["fuzzy_corrections"]:
            lines.append(f"  {fc.get('original')} -> {fc.get('corrected')} (dist={fc.get('distance')})")
        lines.append(sep)

    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))


def save_doc_output(path, data):
    try:
        from docx import Document
        from docx.shared import Pt, Inches, RGBColor
        doc = Document()
        style = doc.styles["Normal"]
        font = style.font
        font.name = "Arial"
        font.size = Pt(11)

        conf = data.get("confidence", {})
        engine_times = data.get("engine_times", {})
        alternatives = data.get("alternatives", {})
        best = data.get("best_engine", {})

        doc.add_heading(f"LibOCR - {data.get('filename', 'unknown')}", 0)

        doc.add_heading("Метаданные", 1)
        meta = [
            ("Тип текста", data.get("text_type", "unknown")),
            ("Тип документа", data.get("document_type", "unknown")),
            ("Язык", data.get("language", "unknown")),
            ("Общая уверенность", f"{conf.get('overall', 0):.0%}"),
            ("Общее время", f"{data.get('processing_time_sec', 0):.1f}с"),
        ]
        for label, value in meta:
            p = doc.add_paragraph()
            p.add_run(f"{label}: ").bold = True
            p.add_run(str(value))

        doc.add_heading("Оценка моделей (0-100)", 1)
        model_names = {"easyocr": "EasyOCR", "tesseract": "Tesseract", "trocr": "TrOCR",
                       "paddleocr": "PaddleOCR", "suryaocr": "Surya OCR", "glmocr": "GLM-OCR"}
        sorted_models = sorted(
            [(k, v) for k, v in conf.items() if k != "overall" and v > 0],
            key=lambda x: -x[1]
        )
        for model_key, score in sorted_models:
            name = model_names.get(model_key, model_key)
            et = engine_times.get(model_key, {})
            dur = et.get("duration", 0) if isinstance(et, dict) else 0
            marker = " ★" if best.get("name") == model_key else ""
            p = doc.add_paragraph(f"{name}: {score*100:.0f}/100 (время: {dur:.1f}с){marker}")

        doc.add_heading("Результаты всех моделей", 1)
        for model_key in ["easyocr", "tesseract", "trocr", "paddleocr", "suryaocr", "glmocr"]:
            alt_text = alternatives.get(f"{model_key}_text", "")
            if not alt_text or not alt_text.strip():
                continue
            name = model_names.get(model_key, model_key)
            score = conf.get(model_key, 0)
            et = engine_times.get(model_key, {})
            start_t = et.get("start", "?") if isinstance(et, dict) else "?"
            end_t = et.get("end", "?") if isinstance(et, dict) else "?"
            dur = et.get("duration", 0) if isinstance(et, dict) else 0

            doc.add_heading(name, 2)
            p = doc.add_paragraph()
            p.add_run("Оценка: ").bold = True
            p.add_run(f"{score*100:.0f}/100")
            p = doc.add_paragraph()
            p.add_run("Время: ").bold = True
            p.add_run(f"{start_t} - {end_t} ({dur:.1f}с)")
            p = doc.add_paragraph()
            p.add_run("Текст: ").bold = True
            doc.add_paragraph(alt_text.strip()[:500])

        doc.add_heading("Полный текст", 1)
        doc.add_paragraph(data.get("formatted_text", data.get("raw_text", "")))

        if data.get("library_terms"):
            doc.add_heading("Library Terms", 1)
            for t in data["library_terms"]:
                doc.add_paragraph(t, style="List Bullet")

        if data.get("possible_noise"):
            doc.add_heading("Possible Noise", 1)
            for n in data["possible_noise"]:
                doc.add_paragraph(n, style="List Bullet")

        if data.get("possible_errors"):
            doc.add_heading("Possible Errors", 1)
            for e in data["possible_errors"]:
                doc.add_paragraph(e, style="List Bullet")

        if data.get("fuzzy_corrections"):
            doc.add_heading("Fuzzy Corrections", 1)
            for fc in data["fuzzy_corrections"]:
                doc.add_paragraph(
                    f"{fc.get('original')} \u2192 {fc.get('corrected')} (dist={fc.get('distance')})"
                )

        if data.get("warnings"):
            doc.add_heading("Warnings", 1)
            for warn in data["warnings"]:
                doc.add_paragraph(warn, style="List Bullet")

        doc.save(str(path))
    except ImportError:
        save_txt_output(path.with_suffix(".txt"), data)


def save_md_output(path, data):
    conf = data.get("confidence", {})
    engine_times = data.get("engine_times", {})
    alternatives = data.get("alternatives", {})
    best = data.get("best_engine", {})
    model_names = {"easyocr": "EasyOCR", "tesseract": "Tesseract", "trocr": "TrOCR",
                   "paddleocr": "PaddleOCR", "suryaocr": "Surya OCR", "glmocr": "GLM-OCR"}

    sep = "---"
    parts = [
        f"# LibOCR: {data.get('filename', 'unknown')}",
        sep,
        "## Метаданные",
        f"- **Тип текста:** {data.get('text_type', 'unknown')}",
        f"- **Тип документа:** {data.get('document_type', 'unknown')}",
        f"- **Язык:** {data.get('language', 'unknown')}",
        f"- **Общая уверенность:** {conf.get('overall', 0):.0%}",
        f"- **Общее время:** {data.get('processing_time_sec', 0):.1f}с",
        sep,
        "## Оценка моделей (0-100)",
    ]
    sorted_models = sorted(
        [(k, v) for k, v in conf.items() if k != "overall" and v > 0],
        key=lambda x: -x[1]
    )
    for i, (model_key, score) in enumerate(sorted_models, 1):
        name = model_names.get(model_key, model_key)
        et = engine_times.get(model_key, {})
        dur = et.get("duration", 0) if isinstance(et, dict) else 0
        marker = " ★" if best.get("name") == model_key else ""
        parts.append(f"{i}. **{name}**: {score*100:.0f}/100 (время: {dur:.1f}с){marker}")

    for model_key in ["easyocr", "tesseract", "trocr", "paddleocr", "suryaocr", "glmocr"]:
        alt_text = alternatives.get(f"{model_key}_text", "")
        if not alt_text or not alt_text.strip():
            continue
        name = model_names.get(model_key, model_key)
        score = conf.get(model_key, 0)
        et = engine_times.get(model_key, {})
        start_t = et.get("start", "?") if isinstance(et, dict) else "?"
        end_t = et.get("end", "?") if isinstance(et, dict) else "?"
        dur = et.get("duration", 0) if isinstance(et, dict) else 0

        parts += [
            sep,
            f"## Модель: {name}",
            f"- **Оценка:** {score*100:.0f}/100",
            f"- **Время начала:** {start_t}",
            f"- **Время завершения:** {end_t}",
            f"- **Длительность:** {dur:.1f}с",
            "",
            "**Определенный текст:**",
            f"```",
            alt_text.strip()[:500],
            f"```",
        ]

    parts += [sep, "## Полный текст", data.get("formatted_text", data.get("raw_text", ""))]
    if data.get("library_terms"):
        parts += [sep, "## Library Terms"]
        parts += [f"- {t}" for t in data["library_terms"]]
    if data.get("possible_noise"):
        parts += [sep, "## Possible Noise"]
        parts += [f"- {n}" for n in data["possible_noise"]]
    if data.get("possible_errors"):
        parts += [sep, "## Possible Errors"]
        parts += [f"- {e}" for e in data["possible_errors"]]
    if data.get("fuzzy_corrections"):
        parts += [sep, "## Fuzzy Corrections"]
        for fc in data["fuzzy_corrections"]:
            parts += [f"- {fc.get('original')} → {fc.get('corrected')} (dist={fc.get('distance')})"]
    if data.get("warnings"):
        parts += [sep, "## Warnings"]
        parts += [f"- {w}" for w in data["warnings"]]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))


def memory_cleanup():
    try:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass


def log_message(level, filename, message, logger=None):
    if logger:
        log_func = {
            "DEBUG": logger.debug,
            "INFO": logger.info,
            "WARNING": logger.warning,
            "ERROR": logger.error,
        }.get(level.upper(), logger.info)
        log_func(f"[{filename}] {message}")


def save_error_report(img_path, error_message, config):
    error_dir = get_project_root() / config["paths"].get("error_reports_dir", "error_reports")
    error_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    stem = img_path.stem if img_path else "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = error_dir / f"{stem}_{timestamp}_error.txt"

    lines = []
    lines.append("=" * 60)
    lines.append(f"ОШИБКА ОБРАБОТКИ")
    lines.append("=" * 60)
    lines.append(f"Файл:     {img_path.name if img_path else 'unknown'}")
    lines.append(f"Путь:     {str(img_path) if img_path else 'unknown'}")
    lines.append(f"Время:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Ошибка:   {error_message}")
    lines.append("=" * 60)

    with open(report_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))

    return report_path
