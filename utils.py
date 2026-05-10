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
    print(f"{color}{msg}{Style.RESET_ALL}")


def get_image_files(directory):
    path = get_project_root() / directory
    if not path.exists():
        return []
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tiff", "*.tif"]
    files = []
    for ext in extensions:
        files.extend(path.glob(ext))
        files.extend(path.glob(ext.upper()))
    return sorted(files)


def save_json_output(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def save_txt_output(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    sep = "=" * 60
    sub = "-" * 60

    lines.append(sep)
    lines.append(f"Файл:          {data.get('filename', 'unknown')}")
    lines.append(f"Язык:          {data.get('language', 'unknown')}")
    lines.append(f"Уверенность:   {data.get('confidence', {}).get('overall', 0):.0%}")
    lines.append(
        f"Время:          {data.get('processing_time_sec', 0):.1f} сек"
    )
    lines.append(
        f"Движки:         EasyOCR ({data.get('confidence', {}).get('easyocr', 0):.0%}), "
        f"Tesseract ({data.get('confidence', {}).get('tesseract', 0):.0%}), "
        f"TrOCR ({data.get('confidence', {}).get('trocr', 0):.0%})"
    )
    lines.append(sep)
    lines.append(
        f"УГОЛ КОРРЕКЦИИ: исходный {data.get('detected_angle', 0):.1f}° → "
        f"{'применена' if data.get('corrected') else 'не требуется'}"
    )
    lines.append(sep)

    def _section(title, items):
        lines.append(title)
        if items:
            for item in items:
                lines.append(f"  • {item}")
        else:
            lines.append("  (не обнаружено)")

    _section("ЧИТАТЕЛИ:", data.get("reader_names", []))
    _section("ВОЗМОЖНЫЕ ЧИТАТЕЛИ:", data.get("possible_readers", []))
    _section("ДАТЫ:", data.get("dates", []))
    _section("ВОЗМОЖНЫЕ ДАТЫ:", data.get("possible_dates", []))
    _section("НОМЕРА:", data.get("inventory_numbers", []))
    _section("ШТАМПЫ:", data.get("possible_stamps", []))
    _section("ШУМ:", data.get("possible_noise", []))
    _section("БИБЛИОТЕЧНЫЕ ТЕРМИНЫ:", data.get("library_terms", []))
    if data.get("fuzzy_corrections"):
        lines.append("КОРРЕКТИРОВКИ СЛОВАРЯ:")
        for fc in data["fuzzy_corrections"]:
            lines.append(f"  • {fc['original']} → {fc['corrected']} (расстояние={fc['distance']})")

    lines.append(sep)
    lines.append("ПОЛНЫЙ ТЕКСТ (raw_text):")
    lines.append(sub)
    lines.append(data.get("raw_text", ""))
    lines.append(sub)
    lines.append(sep)
    lines.append("АЛЬТЕРНАТИВЫ OCR:")
    lines.append(
        f"EasyOCR ({data.get('confidence', {}).get('easyocr', 0):.0%}): "
        f"{data.get('alternatives', {}).get('easyocr_text', '')}"
    )
    lines.append(
        f"Tesseract ({data.get('confidence', {}).get('tesseract', 0):.0%}): "
        f"{data.get('alternatives', {}).get('tesseract_text', '')}"
    )
    lines.append(
        f"TrOCR ({data.get('confidence', {}).get('trocr', 0):.0%}): "
        f"{data.get('alternatives', {}).get('trocr_text', '')}"
    )
    lines.append(sep)

    if data.get("warnings"):
        lines.append("ПРЕДУПРЕЖДЕНИЯ:")
        for warn in data["warnings"]:
            lines.append(f"  • {warn}")
        lines.append(sep)

    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))


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
