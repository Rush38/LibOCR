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
    try:
        print(f"{color}{msg}{Style.RESET_ALL}")
    except UnicodeEncodeError:
        safe = msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        print(f"{color}{safe}{Style.RESET_ALL}")


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


def get_all_image_files(directory):
    path = get_project_root() / directory
    if not path.exists():
        return []
    extensions = ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"]
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

    lines.append(sep)
    lines.append(f"File: {data.get('filename', 'unknown')}")
    lines.append(f"Language: {data.get('language', 'unknown')}")
    lines.append(f"Confidence: {data.get('confidence', {}).get('overall', 0):.0%}")
    lines.append(f"Time: {data.get('processing_time_sec', 0):.1f}s")
    lines.append(f"Angle: {data.get('detected_angle', 0):.1f} deg, corrected: {data.get('corrected', False)}")
    lines.append(sep)
    lines.append("TEXT:")
    lines.append(data.get("raw_text", ""))
    lines.append(sep)
    if data.get("warnings"):
        lines.append("WARNINGS:")
        for warn in data["warnings"]:
            lines.append(f"  {warn}")
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
