#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from pathlib import Path
from tqdm import tqdm

from utils import (
    load_config,
    setup_logging,
    get_all_image_files,
    save_json_output,
    save_txt_output,
    save_doc_output,
    save_md_output,
    print_colored,
    Colors,
    memory_cleanup,
    get_project_root,
    save_error_report,
    make_output_subdir,
)
from batch_state import load_batch_state
from image_processor import ImageProcessor
from ocr_engine import HybridOCREngine
from postprocessor import PostProcessor
from pdf_processor import PDFProcessor


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="OCR для библиотечных карточек"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Путь к папке с изображениями",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Режим отладки (сохраняет промежуточные изображения)",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=None,
        help="Число повторов при ошибке (умолчание из config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить с места остановки (пропустить обработанные)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Сбросить состояние batch и начать сначала",
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Обработать один файл",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["txt", "doc", "md"],
        help="Формат вывода: txt (по умолч.), doc (Word), md (Markdown)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["library", "universal"],
        help="Режим: library (с библиотечными полями) или universal (только текст)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Число параллельных потоков для batch (по умолч. 1)",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Установить все зависимости и дополнительные OCR движки",
    )
    return parser.parse_args()


def process_single_image(
    img_path,
    image_processor,
    ocr_engine,
    post_processor,
    input_dir,
    output_dir,
    debug_dir,
    logger,
    config,
    debug_mode=False,
    max_retries=3,
    retry_delay=2,
    output_format="txt",
):
    filename = img_path.name
    is_pdf = img_path.suffix.lower() == ".pdf"
    json_path = make_output_subdir(img_path, input_dir, output_dir).with_suffix(".json")
    txt_path = json_path.with_suffix(f".{output_format}")
    start_time = time.time()

    for attempt in range(max_retries):
        try:
            is_pdf = img_path.suffix.lower() == ".pdf"
            if is_pdf:
                pdf_proc = PDFProcessor(config)
                page_images = pdf_proc.extract_images_from_pdf(img_path)
                if not page_images:
                    raise ValueError(f"Не удалось извлечь страницы из PDF: {img_path.name}")
                blocks = page_images
                metadata = {
                    "detected_angle": 0.0,
                    "corrected": False,
                    "stamps_detected": 0,
                    "columns_count": 1,
                    "original_size": (page_images[0].shape[1], page_images[0].shape[0]),
                }
                logger.info(f"PDF {img_path.name}: {len(blocks)} pages")
            else:
                blocks, metadata = image_processor.process(img_path)

            if debug_mode and debug_dir:
                import cv2

                for i, block in enumerate(blocks):
                    cv2.imwrite(
                        str(debug_dir / f"{img_path.stem}_block{i}.png"),
                        block,
                    )

            ocr_parts = []
            for block in blocks:
                result = ocr_engine.recognize(block)
                ocr_parts.append(result)

            easy_conf_total = 0.0
            tess_conf_total = 0.0
            trocr_conf_total = 0.0

            for part in ocr_parts:
                easy_conf_total += part["easyocr_conf"]
                tess_conf_total += part["tesseract_conf"]
                trocr_conf_total += part.get("trocr_conf", 0.0)

            n = len(ocr_parts)
            last_part = ocr_parts[-1] if ocr_parts else {}
            combined_result = {
                "easyocr_text": "\n".join(
                    [p["easyocr_text"] for p in ocr_parts]
                ),
                "tesseract_text": "\n".join(
                    [p["tesseract_text"] for p in ocr_parts]
                ),
                "trocr_text": "\n".join(
                    [p.get("trocr_text", "") for p in ocr_parts]
                ),
                "easyocr_conf": easy_conf_total / n if n else 0,
                "tesseract_conf": tess_conf_total / n if n else 0,
                "trocr_conf": trocr_conf_total / n if n else 0,
                "overall_conf": sum(p["overall_conf"] for p in ocr_parts)
                / n
                if n
                else 0,
                "best_engine": last_part.get("best_engine", {}),
                "engine_times": last_part.get("engine_times", {}),
                "warnings": [],
            }

            for part in ocr_parts:
                combined_result["warnings"].extend(part.get("warnings", []))

            result_data = post_processor.process(
                combined_result,
                filename,
                metadata,
                time.time() - start_time,
            )

            json_path = output_dir / f"{img_path.stem}.json"
            txt_path = output_dir / f"{img_path.stem}.{output_format}"

            save_json_output(json_path, result_data)
            if output_format == "txt":
                save_txt_output(txt_path, result_data)
            elif output_format == "doc":
                save_doc_output(txt_path, result_data)
            elif output_format == "md":
                save_md_output(txt_path, result_data)

            elapsed = time.time() - start_time
            overall = result_data["confidence"]["overall"]
            easy = result_data["confidence"]["easyocr"]
            tess = result_data["confidence"]["tesseract"]
            trocr = result_data["confidence"]["trocr"]
            glm = result_data["confidence"]["glmocr"]

            text_type = result_data.get("text_type", "unknown")
            primary_engine = result_data.get("primary_engine", "tesseract")
            best_engine = result_data.get("best_engine", {})
            engine_times = result_data.get("engine_times", {})
            library_terms_count = len(result_data.get("library_terms", []))
            raw_preview = result_data.get("raw_text", "")[:80].replace("\n", " ")

            status = "OK" if overall >= 0.6 else "WARN"
            color = Colors.GREEN if overall >= 0.6 else Colors.YELLOW
            best_name = best_engine.get("name", primary_engine)
            et = engine_times.get(best_name, 0)

            print_colored(
                f"[{status}] {filename} | {text_type} | {primary_engine} | {library_terms_count} терминов | "
                f"T:{tess:.0%} Tr:{trocr:.0%} | Лучший: {best_name} ({et:.1f}s) | {elapsed:.1f}s",
                color,
            )
            if raw_preview.strip():
                print_colored(f"     Text: {raw_preview}...", Colors.BLUE)

            return {
                "file": filename,
                "detected_angle": result_data["detected_angle"],
                "corrected": result_data["corrected"],
                "text_type": result_data.get("text_type", "unknown"),
                "primary_engine": result_data.get("primary_engine", "tesseract"),
                "library_terms": "|".join(result_data.get("library_terms", [])),
                "confidence": f"{overall:.0%}",
                "language": result_data["language"],
                "warnings": "|".join(result_data["warnings"]),
                "status": "OK",
            }

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                memory_cleanup()
                continue
            else:
                print_colored(f"[FAIL] {filename} — {str(e)}", Colors.RED)
                logger.error(f"[{filename}] {str(e)}")
                save_error_report(img_path, str(e), config)
                return {
                    "file": filename,
                    "detected_angle": "",
                    "corrected": "",
                    "confidence": "ERROR",
                    "language": "",
                    "warnings": str(e),
                    "status": "ERROR",
                }


def main():
    args = parse_arguments()

    if args.install:
        print_colored("[INFO] Установка зависимостей...", Colors.BLUE)
        install_deps = [
            ("PyMuPDF", "PyMuPDF>=1.23.0"),
            ("python-docx", "python-docx>=1.0.0"),
        ]
        for name, pkg in install_deps:
            try:
                __import__(name.replace("-", "_").split(">")[0].split("=")[0])
                print_colored(f"  [OK] {name}", Colors.GREEN)
            except ImportError:
                import subprocess, sys
                print_colored(f"  [..] Установка {name}...", Colors.YELLOW)
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print_colored(f"  [OK] {name} установлен", Colors.GREEN)

        print_colored("[INFO] Дополнительные OCR движки (опционально):", Colors.BLUE)
        extras = [
            ("paddleocr", "paddleocr", "PaddleOCR — высокая точность, свой DL-фреймворк"),
            ("surya_ocr", "surya-ocr", "Surya OCR — современная архитектура"),
        ]
        for mod_name, pip_name, desc in extras:
            try:
                __import__(mod_name)
                print_colored(f"  [OK] {desc}", Colors.GREEN)
            except ImportError:
                print_colored(f"  [- ] {desc} — пропущен (pip install {pip_name})", Colors.GRAY)

        print_colored("[OK] Установка завершена", Colors.GREEN)
        return

    try:
        from datetime import datetime
        log_dir = get_project_root() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        run_log = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(run_log, "w", encoding="utf-8") as rf:
            rf.write(f"LibOCR run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception:
        pass

    try:
        args = parse_arguments()
        config = load_config()
        logger = setup_logging(config)
        output_format = args.format or config.get("output", {}).get("format", "txt")
        app_mode = args.mode or config.get("output", {}).get("mode", "library")
        num_workers = args.workers or config.get("processing", {}).get("max_workers", 1)
    except Exception as e:
        import sys
        from pathlib import Path
        from datetime import datetime

        print(f"[FAIL] {e}")
        try:
            log_dir = Path(__file__).parent / "logs"
            log_dir.mkdir(exist_ok=True)
            with open(log_dir / "ocr.log", "w", encoding="utf-8") as f:
                f.write(f"[{datetime.now()}] [ERROR] [main] {e}\n")
        except Exception:
            pass
        sys.exit(1)

    root = get_project_root()

    if args.input:
        input_dir = root / args.input
    else:
        input_dir = root / config["paths"]["input_dir"]

    output_dir = root / config["paths"]["output_dir"]

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    (root / config["paths"]["log_dir"]).mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if args.debug:
        debug_dir = root / config["paths"]["debug_dir"]
        debug_dir.mkdir(parents=True, exist_ok=True)

    batch_state = load_batch_state(output_dir)

    if args.reset:
        batch_state.reset()
        print_colored("[INFO] Состояние сброшено", Colors.YELLOW)

    if args.single:
        single_path = input_dir / args.single
        if not single_path.exists():
            print_colored(f"[FAIL] File not found: {single_path}", Colors.RED)
            return
        images = [single_path]
    else:
        all_images = get_all_image_files(input_dir)
        if args.resume:
            images = batch_state.get_remaining(all_images)
            skipped = len(all_images) - len(images)
            if skipped > 0:
                print_colored(
                    f"[INFO] Skipped {skipped} processed files", Colors.BLUE
                )
        else:
            images = all_images

        skipped_existing = 0
        images_to_process = []
        for img_path in images:
            json_path = make_output_subdir(img_path, input_dir, output_dir).with_suffix(".json")
            if json_path.exists():
                skipped_existing += 1
            else:
                images_to_process.append(img_path)
        if skipped_existing > 0:
            print_colored(
                f"[INFO] Skipped {skipped_existing} files with existing results", Colors.BLUE
            )
        images = images_to_process

    if not images:
        print_colored(
            f"[WARN] Нет изображений в {input_dir}", Colors.YELLOW
        )
        return

    print_colored(
        f"[INFO] Найдено изображений: {len(images)}", Colors.BLUE
    )

    max_retries = (
        args.retry
        if args.retry is not None
        else config["processing"].get("retry", 3)
    )
    retry_delay = config["processing"].get("retry_delay", 2)

    image_processor = ImageProcessor(config, debug_dir)
    ocr_engine = HybridOCREngine(config)
    post_processor = PostProcessor()

    try:
        from language_corrector import LanguageModelCorrector
        bigrams_path = get_project_root() / "models" / "ru_bigrams.json"
        corr = LanguageModelCorrector(bigrams_path)
        post_processor.register_language_corrector(corr)
        logger.info("LanguageModelCorrector: loaded")
    except Exception as e:
        logger.warning(f"LanguageModelCorrector: not loaded ({e})")

    results = []

    total_start = time.time()

    with tqdm(
        total=len(images),
        desc="Обработка",
        unit="файл",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:
        if num_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {}
                for img_path in images:
                    future = pool.submit(
                        process_single_image,
                        img_path, image_processor, ocr_engine, post_processor,
                        input_dir, output_dir, debug_dir, logger, config,
                        debug_mode=args.debug, max_retries=max_retries,
                        retry_delay=retry_delay, output_format=output_format,
                    )
                    futures[future] = img_path
                for future in as_completed(futures):
                    img_path = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {"file": img_path.name, "status": "ERROR", "warnings": str(e)}
                    results.append(result)
                    if result["status"] == "OK":
                        batch_state.mark_completed(img_path)
                    else:
                        batch_state.mark_failed(img_path, result.get("warnings", str(e)))
                    pbar.update(1)
                    memory_cleanup()
        else:
            for img_path in images:
                result = process_single_image(
                img_path,
                image_processor,
                ocr_engine,
                post_processor,
                input_dir,
                output_dir,
                debug_dir,
                logger,
                config,
                debug_mode=args.debug,
                max_retries=max_retries,
                retry_delay=retry_delay,
                output_format=output_format,
            )
            results.append(result)
            if result["status"] == "OK":
                batch_state.mark_completed(img_path)
            else:
                batch_state.mark_failed(img_path, result["warnings"])
            pbar.update(1)
            memory_cleanup()

    total_elapsed = time.time() - total_start

    if args.single:
        print_colored("[INFO] Обработка завершена", Colors.GREEN)
        return

    csv_path = output_dir / "summary_report.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write(
            "file;detected_angle;corrected;text_type;primary_engine;"
            "library_terms;confidence;language;warnings;"
            "needs_review_flag;consensus_validated\n"
        )
        for r in results:
            needs_review = "YES" if r.get("confidence", "0%").replace("%", "").replace(",", ".").isdigit() and float(r["confidence"].replace("%", "").replace(",", ".")) < 60 else ""
            f.write(
                f"{r['file']};{r['detected_angle']};{r['corrected']};"
                f"{r.get('text_type', '')};{r.get('primary_engine', '')};"
                f"{r.get('library_terms', '')};"
                f"{r['confidence']};{r['language']};{r['warnings']};"
                f"{needs_review};{r.get('consensus_validated', '')}\n"
            )

    avg_conf = {}
    engine_success = {}
    for r in results:
        if r["status"] == "OK":
            ttype = r.get("text_type", "unknown")
            conf = r.get("confidence", "0%").replace("%", "")
            try:
                conf_val = float(conf)
            except ValueError:
                conf_val = 0.0
            avg_conf[ttype] = avg_conf.get(ttype, []) + [conf_val]
            engine = r.get("primary_engine", "unknown")
            engine_success[engine] = engine_success.get(engine, 0) + 1

    if avg_conf:
        print_colored("  Статистика по типам текста:", Colors.MAGENTA)
        for ttype, confs in sorted(avg_conf.items()):
            avg = sum(confs) / len(confs)
            print_colored(f"    {ttype}: {avg:.1f}% ({len(confs)} файлов)", Colors.BLUE)

    if engine_success:
        print_colored("  Успешность по движкам:", Colors.MAGENTA)
        for eng, cnt in sorted(engine_success.items()):
            print_colored(f"    {eng}: {cnt} файлов", Colors.BLUE)

    ok_count = sum(1 for r in results if r["status"] == "OK")
    err_count = sum(1 for r in results if r["status"] == "ERROR")

    print_colored(
        f"\n[{'OK' if err_count == 0 else 'DONE'}] "
        f"Готово! Обработано: {ok_count}/{len(images)} "
        f"({err_count} ошибок) за {total_elapsed:.1f}с",
        Colors.GREEN if err_count == 0 else Colors.YELLOW,
    )
    print_colored(
        "Results in " + str(output_dir.relative_to(root)),
        Colors.BLUE,
    )

    try:
        with open(run_log, "a", encoding="utf-8") as rf:
            from datetime import datetime
            rf.write(f"Run finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            rf.write(f"Processed: {ok_count}/{len(images)}, errors: {err_count}, time: {total_elapsed:.1f}s\n")
            rf.write(f"Results: {output_dir.relative_to(root)}\n")
    except Exception:
        pass

    try:
        import os
        os.startfile(output_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
