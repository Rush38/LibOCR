#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from pathlib import Path
from tqdm import tqdm

from utils import (
    load_config,
    setup_logging,
    get_image_files,
    save_json_output,
    save_txt_output,
    print_colored,
    Colors,
    memory_cleanup,
    get_project_root,
    save_error_report,
)
from batch_state import load_batch_state, save_batch_state


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
    return parser.parse_args()


def process_single_image(
    img_path,
    image_processor,
    ocr_engine,
    post_processor,
    output_dir,
    debug_dir,
    logger,
    debug_mode=False,
    max_retries=3,
    retry_delay=2,
):
    filename = img_path.name
    start_time = time.time()

    for attempt in range(max_retries):
        try:
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

            for part in ocr_parts:
                easy_conf_total += part["easyocr_conf"]
                tess_conf_total += part["tesseract_conf"]

            n = len(ocr_parts)
            combined_result = {
                "easyocr_text": "\n".join(
                    [p["easyocr_text"] for p in ocr_parts]
                ),
                "tesseract_text": "\n".join(
                    [p["tesseract_text"] for p in ocr_parts]
                ),
                "easyocr_conf": easy_conf_total / n if n else 0,
                "tesseract_conf": tess_conf_total / n if n else 0,
                "overall_conf": sum(p["overall_conf"] for p in ocr_parts)
                / n
                if n
                else 0,
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
            txt_path = output_dir / f"{img_path.stem}.txt"

            save_json_output(json_path, result_data)
            save_txt_output(txt_path, result_data)

            elapsed = time.time() - start_time
            overall = result_data["confidence"]["overall"]
            easy = result_data["confidence"]["easyocr"]
            tess = result_data["confidence"]["tesseract"]
            trocr = result_data["confidence"]["trocr"]

            if overall >= 0.6:
                print_colored(
                    f"[ OK ] {filename} — {elapsed:.1f}с "
                    f"(E:{easy:.0%} T:{tess:.0%} Tr:{trocr:.0%} → {overall:.0%})",
                    Colors.GREEN,
                )
            else:
                print_colored(
                    f"[WARN] {filename} — низкая уверенность ({overall:.0%}), "
                    "проверьте результат",
                    Colors.YELLOW,
                )

            return {
                "file": filename,
                "detected_angle": result_data["detected_angle"],
                "corrected": result_data["corrected"],
                "reader_names": "|".join(result_data["reader_names"]),
                "dates": "|".join(result_data["dates"]),
                "possible_dates": "|".join(result_data["possible_dates"]),
                "inventory_numbers": "|".join(
                    result_data["inventory_numbers"]
                ),
                "possible_stamps": "|".join(result_data["possible_stamps"]),
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
                    "reader_names": "",
                    "dates": "",
                    "possible_dates": "",
                    "inventory_numbers": "",
                    "possible_stamps": "",
                    "confidence": "ERROR",
                    "language": "",
                    "warnings": str(e),
                    "status": "ERROR",
                }


def main():
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
            print_colored(f"[FAIL] Файл не найден: {single_path}", Colors.RED)
            return
        images = [single_path]
    else:
        all_images = get_image_files(input_dir)
        if args.resume:
            images = batch_state.get_remaining(all_images)
            skipped = len(all_images) - len(images)
            if skipped > 0:
                print_colored(
                    f"[INFO] Пропуск {skipped} уже обработанных файлов", Colors.BLUE
                )
        else:
            images = all_images

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

    results = []

    total_start = time.time()

    with tqdm(
        total=len(images),
        desc="Обработка",
        unit="файл",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:
        for img_path in images:
            result = process_single_image(
                img_path,
                image_processor,
                ocr_engine,
                post_processor,
                output_dir,
                debug_dir,
                logger,
                debug_mode=args.debug,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            results.append(result)
            if result["status"] == "OK":
                batch_state.mark_completed(result["file"], result)
            else:
                batch_state.mark_failed(result["file"], result["warnings"])
            pbar.update(1)
            memory_cleanup()

    total_elapsed = time.time() - total_start

    if args.single:
        print_colored("[INFO] Обработка завершена", Colors.GREEN)
        return

    csv_path = output_dir / "summary_report.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write(
            "file;detected_angle;corrected;reader_names;dates;"
            "possible_dates;inventory_numbers;possible_stamps;"
            "library_terms;confidence;language;warnings\n"
        )
        for r in results:
            f.write(
                f"{r['file']};{r['detected_angle']};{r['corrected']};"
                f"{r['reader_names']};{r['dates']};{r['possible_dates']};"
                f"{r['inventory_numbers']};{r['possible_stamps']};"
                f"{r.get('library_terms', '')};"
                f"{r['confidence']};{r['language']};{r['warnings']}\n"
            )

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


if __name__ == "__main__":
    main()
