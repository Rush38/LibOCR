import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional


class ImageProcessor:
    def __init__(self, config, debug_dir: Optional[Path] = None):
        self.config = config
        self.prep = config.get("preprocessing", {})
        self.debug_dir = debug_dir
        self._debug_count = 0

    def process(
        self, image_path: Path
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        img_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Не удалось открыть изображение: {image_path}")

        original_h, original_w = img.shape[:2]

        metadata = {
            "detected_angle": 0.0,
            "corrected": False,
            "stamps_detected": 0,
            "columns_count": 1,
            "original_size": (original_w, original_h),
        }

        img = self._preprocess_pipeline(img, image_path.stem, metadata)

        blocks, metadata["columns_count"] = self._split_columns(img)

        return blocks, metadata

    def _preprocess_pipeline(
        self, img: np.ndarray, stem: str, metadata: Dict
    ) -> np.ndarray:
        img = self._denoise(img)

        img = self._correct_angle(img, metadata)

        img = self._enhance_contrast(img)

        img = self._remove_lines(img)

        img, metadata["stamps_detected"] = self._suppress_stamps(img)

        if self.debug_dir:
            cv2.imwrite(
                str(self.debug_dir / f"{stem}_01_preprocessed.png"), img
            )

        return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    def _correct_angle(
        self, img: np.ndarray, metadata: Dict
    ) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        angle = 0.0
        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                a = np.rad2deg(theta)
                if a > 90:
                    a -= 180
                angles.append(a)
            if angles:
                angle = float(np.median(angles))

        max_dev = self.prep.get("angle_correction", {}).get(
            "max_deviation", 45.0
        )

        if abs(angle) > 0.5 and abs(angle) <= max_dev:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(
                img, M, (w, h), borderMode=cv2.BORDER_REPLICATE
            )
            metadata["corrected"] = True

        metadata["detected_angle"] = angle
        return img

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        clahe_cfg = self.prep.get("clahe", {})
        clip_limit = clahe_cfg.get("clip_limit", 2.0)
        tile_size = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _remove_lines(self, img: np.ndarray) -> np.ndarray:
        line_cfg = self.prep.get("line_removal", {})
        h_thresh = line_cfg.get("h_thresh", 0.4)
        v_thresh = line_cfg.get("v_thresh", 0.3)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        h, w = binary.shape

        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(int(w * h_thresh), 5), 1)
        )
        v_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(int(h * v_thresh), 5))
        )

        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

        lines_mask = cv2.bitwise_or(h_lines, v_lines)
        lines_mask = cv2.dilate(lines_mask, np.ones((3, 3), np.uint8), iterations=1)

        result = cv2.inpaint(
            img, lines_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
        )

        return result

    def _suppress_stamps(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        stamp_cfg = self.prep.get("stamp_suppression", {})
        hsv_ranges = stamp_cfg.get("hsv_ranges", [])
        sat_reduction = stamp_cfg.get("sat_reduction", 0.0)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        stamps_count = 0
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for hsv_range in hsv_ranges:
            lower = np.array(
                [hsv_range["h"][0], hsv_range["s"][0], hsv_range["v"][0]]
            )
            upper = np.array(
                [hsv_range["h"][1], hsv_range["s"][1], hsv_range["v"][1]]
            )
            mask = cv2.inRange(hsv, lower, upper)

            if np.sum(mask) > 100:
                stamps_count += 1
                mask_total = cv2.bitwise_or(mask_total, mask)

        if stamps_count > 0:
            sat_val = int(sat_reduction * 255)
            hsv[:, :, 1] = np.where(
                mask_total > 0, sat_val, hsv[:, :, 1]
            )
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img, stamps_count

    def _split_columns(
        self, img: np.ndarray
    ) -> Tuple[List[np.ndarray], int]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        h, w = binary.shape
        projection = np.sum(binary, axis=0).astype(float)

        normalized = projection / (projection.max() + 1e-6)
        threshold = 0.1

        min_width = int(w * 0.08)

        gaps = []
        gap_start = None

        for i in range(w):
            if normalized[i] < threshold:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gap_width = i - gap_start
                    if gap_width >= min_width:
                        peaks = []
                        for j in range(gap_start, i):
                            peaks.append(normalized[j])
                        if peaks:
                            avg = sum(peaks) / len(peaks)
                            if avg < threshold * 2:
                                gaps.append((gap_start, i))
                    gap_start = None

        if len(gaps) >= 1:
            if len(gaps) == 1:
                mid = (gaps[0][0] + gaps[0][1]) // 2
                left = img[:, :mid]
                right = img[:, mid:]
                return [left, right], 2
            else:
                boundaries = [0] + [g[1] for g in gaps] + [w]
                blocks = []
                for i in range(len(boundaries) - 1):
                    col = img[:, boundaries[i] : boundaries[i + 1]]
                    if col.shape[1] > 10:
                        blocks.append(col)
                return blocks, len(blocks)

        return [img], 1
