import cv2
import numpy as np
from typing import Optional


def sauvola_threshold(
    image: np.ndarray, window_size: int = 31, k: float = 0.2, r: Optional[float] = None
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
    sq_mean = cv2.boxFilter(gray**2, cv2.CV_32F, (window_size, window_size))
    variance = sq_mean - mean**2
    variance = np.maximum(variance, 0)
    std = np.sqrt(variance)
    r_val = r if r is not None else (std.max() if std.max() > 0 else 128)
    threshold = mean * (1.0 + k * ((std / r_val) - 1.0))
    binary = (gray > threshold).astype(np.uint8) * 255
    return binary


def niblack_threshold(
    image: np.ndarray, window_size: int = 31, k: float = -0.2
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
    sq_mean = cv2.boxFilter(gray**2, cv2.CV_32F, (window_size, window_size))
    std = np.sqrt(np.maximum(sq_mean - mean**2, 0))
    threshold = mean + k * std
    binary = (gray > threshold).astype(np.uint8) * 255
    return binary


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_profile(image: np.ndarray, profile: str) -> np.ndarray:
    if profile == "old_document":
        return sauvola_threshold(image, window_size=41, k=0.25)
    elif profile == "photo":
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return image
