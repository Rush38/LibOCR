import hashlib
from collections import OrderedDict
from typing import Optional, Any


class PreprocessingCache:
    def __init__(self, maxsize: int = 100):
        self._maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def _make_key(image_bytes: bytes) -> str:
        return hashlib.md5(image_bytes[:10240]).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()
