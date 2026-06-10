import json
from pathlib import Path
from typing import Dict, Any, List


class BatchState:
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8-sig") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"completed": [], "failed": [], "total": 0}

    def save(self):
        with open(self.state_file, "w", encoding="utf-8-sig") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def mark_completed(self, filepath):
        self.data.setdefault("completed", [])
        key = str(filepath)
        if key not in self.data["completed"]:
            self.data["completed"].append(key)
        self.data["total"] = len(self.data["completed"]) + len(
            self.data.get("failed", [])
        )
        self.save()

    def mark_failed(self, filepath, error):
        self.data.setdefault("failed", [])
        key = str(filepath)
        if key not in [f["file"] for f in self.data["failed"]]:
            self.data["failed"].append({"file": key, "error": error})
        self.save()

    def is_completed(self, filepath):
        return str(filepath) in self.data.get("completed", [])

    def get_remaining(self, all_files):
        completed = set(self.data.get("completed", []))
        return [f for f in all_files if str(f) not in completed]

    def reset(self):
        self.data = {"completed": [], "failed": [], "total": 0}
        self.save()


def load_batch_state(output_dir: Path) -> BatchState:
    state_file = Path(__file__).parent / ".batch_state.json"
    return BatchState(state_file)
