import json
from pathlib import Path
from typing import Dict, Any, List, Optional


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
        return {"completed": [], "failed": [], "total": 0, "started": False}

    def save(self):
        with open(self.state_file, "w", encoding="utf-8-sig") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def mark_completed(self, filename: str, result_data: Dict):
        self.data.setdefault("completed", [])
        if filename not in self.data["completed"]:
            self.data["completed"].append(filename)
        self.data["total"] = len(self.data["completed"]) + len(
            self.data.get("failed", [])
        )
        self.save()

    def mark_failed(self, filename: str, error: str):
        self.data.setdefault("failed", [])
        if filename not in [f["file"] for f in self.data["failed"]]:
            self.data["failed"].append({"file": filename, "error": error})
        self.save()

    def is_completed(self, filename: str) -> bool:
        return filename in self.data.get("completed", [])

    def get_remaining(self, all_files: List[Path]) -> List[Path]:
        completed = set(self.data.get("completed", []))
        return [f for f in all_files if f.name not in completed]

    def reset(self):
        self.data = {"completed": [], "failed": [], "total": 0, "started": False}
        self.save()


def save_batch_state(output_dir: Path, state: BatchState):
    state.save()


def load_batch_state(output_dir: Path) -> BatchState:
    state_file = output_dir / ".batch_state.json"
    return BatchState(state_file)
