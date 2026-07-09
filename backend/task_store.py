"""Thread-safe, JSON file-backed task status store.

Replaces in-memory dicts (ingestion_tasks, batch_tasks, polishing_tasks)
so that task progress survives server restarts on Railway.
"""

import json
import os
import threading
import time


class TaskStore:
    """Persistent task status storage backed by a JSON file.

    Features:
        - Thread-safe via threading.Lock (FastAPI BackgroundTasks use thread pool)
        - Atomic writes (temp file + os.replace) to prevent corruption on crash
        - Auto-cleanup of completed/failed tasks older than `cleanup_hours`
    """

    def __init__(self, filepath, cleanup_hours=1):
        self._filepath = filepath
        self._lock = threading.Lock()
        self._cleanup_hours = cleanup_hours
        self._data = self._load()
        self._cleanup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key, default=None):
        """Returns the task dict for `key`, or `default` if not found."""
        with self._lock:
            return self._data.get(key, default)

    def set(self, key, value):
        """Creates or overwrites the task entry for `key`."""
        with self._lock:
            value["_updated_at"] = time.time()
            self._data[key] = value
            self._save()

    def update(self, key, partial):
        """Merges `partial` dict into the existing task entry for `key`."""
        with self._lock:
            if key not in self._data:
                self._data[key] = {}
            self._data[key].update(partial)
            self._data[key]["_updated_at"] = time.time()
            self._save()

    def update_nested(self, key, path, partial):
        """Updates a nested dict inside the task entry.

        Example:
            update_nested("task_1", ["videos", 0], {"status": "done"})
            # Equivalent to: data["task_1"]["videos"][0].update({"status": "done"})
        """
        with self._lock:
            if key not in self._data:
                return
            target = self._data[key]
            for step in path:
                try:
                    target = target[step]
                except (KeyError, IndexError, TypeError):
                    return
            if isinstance(target, dict):
                target.update(partial)
            self._data[key]["_updated_at"] = time.time()
            self._save()

    def delete(self, key):
        """Removes a task entry."""
        with self._lock:
            self._data.pop(key, None)
            self._save()

    def __contains__(self, key):
        """Supports `key in store` syntax."""
        with self._lock:
            return key in self._data

    def keys(self):
        """Returns a list of all task keys."""
        with self._lock:
            return list(self._data.keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self):
        """Loads data from the JSON file, returning empty dict on failure."""
        if os.path.exists(self._filepath):
            try:
                with open(self._filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load task store {self._filepath}: {e}")
        return {}

    def _save(self):
        """Atomically writes data to disk (write tmp → os.replace)."""
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        tmp_path = self._filepath + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp_path, self._filepath)
        except Exception as e:
            print(f"Warning: Failed to save task store {self._filepath}: {e}")
            # Clean up temp file on failure
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _cleanup(self):
        """Removes completed/failed tasks older than `cleanup_hours`."""
        if not self._cleanup_hours:
            return
        now = time.time()
        cutoff = now - (self._cleanup_hours * 3600)
        keys_to_remove = []
        for key, value in self._data.items():
            if not isinstance(value, dict):
                continue
            status = value.get("status", "")
            updated = value.get("_updated_at", 0)
            if status in ("completed", "failed", "completed_with_errors") and updated < cutoff:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._data[key]
        if keys_to_remove:
            self._save()
