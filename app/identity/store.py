"""Saves registered face templates on this computer.

Only names and numeric templates are stored, never photos. The file is outside the
project folder, so it can't be committed to Git by accident. It isn't encrypted.
"""

import json
import os
import threading

import numpy as np

from app import config

_FORMAT_VERSION = 1
_EMBEDDING_SIZE = 128


class FaceStore:
    def __init__(self, path=config.FACE_STORE_PATH):
        self.path = path
        self.problem = None
        self._lock = threading.Lock()
        self._profiles = {}
        self._load()

    def _load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._profiles = {
                str(name): np.asarray(values, dtype=np.float64)
                for name, values in data["profiles"].items()
                if len(values) == _EMBEDDING_SIZE
            }
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            # Keep the damaged file aside so it isn't overwritten by the next save.
            self.problem = f"Saved faces could not be read ({type(exc).__name__}). Please register again."
            try:
                os.replace(self.path, self.path.with_name(self.path.name + ".damaged"))
            except OSError:
                pass

    def _save(self):
        """Write to a temporary file first, so a crash can't leave a half-written file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": _FORMAT_VERSION,
            "model": config.RECOGNITION_MODEL_PATH.name,
            "profiles": {name: values.tolist() for name, values in self._profiles.items()},
        }
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(data), encoding="utf-8")
        os.replace(temporary, self.path)

    def _existing_name(self, name):
        return next((n for n in self._profiles if n.casefold() == name.casefold()), None)

    def names(self):
        with self._lock:
            return sorted(self._profiles, key=str.casefold)

    def add(self, name, template):
        """Save a template. Registering an existing name replaces its template."""
        with self._lock:
            existing = self._existing_name(name)
            if existing is not None:
                del self._profiles[existing]
            self._profiles[name] = np.asarray(template, dtype=np.float64)
            self._save()
            self.problem = None

    def remove(self, name):
        with self._lock:
            existing = self._existing_name(name)
            if existing is None:
                return False
            del self._profiles[existing]
            self._save()
            return True

    def best_match(self, template):
        """Return (name, similarity) for the closest profile, or None if there are none."""
        with self._lock:
            if not self._profiles:
                return None
            scores = {name: float(np.dot(values, template)) for name, values in self._profiles.items()}
        best = max(scores, key=scores.get)
        return best, scores[best]
