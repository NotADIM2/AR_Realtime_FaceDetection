"""Frame rate measurement."""

import time
from collections import deque


class FpsCounter:
    """Measures frames per second over a sliding time window."""

    def __init__(self, window_s=1.5):
        self._window_s = window_s
        self._times = deque()

    def tick(self):
        now = time.monotonic()
        self._times.append(now)
        while now - self._times[0] > self._window_s:
            self._times.popleft()

    @property
    def fps(self):
        if len(self._times) < 2:
            return 0.0
        elapsed = time.monotonic() - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
