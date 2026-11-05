"""One Euro filter for reducing landmark jitter.

This is a low-pass filter whose cutoff frequency adapts to speed:

    cutoff = min_cutoff + beta * |speed|
    a      = 1 / (1 + (1 / (2 * pi * cutoff)) / dt)
    x_hat  = a * x + (1 - a) * x_hat_previous

A still face gets a low cutoff, which removes jitter. A moving face gets a high
cutoff, which keeps lag small. Reference: Casiez, Roussel and Vogel, CHI 2012.
"""

import numpy as np


def _smoothing_factor(cutoff, dt):
    return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))


class OneEuroFilter:
    """Filters an array of any shape, with each element adapting independently."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._value = None
        self._derivative = None
        self._time = None

    def __call__(self, value, time_s):
        if self._value is None:
            self._value = value.copy()
            self._derivative = np.zeros_like(value)
            self._time = time_s
            return self._value.copy()

        dt = max(time_s - self._time, 1e-3)
        self._time = time_s

        raw_derivative = (value - self._value) / dt
        a_d = _smoothing_factor(self.d_cutoff, dt)
        self._derivative = a_d * raw_derivative + (1.0 - a_d) * self._derivative

        cutoff = self.min_cutoff + self.beta * np.abs(self._derivative)
        a = _smoothing_factor(cutoff, dt)
        self._value = a * value + (1.0 - a) * self._value
        return self._value.copy()
