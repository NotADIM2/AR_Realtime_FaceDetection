"""The filter list, and where each overlay sits on the face."""

from app import config
from app.filters.drawn import BlushFilter, MeshFilter, NormalFilter, OutlineFilter
from app.filters.overlay import OverlayFilter, OverlaySpec
from app.vision import landmarks as lm

OVERLAY_SPECS = (
    OverlaySpec("ar_glasses", "AR Glasses", "ar_glasses.png", lm.NOSE_BRIDGE, width=1.72),
    OverlaySpec("sunglasses", "Sunglasses", "sunglasses.png", lm.NOSE_BRIDGE, width=1.72, offset=(0, 0.04, 0)),
    OverlaySpec("red_glasses", "Red Glasses", "red_glasses.png", lm.NOSE_BRIDGE, width=1.72),
    OverlaySpec("moustache", "Moustache", "moustache.png", lm.PHILTRUM, width=0.78, pivot=(0.5, 0.42)),
    OverlaySpec("halo", "Halo", "halo.png", lm.FOREHEAD_TOP, width=1.6, offset=(0, -0.95, 0.35)),
    OverlaySpec("crown", "Crown", "crown.png", lm.FOREHEAD_TOP, width=1.45, pivot=(0.5, 1.0), offset=(0, -0.08, 0), lean=0.15),
)


class FilterManager:
    def __init__(self, filter_dir=config.FILTER_DIR):
        overlays = [OverlayFilter(spec, filter_dir / spec.asset) for spec in OVERLAY_SPECS]
        self._filters = [NormalFilter(), *overlays, BlushFilter(), MeshFilter(), OutlineFilter()]
        self._by_key = {f.key: f for f in self._filters}

    def get(self, key):
        return self._by_key.get(key)

    def is_usable(self, key):
        item = self.get(key)
        return item is not None and item.problem is None

    def describe(self):
        """A missing asset disables that filter rather than crashing the app."""
        return [
            {"key": f.key, "label": f.label, "available": f.problem is None, "problem": f.problem}
            for f in self._filters
        ]
