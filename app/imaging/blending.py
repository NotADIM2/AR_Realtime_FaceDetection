"""Alpha compositing with NumPy.

Standard "over" compositing of foreground F onto background B is:

    C = alpha * F + (1 - alpha) * B

Sprites are stored with premultiplied alpha (F' = alpha * F), which reduces it to:

    C = F' + (1 - alpha) * B

Premultiplying matters during warping. Interpolating straight-alpha pixels mixes
in the colour of fully transparent pixels and leaves a dark fringe around the
sprite. Premultiplied pixels interpolate correctly.
"""

import numpy as np


def premultiply(bgra):
    """Convert a straight-alpha BGRA uint8 image to premultiplied alpha."""
    alpha = bgra[..., 3:4].astype(np.uint16)
    color = (bgra[..., :3].astype(np.uint16) * alpha + 127) // 255
    return np.dstack((color.astype(np.uint8), bgra[..., 3]))


def composite_premultiplied(roi, overlay):
    """Composite a premultiplied BGRA overlay onto a BGR ROI, in place.

    Integer arithmetic is used because it runs once per frame on every ROI pixel.
    """
    inverse_alpha = 255 - overlay[..., 3:4].astype(np.uint16)
    background = (roi.astype(np.uint16) * inverse_alpha + 127) // 255
    roi[:] = np.minimum(background + overlay[..., :3], 255).astype(np.uint8)
