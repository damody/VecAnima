"""Reproducible flat-color baseline; dark-mask outlines are NOT fitted strokes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import cv2
import numpy as np
import resvg_py


@dataclass(frozen=True)
class Settings:
    colors: int = 24
    epsilon: float = 0.65
    stroke_threshold: int = 65
    seed: int = 7


def smooth(image):
    return cv2.bilateralFilter(image, 5, 25, 25)


def fit_palette(images, settings: Settings):
    samples = []
    for image in images:
        lab = cv2.cvtColor(smooth(image), cv2.COLOR_BGR2LAB)
        points = lab.reshape(-1, 3)
        samples.append(points[::max(1, len(points) // 4000)])
    data = np.concatenate(samples).astype(np.float32)
    k = min(settings.colors, len(np.unique(data, axis=0)))
    cv2.setRNGSeed(settings.seed)
    _, _, palette = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                              50, 0.1), 1, cv2.KMEANS_PP_CENTERS)
    return palette


def mask_rings(mask, epsilon):
    # Trace exact pixel-cell edges instead of pixel centers. Center contours shrink
    # single-pixel strokes and leave cracks between adjacent color masks.
    foreground = mask > 0
    height, width = foreground.shape
    padded = np.pad(foreground, 1)
    stride = width + 1
    edges = {}
    neighbors = (padded[:-2,1:-1], padded[1:-1,2:], padded[2:,1:-1], padded[1:-1,:-2])
    for direction, neighbor in enumerate(neighbors):
        yy, xx = np.nonzero(foreground & ~neighbor)
        if direction == 1:
            xx = xx + 1
        elif direction == 2:
            xx, yy = xx + 1, yy + 1
        elif direction == 3:
            yy = yy + 1
        for x, y in zip(xx.tolist(), yy.tolist()):
            edges.setdefault(y * stride + x, []).append(direction)
    offsets = (1, stride, -1, -stride)
    contours = []
    while edges:
        start = current = next(iter(edges))
        previous = edges[start][0]
        points = []
        while True:
            points.append((current % stride, current // stride))
            options = edges[current]
            # At a diagonal contact, take the right turn to keep 4-connected
            # components separate rather than generating a self-touching ring.
            rank = {1:0, 0:1, 3:2, 2:3}
            direction = min(options, key=lambda d: rank[(d-previous) % 4])
            options.remove(direction)
            if not options:
                del edges[current]
            current += offsets[direction]
            previous = direction
            if current == start:
                break
        contours.append(np.array(points, dtype=np.float32).reshape(-1,1,2))
    rings = []
    for contour in contours:
        approximated = cv2.approxPolyDP(contour, epsilon, True)
        ring = approximated.reshape(-1, 2).astype(float)
        if len(ring) < 3 or abs(cv2.contourArea(ring.astype(np.float32))) < 0.1:
            ring = contour.reshape(-1, 2).astype(float)
        if len(ring) >= 3:
            rings.append(ring.tolist())
    return rings


def vectorize(image, palette, settings):
    height, width = image.shape[:2]
    filtered = smooth(image)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    labels = np.empty(len(lab), dtype=np.int32)
    # Bound temporary memory for longer, larger experiments.
    for begin in range(0, len(lab), 16000):
        d = lab[begin:begin + 16000, None] - palette[None]
        labels[begin:begin + 16000] = np.argmin(np.sum(d*d, axis=2), axis=1)
    labels = labels.reshape(height, width)
    colors = cv2.cvtColor(np.uint8(np.clip(palette, 0, 255))[None], cv2.COLOR_LAB2BGR)[0]
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    dark = np.uint8(gray < settings.stroke_threshold) * 255
    # Fill under dark pixels, so dark lines do not punch holes in region colors.
    filled_labels = labels.copy()
    if np.any(dark) and np.any(dark == 0):
        # Nearest non-dark label with OpenCV distance-transform label lookup.
        _, nearest = cv2.distanceTransformWithLabels(dark, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        lookup = labels[dark == 0]
        filled_labels[dark > 0] = lookup[nearest[dark > 0] - 1]
    layers = []
    for i, color in enumerate(colors):
        rings = mask_rings(np.uint8(filled_labels == i) * 255, settings.epsilon)
        if rings:
            layers.append({"kind": "region", "fill": "#" + "".join(f"{int(c):02x}" for c in color[::-1]),
                           "rings": rings})
    rings = mask_rings(dark, settings.epsilon / 2)
    if rings:
        color = np.median(image[dark > 0], axis=0).astype(int)
        layers.append({"kind": "dark-mask-baseline", "fill": "#" + "".join(f"{c:02x}" for c in color[::-1]),
                       "rings": rings})
    # Solid underlay prevents transparent cracks; baseline contours are not yet shared.
    dominant = colors[np.argmax(np.bincount(filled_labels.ravel(), minlength=len(colors)))]
    background = "#" + "".join(f"{int(c):02x}" for c in dominant[::-1])
    return {"version": 1, "model": "flat-regions-and-dark-mask-baseline", "width": width,
            "height": height, "background": background, "settings": asdict(settings), "layers": layers}


def svg(project):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{project["width"]}" height="{project["height"]}" '
             f'viewBox="0 0 {project["width"]} {project["height"]}">',
             f'<rect width="100%" height="100%" fill="{project["background"]}"/>']
    for kind in ("region", "dark-mask-baseline"):
        parts.append(f'<g id="{kind}">')
        for layer in project["layers"]:
            if layer["kind"] != kind:
                continue
            commands = []
            for ring in layer["rings"]:
                commands.append("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in ring) + " Z")
            parts.append(f'<path fill="{layer["fill"]}" fill-rule="evenodd" d="{" ".join(commands)}"/>')
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def rasterize(text):
    png = resvg_py.svg_to_bytes(svg_string=text, skip_system_fonts=True)
    return png, cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
