"""Decoding uploaded images and writing the PNGs the parsers hand to the index."""

from pathlib import Path

from app.kb.parsers.base import ImageRef, ParseError

# Largest image (in pixels) that is ever decoded: a 50 MP RGB image is ~150 MB in memory.
MAX_PIXELS = 50_000_000

# Upload extension -> the PIL format the content must actually be.
IMAGE_FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}

UNREADABLE = "無法讀取檔案內容"


def check_pixels(width: int, height: int) -> None:
    if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
        raise ParseError(UNREADABLE)


def open_image(path: str | Path, suffix: str | None = None):
    """Fully decode an image file whose content matches its extension (`suffix`, by
    default the path's own).

    Returns an upright (EXIF transposed) RGB PIL image; raises ParseError otherwise.
    The size is checked from the header before any pixel is decoded.
    """
    from PIL import Image, ImageOps

    expected = IMAGE_FORMATS.get((suffix or Path(path).suffix).lower())
    if expected is None:
        raise ParseError(UNREADABLE)
    try:
        with Image.open(path) as image:
            if image.format != expected:
                raise ParseError(UNREADABLE)
            check_pixels(image.width, image.height)
            image.load()
            upright = ImageOps.exif_transpose(image)
            return to_rgb(upright)
    except ParseError:
        raise
    except Exception as e:  # PIL raises many kinds (including decompression bombs)
        raise ParseError(UNREADABLE) from e


def to_rgb(image):
    """RGB, with any transparency flattened onto white (a transparent PNG stays readable)."""
    from PIL import Image

    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def fit(image, longest_edge: int):
    """Downscale so that max(width, height) <= longest_edge, keeping the aspect ratio.

    Never upscales (the same rule as the official LightOnOCR client's render.fit).
    """
    from PIL import Image

    w, h = image.size
    scale = longest_edge / max(w, h)
    if scale >= 1:
        return image
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def save_png(image, out_dir: Path, name: str) -> ImageRef:
    """Write `image` as out_dir/name and return its reference (the name is validated)."""
    ref = ImageRef(name, image.width, image.height)
    image.save(Path(out_dir) / name, format="PNG")
    return ref
