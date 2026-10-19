"""Which uploads the knowledge base accepts and how their files are named."""

from pathlib import Path

PDF_EXTENSIONS = frozenset({".pdf"})
TEXT_EXTENSIONS = frozenset({".txt", ".md"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
UPLOAD_EXTENSIONS = PDF_EXTENSIONS | TEXT_EXTENSIONS | IMAGE_EXTENSIONS


def file_extension(filename: str) -> str:
    """The lower-case upload extension of a filename, ".pdf" for anything else.

    Every stored file was accepted by the upload whitelist, so documents uploaded before
    other types existed (and names cut short) are PDFs.
    """
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in UPLOAD_EXTENSIONS else ".pdf"
