import hashlib
import re


class FakeEmbedder:
    """Deterministic bag-of-words embedder so retrieval order is predictable in tests.

    Subclasses override `query_vector` / `document_vector` to shape the vectors.
    """

    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        if not any(v):
            v[0] = 1.0
        return v

    def query_vector(self, text: str) -> list[float]:
        return self._vec(text)

    def document_vector(self, title: str, text: str) -> list[float]:
        return self._vec(f"{title} {text}")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self.query_vector(t) for t in texts]

    def embed_documents(self, title: str, texts: list[str]) -> list[list[float]]:
        return [self.document_vector(title, t) for t in texts]

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        return [self._vec(hashlib.md5(i).hexdigest()) for i in images]

    def embed_image_texts(self, items: list[tuple[bytes, str]]) -> list[list[float]]:
        return [self._vec(hashlib.md5(i).hexdigest() + " " + t) for i, t in items]
