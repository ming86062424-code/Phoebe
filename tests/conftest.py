import os
from pathlib import Path

os.environ["ENV_FILE"] = str(Path(__file__).parent / "no-such.env")

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def settings(tmp_path):
    return Settings(
        db_path=str(tmp_path / "test.db"),
        frontend_dist=str(tmp_path / "no-dist"),
        kb_dir=str(tmp_path / "kb"),
        chroma_dir=str(tmp_path / "chroma"),
    )


@pytest.fixture
def fake_parser():
    from tests.fakes import FakeParser

    return FakeParser()


@pytest.fixture
def kb_services(settings, fake_parser):
    from app.kb.parsers.liteparse_parser import LiteParseParser
    from app.kb.services import build_kb_services
    from tests.fakes import KeywordEmbedding

    return build_kb_services(
        settings,
        embed_model=KeywordEmbedding(),
        parsers=[LiteParseParser("eng"), fake_parser],
        threaded=False,
    )


@pytest.fixture
def app(settings, kb_services):
    return create_app(settings, kb_services)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def make_user(app):
    def _make(username):
        c = TestClient(app)
        creds = {"username": username, "password": "password123"}
        assert c.post("/api/auth/register", json=creds).status_code == 201
        assert c.post("/api/auth/login", json=creds).status_code == 200
        return c

    return _make


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def fake_chat(app):
    from app.deps import get_chat_client
    from tests.fakes import FakeChatClient

    fake = FakeChatClient()
    app.dependency_overrides[get_chat_client] = lambda: fake
    return fake
