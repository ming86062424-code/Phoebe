import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def settings(tmp_path):
    return Settings(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def app(settings):
    return create_app(settings)


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
