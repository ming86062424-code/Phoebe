"""Monitoring: task events, resource samples, the collector and the admin API."""

import json
import subprocess
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.deps import get_chat_client
from app.main import create_app
from app.monitor import store
from app.monitor.chat_trace import ChatTrace
from app.monitor.collector import (
    Collector,
    CpuTracker,
    memory_used,
    parse_nvidia_smi,
    read_gpus,
)
from app.monitor.store import Sample
from app.ollama_client import OllamaStreamError
from tests.fakes import FakeChatClient
from tests.pdf_factory import make_pdf


@pytest.fixture
def db(settings):
    conn = store.open_db(settings.db_path)
    yield conn
    conn.close()


def _events(settings, kind=None):
    conn = store.open_db(settings.db_path)
    try:
        return store.list_events(conn, kind)
    finally:
        conn.close()


# ---- store ----


def test_monitor_db_sits_next_to_app_db(tmp_path):
    assert store.monitor_db_path(str(tmp_path / "x" / "app.db")) == str(tmp_path / "x" / "monitor.db")


def test_record_and_list_events(settings, db):
    store.record_event(settings.db_path, "document", "ok", 100.0, 112.5, "a.pdf",
                       {"parse": 10.12345, "embed": 2.0}, {"pages": 3})
    store.record_event(settings.db_path, "chat", "error", 200.0, 201.0, "嗨", {}, {"error": "x"})
    newest, oldest = store.list_events(db)
    assert (newest["kind"], newest["status"], newest["details"]) == ("chat", "error", {"error": "x"})
    assert oldest["duration"] == 12.5
    assert oldest["phases"] == {"parse": 10.123, "embed": 2.0}
    assert [e["kind"] for e in store.list_events(db, kind="document")] == ["document"]
    assert store.list_events(db, before_id=newest["id"]) == [oldest]
    # events overlapping a time window
    assert [e["kind"] for e in store.list_events(db, since=150, until=300)] == ["chat"]
    assert [e["kind"] for e in store.list_events(db, since=110, until=120)] == ["document"]


def test_record_event_never_raises(tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    store.record_event(str(blocker / "app.db"), "chat", "ok", 1, 2, "x")  # cannot open
    assert "無法記錄監控事件" in caplog.text


def test_series_buckets_keep_the_peak(db):
    for i, mem in enumerate([100, 900, 300, 200]):
        store.insert_samples(db, 1000.0 + i, [Sample("web", "container", mem, 4000, cpu_pct=i)])
    store.insert_samples(db, 1000.0, [Sample("GPU", "gpu", 3000, 6144, gpu_util_pct=50)])
    data = store.series(db, 1000.0, 1000.0 + 720)  # 2 s per bucket
    assert data["step"] == 2.0
    web = next(s for s in data["sources"] if s["source"] == "web")
    assert [p["mem_mib"] for p in web["points"]] == [900, 300]
    assert [p["cpu_pct"] for p in web["points"]] == [1, 3]
    gpu = next(s for s in data["sources"] if s["source"] == "GPU")
    assert (gpu["kind"], gpu["points"][0]["gpu_util_pct"]) == ("gpu", 50)


def test_latest_samples_raw_samples_and_prune(db):
    store.insert_samples(db, 10.0, [Sample("web", "container", 1)])
    store.insert_samples(db, 20.0, [Sample("web", "container", 2), Sample("GPU", "gpu", 3)])
    assert [s["source"] for s in store.latest_samples(db)] == ["web", "GPU"]
    assert [s["ts"] for s in store.list_samples(db, since=0, source="web")] == [20.0, 10.0]
    store.prune(db, sample_hours=1, event_days=1, now=10.0 + 3600 + 1)
    assert [s["ts"] for s in store.list_samples(db, since=0)] == [20.0, 20.0]


# ---- collector ----


def test_memory_used_matches_docker_stats():
    v1 = {"usage": 1000, "stats": {"total_inactive_file": 300}}
    v2 = {"usage": 1000, "stats": {"inactive_file": 100}}
    assert memory_used(v1) == 700
    assert memory_used(v2) == 900
    assert memory_used({"usage": 50, "stats": {"inactive_file": 80}}) == 50
    assert memory_used({}) == 0


def test_cpu_percent_needs_two_readings():
    cpu = CpuTracker()
    first = {"cpu_usage": {"total_usage": 1_000}, "system_cpu_usage": 10_000, "online_cpus": 4}
    second = {"cpu_usage": {"total_usage": 1_500}, "system_cpu_usage": 12_000, "online_cpus": 4}
    assert cpu.percent("a", first) is None
    assert cpu.percent("a", second) == pytest.approx(100.0)  # 500/2000 * 4 cpus
    cpu.forget_except(set())
    assert cpu.percent("a", second) is None


def test_parse_nvidia_smi_and_missing_gpu():
    (gpu,) = parse_nvidia_smi("3444, 6144, 3\n")
    assert (gpu.source, gpu.kind, gpu.mem_mib, gpu.mem_limit_mib, gpu.gpu_util_pct) == (
        "GPU", "gpu", 3444, 6144, 3
    )

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    assert read_gpus(missing) == []

    def failing(*a, **k):
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    assert read_gpus(failing) == []


def _docker(containers):
    """A fake Docker Engine API with `containers`: {name: (usage, inactive, cpu_total)}."""
    calls = {"n": 0}

    def handler(request: httpx.Request):
        if request.url.path == "/containers/json":
            return httpx.Response(200, json=[
                {"Id": name, "Names": [f"/{name}"],
                 "Labels": {"com.docker.compose.project": "phoebe"} if name != "ragflow" else {}}
                for name in containers
            ])
        name = request.url.path.split("/")[2]
        assert request.url.params["one-shot"] == "true"
        usage, inactive, cpu_total = containers[name]
        calls["n"] += 1
        return httpx.Response(200, json={
            "memory_stats": {"usage": usage, "limit": 16 * 2**30, "stats": {"inactive_file": inactive}},
            "cpu_stats": {"cpu_usage": {"total_usage": cpu_total + calls["n"] * 100},
                          "system_cpu_usage": calls["n"] * 1000, "online_cpus": 2},
        })

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://docker")


def test_collector_samples_containers_and_gpu(settings, db):
    gpu = [Sample("GPU", "gpu", 3000, 6144, gpu_util_pct=10)]
    clock = iter([100.0, 105.0])
    collector = Collector(settings.db_path, 5, 48, 30, docker=_docker(
        {"phoebe-web-1": (300 * 2**20, 100 * 2**20, 0), "ragflow": (2**30, 0, 0)}
    ), gpus=lambda: gpu, clock=lambda: next(clock))
    collector.sample_once(db)
    latest = {s["source"]: s for s in store.latest_samples(db)}
    assert latest["phoebe-web-1"]["mem_mib"] == 200
    assert latest["phoebe-web-1"]["project"] == "phoebe"
    assert latest["ragflow"]["project"] == ""
    assert latest["phoebe-web-1"]["cpu_pct"] is None  # first reading
    assert latest["GPU"]["mem_mib"] == 3000
    assert store.collector_status(db) == {"last_sample_at": 100.0, "interval_seconds": 5, "error": ""}
    collector.sample_once(db)
    latest = {s["source"]: s for s in store.latest_samples(db)}
    assert latest["phoebe-web-1"]["cpu_pct"] == pytest.approx(20.0)  # 200/2000 * 2 cpus


def test_collector_reports_docker_errors_and_keeps_the_gpu(settings, db):
    def refused(request):
        raise httpx.ConnectError("permission denied")

    docker = httpx.Client(transport=httpx.MockTransport(refused), base_url="http://docker")
    collector = Collector(settings.db_path, 5, 48, 30, docker=docker,
                          gpus=lambda: [Sample("GPU", "gpu", 1)], clock=lambda: 50.0)
    collector.sample_once(db)
    status = store.collector_status(db)
    assert status["last_sample_at"] == 50.0
    assert "無法讀取 Docker" in status["error"]
    assert [s["source"] for s in store.latest_samples(db)] == ["GPU"]


# ---- task events ----


def test_document_processing_records_an_event(make_user, settings):
    alice = make_user("alice")
    kb = alice.post("/api/kbs", json={"name": "手冊", "parser": "fake"}).json()
    r = alice.post(f"/api/kbs/{kb['id']}/documents",
                   files={"file": ("a.pdf", make_pdf(["hello"]), "application/pdf")})
    assert r.json()["status"] == "ready"
    (event,) = _events(settings, "document")
    assert (event["status"], event["title"]) == ("ok", "a.pdf")
    assert set(event["phases"]) == {"parse", "embed", "store"}
    assert event["details"] == {"kb_id": kb["id"], "document_id": r.json()["id"],
                                "parser": "fake", "pages": 2, "chunks": 2}


def test_failed_document_records_the_error(make_user, settings, fake_parser):
    from app.kb.parsers.base import ParseError

    fake_parser.error = ParseError("壞掉的 PDF")
    alice = make_user("alice")
    kb = alice.post("/api/kbs", json={"name": "手冊", "parser": "fake"}).json()
    alice.post(f"/api/kbs/{kb['id']}/documents",
               files={"file": ("a.pdf", make_pdf(["hello"]), "application/pdf")})
    (event,) = _events(settings, "document")
    assert event["status"] == "error"
    assert event["details"]["error"] == "壞掉的 PDF"


def test_search_test_records_an_event(make_user, settings):
    alice = make_user("alice")
    kb = alice.post("/api/kbs", json={"name": "手冊", "parser": "fake"}).json()
    alice.post(f"/api/kbs/{kb['id']}/documents",
               files={"file": ("a.pdf", make_pdf(["hello"]), "application/pdf")})
    hits = alice.get(f"/api/kbs/{kb['id']}/search", params={"q": "cherry"}).json()
    (event,) = _events(settings, "search")
    assert (event["status"], event["title"]) == ("ok", "搜尋測試：手冊")
    assert set(event["phases"]) == {"embed", "search"}
    assert event["details"]["results"] == len(hits)
    assert event["details"]["user"] == "alice"


def _chat(make_user):
    c = make_user("alice")
    cid = c.post("/api/conversations").json()["id"]
    r = c.post(f"/api/conversations/{cid}/messages", json={"content": "今天天氣如何"})
    return c, cid, r


def test_chat_reply_records_phases_without_the_message(make_user, fake_chat, settings):
    _, cid, r = _chat(make_user)
    assert json.loads(r.text.splitlines()[-1])["type"] == "done"
    (event,) = _events(settings, "chat")
    assert event["status"] == "ok"
    assert event["title"] == "今天天氣如何"  # the conversation title
    assert set(event["phases"]) == {"first_token", "thinking", "answer"}
    assert event["details"] == {"conversation_id": cid, "user": "alice",
                                "provider": "ollama", "model": settings.chat_model}
    assert "天氣" not in json.dumps(event["details"], ensure_ascii=False)


def test_chat_with_a_knowledge_base_records_retrieval(make_user, fake_chat, settings):
    c = make_user("alice")
    kb = c.post("/api/kbs", json={"name": "手冊", "parser": "fake"}).json()
    c.post(f"/api/kbs/{kb['id']}/documents",
           files={"file": ("a.pdf", make_pdf(["hello"]), "application/pdf")})
    cid = c.post("/api/conversations").json()["id"]
    c.patch(f"/api/conversations/{cid}", json={"kb_id": kb["id"]})
    c.post(f"/api/conversations/{cid}/messages", json={"content": "cherry"})
    (event,) = _events(settings, "chat")
    assert {"embed", "search", "images", "first_token", "answer"} <= set(event["phases"])
    assert event["details"]["kb"] == "手冊"
    assert event["details"]["sources"] >= 1


def test_chat_error_mid_stream_is_recorded(make_user, app, settings):
    app.dependency_overrides[get_chat_client] = lambda: FakeChatClient(
        error=OllamaStreamError("boom"), error_after=2
    )
    _chat(make_user)
    (event,) = _events(settings, "chat")
    assert (event["status"], event["details"]["error"]) == ("error", "boom")


def test_chat_trace_finishes_once(settings):
    trace = ChatTrace(settings.db_path, "t", {})
    trace.llm_started()
    trace.delta("content")
    trace.finish("ok")
    trace.finish("cancelled")
    (event,) = _events(settings, "chat")
    assert event["status"] == "ok"
    assert "thinking" not in event["phases"]  # the first delta was already the answer


# ---- API ----


def _admin(settings, kb_services):
    app = create_app(
        replace(settings, admin_username="admin", admin_password="password123"), kb_services
    )
    c = TestClient(app)
    assert c.post("/api/auth/login",
                  json={"username": "admin", "password": "password123"}).status_code == 200
    return c


def test_monitor_api_is_for_admins_only(make_user, client):
    alice = make_user("alice")
    for path in ("events", "status", "series", "samples"):
        assert alice.get(f"/api/monitor/{path}").status_code == 403
        assert client.get(f"/api/monitor/{path}").status_code == 401


def test_monitor_api(settings, kb_services, db):
    import time

    now = time.time()
    store.record_event(settings.db_path, "document", "ok", now - 30, now - 20, "a.pdf")
    store.insert_samples(db, now - 5, [Sample("phoebe-web-1", "container", 200, 4000, 1.0, project="phoebe")])
    admin = _admin(settings, kb_services)

    events = admin.get("/api/monitor/events", params={"kind": "document"}).json()["events"]
    assert [e["title"] for e in events] == ["a.pdf"]
    assert admin.get("/api/monitor/events", params={"kind": "nope"}).status_code == 422

    status = admin.get("/api/monitor/status").json()
    assert status["running"] is False  # the collector never reported
    assert [s["source"] for s in status["latest"]] == ["phoebe-web-1"]
    store.set_collector_status(db, 5, now - 2, "")
    assert admin.get("/api/monitor/status").json()["running"] is True

    series = admin.get("/api/monitor/series", params={"minutes": 1}).json()
    assert [s["source"] for s in series["sources"]] == ["phoebe-web-1"]
    assert [e["title"] for e in series["events"]] == ["a.pdf"]

    samples = admin.get("/api/monitor/samples", params={"source": "phoebe-web-1"}).json()["samples"]
    assert samples[0]["mem_mib"] == 200
