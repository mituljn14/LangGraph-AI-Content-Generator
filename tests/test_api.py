"""Run: pytest -q   (LangGraph is mocked; no network/API key needed)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("GEMINI_API_KEY", "test")

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from backend import main
from prompts.prompts import build_prompt, route

client = TestClient(main.app)
TOPICS = ["Artificial Intelligence in Healthcare", "Sustainable Technology", "Benefits of Remote Learning"]
CASES = list(zip(TOPICS, ["Blog Post", "Social Media Caption", "Text Summary"], ["Professional", "Inspirational", "Friendly"]))


@pytest.fixture(autouse=True)
def fake_langgraph(monkeypatch):
    seen = []
    async def fake(payload):
        seen.append(payload)
        return f"[{payload['mode']}] fake content for words test"
    monkeypatch.setattr(main, "call_langgraph", fake)
    return seen


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


@pytest.mark.parametrize("topic,ctype,tone", CASES)
def test_generate_regenerate_refine(topic, ctype, tone):
    body = {"content_type": ctype, "topic": topic, "tone": tone, "instructions": ""}
    r = client.post("/api/generate", json=body); d = r.json()
    assert r.status_code == 200 and d["success"] and d["word_count"] == len(d["content"].split())
    assert d["character_count"] == len(d["content"])
    assert client.post("/api/regenerate", json={**body, "previous_content": d["content"]}).status_code == 200
    ref = {"content_type": ctype, "topic": topic, "tone": tone, "previous_content": d["content"],
           "refinement_instruction": "Make it shorter"}
    assert client.post("/api/refine", json=ref).status_code == 200


def test_regenerate_gets_new_seed(fake_langgraph):
    body = {"content_type": "Blog Post", "topic": "x"}
    client.post("/api/regenerate", json=body); client.post("/api/regenerate", json=body)
    assert fake_langgraph[0]["variation_seed"] != fake_langgraph[1]["variation_seed"]


def test_empty_topic():
    r = client.post("/api/generate", json={"content_type": "Blog Post", "topic": "  "})
    assert r.status_code == 400 and "Topic" in r.json()["detail"]


def test_invalid_content_type():
    r = client.post("/api/generate", json={"content_type": "Poem", "topic": "x"})
    assert r.status_code == 400 and "Invalid content type" in r.json()["detail"]


def test_invalid_request():
    assert client.post("/api/generate", content="not json", headers={"Content-Type": "application/json"}).status_code == 400


def test_refine_requires_fields():
    assert client.post("/api/refine", json={"content_type": "Blog Post"}).status_code == 400


def test_api_failure(monkeypatch):
    async def boom(_): raise HTTPException(502, "LangGraph/Gemini error (500): quota")
    monkeypatch.setattr(main, "call_langgraph", boom)
    r = client.post("/api/generate", json={"content_type": "Blog Post", "topic": "x"})
    assert r.status_code == 502 and "quota" in r.json()["detail"]


def test_prompt_routing():
    p = {"mode": "generate", "content_type": "Text Summary", "topic": "T", "tone": "Formal"}
    assert route(p) == "summary" and "summarization assistant" in build_prompt(p, "summary")
    assert route({"mode": "refine"}) == "refine"
    with pytest.raises(ValueError): route({"content_type": "bad"})


