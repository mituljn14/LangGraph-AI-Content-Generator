"""Compiled LangGraph orchestration for the AI Content Generator."""
from __future__ import annotations
import os, re
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from google import genai
from google.genai import types
from prompts.prompts import build_prompt, route

load_dotenv()

class ContentState(TypedDict, total=False):
    mode: str
    content_type: str
    topic: str
    tone: str
    instructions: str
    previous_content: str | None
    refinement_instruction: str | None
    variation_seed: str
    branch: str
    prompt: str
    response: str
    content: str

def route_node(state: ContentState) -> dict:
    return {"branch": route(state)}

def blog_prompt_node(state: ContentState) -> dict:
    return {"prompt": build_prompt(state, "blog")}

def social_prompt_node(state: ContentState) -> dict:
    return {"prompt": build_prompt(state, "social")}

def summary_prompt_node(state: ContentState) -> dict:
    return {"prompt": build_prompt(state, "summary")}

def refine_prompt_node(state: ContentState) -> dict:
    return {"prompt": build_prompt(state, "refine")}

def generate_node(state: ContentState) -> dict:
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_api_key":
        raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env.")
    client = genai.Client(api_key=key)
    result = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        contents=state["prompt"],
        config=types.GenerateContentConfig(
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.9"))
        ),
    )
    text = (result.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty output.")
    return {"response": text}

def format_node(state: ContentState) -> dict:
    text = re.sub(r"^```(?:markdown|md|text)?\s*|\s*```$", "", state.get("response", "").strip())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise RuntimeError("Output formatter received empty output.")
    return {"content": text}

def build_graph():
    builder = StateGraph(ContentState)
    builder.add_node("router", route_node)
    builder.add_node("blog_prompt", blog_prompt_node)
    builder.add_node("social_prompt", social_prompt_node)
    builder.add_node("summary_prompt", summary_prompt_node)
    builder.add_node("refine_prompt", refine_prompt_node)
    builder.add_node("gemini", generate_node)
    builder.add_node("formatter", format_node)
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        lambda state: state["branch"],
        {"blog": "blog_prompt", "social": "social_prompt",
         "summary": "summary_prompt", "refine": "refine_prompt"},
    )
    builder.add_edge("blog_prompt", "gemini")
    builder.add_edge("social_prompt", "gemini")
    builder.add_edge("summary_prompt", "gemini")
    builder.add_edge("refine_prompt", "gemini")
    builder.add_edge("gemini", "formatter")
    builder.add_edge("formatter", END)
    return builder.compile()

GRAPH = build_graph()

def run_graph(payload: dict) -> str:
    return GRAPH.invoke(payload)["content"]
