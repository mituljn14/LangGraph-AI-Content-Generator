"""Prompt templates + prompt builder.

Single source of truth: used by the LangGraph orchestration
(backend/graph.py and langgraph/content_generator.ipynb) so the prompts live in one place.
"""
import json

BLOG_PROMPT = """You are an expert professional blog writer.

Generate a high-quality, original, coherent and engaging blog post.

Topic:
{topic}

Tone:
{tone}

Additional Instructions:
{instructions}

Requirements:
- Create an engaging title.
- Write a strong introduction.
- Use clear headings and subheadings.
- Provide useful and relevant information.
- Maintain the requested tone.
- Include examples where appropriate.
- Avoid unnecessary repetition.
- End with a concise conclusion.
- Do not mention that you are an AI.
- Do not explain how the content was generated.
- Return only the final blog post.
- Target approximately 600-800 words unless another length is specified.
"""

SOCIAL_PROMPT = """You are an expert social media content strategist.

Create an engaging social media caption.

Topic:
{topic}

Tone:
{tone}

Additional Instructions:
{instructions}

Requirements:
- Start with a strong attention-grabbing hook.
- Clearly communicate the main idea.
- Keep the caption concise and engaging.
- Match the requested tone.
- Use emojis only when appropriate.
- Add 3-6 relevant hashtags.
- Avoid generic filler and excessive hashtags.
- Do not mention that you are an AI.
- Return only the final caption.
"""

SUMMARY_PROMPT = """You are an expert summarization assistant.

Create a concise and accurate summary.

Topic/Content:
{topic}

Tone:
{tone}

Additional Instructions:
{instructions}

Requirements:
- Extract the most important information.
- Preserve the original meaning.
- Remove unnecessary repetition.
- Use clear and simple language.
- Do not introduce unsupported information.
- Return only the final summary.
"""

REFINE_PROMPT = """You are an expert content editor.

Improve the following previously generated content according to the user's refinement instruction.

Previous Content:
{previous_content}

Refinement Instruction:
{refinement_instruction}

Requirements:
- Preserve the useful information from the original.
- Apply the requested changes carefully.
- Improve clarity, readability and quality.
- Do not introduce unrelated information.
- Maintain the original content type ({content_type}).
- Maintain the requested tone ({tone}).
- Return only the refined content.
"""

REGEN_SUFFIX = """

Regeneration Notice:
This is a regeneration request (variation id: {seed}). Write a completely NEW version:
use a different angle, structure, opening and wording. Do not reuse sentences from the
previous version below.

Previous Version:
{previous_content}
"""

BRANCHES = {
    "Blog Post": "blog",
    "Social Media Caption": "social",
    "Text Summary": "summary",
}
TEMPLATES = {"blog": BLOG_PROMPT, "social": SOCIAL_PROMPT, "summary": SUMMARY_PROMPT}


def parse_payload(raw) -> dict:
    """Payload arrives from FastAPI as a JSON string."""
    if isinstance(raw, dict):
        return raw
    return json.loads(str(raw))


def route(payload: dict) -> str:
    """Return which graph branch handles this request."""
    if payload.get("mode") == "refine":
        return "refine"
    ctype = payload.get("content_type")
    if ctype not in BRANCHES:
        raise ValueError(f"Invalid content type: {ctype}")
    return BRANCHES[ctype]


def build_prompt(payload: dict, branch: str) -> str:
    instructions = (payload.get("instructions") or "").strip() or "None"
    if branch == "refine":
        return REFINE_PROMPT.format(
            previous_content=payload.get("previous_content", ""),
            refinement_instruction=payload.get("refinement_instruction", ""),
            content_type=payload.get("content_type", "the same type"),
            tone=payload.get("tone", "Professional"),
        )
    prompt = TEMPLATES[branch].format(
        topic=payload.get("topic", ""),
        tone=payload.get("tone", "Professional"),
        instructions=instructions,
    )
    if payload.get("mode") == "regenerate":
        prompt += REGEN_SUFFIX.format(
            seed=payload.get("variation_seed", "n/a"),
            previous_content=payload.get("previous_content") or "(not provided)",
        )
    return prompt
