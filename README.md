# ✍️ AI Content Generator

A modular, production-ready AI content generation platform powered by **LangGraph**, **FastAPI**, **Streamlit**, and the modern **Google GenAI SDK** (`gemini-3.5-flash`).

Built with an orchestrated state graph in native Python, this application dynamically routes content creation tasks into specialized prompt pipelines for blog posts, social media captions, summaries, iterative refinements, and alternate variations.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [LangGraph Workflow & State Management](#-langgraph-workflow--state-management)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [Interactive Notebook](#-interactive-notebook)
- [Running Tests](#-running-tests)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🌟 Overview

The **AI Content Generator** provides an end-to-end multi-tier architecture designed for flexible text generation:

1. **Streamlit Frontend**: An intuitive dual-panel dashboard allowing users to select content formats, configure tone and custom instructions, inspect output statistics (word, character, and paragraph counts), download output files, and iteratively refine or regenerate drafts.
2. **FastAPI Backend**: An asynchronous REST API that validates input payloads via Pydantic, applies CORS policies, and delegates graph invocations asynchronously using Python worker threads.
3. **LangGraph Engine**: A compiled directed state graph that inspects incoming generation modes (`generate`, `regenerate`, `refine`) and content types to execute specialized prompt graph branches before querying Google Gemini and sanitizing responses.
4. **Google GenAI Integration**: Direct integration with Gemini models using the official `google-genai` SDK with temperature controls and structured generation.

> **Note on Architecture:** This project runs pure, native Python LangGraph (`StateGraph`). It does **not** rely on external workflow servers, LangFlow servers, or third-party cloud flow runners. The graph is defined, compiled, and executed locally within the backend process.

---

## ✨ Key Features

- **Multi-Format Generation**:
  - **Blog Posts**: Structured, long-form articles (600–800 words) with headlines, introductions, body sections, and conclusions.
  - **Social Media Captions**: High-impact hooks, platform-ready copy, concise messaging, and relevant hashtags (3–6 tags).
  - **Text Summaries**: Concise synthesis preserving core points without hallucinations or repetitive filler.
- **Dynamic Graph Routing**:
  - A dedicated `router` node routes states into specialized prompt nodes (`blog_prompt`, `social_prompt`, `summary_prompt`, `refine_prompt`) through conditional edges.
- **Intelligent Refinement**:
  - Edit existing drafts with natural-language instructions (e.g., *"Make it punchier"*, *"Shorten the introduction"*) while preserving context and tone.
- **Smart Regeneration**:
  - Generates distinct alternative versions using unique variation seeds (`variation_seed`) and explicit divergence prompts.
- **Text Analytics**:
  - Real-time word count, character count, and paragraph metrics returned via API and displayed in the frontend.
- **Clean Markdown Output**:
  - Automatic post-processing node that cleans code block wrappers, unescapes formatting, and normalizes spacing.
- **Thoroughly Tested**:
  - Pytest test suite with mock adapters verifying validation, prompt assembly, seed handling, and API failure modes.

---

## 🏗️ System Architecture

```text
┌────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                   │
│             (http://localhost:8501)                    │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP POST (JSON)
                            ▼
┌────────────────────────────────────────────────────────┐
│                    FastAPI Backend                     │
│             (http://localhost:8000)                    │
│   • Request Validation (Pydantic)                      │
│   • CORS & Error Handling Middleware                   │
│   • Asynchronous Execution (asyncio.to_thread)         │
└───────────────────────────┬────────────────────────────┘
                            │ GRAPH.invoke(payload)
                            ▼
┌────────────────────────────────────────────────────────┐
│               Compiled LangGraph Engine                │
│                                                        │
│                       [ START ]                        │
│                           │                            │
│                           ▼                            │
│                    [ Router Node ]                     │
│                           │                            │
│         ┌─────────────┬───┴─────────────┬────────────┐ │
│         ▼             ▼                 ▼            ▼ │
│    [Blog Node]  [Social Node]     [Summary Node] [Refine Node]
│         └─────────────┬───┬─────────────┴────────────┘ │
│                           │                            │
│                           ▼                            │
│                  [ Gemini LLM Node ]                   │
│                  (google-genai SDK)                    │
│                           │                            │
│                           ▼                            │
│               [ Formatter / Sanitize Node ]            │
│                           │                            │
│                           ▼                            │
│                        [ END ]                         │
└────────────────────────────────────────────────────────┘
```

---

## 🔄 LangGraph Workflow & State Management

The core graph is implemented in [`backend/graph.py`](file:///c:/Users/mitul%20jain/Desktop/AI%20INTERN/ai-content-generator-final-local/backend/graph.py) using `langgraph.graph.StateGraph`.

### State Schema (`ContentState`)

```python
class ContentState(TypedDict, total=False):
    mode: str                          # "generate", "regenerate", or "refine"
    content_type: str                  # "Blog Post", "Social Media Caption", "Text Summary"
    topic: str                         # Content subject
    tone: str                          # "Professional", "Casual", "Creative", etc.
    instructions: str                  # User instructions / constraints
    previous_content: str | None       # Pre-existing draft (for regenerate/refine)
    refinement_instruction: str | None # Directive for refinement
    variation_seed: str                # Unique identifier to ensure variance
    branch: str                        # Routed branch identifier
    prompt: str                        # Compiled prompt string
    response: str                      # Raw LLM response from Gemini
    content: str                       # Sanitized final output
```

### Graph Lifecycle

1. **`router`**: Inspects `state["mode"]` and `state["content_type"]`. Maps requests to `"blog"`, `"social"`, `"summary"`, or `"refine"`.
2. **Branch Prompt Nodes**:
   - `blog_prompt`: Loads structured long-form article template with target word counts and outline guidelines.
   - `social_prompt`: Formats high-engagement caption prompt with hook and hashtag constraints.
   - `summary_prompt`: Applies precision extraction template.
   - `refine_prompt`: Injects existing text along with modification instructions.
   - *Regeneration Mode*: Appends a unique divergence suffix referencing the previous version and variation seed.
3. **`gemini`**: Calls `google.genai.Client.models.generate_content` using the specified model (e.g. `gemini-3.5-flash`) and configured temperature.
4. **`formatter`**: Strips markdown code wrapper blocks (```markdown ... ```), standardizes newline whitespace, and validates payload presence.

---

## 📂 Repository Structure

```text
ai-content-generator-final-local/
├── backend/
│   ├── __init__.py
│   ├── graph.py               # Native LangGraph graph definition, compilation & invoke handler
│   └── main.py                # FastAPI REST endpoints, Pydantic schemas, and error handlers
├── frontend/
│   └── app.py                 # Streamlit UI with responsive dual-pane layout & state management
├── langgraph/
│   └── content_generator.ipynb# Standalone Jupyter notebook for graph visualization & step-by-step runs
├── prompts/
│   ├── __init__.py
│   └── prompts.py             # Reusable prompt templates, branch routing logic & prompt builder
├── tests/
│   └── test_api.py            # Comprehensive Pytest suite with mocked LangGraph execution
├── .env.example               # Template environment configuration file
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```

---

## 📋 Prerequisites

- **Python**: `3.10` or higher
- **Gemini API Key**: Obtain a key from [Google AI Studio](https://aistudio.google.com/)

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-content-generator-final-local
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Copy the example environment file and configure your credentials:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Open `.env` and configure the following variables:

```env
# Required: Google Gemini API key
GEMINI_API_KEY=your_gemini_api_key_here

# Optional model selection (default: gemini-3.5-flash)
GEMINI_MODEL=gemini-3.5-flash

# Optional temperature (default: 0.9 for creative generation)
GEMINI_TEMPERATURE=0.9

# FastAPI backend URL for Streamlit
API_URL=http://localhost:8000
```

---

## 🏃 Running the Application

To run the full stack, start both the backend API and frontend client in separate terminal windows with your virtual environment activated.

### Terminal 1 — Start the FastAPI Backend

```powershell
uvicorn backend.main:app --reload --port 8000
```

- API Base URL: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`
- Redoc Documentation: `http://localhost:8000/redoc`

### Terminal 2 — Start the Streamlit Frontend

```powershell
streamlit run frontend/app.py
```

- The UI will open automatically at: `http://localhost:8501`

---

## 📡 API Reference

All requests and responses use JSON.

### 1. Health Check

Checks service status and validates whether `GEMINI_API_KEY` is loaded.

- **Endpoint**: `GET /api/health`
- **Response**:
```json
{
  "status": "ok",
  "gemini_key_configured": true,
  "orchestration": "LangGraph",
  "gemini_model": "gemini-3.5-flash"
}
```

---

### 2. Generate Content

Generates fresh content based on topic, content type, tone, and optional instructions.

- **Endpoint**: `POST /api/generate`
- **Request Body**:
```json
{
  "content_type": "Blog Post",
  "topic": "The Impact of Quantum Computing on Cryptography",
  "tone": "Professional",
  "instructions": "Include a brief real-world example of post-quantum encryption standards."
}
```

- **Supported Content Types**:
  - `"Blog Post"`
  - `"Social Media Caption"`
  - `"Text Summary"`

- **Supported Tones**:
  - `"Professional"`, `"Friendly"`, `"Casual"`, `"Creative"`, `"Formal"`, `"Persuasive"`, `"Inspirational"`

- **Response**:
```json
{
  "success": true,
  "content": "# The Quantum Shift: Safeguarding Tomorrow's Cryptography\n\n...",
  "word_count": 684,
  "character_count": 4512
}
```

---

### 3. Regenerate Content

Creates a completely distinct alternative version using a randomized variation seed.

- **Endpoint**: `POST /api/regenerate`
- **Request Body**:
```json
{
  "content_type": "Blog Post",
  "topic": "The Impact of Quantum Computing on Cryptography",
  "tone": "Professional",
  "instructions": "Focus on real-world impact.",
  "previous_content": "Earlier generated draft content here..."
}
```
- **Response**: Same format as `/api/generate`.

---

### 4. Refine Content

Improves an existing draft using targeted user instructions.

- **Endpoint**: `POST /api/refine`
- **Request Body**:
```json
{
  "content_type": "Blog Post",
  "topic": "The Impact of Quantum Computing on Cryptography",
  "tone": "Professional",
  "previous_content": "Full previous draft here...",
  "refinement_instruction": "Condense the introduction and emphasize key industry takeaways as bullet points."
}
```
- **Response**: Same format as `/api/generate`.

---

### CLI Testing via cURL

```bash
# Test generation
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "content_type": "Social Media Caption",
    "topic": "Launching a New AI Productivity Tool",
    "tone": "Creative",
    "instructions": "Include an engaging call-to-action."
  }'
```

---

## 📓 Interactive Notebook

A standalone Jupyter notebook is available at [`langgraph/content_generator.ipynb`](file:///c:/Users/mitul%20jain/Desktop/AI%20INTERN/ai-content-generator-final-local/langgraph/content_generator.ipynb) for developers who wish to inspect or debug the LangGraph topology interactively.

To launch the notebook:

```powershell
jupyter lab langgraph/content_generator.ipynb
```

The notebook:
- Imports the compiled `GRAPH` object directly from `backend/graph.py`
- Prints the compiled graph schema representation (`GRAPH.get_graph()`)
- Demonstrates sample invocations for generate, regenerate, and refine workflows

---

## 🧪 Running Tests

The test suite is built with `pytest` and uses FastAPI's `TestClient` to validate endpoints, payload schemas, error handling, and routing logic without consuming live Gemini API quota.

Run the test suite from the project root:

```powershell
pytest -q
```

### Coverage Highlights

- **Health Endpoint**: Verifies API status and health response.
- **Parametrized Generation**: Verifies `Blog Post`, `Social Media Caption`, and `Text Summary` across distinct tones.
- **Variation Seed Generation**: Confirms that consecutive regeneration requests produce distinct random seeds.
- **Input Validation**: Tests rejection of empty topics, unknown content types, and malformed JSON payloads.
- **Refinement Validation**: Ensures missing `previous_content` or `refinement_instruction` returns HTTP 400.
- **Error Simulation**: Tests graceful handling of upstream API failures (HTTP 502).
- **Prompt Routing**: Tests branch routing logic and template injection in `prompts.prompts`.

---

## 🛠️ Troubleshooting

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| **`GEMINI_API_KEY is missing. Set it in .env.`** | The `.env` file does not exist, or the API key remains set to `"your_api_key"`. | Create `.env` from `.env.example` and set your actual Gemini API key. |
| **`Cannot connect to FastAPI at http://localhost:8000`** | The FastAPI backend is not running. | Start the backend using `uvicorn backend.main:app --reload --port 8000` before using Streamlit. |
| **`HTTP 502: LangGraph/Gemini failure`** | Network connectivity issues or Gemini API rate limit / quota exceeded. | Check your internet connection and inspect quota usage in Google AI Studio. |
| **`Invalid content type`** | Payload `content_type` does not match allowed options. | Use one of `"Blog Post"`, `"Social Media Caption"`, or `"Text Summary"`. |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
