"""
AI Content Generator
--------------------
Streamlit frontend for:
Streamlit → FastAPI → LangGraph → Gemini 3.5 Flash

The Gemini API key is NOT used in this frontend.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
).rstrip("/")


CONTENT_TYPES = [
    "Blog Post",
    "Social Media Caption",
    "Text Summary",
]

TONES = [
    "Professional",
    "Friendly",
    "Casual",
    "Creative",
    "Formal",
    "Persuasive",
    "Inspirational",
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Content Generator",
    page_icon="✍️",
    layout="wide",
)


# ============================================================
# SIMPLE PROFESSIONAL STYLING
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 750;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #8b93a3;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .section-description {
        color: #7e8797;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    .content-info {
        padding: 0.7rem 0.9rem;
        border-radius: 9px;
        border: 1px solid #303642;
        background: #11141b;
        color: #929aaa;
        font-size: 0.78rem;
        margin-top: -0.25rem;
        margin-bottom: 1rem;
    }

    .generated-meta {
        color: #858e9e;
        font-size: 0.78rem;
        margin-bottom: 1rem;
    }

    .empty-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .empty-description {
        color: #7e8797;
        font-size: 0.82rem;
        line-height: 1.6;
    }

    .footer-text {
        text-align: center;
        color: #687181;
        font-size: 0.72rem;
        margin-top: 2rem;
    }

    /* Better markdown output */

    [data-testid="stMarkdownContainer"] h1 {
        font-size: 2rem;
        line-height: 1.25;
    }

    [data-testid="stMarkdownContainer"] h2 {
        font-size: 1.45rem;
        margin-top: 1.5rem;
    }

    [data-testid="stMarkdownContainer"] h3 {
        font-size: 1.15rem;
        margin-top: 1.2rem;
    }

    [data-testid="stMarkdownContainer"] p {
        line-height: 1.75;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "content" not in st.session_state:
    st.session_state.content = ""

if "params" not in st.session_state:
    st.session_state.params = None


# ============================================================
# API HELPER
# ============================================================

def call_api(endpoint: str, payload: dict):
    """
    Send request to FastAPI backend.

    Returns:
        data, error
    """

    try:

        response = requests.post(
            f"{API_URL}/api/{endpoint}",
            json=payload,
            timeout=120,
        )

    except requests.exceptions.ConnectionError:

        return (
            None,
            f"Cannot connect to FastAPI at {API_URL}. "
            "Please start the backend first.",
        )

    except requests.exceptions.Timeout:

        return (
            None,
            "The request timed out. Please try again.",
        )

    except requests.exceptions.RequestException as exc:

        return (
            None,
            f"Backend request failed: {exc}",
        )

    if response.status_code != 200:

        try:

            detail = response.json().get(
                "detail",
                response.text,
            )

        except Exception:

            detail = response.text

        return None, str(detail)

    try:

        return response.json(), None

    except Exception:

        return (
            None,
            "The backend returned an invalid response.",
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">✍️ AI Content Generator</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Create polished blog posts, social media captions and
        concise summaries with AI.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# MAIN LAYOUT
# ============================================================

settings_col, output_col = st.columns(
    [0.82, 1.45],
    gap="large",
)


# ============================================================
# SETTINGS PANEL
# ============================================================

with settings_col:

    st.markdown(
        '<div class="section-title">Create content</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Configure your content before generating it.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # CONTENT TYPE
    # --------------------------------------------------------

    content_type = st.selectbox(
        "Content type",
        CONTENT_TYPES,
    )

    if content_type == "Blog Post":

        description = (
            "Create a structured long-form article "
            "with headings and useful explanations."
        )

    elif content_type == "Social Media Caption":

        description = (
            "Create a short, engaging caption "
            "suitable for social media."
        )

    else:

        description = (
            "Create a concise summary focused on "
            "the most important information."
        )

    st.markdown(
        f"""
        <div class="content-info">
            {description}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TOPIC
    # --------------------------------------------------------

    topic = st.text_input(
        "Topic",
        placeholder="e.g. Artificial Intelligence in Healthcare",
    )

    # --------------------------------------------------------
    # TONE
    # --------------------------------------------------------

    tone = st.selectbox(
        "Tone",
        TONES,
    )

    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    instructions = st.text_area(
        "Additional instructions",
        placeholder=(
            "Optional: target audience, length, key points, "
            "examples, keywords, etc."
        ),
        height=120,
    )

    st.write("")

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    generate = st.button(
        "🚀 Generate Content",
        type="primary",
        use_container_width=True,
    )

    if generate:

        if not topic.strip():

            st.warning(
                "Please enter a topic."
            )

        else:

            payload = {
                "content_type": content_type,
                "topic": topic.strip(),
                "tone": tone,
                "instructions": instructions.strip(),
            }

            st.session_state.params = payload

            with st.spinner(
                "Generating your content..."
            ):

                data, error = call_api(
                    "generate",
                    payload,
                )

            if error:

                st.error(error)

            else:

                st.session_state.content = data.get(
                    "content",
                    "",
                )

                st.rerun()


# ============================================================
# OUTPUT PANEL
# ============================================================

with output_col:

    st.markdown(
        '<div class="section-title">Generated Content</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Review your result, regenerate it or refine it.
        </div>
        """,
        unsafe_allow_html=True,
    )

    content = st.session_state.content
    params = st.session_state.params

    # ========================================================
    # EMPTY STATE
    # ========================================================

    if not content:

        st.info(
            "Your generated content will appear here."
        )

        st.markdown(
            """
            <div class="empty-title">
                Ready when you are.
            </div>

            <div class="empty-description">
                Choose a content type, enter your topic,
                select a tone and click
                <strong>Generate Content</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # GENERATED CONTENT
    # ========================================================

    else:

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        if params:

            st.markdown(
                f"""
                <div class="generated-meta">
                    {params["content_type"]}
                    &nbsp;•&nbsp;
                    {params["tone"]}
                    &nbsp;•&nbsp;
                    {params["topic"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # CLEAN DOCUMENT OUTPUT
        # ----------------------------------------------------

        # IMPORTANT:
        # We intentionally use st.markdown here.
        # There is NO st.code() and NO dark code box.

        st.markdown(content)

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        st.divider()

        words = len(content.split())
        characters = len(content)

        paragraphs = len(
            [
                paragraph
                for paragraph in content.split("\n\n")
                if paragraph.strip()
            ]
        )

        stat1, stat2, stat3 = st.columns(3)

        with stat1:

            st.metric(
                "Words",
                words,
            )

        with stat2:

            st.metric(
                "Characters",
                characters,
            )

        with stat3:

            st.metric(
                "Paragraphs",
                paragraphs,
            )

        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        st.divider()

        action1, action2 = st.columns(2)

        # ----------------------------------------------------
        # REGENERATE
        # ----------------------------------------------------

        with action1:

            regenerate = st.button(
                "🔄 Regenerate",
                use_container_width=True,
            )

            if regenerate:

                payload = {
                    **params,
                    "previous_content": content,
                }

                with st.spinner(
                    "Creating another version..."
                ):

                    data, error = call_api(
                        "regenerate",
                        payload,
                    )

                if error:

                    st.error(error)

                else:

                    st.session_state.content = data[
                        "content"
                    ]

                    st.rerun()

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        with action2:

            st.download_button(
                "⬇️ Download Content",
                data=content,
                file_name="generated_content.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # ====================================================
        # REFINE
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">Refine Content</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-description">
                Tell the AI how you want the current result improved.
            </div>
            """,
            unsafe_allow_html=True,
        )

        refinement = st.text_input(
            "Refinement instruction",
            placeholder=(
                "e.g. Make it shorter and more professional"
            ),
        )

        refine = st.button(
            "✨ Refine Content",
            use_container_width=True,
        )

        if refine:

            if not refinement.strip():

                st.warning(
                    "Please enter a refinement instruction."
                )

            else:

                payload = {
                    "content_type": params[
                        "content_type"
                    ],
                    "topic": params[
                        "topic"
                    ],
                    "tone": params[
                        "tone"
                    ],
                    "previous_content": content,
                    "refinement_instruction": (
                        refinement.strip()
                    ),
                }

                with st.spinner(
                    "Refining your content..."
                ):

                    data, error = call_api(
                        "refine",
                        payload,
                    )

                if error:

                    st.error(error)

                else:

                    st.session_state.content = data[
                        "content"
                    ]

                    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
        AI Content Generator
        &nbsp;•&nbsp;
        Streamlit → FastAPI → LangGraph → Gemini 3.5 Flash
    </div>
    """,
    unsafe_allow_html=True,
)