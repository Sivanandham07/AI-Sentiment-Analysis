"""

Main Streamlit application for the AI-Powered Sentiment Analysis project, responsible for rendering the user interface and managing application workflow.
Handles user interactions, invokes the AI analyzer and visualization modules, and provides robust error handling with persistent session state.
"""


from typing import Optional

import pandas as pd
import streamlit as st


from analyzer import (
    AnalysisTimeoutError,
    AnalyzerError,
    APIKeyError,
    NetworkError,
    RateLimitError,
    get_analyzer,
)
from charts import (
    create_emotion_bar_chart,
    create_sentiment_bar_chart,
    create_sentiment_pie_chart,
)
from utils import (
    calculate_summary_stats,
    check_api_key_configured,
    convert_df_to_csv_bytes,
    get_text_column_candidates,
    validate_dataframe,
    validate_text_input,
)

APP_TITLE: str = "AI-Powered Sentiment Analysis"
APP_ICON: str = "💬"
RESULT_COLUMNS = [
    "Sentiment",
    "Emotion",
    "Confidence",
    "Summary",
    "Key Topics",
    "Suggested Response",
]

COLOR_BG = "#FAFAF9"
COLOR_SURFACE = "#F1F3F5"
COLOR_BORDER = "#E2E4E8"
COLOR_TEXT = "#374151"
COLOR_TEXT_MUTED = "#6B7280"
COLOR_PRIMARY = "#2563EB"
COLOR_PRIMARY_HOVER = "#1D4ED8"



st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    """Inject CSS overrides so the app renders as a clean light-mode UI.

    A `.streamlit/config.toml` with `base = "light"` is the primary switch
    (see project root), but we layer a small amount of extra CSS on top to
    polish surfaces Streamlit's base theme doesn't fully cover (metric
    cards, expanders, tabs, buttons, dataframes) so everything reads as one
    consistent light design instead of a mix of light/dark widgets.
    """
    st.markdown(
        f"""
        <style>
            /* App background & base text */
            .stApp {{
                background-color: {COLOR_BG};
                color: {COLOR_TEXT};
            }}

            /* Top header/toolbar (the bar with the "Deploy" button) and the
               thin decoration strip above it both default to a dark color
               that clashes with a light theme — force them to match. */
            header[data-testid="stHeader"] {{
                background-color: {COLOR_BG};
            }}
            header[data-testid="stHeader"] * {{
                color: {COLOR_TEXT} !important;
            }}
            div[data-testid="stDecoration"] {{
                background-image: none;
                background-color: {COLOR_PRIMARY};
            }}
            div[data-testid="stToolbar"] {{
                color: {COLOR_TEXT};
            }}

            /* Sidebar */
            section[data-testid="stSidebar"] {{
                background-color: {COLOR_SURFACE};
                border-right: 1px solid {COLOR_BORDER};
            }}
            section[data-testid="stSidebar"] * {{
                color: {COLOR_TEXT};
            }}

            /* Headings */
            h1, h2, h3, h4, h5, h6 {{
                color: {COLOR_TEXT} !important;
            }}

            /* Captions / muted text */
            .stCaption, [data-testid="stCaptionContainer"] {{
                color: {COLOR_TEXT_MUTED} !important;
            }}

            /* Buttons */
            .stButton > button {{
                background-color: {COLOR_PRIMARY};
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                transition: background-color 0.15s ease-in-out;
            }}
            .stButton > button:hover {{
                background-color: {COLOR_PRIMARY_HOVER};
                color: #FFFFFF;
            }}

            /* Tabs */
            button[data-baseweb="tab"] {{
                color: {COLOR_TEXT_MUTED};
                font-weight: 500;
            }}
            button[data-baseweb="tab"][aria-selected="true"] {{
                color: {COLOR_PRIMARY};
                font-weight: 700;
            }}
            div[data-baseweb="tab-highlight"] {{
                background-color: {COLOR_PRIMARY};
            }}

            /* Metric cards */
            div[data-testid="stMetric"] {{
                background-color: {COLOR_SURFACE};
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                padding: 1rem;
            }}
            div[data-testid="stMetricLabel"] {{
                color: {COLOR_TEXT_MUTED} !important;
            }}
            div[data-testid="stMetricValue"] {{
                color: {COLOR_TEXT} !important;
            }}

            /* Expanders */
            details {{
                background-color: {COLOR_SURFACE};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
            }}
            summary {{
                color: {COLOR_TEXT} !important;
            }}

            /* Text areas / inputs / selects */
            .stTextArea textarea,
            .stTextInput input,
            div[data-baseweb="select"] > div {{
                background-color: {COLOR_BG};
                color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER};
            }}

            /* File uploader */
            section[data-testid="stFileUploaderDropzone"] {{
                background-color: {COLOR_SURFACE};
                border: 1px dashed {COLOR_BORDER};
            }}

            /* DataFrames / tables */
            div[data-testid="stDataFrame"] {{
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
            }}

            /* Progress bar */
            div[data-testid="stProgress"] > div > div {{
                background-color: {COLOR_PRIMARY};
            }}

            /* Alert boxes (info/success/warning/error) keep their semantic
               colors but get consistent rounded, bordered styling */
            div[data-testid="stAlert"] {{
                border-radius: 8px;
                border: 1px solid {COLOR_BORDER};
            }}

            /* Divider */
            hr {{
                border-color: {COLOR_BORDER};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Ensure all session-state keys used by this app exist with sane defaults.

    Streamlit reruns the entire script on every interaction, so any data we
    want to persist across reruns (like an analyzed DataFrame) must live in
    `st.session_state` rather than a plain local variable.
    """
    if "analyzed_df" not in st.session_state:
        
        st.session_state.analyzed_df = None

    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None

    if "single_result" not in st.session_state:
        
        st.session_state.single_result = None


@st.cache_resource(show_spinner=False)
def load_analyzer():
    """Instantiate and cache the configured sentiment analyzer.

    Using `st.cache_resource` ensures the underlying AI client (e.g. the
    Gemini SDK client) is created only once per app session rather than on
    every rerun, which would be wasteful and could trigger unnecessary
    re-authentication.

    Returns:
        An instance of a class implementing `BaseSentimentAnalyzer`.

    Raises:
        APIKeyError: If no valid API key is configured. This is allowed to
            propagate to the caller, which is expected to catch it and show
            a friendly Streamlit message.
    """
    return get_analyzer()


def get_analyzer_safe():
    """Attempt to load the analyzer, capturing any configuration errors.

    Returns:
        A tuple of (analyzer_instance_or_None, error_message_or_None).
    """
    try:
        return load_analyzer(), None
    except APIKeyError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        return None, f"Failed to initialize the AI analyzer: {exc}"



def render_sidebar() -> None:
    """Render the sidebar with app branding, status, and instructions."""
    with st.sidebar:
        st.markdown(f"## {APP_ICON} {APP_TITLE}")
        st.caption(
            "Analyze customer sentiment, emotion, and intent using AI — "
            "one review at a time or in bulk via CSV."
        )

        st.divider()

        st.markdown("### ⚙️ Configuration Status")
        if check_api_key_configured():
            st.success("API key detected", icon="✅")
        else:
            st.error("No API key found", icon="🚫")
            st.caption(
                "Add your key to a `.env` file. See `.env.example` for the "
                "expected variable names."
            )

        st.divider()

        st.markdown("### 📖 How to Use")
        st.markdown(
            "- **Single Text Analysis**: paste one review and get instant "
            "sentiment, emotion, and a suggested business response.\n"
            "- **CSV Batch Analysis**: upload a CSV of reviews to analyze "
            "them all at once.\n"
            "- **Dashboard**: visualize sentiment/emotion distribution "
            "after a batch analysis.\n"
        )

        st.divider()

        st.caption("Built with Streamlit, Plotly, and Google Gemini.")
        st.caption("© 2025 · Portfolio Project")



def render_single_text_tab() -> None:
    """Render the UI for analyzing a single piece of text on demand."""
    st.subheader("🔍 Single Text Analysis")
    st.write(
        "Paste a customer review, support ticket, or any piece of "
        "feedback below to get an instant AI-powered analysis."
    )

    text_input: str = st.text_area(
        label="Enter text to analyze",
        placeholder="e.g. 'The product arrived late and the packaging was damaged...'",
        height=150,
        key="single_text_input",
    )

    analyze_clicked = st.button(
        "Analyze Text", type="primary", key="analyze_single_button"
    )

    if analyze_clicked:
        is_valid, error_message = validate_text_input(text_input)
        if not is_valid:
            st.warning(error_message, icon="⚠️")
            return

        analyzer, init_error = get_analyzer_safe()
        if init_error:
            st.error(init_error, icon="🚫")
            return

        with st.spinner("Analyzing sentiment..."):
            try:
                result = analyzer.analyze_text(text_input)
                st.session_state.single_result = result
            except APIKeyError as exc:
                st.error(f"Authentication failed: {exc}", icon="🔑")
                return
            except RateLimitError as exc:
                st.warning(
                    f"Rate limit reached: {exc}. Please wait a moment and "
                    "try again.",
                    icon="⏳",
                )
                return
            except AnalysisTimeoutError as exc:
                st.error(
                    f"The request timed out: {exc}. Please try again.",
                    icon="⏱️",
                )
                return
            except NetworkError as exc:
                st.error(
                    f"Network error: {exc}. Please check your connection "
                    "and try again.",
                    icon="🌐",
                )
                return
            except AnalyzerError as exc:
                st.error(f"Analysis failed: {exc}", icon="❌")
                return
            except Exception as exc:  # noqa: BLE001
                st.error(f"An unexpected error occurred: {exc}", icon="❌")
                return

 
    if st.session_state.single_result:
        render_single_result(st.session_state.single_result)


def render_single_result(result: dict) -> None:
    """Render a single analysis result dict in a polished, readable layout.

    Args:
        result: A dict matching the `analyze_text` return contract.
    """
    st.success("Analysis complete", icon="✅")

    sentiment: str = result.get("sentiment", "Unknown")
    emotion: str = result.get("emotion", "Unknown")
    confidence: float = float(result.get("confidence", 0.0))

    sentiment_icon = {"Positive": "😊", "Negative": "😟", "Neutral": "😐"}.get(
        sentiment, "❔"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment", f"{sentiment_icon} {sentiment}")
    col2.metric("Emotion", emotion)
    col3.metric("Confidence", f"{confidence * 100:.1f}%")

    st.progress(min(max(confidence, 0.0), 1.0))

    with st.expander("📝 Explanation", expanded=True):
        st.write(result.get("explanation", "No explanation provided."))

    with st.expander("📄 Summary"):
        st.write(result.get("summary", "No summary provided."))

    with st.expander("🏷️ Key Topics"):
        topics = result.get("key_topics", [])
        if topics:
            st.write(", ".join(topics))
        else:
            st.write("No key topics identified.")

    with st.expander("💬 Suggested Business Response", expanded=True):
        st.write(result.get("suggested_response", "No suggestion available."))


def render_csv_batch_tab() -> None:
    """Render the UI for uploading and analyzing a CSV of reviews."""
    st.subheader("📂 CSV Batch Analysis")
    st.write(
        "Upload a CSV file containing customer reviews. Each row will be "
        "analyzed and enriched with sentiment, emotion, and more."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file", type=["csv"], key="csv_uploader"
    )

    if uploaded_file is None:
        st.info("Awaiting a CSV file upload.", icon="ℹ️")
        return

    try:
        raw_df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.", icon="🚫")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the CSV file: {exc}", icon="🚫")
        return

    is_valid, error_message = validate_dataframe(raw_df)
    if not is_valid:
        st.error(error_message, icon="🚫")
        return

    st.success(f"Loaded {len(raw_df)} rows from `{uploaded_file.name}`.", icon="✅")
    with st.expander("Preview uploaded data"):
        st.dataframe(raw_df.head(10), use_container_width=True)


    candidate_columns = get_text_column_candidates(raw_df)
    text_column = st.selectbox(
        "Which column contains the review text?",
        options=candidate_columns,
        key="text_column_select",
    )

    analyze_clicked = st.button(
        "Analyze CSV", type="primary", key="analyze_csv_button"
    )

    if analyze_clicked:
        analyzer, init_error = get_analyzer_safe()
        if init_error:
            st.error(init_error, icon="🚫")
            return

        progress_bar = st.progress(0.0, text="Starting analysis...")
        status_placeholder = st.empty()

        def _update_progress(current: int, total: int) -> None:
            """Callback passed into analyzer.analyze_dataframe for UI feedback."""
            fraction = current / total if total else 0.0
            progress_bar.progress(
                min(fraction, 1.0), text=f"Analyzed {current} of {total} reviews..."
            )

        try:
            with st.spinner("Running batch analysis..."):
                analyzed_df = analyzer.analyze_dataframe(
                    raw_df, text_column, progress_callback=_update_progress
                )
            st.session_state.analyzed_df = analyzed_df
            st.session_state.uploaded_file_name = uploaded_file.name
            progress_bar.progress(1.0, text="Analysis complete!")
            status_placeholder.success(
                f"Successfully analyzed {len(analyzed_df)} reviews.", icon="✅"
            )
        except APIKeyError as exc:
            status_placeholder.error(f"Authentication failed: {exc}", icon="🔑")
            return
        except RateLimitError as exc:
            status_placeholder.warning(
                f"Rate limit reached: {exc}. Some rows may be incomplete. "
                "Try again shortly.",
                icon="⏳",
            )
            return
        except AnalysisTimeoutError as exc:
            status_placeholder.error(
                f"The request timed out: {exc}. Please try again.", icon="⏱️"
            )
            return
        except NetworkError as exc:
            status_placeholder.error(
                f"Network error: {exc}. Please check your connection and "
                "try again.",
                icon="🌐",
            )
            return
        except AnalyzerError as exc:
            status_placeholder.error(f"Batch analysis failed: {exc}", icon="❌")
            return
        except Exception as exc:  # noqa: BLE001
            status_placeholder.error(f"An unexpected error occurred: {exc}", icon="❌")
            return


    if st.session_state.analyzed_df is not None:
        st.markdown("### 📋 Analyzed Results")
        st.dataframe(st.session_state.analyzed_df, use_container_width=True)

        csv_bytes = convert_df_to_csv_bytes(st.session_state.analyzed_df)
        download_name = (
            f"analyzed_{st.session_state.uploaded_file_name}"
            if st.session_state.uploaded_file_name
            else "analyzed_reviews.csv"
        )
        st.download_button(
            label="⬇️ Download Analyzed CSV",
            data=csv_bytes,
            file_name=download_name,
            mime="text/csv",
            key="download_csv_button",
        )



def render_dashboard_tab() -> None:
    """Render sentiment/emotion visualizations for the analyzed CSV data."""
    st.subheader("📊 Sentiment Dashboard")

    analyzed_df: Optional[pd.DataFrame] = st.session_state.analyzed_df

    if analyzed_df is None or analyzed_df.empty:
        st.info(
            "No analyzed data yet. Upload and analyze a CSV in the "
            "'CSV Batch Analysis' tab to populate this dashboard.",
            icon="ℹ️",
        )
        return

    stats = calculate_summary_stats(analyzed_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", stats["total"])
    col2.metric("Positive %", f"{stats['positive_pct']:.1f}%")
    col3.metric("Negative %", f"{stats['negative_pct']:.1f}%")
    col4.metric("Neutral %", f"{stats['neutral_pct']:.1f}%")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(
            create_sentiment_pie_chart(analyzed_df),
            use_container_width=True,
            key="dashboard_sentiment_pie_chart",
        )
    with chart_col2:
        st.plotly_chart(
            create_sentiment_bar_chart(analyzed_df),
            use_container_width=True,
            key="dashboard_sentiment_bar_chart",
        )

    st.plotly_chart(
        create_emotion_bar_chart(analyzed_df),
        use_container_width=True,
        key="dashboard_emotion_bar_chart",
    )



def main() -> None:
    """Application entry point: wires up session state, sidebar, and tabs."""
    inject_custom_css()
    initialize_session_state()
    render_sidebar()

    st.title(f"{APP_ICON} {APP_TITLE}")
    st.write(
        "A portfolio-quality demonstration of AI-powered sentiment analysis "
        "for customer feedback, built with a swappable LLM backend."
    )

    tab_single, tab_csv, tab_dashboard = st.tabs(
        ["🔍 Single Text Analysis", "📂 CSV Batch Analysis", "📊 Dashboard"]
    )

    with tab_single:
        render_single_text_tab()

    with tab_csv:
        render_csv_batch_tab()

    with tab_dashboard:
        render_dashboard_tab()


if __name__ == "__main__":
    main()