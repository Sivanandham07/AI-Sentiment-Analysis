"""
utils.py
========

General-purpose helper functions for the Sentiment Analyzer application.

This module intentionally has NO dependency on Streamlit or any AI SDK
(Gemini/OpenAI/Claude). It only handles:
    * Configuration/environment checks
    * Input validation (text and DataFrames)
    * Data-wrangling helpers (column detection, CSV export, summary stats)

Keeping this module dependency-light means every function here can be
unit-tested in isolation, without mocking a UI or an AI provider.
"""

import os
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv

# Ensure environment variables from a local .env file are loaded whenever
# this module is imported, so `check_api_key_configured` works even if
# app.py happens to import utils before analyzer.
load_dotenv()


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Minimum/maximum lengths we consider reasonable for a single text analysis
# input. These are generous bounds meant to catch obviously bad input
# (empty strings, accidental multi-megabyte pastes) rather than to be
# restrictive.
_MIN_TEXT_LENGTH = 1
_MAX_TEXT_LENGTH = 5000

# Column name substrings (case-insensitive) that suggest a column contains
# review/feedback text. Used to rank candidate columns for the CSV upload
# text-column selector.
_LIKELY_TEXT_COLUMN_HINTS = (
    "review",
    "text",
    "comment",
    "feedback",
    "description",
    "message",
    "content",
)


# --------------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------------

def check_api_key_configured() -> bool:
    """Check whether an AI provider API key is present in the environment.

    This is a lightweight, side-effect-free check intended for display
    purposes (e.g. a sidebar status indicator). It does NOT validate that
    the key is actually correct -- only that something is set.

    Returns:
        True if `GEMINI_API_KEY` is set and non-empty, False otherwise.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    return bool(api_key and api_key.strip())


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def validate_text_input(text: str) -> Tuple[bool, str]:
    """Validate user-provided text before sending it for analysis.

    Args:
        text: The raw text entered by the user.

    Returns:
        A tuple of (is_valid, error_message). If is_valid is True,
        error_message is an empty string.
    """
    if text is None or not text.strip():
        return False, "Please enter some text to analyze."

    stripped_length = len(text.strip())

    if stripped_length < _MIN_TEXT_LENGTH:
        return False, "Please enter some text to analyze."

    if stripped_length > _MAX_TEXT_LENGTH:
        return (
            False,
            f"Text is too long ({stripped_length} characters). "
            f"Please limit input to {_MAX_TEXT_LENGTH} characters.",
        )

    return True, ""


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate an uploaded CSV's DataFrame before analysis.

    Covers the "empty CSV" and "missing usable columns" failure modes
    called out in the project requirements, without assuming any specific
    required column name (real-world CSVs vary widely).

    Args:
        df: The DataFrame parsed from the uploaded CSV.

    Returns:
        A tuple of (is_valid, error_message).
    """
    if df is None or df.empty:
        return False, "The uploaded CSV file contains no data."

    if len(df.columns) == 0:
        return False, "The uploaded CSV file has no columns."

    # Ensure there is at least one column that could plausibly contain
    # text (i.e. not every column is purely numeric).
    text_like_columns = [
        column for column in df.columns if df[column].dtype == object
    ]
    if not text_like_columns:
        return (
            False,
            "The uploaded CSV does not appear to contain any text columns "
            "to analyze.",
        )

    return True, ""


# --------------------------------------------------------------------------
# Column detection
# --------------------------------------------------------------------------

def get_text_column_candidates(df: pd.DataFrame) -> List[str]:
    """Return the DataFrame's columns ordered by likelihood of being review text.

    Columns whose names contain a hint like "review", "text", "comment",
    etc. AND whose dtype is text-like (`object`) are moved to the front of
    the list, so the Streamlit selectbox defaults to a sensible guess.

    Matching on name alone is not enough: a column like "review_id" also
    contains the hint substring "review" but holds integers, not text. If
    such a column were ranked ahead of the real text column (e.g.
    "review_text"), it could get auto-selected and silently cause every
    row to be marked "Skipped" during analysis (since integers fail the
    `isinstance(value, str)` check in `analyzer.py`). Requiring `object`
    dtype for the "likely" bucket avoids that trap while still leaving
    every column selectable.

    Args:
        df: The DataFrame to inspect.

    Returns:
        A list of column names, most-likely-to-be-text-column first.
    """
    all_columns = list(df.columns)

    def _is_text_dtype(column_name: str) -> bool:
        return df[column_name].dtype == object

    def _is_likely_text_column(column_name: str) -> bool:
        lowered = str(column_name).lower()
        return any(hint in lowered for hint in _LIKELY_TEXT_COLUMN_HINTS)

    likely_columns = [
        col for col in all_columns
        if _is_text_dtype(col) and _is_likely_text_column(col)
    ]
    other_text_columns = [
        col for col in all_columns
        if _is_text_dtype(col) and col not in likely_columns
    ]
    non_text_columns = [
        col for col in all_columns
        if col not in likely_columns and col not in other_text_columns
    ]

    return likely_columns + other_text_columns + non_text_columns


# --------------------------------------------------------------------------
# Export helpers
# --------------------------------------------------------------------------

def convert_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame into UTF-8 encoded CSV bytes for download.

    Centralizing this logic ensures every download button in the app
    produces a consistently formatted CSV (no index column, UTF-8 encoding).

    Args:
        df: The DataFrame to export.

    Returns:
        The CSV file contents as bytes, suitable for `st.download_button`.
    """
    return df.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

def calculate_summary_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the sentiment dashboard.

    Args:
        df: The analyzed DataFrame, expected to contain a "Sentiment"
            column with values like "Positive", "Negative", "Neutral"
            (rows with "Skipped" are excluded from percentage calculations
            but included in the total count).

    Returns:
        A dict with keys: total, positive_pct, negative_pct, neutral_pct.
    """
    total = len(df)

    if total == 0 or "Sentiment" not in df.columns:
        return {
            "total": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
        }

    # Only count rows that were actually analyzed (exclude "Skipped" rows
    # from percentage denominators so they don't artificially deflate
    # real sentiment percentages).
    analyzed_rows = df[df["Sentiment"] != "Skipped"]
    analyzed_total = len(analyzed_rows)

    if analyzed_total == 0:
        return {
            "total": total,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
        }

    sentiment_counts = analyzed_rows["Sentiment"].value_counts()

    positive_count = int(sentiment_counts.get("Positive", 0))
    negative_count = int(sentiment_counts.get("Negative", 0))
    neutral_count = int(sentiment_counts.get("Neutral", 0))

    return {
        "total": total,
        "positive_pct": (positive_count / analyzed_total) * 100,
        "negative_pct": (negative_count / analyzed_total) * 100,
        "neutral_pct": (neutral_count / analyzed_total) * 100,
    }