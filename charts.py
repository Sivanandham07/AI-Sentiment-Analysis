"""
Provides reusable Plotly chart functions for visualizing sentiment and emotion analysis results in the dashboard.
Generates interactive figures from analyzed data without depending on Streamlit, ensuring modularity and reusability.
"""
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# Shared styling constants
# --------------------------------------------------------------------------

# A consistent color mapping for sentiment values, reused across every
# chart so the dashboard reads coherently (green = positive, red =
# negative, gray = neutral) no matter which chart the user looks at.
SENTIMENT_COLOR_MAP: Dict[str, str] = {
    "Positive": "#2ECC71",
    "Negative": "#E74C3C",
    "Neutral": "#95A5A6",
}


def _filter_analyzed_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that were skipped during analysis (e.g. empty text).

    Args:
        df: The full analyzed DataFrame.

    Returns:
        A DataFrame containing only rows with a real sentiment value.
    """
    if "Sentiment" not in df.columns:
        return df.iloc[0:0]  # empty DataFrame with same columns
    return df[df["Sentiment"] != "Skipped"]


def _empty_figure(message: str) -> go.Figure:
    """Build a placeholder figure for when there is no data to chart.

    Args:
        message: The message to display in place of a chart.

    Returns:
        A Plotly figure with no traces and a centered annotation.
    """
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 16, "color": "#888888"},
    )
    figure.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=350,
    )
    return figure


# --------------------------------------------------------------------------
# Sentiment charts
# --------------------------------------------------------------------------

def create_sentiment_pie_chart(df: pd.DataFrame) -> go.Figure:
    """Build a pie chart showing the distribution of sentiment labels.

    Args:
        df: The analyzed DataFrame containing a "Sentiment" column.

    Returns:
        A Plotly pie chart figure.
    """
    analyzed_df = _filter_analyzed_rows(df)

    if analyzed_df.empty:
        return _empty_figure("No sentiment data available.")

    sentiment_counts = analyzed_df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    figure = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        title="Sentiment Distribution",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLOR_MAP,
        hole=0.35,
    )
    figure.update_traces(textposition="inside", textinfo="percent+label")
    figure.update_layout(legend_title_text="Sentiment")
    return figure


def create_sentiment_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Build a bar chart showing the count of reviews per sentiment label.

    Args:
        df: The analyzed DataFrame containing a "Sentiment" column.

    Returns:
        A Plotly bar chart figure.
    """
    analyzed_df = _filter_analyzed_rows(df)

    if analyzed_df.empty:
        return _empty_figure("No sentiment data available.")

    sentiment_counts = analyzed_df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    figure = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        title="Review Count by Sentiment",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLOR_MAP,
        text="Count",
    )
    figure.update_traces(textposition="outside")
    figure.update_layout(showlegend=False, xaxis_title="Sentiment", yaxis_title="Number of Reviews")
    return figure


# --------------------------------------------------------------------------
# Emotion chart
# --------------------------------------------------------------------------

def create_emotion_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Build a bar chart showing the frequency of each detected emotion.

    Emotions are sorted by frequency descending so the most common
    emotions are immediately visible.

    Args:
        df: The analyzed DataFrame containing an "Emotion" column.

    Returns:
        A Plotly bar chart figure.
    """
    analyzed_df = _filter_analyzed_rows(df)

    if analyzed_df.empty or "Emotion" not in analyzed_df.columns:
        return _empty_figure("No emotion data available.")

    emotion_counts = analyzed_df["Emotion"].value_counts().reset_index()
    emotion_counts.columns = ["Emotion", "Count"]
    emotion_counts = emotion_counts.sort_values("Count", ascending=False)

    figure = px.bar(
        emotion_counts,
        x="Emotion",
        y="Count",
        title="Emotion Distribution",
        text="Count",
        color="Count",
        color_continuous_scale="Blues",
    )
    figure.update_traces(textposition="outside")
    figure.update_layout(
        showlegend=False,
        xaxis_title="Emotion",
        yaxis_title="Number of Reviews",
        coloraxis_showscale=False,
    )
    return figure 