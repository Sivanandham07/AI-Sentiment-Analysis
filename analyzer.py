"""
analyzer.py
===========

AI abstraction layer for the Sentiment Analyzer application.

This module defines a provider-agnostic interface (`BaseSentimentAnalyzer`)
for performing sentiment analysis, along with a concrete implementation
backed by Google's Gemini API (`GeminiSentimentAnalyzer`).

The rest of the application (`app.py`) interacts ONLY with:
    * `get_analyzer()`      -- factory that returns the configured analyzer
    * The custom exceptions defined below

This means switching the underlying AI provider (e.g. to OpenAI or Claude)
requires changes ONLY in this file:
    1. Create a new class, e.g. `OpenAISentimentAnalyzer(BaseSentimentAnalyzer)`,
       implementing `analyze_text` and `analyze_dataframe`.
    2. Add one branch to `get_analyzer()` to instantiate it based on the
       `AI_PROVIDER` environment variable.

No other file needs to know an AI provider was ever changed.
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

# Load environment variables from a local .env file (if present) as early
# as possible so GEMINI_API_KEY / AI_PROVIDER are available below.
load_dotenv()


# --------------------------------------------------------------------------
# Custom exception hierarchy
# --------------------------------------------------------------------------
# These normalize errors from ANY underlying AI SDK into a consistent set
# of exception types that app.py can catch and translate into friendly
# Streamlit messages, without needing to know which provider raised them.

class AnalyzerError(Exception):
    """Base exception for all sentiment-analyzer errors."""


class APIKeyError(AnalyzerError):
    """Raised when the API key is missing, invalid, or unauthorized."""


class RateLimitError(AnalyzerError):
    """Raised when the AI provider reports a rate limit or quota error."""


class AnalysisTimeoutError(AnalyzerError):
    """Raised when a request to the AI provider times out."""


class NetworkError(AnalyzerError):
    """Raised when a network-level failure prevents reaching the AI provider."""


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Default field values used when the model's response is missing a key,
# so downstream code (app.py, charts.py) never has to guard against KeyError.
_DEFAULT_RESULT: Dict[str, Any] = {
    "sentiment": "Neutral",
    "emotion": "Unknown",
    "confidence": 0.0,
    "explanation": "No explanation was returned by the model.",
    "summary": "No summary was returned by the model.",
    "key_topics": [],
    "suggested_response": "No suggested response was returned by the model.",
}

_VALID_SENTIMENTS = {"Positive", "Negative", "Neutral"}

# Request timeout (seconds) applied to each Gemini call.
_REQUEST_TIMEOUT_SECONDS = 30


# --------------------------------------------------------------------------
# Abstract base class -- the provider-agnostic contract
# --------------------------------------------------------------------------

class BaseSentimentAnalyzer(ABC):
    """Defines the interface every sentiment-analysis backend must implement.

    Any new AI provider (OpenAI, Claude, local model, etc.) should subclass
    this and implement both methods. `app.py` depends only on this
    interface, never on a specific provider's SDK.
    """

    @abstractmethod
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze a single piece of text and return a structured result.

        Args:
            text: The raw text to analyze (e.g. a customer review).

        Returns:
            A dict with keys: sentiment, emotion, confidence, explanation,
            summary, key_topics, suggested_response.

        Raises:
            APIKeyError, RateLimitError, AnalysisTimeoutError, NetworkError,
            AnalyzerError: on any failure to complete the analysis.
        """
        raise NotImplementedError

    @abstractmethod
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """Analyze every row of a DataFrame and append result columns.

        Args:
            df: The input DataFrame containing a column of review text.
            text_column: The name of the column containing text to analyze.
            progress_callback: Optional callable invoked as
                `progress_callback(current_row_index, total_rows)` after
                each row is processed, allowing the UI to show progress.

        Returns:
            A new DataFrame equal to `df` plus the columns: Sentiment,
            Emotion, Confidence, Summary, Key Topics, Suggested Response.

        Raises:
            APIKeyError, RateLimitError, AnalysisTimeoutError, NetworkError,
            AnalyzerError: on any failure to complete the analysis.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# Gemini implementation
# --------------------------------------------------------------------------

class GeminiSentimentAnalyzer(BaseSentimentAnalyzer):
    """Sentiment analyzer backed by Google's Gemini API.

    This class handles:
        * Authenticating with the Gemini SDK using an API key from the
          environment.
        * Building a structured prompt that asks the model to return JSON.
        * Parsing and validating the model's JSON response defensively.
        * Translating Gemini/SDK-level errors into our custom exception
          hierarchy so callers don't need to know about Gemini internals.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.1-flash-lite") -> None:
        """Configure the Gemini client.

        Args:
            api_key: Explicit API key. If omitted, read from the
                `GEMINI_API_KEY` environment variable.
            model_name: The Gemini model identifier to use.

        Raises:
            APIKeyError: If no API key is available from either source.
        """
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise APIKeyError(
                "GEMINI_API_KEY is not set. Add it to your .env file "
                "(see .env.example) or your environment variables."
            )

        self._client = genai.Client(api_key=resolved_key)
        self._model_name = model_name

    # ---- Public interface -------------------------------------------------

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """See `BaseSentimentAnalyzer.analyze_text`."""
        prompt = self._build_prompt(text)
        raw_response_text = self._call_model(prompt)
        return self._parse_response(raw_response_text)

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """See `BaseSentimentAnalyzer.analyze_dataframe`."""
        if text_column not in df.columns:
            raise AnalyzerError(
                f"Column '{text_column}' was not found in the uploaded CSV."
            )

        result_df = df.copy()

        # Pre-create result columns so partial failures still leave a
        # well-formed DataFrame (rows not yet processed stay as None).
        result_df["Sentiment"] = None
        result_df["Emotion"] = None
        result_df["Confidence"] = None
        result_df["Summary"] = None
        result_df["Key Topics"] = None
        result_df["Suggested Response"] = None

        total_rows = len(result_df)

        for position, (row_index, row) in enumerate(result_df.iterrows(), start=1):
            text_value = row[text_column]

            # Skip rows with empty/missing text rather than failing the
            # entire batch -- mark them clearly as skipped.
            if not isinstance(text_value, str) or not text_value.strip():
                result_df.at[row_index, "Sentiment"] = "Skipped"
                result_df.at[row_index, "Emotion"] = "N/A"
                result_df.at[row_index, "Confidence"] = 0.0
                result_df.at[row_index, "Summary"] = "No text provided."
                result_df.at[row_index, "Key Topics"] = ""
                result_df.at[row_index, "Suggested Response"] = "N/A"
            else:
                result = self.analyze_text(text_value)
                result_df.at[row_index, "Sentiment"] = result["sentiment"]
                result_df.at[row_index, "Emotion"] = result["emotion"]
                result_df.at[row_index, "Confidence"] = result["confidence"]
                result_df.at[row_index, "Summary"] = result["summary"]
                result_df.at[row_index, "Key Topics"] = ", ".join(
                    result["key_topics"]
                ) if isinstance(result["key_topics"], list) else result["key_topics"]
                result_df.at[row_index, "Suggested Response"] = result[
                    "suggested_response"
                ]

            if progress_callback is not None:
                progress_callback(position, total_rows)

        return result_df

    # ---- Internal helpers ---------------------------------------------------

    @staticmethod
    def _build_prompt(text: str) -> str:
        """Construct a prompt instructing Gemini to return structured JSON.

        Asking explicitly for JSON-only output (and providing an example
        schema) dramatically improves parse reliability compared to
        free-form prompts.
        """
        # Escape the user's text minimally by wrapping it in triple quotes
        # inside the prompt; we do not execute or evaluate this text.
        return f"""
You are an expert customer feedback analyst. Analyze the following text and
respond with ONLY a valid JSON object (no markdown formatting, no code
fences, no extra commentary) matching exactly this schema:

{{
  "sentiment": "Positive" | "Negative" | "Neutral",
  "emotion": "<single dominant emotion, e.g. Joy, Frustration, Anger, Sadness, Surprise, Satisfaction>",
  "confidence": <float between 0.0 and 1.0 representing your confidence in the sentiment>,
  "explanation": "<a detailed explanation, about 8-10 sentences, thoroughly discussing why you chose this sentiment, what specific words/phrases signal it, and any nuance or mixed signals in the text>",
  "summary": "<a detailed summary, about 8-10 sentences, covering the main points, context, and any secondary issues or praise mentioned in the text>",
  "key_topics": ["<topic1>", "<topic2>", "..."],
  "suggested_response": "<a short, professional business response addressing the feedback, max 2-3 sentences>"
}}

Text to analyze:
\"\"\"{text}\"\"\"

Write the explanation and summary as substantial, well-developed paragraphs
(approximately 8-10 sentences / lines each) rather than single-line answers.
Respond with ONLY the JSON object.
""".strip()

    def _call_model(self, prompt: str) -> str:
        """Call the Gemini model and return the raw text response.

        Translates Gemini/SDK exceptions into our custom exception
        hierarchy based on message content, since the SDK does not always
        expose distinct exception classes for every failure mode.

        Raises:
            APIKeyError, RateLimitError, AnalysisTimeoutError, NetworkError,
            AnalyzerError
        """
        start_time = time.monotonic()
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
        except Exception as exc:  # noqa: BLE001 - broad on purpose; we classify below
            self._raise_classified_error(exc, start_time)
        else:
            if not response or not getattr(response, "text", None):
                raise AnalyzerError(
                    "The AI model returned an empty response. Please try again."
                )
            return response.text

    def _raise_classified_error(self, exc: Exception, start_time: float) -> None:
        """Inspect an exception and raise the appropriate custom exception.

        Args:
            exc: The original exception raised by the Gemini SDK.
            start_time: `time.monotonic()` value captured before the call,
                used to distinguish timeouts from other failures.

        Raises:
            One of APIKeyError, RateLimitError, AnalysisTimeoutError,
            NetworkError, or AnalyzerError -- always raises, never returns.
        """
        message = str(exc).lower()
        elapsed = time.monotonic() - start_time

        # The current google-genai SDK raises structured APIError instances
        # with a numeric HTTP-style status code, which is far more reliable
        # to branch on than sniffing the exception message.
        status_code: Optional[int] = getattr(exc, "code", None) if isinstance(
            exc, genai_errors.APIError
        ) else None

        if status_code in (401, 403) or "api key" in message or "api_key" in message or "unauthorized" in message or "permission" in message:
            raise APIKeyError(
                "The provided API key was rejected. Please check that "
                "GEMINI_API_KEY is correct and active."
            ) from exc

        if status_code == 404 or "not found" in message or "is not supported for generatecontent" in message:
            raise AnalyzerError(
                f"The configured model '{self._model_name}' was not found or is "
                "no longer supported. It may have been deprecated -- check "
                "https://ai.google.dev/gemini-api/docs/models for current "
                "model names."
            ) from exc

        if status_code == 429 or "quota" in message or "rate limit" in message:
            raise RateLimitError(
                "The AI provider's rate limit or quota has been reached."
            ) from exc

        if "timeout" in message or "deadline" in message or elapsed >= _REQUEST_TIMEOUT_SECONDS:
            raise AnalysisTimeoutError(
                f"The request took too long to complete ({elapsed:.1f}s)."
            ) from exc

        if (
            "connection" in message
            or "network" in message
            or "dns" in message
            or "unreachable" in message
        ):
            raise NetworkError(
                "Could not reach the AI provider. Please check your "
                "internet connection."
            ) from exc

        raise AnalyzerError(f"An unexpected error occurred during analysis: {exc}") from exc

    @staticmethod
    def _parse_response(raw_text: str) -> Dict[str, Any]:
        """Parse and validate the model's raw text into our result schema.

        Defensive by design: strips markdown code fences if present,
        falls back to defaults for any missing/invalid fields rather than
        raising, and clamps confidence into [0.0, 1.0].

        Raises:
            AnalyzerError: If the response cannot be parsed as JSON at all.
        """
        cleaned_text = raw_text.strip()

        # Some models wrap JSON in ```json ... ``` code fences despite
        # instructions not to -- strip those defensively.
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

        try:
            parsed: Dict[str, Any] = json.loads(cleaned_text)
        except json.JSONDecodeError as exc:
            raise AnalyzerError(
                "The AI model's response could not be parsed as valid JSON."
            ) from exc

        result = dict(_DEFAULT_RESULT)  # start from safe defaults
        result.update(
            {key: parsed[key] for key in _DEFAULT_RESULT if key in parsed}
        )

        # Validate and normalize sentiment.
        if result["sentiment"] not in _VALID_SENTIMENTS:
            result["sentiment"] = "Neutral"

        # Validate and clamp confidence.
        try:
            confidence = float(result["confidence"])
        except (TypeError, ValueError):
            confidence = 0.0
        result["confidence"] = max(0.0, min(1.0, confidence))

        # Ensure key_topics is always a list of strings.
        if not isinstance(result["key_topics"], list):
            result["key_topics"] = [str(result["key_topics"])] if result["key_topics"] else []

        return result


# --------------------------------------------------------------------------
# Factory function -- the ONLY place that needs to change to add a provider
# --------------------------------------------------------------------------

def get_analyzer(provider: Optional[str] = None) -> BaseSentimentAnalyzer:
    """Return a configured sentiment analyzer instance for the given provider.

    Args:
        provider: The AI provider to use ("gemini", "openai", "claude").
            If omitted, read from the `AI_PROVIDER` environment variable,
            defaulting to "gemini".

    Returns:
        An instance implementing `BaseSentimentAnalyzer`.

    Raises:
        APIKeyError: If the selected provider's API key is missing.
        AnalyzerError: If an unsupported provider name is given.

    To add a new provider (e.g. OpenAI):
        1. Implement `OpenAISentimentAnalyzer(BaseSentimentAnalyzer)` above.
        2. Add an `elif resolved_provider == "openai":` branch below.
        No other file in the project needs to change.
    """
    resolved_provider = (provider or os.getenv("AI_PROVIDER", "gemini")).strip().lower()

    if resolved_provider == "gemini":
        return GeminiSentimentAnalyzer()

    # Placeholder branches for future providers -- intentionally left as
    # clear extension points rather than implemented, per current scope.
    # elif resolved_provider == "openai":
    #     return OpenAISentimentAnalyzer()
    # elif resolved_provider == "claude":
    #     return ClaudeSentimentAnalyzer()

    raise AnalyzerError(
        f"Unsupported AI_PROVIDER '{resolved_provider}'. Supported "
        "providers: 'gemini'."
    )