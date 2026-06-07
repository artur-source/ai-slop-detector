# Tests for web page scraping and text extraction behavior.

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from app.scraper import clean_text, extract_text, scrape_url


def test_scrape_url_valid_url_returns_text() -> None:
    """Valid URLs should return readable non-empty text from paragraph tags."""

    paragraph = "This is a long paragraph with enough readable content to pass the minimum length filter."
    html = f"<html><body><p>{paragraph}</p><p>{paragraph}</p><p>{paragraph}</p></body></html>"
    response = Mock(text=html)
    response.raise_for_status = Mock()

    with patch("app.scraper.requests.get", return_value=response):
        result = scrape_url("https://example.com/article")

    assert isinstance(result, str)
    assert result


def test_scrape_url_invalid_url_raises_value_error() -> None:
    """Invalid URLs should raise a clear ValueError."""

    with pytest.raises(ValueError, match="Invalid URL"):
        scrape_url("not-a-url")


def test_scrape_url_without_paragraphs_raises_value_error() -> None:
    """Pages without readable paragraphs should raise ValueError."""

    response = Mock(text="<html><body><main>No paragraph tags here.</main></body></html>")
    response.raise_for_status = Mock()

    with patch("app.scraper.requests.get", return_value=response):
        with pytest.raises(ValueError, match="No readable content found"):
            scrape_url("https://example.com/empty")


def test_extract_text_direct_text_returns_text_source() -> None:
    """Direct text input should return the cleaned text and text source."""

    result, source = extract_text("This is direct text input.")

    assert result == "This is direct text input."
    assert source == "text"


def test_extract_text_url_returns_url_source() -> None:
    """URL input should call scrape_url and return the url source."""

    with patch("app.scraper.scrape_url", return_value="Scraped article text."):
        result, source = extract_text("https://example.com/article")

    assert result == "Scraped article text."
    assert source == "url"


def test_clean_text_normalizes_multiple_newlines() -> None:
    """Clean text should reduce long blank-line runs to a maximum of two newlines."""

    result = clean_text("First paragraph.\n\n\n\nSecond paragraph.")

    assert "\n\n\n" not in result
    assert result == "First paragraph.\n\nSecond paragraph."

