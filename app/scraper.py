# Utilities for extracting readable text from web pages.

from __future__ import annotations

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests import RequestException


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT = 10
TAGS_TO_REMOVE = ["script", "style", "nav", "footer", "header", "aside", "form"]
MIN_PARAGRAPH_LENGTH = 40


def scrape_url(url: str) -> str:
    """Fetch a URL and extract readable paragraph text from its HTML.

    Args:
        url: HTTP or HTTPS URL to scrape.

    Returns:
        Clean readable text joined by blank lines.

    Raises:
        ValueError: If the URL is invalid or no readable content is found.
        requests.RequestException: If the network request fails.
        requests.HTTPError: If the server returns a non-success status code.
    """

    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Invalid URL. Only http and https URLs are supported.")

    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
    except RequestException as exc:
        raise type(exc)(f"Could not fetch URL: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")

    for tag_name in TAGS_TO_REMOVE:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    paragraphs = [
        paragraph.get_text(" ", strip=True)
        for paragraph in soup.find_all("p")
        if len(paragraph.get_text(strip=True)) >= MIN_PARAGRAPH_LENGTH
    ]
    text = clean_text("\n\n".join(paragraphs))

    if not text:
        raise ValueError("No readable content found at this URL")

    return text


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks.

    Args:
        text: Raw text to normalize.

    Returns:
        Text with stripped lines and at most one blank line between paragraphs.
    """

    stripped_lines = [line.strip() for line in text.splitlines()]
    normalized_text = "\n".join(stripped_lines).strip()
    return re.sub(r"\n{3,}", "\n\n", normalized_text)


def extract_text(input_data: str) -> tuple[str, str]:
    """Extract clean text from either a direct text input or a URL.

    Args:
        input_data: User-provided text or an HTTP/HTTPS URL.

    Returns:
        A tuple containing cleaned text and source type, either "text" or "url".
    """

    stripped_input = input_data.strip()
    if stripped_input.startswith(("http://", "https://")):
        raw_text = scrape_url(stripped_input)
        source = "url"
    else:
        raw_text = stripped_input
        source = "text"

    return clean_text(raw_text), source

