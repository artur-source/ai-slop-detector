# Web routes for text submission and detector results.

from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for

from app.detector import detect
from app.scraper import extract_text


main = Blueprint("main", __name__)


@main.get("/")
def index() -> str:
    """Render the main text or URL submission form.

    Returns:
        Rendered index page HTML.
    """

    return render_template("index.html")


@main.post("/analyze")
def analyze() -> str:
    """Analyze submitted text or URL and render detection results.

    Returns:
        Rendered results page HTML, or a redirect to the form when validation fails.
    """

    input_data = request.form.get("input_data", "").strip()
    if not input_data:
        flash("Please enter text or a URL.")
        return redirect(url_for("main.index"))

    try:
        text, source = extract_text(input_data)
        results = detect(text)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("main.index"))
    except Exception:
        flash("An unexpected error occurred.")
        return redirect(url_for("main.index"))

    return render_template(
        "result.html",
        results=results,
        source=source,
        input_preview=text[:300],
        input_data=input_data,
    )

