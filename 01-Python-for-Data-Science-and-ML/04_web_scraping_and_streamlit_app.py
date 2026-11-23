import re
from typing import Dict, List
import pandas as pd
import requests
from bs4 import BeautifulSoup


SAMPLE_HTML = """
<div class="course-grid">
  <article class="module-card" data-id="m1">
    <h2>Module 1: NumPy Vectorization</h2>
    <span class="hours">3.5 hrs</span>
  </article>
  <article class="module-card" data-id="m2">
    <h3>Module 2: Pandas Pipelines</h3>
    <span class="hours">4.0 hrs</span>
  </article>
</div>
"""


def fetch_html_safely(url: str, timeout: int = 10) -> str:
    """Fetches webpage HTML with realistic headers and strict HTTP status validation."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AIEngineeringBot/1.0)"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    if "text/html" not in response.headers.get("Content-Type", ""):
        raise ValueError("Unexpected Content-Type received")
    return response.text


def parse_modules(html: str) -> pd.DataFrame:
    """Extracts structured records using BeautifulSoup and regex heading matching."""
    soup = BeautifulSoup(html, "html.parser")
    records: List[Dict[str, str]] = []

    for card in soup.find_all("article", class_="module-card"):
        heading = card.find(re.compile(r"^h[1-6]$"))
        duration = card.find("span", class_="hours")
        records.append({
            "module_id": card.get("data-id", "unknown"),
            "title": heading.get_text(strip=True) if heading else "N/A",
            "duration": duration.get_text(strip=True) if duration else "0 hrs",
        })

    return pd.DataFrame(records)


def run_streamlit_dashboard(df: pd.DataFrame) -> None:
    """Interactive stakeholder view (launch with: streamlit run 04_web_scraping_and_streamlit_app.py)."""
    import streamlit as st

    st.set_page_config(page_title="AI Engineering Ingestion Telemetry", layout="centered")
    st.title("Course 1: Ingestion & Analytics Telemetry")
    st.dataframe(df, use_container_width=True)

    df["numeric_hours"] = df["duration"].str.replace(" hrs", "", regex=False).astype(float)
    st.bar_chart(df.set_index("title")["numeric_hours"])


if __name__ == "__main__":
    parsed_df = parse_modules(SAMPLE_HTML)
    print("=== Parsed Web Modules ===")
    print(parsed_df.to_string(index=False))
