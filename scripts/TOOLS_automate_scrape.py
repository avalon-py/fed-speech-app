#!/usr/bin/env python3
"""
Federal Reserve Speech & Testimony Scraper — Incremental Updater
==================================================================

Pulls the Fed's official speeches-and-testimony RSS feed, figures out which
entries haven't been scraped yet, scrapes their full text, and appends new
rows to dataset/fed_speech.csv (columns: id, date, title, speaker, content).

Designed to be run repeatedly (e.g. via GitHub Actions cron) — it is
idempotent and safe to re-run: already-seen links are skipped.

Dedup uses two layers:
  1. dataset/.seen_links.txt — a plain list of speech/testimony URLs
     already scraped. Kept separate from fed_speech.csv because that CSV's
     schema (id, date, title, speaker, content) has no link column.
  2. The most recent date already present in fed_speech.csv — used as a
     fallback cutoff, since .seen_links.txt starts empty and the historical
     CSV has no links to seed it with.

CSV OUTPUT FORMAT: only the 'content' field is quoted. id, date, title, and
speaker are written unquoted (safe because clean_text() strips commas and
quote characters from every field before writing).
"""

import csv
import datetime
import re
import sys
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

# The csv module defaults to a 128KB per-field limit, which some longer
# speech/testimony transcripts exceed. Raise it as high as the platform
# allows (Windows' underlying C long can be smaller than sys.maxsize).
_max_field_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_field_size)
        break
    except OverflowError:
        _max_field_size = int(_max_field_size / 10)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RSS_URL = "https://www.federalreserve.gov/feeds/speeches_and_testimony.xml"
DATASET_DIR = Path("dataset")
CSV_PATH = DATASET_DIR / "fed_speech.csv"
SEEN_LINKS_PATH = DATASET_DIR / ".seen_links.txt"
CSV_FIELDS = ["id", "date", "title", "speaker", "content"]

HEADERS = {
    "User-Agent": "fed-speech-tracker/1.0 (personal research project; contact: <your-email-here>)"
}
REQUEST_TIMEOUT = 15
REQUEST_DELAY_SECONDS = 1.0  # be polite between page fetches


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace(",", "")
    text = text.replace('"', "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# RSS feed parsing
# ---------------------------------------------------------------------------

def fetch_speech_links_from_rss(rss_url: str) -> list[dict]:
    """
    Returns a list of dicts: {"link": ..., "title": ..., "pub_date": ...}
    for every entry currently in the Fed's speeches-and-testimony RSS feed.
    """
    resp = requests.get(rss_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    root = ElementTree.fromstring(resp.content)
    items = []
    for item in root.findall(".//item"):
        link_el = item.find("link")
        title_el = item.find("title")
        date_el = item.find("pubDate")

        link = link_el.text.strip() if link_el is not None and link_el.text else ""
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        pub_date = date_el.text.strip() if date_el is not None and date_el.text else ""

        if link:
            items.append({"link": link, "title": title, "pub_date": pub_date})

    return items


# ---------------------------------------------------------------------------
# Page scraping (mirrors scrape_newsevents from the original notebook;
# verified to work correctly against both speech and testimony pages, which
# share the same Fed site template)
# ---------------------------------------------------------------------------

def scrape_newsevents(soup: BeautifulSoup) -> tuple[str, str, str, str]:
    """
    Extract (title, date, speaker, content) from a modern
    federalreserve.gov/newsevents/speech/... or /newsevents/testimony/...
    page.
    """
    title = ""
    date = ""
    speaker = ""
    content = ""

    title_tag = soup.find("h3", class_="title")
    if title_tag:
        title = clean_text(title_tag.get_text())

    date_tag = soup.find("p", class_="article__time")
    if date_tag:
        date = clean_text(date_tag.get_text())

    speaker_tag = soup.find("p", class_="speaker")
    if speaker_tag:
        speaker = clean_text(speaker_tag.get_text())

    content_div = soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
    if content_div:
        paragraphs = content_div.find_all("p")
        content = clean_text(
            " ".join(
                p.get_text()
                for p in paragraphs
                if p.get_text().strip() != ""
            )
        )

    return title, date, speaker, content


SCRAPE_MAX_RETRIES = 3
SCRAPE_BACKOFF_SECONDS = 3.0  # multiplied by attempt number between retries


def scrape_speech_page(link: str) -> dict:
    """
    Fetch and parse a single speech/testimony page. Returns a row dict with
    an "ok" flag: True if the page was fetched and parsed, False if every
    attempt failed (network error, timeout, etc.).

    Retries transient failures (timeouts, connection resets) up to
    SCRAPE_MAX_RETRIES times with linear backoff before giving up, since a
    single timeout on one request out of dozens shouldn't kill that item
    permanently. Callers MUST check "ok" before writing a row or marking
    the link as seen — on failure, title/date/speaker/content are all ""
    and should not be treated as a real (empty) result.
    """
    last_error = None
    for attempt in range(1, SCRAPE_MAX_RETRIES + 1):
        try:
            resp = requests.get(link, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            if "newsevents" in link:
                title, date, speaker, content = scrape_newsevents(soup)
            else:
                print(f"  ! Unrecognized URL pattern, skipping parse: {link}", file=sys.stderr)
                title, date, speaker, content = "", "", "", ""

            return {
                "link": link, "title": title, "date": date,
                "speaker": speaker, "content": content, "ok": True,
            }

        except requests.RequestException as e:
            last_error = e
            print(f"  ! Attempt {attempt}/{SCRAPE_MAX_RETRIES} failed for {link}: {e}", file=sys.stderr)
            if attempt < SCRAPE_MAX_RETRIES:
                time.sleep(SCRAPE_BACKOFF_SECONDS * attempt)

    print(f"  ! Giving up on {link} after {SCRAPE_MAX_RETRIES} attempts ({last_error})", file=sys.stderr)
    return {"link": link, "title": "", "date": "", "speaker": "", "content": "", "ok": False}


# ---------------------------------------------------------------------------
# CSV + state helpers
# ---------------------------------------------------------------------------

def load_seen_links(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_seen_links(path: Path, links: list[str]) -> None:
    with path.open("a") as f:
        for link in links:
            f.write(link + "\n")


def get_next_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 1
    max_id = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_id = max(max_id, int(row["id"]))
            except (ValueError, KeyError):
                continue
    return max_id + 1


def get_last_date_in_csv(csv_path: Path) -> "datetime.date | None":
    """
    Returns the most recent date already present in the CSV, so the script
    can skip anything on or before it. This matters because the CSV has no
    link column (dropped during the original historical scrape), so date is
    the only reliable bootstrap signal — .seen_links.txt starts empty and
    only protects future runs, not this first one.
    """
    if not csv_path.exists():
        return None
    latest = None
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("date") or "").strip()
            if not raw:
                continue
            try:
                d = datetime.datetime.strptime(raw, "%Y-%m-%d").date()
            except ValueError:
                continue
            if latest is None or d > latest:
                latest = d
    return latest


def parse_rss_pubdate(pub_date: str) -> "datetime.date | None":
    """Parses an RFC-822 style RSS pubDate string into a date object."""
    if not pub_date:
        return None
    try:
        return parsedate_to_datetime(pub_date).date()
    except (TypeError, ValueError):
        return None


def normalize_date(raw_date: str) -> str:
    """
    Try to normalize the scraped date string (e.g. 'September 3, 2026')
    to ISO format (YYYY-MM-DD). Falls back to the raw string if parsing
    fails, so a format surprise on the Fed's site doesn't crash the run.
    """
    raw_date = raw_date.strip()
    for fmt in ("%B %d, %Y", "%B %d %Y"):
        try:
            return datetime.datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw_date


def append_rows_to_csv(csv_path: Path, rows: list[dict]) -> None:
    """
    Appends rows with ONLY the 'content' field quoted. id/date/title/speaker
    are written unquoted. Safe because clean_text() strips commas and quote
    characters from every field, so no field can accidentally break the
    unquoted CSV structure.
    """
    file_exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(CSV_FIELDS) + "\n")
        for row in rows:
            line = (
                f'{row["id"]},{row["date"]},{row["title"]},'
                f'{row["speaker"]},"{row["content"]}"'
            )
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Fetching speech feed: {RSS_URL}")
    feed_items = fetch_speech_links_from_rss(RSS_URL)
    print(f"Feed contains {len(feed_items)} entries")

    seen_links = load_seen_links(SEEN_LINKS_PATH)
    last_date = get_last_date_in_csv(CSV_PATH)
    if last_date:
        print(f"Most recent date already in CSV: {last_date.isoformat()}")

    new_items = []
    for item in feed_items:
        if item["link"] in seen_links:
            continue
        # Belt-and-suspenders: also skip anything on or before the last
        # date already in the CSV. This matters most on the very first run,
        # since .seen_links.txt starts empty and can't yet protect against
        # re-adding speeches you already have.
        if last_date:
            item_date = parse_rss_pubdate(item["pub_date"])
            if item_date and item_date <= last_date:
                continue
        new_items.append(item)

    if not new_items:
        print("No new speeches found. Nothing to do.")
        return

    print(f"Found {len(new_items)} new speech(es)/testimony to scrape:")
    for item in new_items:
        print(f"  - {item['title']}")

    next_id = get_next_id(CSV_PATH)
    new_rows = []
    newly_seen_links = []
    failed_links = []

    for i, item in enumerate(new_items):
        link = item["link"]
        print(f"[{i + 1}/{len(new_items)}] Scraping {link}")
        scraped = scrape_speech_page(link)

        if not scraped["ok"]:
            # Don't write a row and don't mark it seen — a page we never
            # actually fetched isn't "done", it's just missing. Leaving it
            # off .seen_links.txt means the next run (or a manual rerun)
            # will pick it up again instead of silently losing it forever.
            failed_links.append(link)
            if i < len(new_items) - 1:
                time.sleep(REQUEST_DELAY_SECONDS)
            continue

        # Fall back to RSS title if page parsing came up empty.
        title = scraped["title"] or item["title"]
        date = normalize_date(scraped["date"]) if scraped["date"] else ""

        if not scraped["content"]:
            print(f"  ! Warning: no content extracted for {link}", file=sys.stderr)

        new_rows.append({
            "id": next_id,
            "date": date,
            "title": title,
            "speaker": scraped["speaker"],
            "content": scraped["content"],
        })
        newly_seen_links.append(link)
        next_id += 1

        if i < len(new_items) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    if new_rows:
        append_rows_to_csv(CSV_PATH, new_rows)
        append_seen_links(SEEN_LINKS_PATH, newly_seen_links)
        print(f"Appended {len(new_rows)} new row(s) to {CSV_PATH}")

    if failed_links:
        print(f"\n{len(failed_links)} link(s) failed after retries and were skipped (not written, not marked seen):", file=sys.stderr)
        for link in failed_links:
            print(f"  - {link}", file=sys.stderr)
        print("They'll be retried on the next run since they're still in the RSS feed and not in .seen_links.txt.", file=sys.stderr)


if __name__ == "__main__":
    main()