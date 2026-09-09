#!/usr/bin/env python3
"""
Federal Reserve Speech & Testimony Scraper — Incremental Updater (Postgres)
============================================================================

Same job as the CSV version, but appends to a Supabase Postgres table
instead of a CSV file. This removes two things the CSV version needed:

  - dataset/.seen_links.txt: replaced by a UNIQUE constraint on the
    'link' column plus ON CONFLICT (link) DO NOTHING on every insert —
    the database itself now enforces "don't insert this link twice"
    instead of a separate text file you had to keep in sync.
  - The git commit/push step in the GitHub Actions workflow: state lives
    in the database now, not in a file that needs to be committed back
    to the repo every run.

Requires the DB_URL environment variable — the Supabase "Transaction
pooler" connection string (port 6543), not the direct one. The pooler is
built for exactly this pattern (many short-lived connections from a
serverless-style runner like GitHub Actions); the direct connection has a
much smaller limit and isn't meant for that.

Set DB_URL as a GitHub Actions secret, never commit it to the repo.
"""

import datetime
import re
import sys
import time
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree
from pipeline_logging import start_run, finish_run

import psycopg2
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

import os

RSS_URL = "https://www.federalreserve.gov/feeds/speeches_and_testimony.xml"
DB_URL = os.environ["DB_URL"]  # fails loudly if the secret isn't set — better than silently writing nowhere

HEADERS = {
    "User-Agent": "fed-speech-tracker/1.0 (personal research project; contact: <your-email-here>)"
}
REQUEST_TIMEOUT = 15
REQUEST_DELAY_SECONDS = 1.0

SCRAPE_MAX_RETRIES = 3
SCRAPE_BACKOFF_SECONDS = 3.0


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
# Page scraping (unchanged from the CSV version)
# ---------------------------------------------------------------------------

def scrape_newsevents(soup: BeautifulSoup) -> tuple[str, str, str, str]:
    title = date = speaker = content = ""

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
            " ".join(p.get_text() for p in paragraphs if p.get_text().strip() != "")
        )

    return title, date, speaker, content


def scrape_speech_page(link: str) -> dict:
    """Same retry-then-give-up behavior as the CSV version, with an 'ok' flag."""
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

            return {"link": link, "title": title, "date": date, "speaker": speaker, "content": content, "ok": True}

        except requests.RequestException as e:
            last_error = e
            print(f"  ! Attempt {attempt}/{SCRAPE_MAX_RETRIES} failed for {link}: {e}", file=sys.stderr)
            if attempt < SCRAPE_MAX_RETRIES:
                time.sleep(SCRAPE_BACKOFF_SECONDS * attempt)

    print(f"  ! Giving up on {link} after {SCRAPE_MAX_RETRIES} attempts ({last_error})", file=sys.stderr)
    return {"link": link, "title": "", "date": "", "speaker": "", "content": "", "ok": False}


def parse_rss_pubdate(pub_date: str) -> "datetime.date | None":
    if not pub_date:
        return None
    try:
        return parsedate_to_datetime(pub_date).date()
    except (TypeError, ValueError):
        return None


def normalize_date(raw_date: str) -> "datetime.date | None":
    """Returns a date object (or None) instead of a string, since the
    Postgres column is typed 'date' — psycopg2 handles date objects
    directly without needing an ISO string round-trip."""
    raw_date = raw_date.strip()
    for fmt in ("%B %d, %Y", "%B %d %Y"):
        try:
            return datetime.datetime.strptime(raw_date, fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_last_date(conn) -> "datetime.date | None":
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM fed_speech")
        (last_date,) = cur.fetchone()
        return last_date


def link_exists(conn, link: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM fed_speech WHERE link = %s", (link,))
        return cur.fetchone() is not None


def insert_row(conn, row: dict) -> bool:
    """Returns True if a row was actually inserted, False if it was
    skipped due to a link conflict (already present)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO fed_speech (link, date, title, speaker, content)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (link) DO NOTHING
            """,
            (row["link"], row["date"], row["title"], row["speaker"], row["content"]),
        )
        inserted = cur.rowcount > 0
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conn = psycopg2.connect(DB_URL)
    run_id = start_run(conn, "scraper")

    try:
        print(f"Fetching speech feed: {RSS_URL}")
        feed_items = fetch_speech_links_from_rss(RSS_URL)
        print(f"Feed contains {len(feed_items)} entries")

        last_date = get_last_date(conn)
        if last_date:
            print(f"Most recent date already in DB: {last_date.isoformat()}")

        new_items = []
        for item in feed_items:
            if link_exists(conn, item["link"]):
                continue
            if last_date:
                item_date = parse_rss_pubdate(item["pub_date"])
                if item_date and item_date <= last_date:
                    continue
            new_items.append(item)

        if not new_items:
            print("No new speeches found. Nothing to do.")
            finish_run(conn, run_id, status="success", rows_processed=0,
                       message="Nothing to do.")
            return

        new_items.sort(key=lambda item: (parse_rss_pubdate(item["pub_date"]) is None,
                                          parse_rss_pubdate(item["pub_date"])))

        print(f"Found {len(new_items)} new speech(es)/testimony to scrape (oldest first):")
        for item in new_items:
            print(f"  - {item['title']}")

        inserted_count = 0
        failed_links = []

        for i, item in enumerate(new_items):
            link = item["link"]
            print(f"[{i + 1}/{len(new_items)}] Scraping {link}")
            scraped = scrape_speech_page(link)

            if not scraped["ok"]:
                failed_links.append(link)
                if i < len(new_items) - 1:
                    time.sleep(REQUEST_DELAY_SECONDS)
                continue

            title = scraped["title"] or item["title"]
            date = normalize_date(scraped["date"]) if scraped["date"] else None

            if not scraped["content"]:
                print(f"  ! Warning: no content extracted for {link}", file=sys.stderr)

            row = {
                "link": link, "date": date, "title": title,
                "speaker": scraped["speaker"], "content": scraped["content"],
            }
            if insert_row(conn, row):
                inserted_count += 1

            if i < len(new_items) - 1:
                time.sleep(REQUEST_DELAY_SECONDS)

        print(f"Inserted {inserted_count} new row(s) into fed_speech")

        if failed_links:
            print(f"\n{len(failed_links)} link(s) failed after retries and were skipped:", file=sys.stderr)
            for link in failed_links:
                print(f"  - {link}", file=sys.stderr)
            print("They're still absent from the table, so the next run will retry them.", file=sys.stderr)

        finish_run(
            conn, run_id, status="success", rows_processed=inserted_count,
            message=f"Inserted {inserted_count} new speech(es)",
            details={"failed_links": failed_links} if failed_links else None,
        )

    except Exception as e:
        finish_run(conn, run_id, status="failed", error_message=str(e))
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
