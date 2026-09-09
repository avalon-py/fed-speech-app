# pipeline_logging.py
import os
from datetime import datetime, timezone

import psycopg2


def start_run(conn, job_name: str) -> int:
    """Insert a 'running' row at the start of a job. Returns the row id
    so the same script can update it when it finishes."""
    github_run_url = None
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if repo and run_id:
        github_run_url = f"https://github.com/{repo}/actions/runs/{run_id}"

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline_runs (job_name, status, started_at, github_run_url)
            VALUES (%s, 'running', %s, %s)
            RETURNING id
            """,
            (job_name, datetime.now(timezone.utc), github_run_url),
        )
        (run_id,) = cur.fetchone()
    conn.commit()
    return run_id


def finish_run(conn, run_id: int, status: str, rows_processed: int = 0,
                message: str = None, error_message: str = None, details: dict = None):
    import json
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_runs
            SET status = %s, finished_at = %s, rows_processed = %s,
                message = %s, error_message = %s, details = %s
            WHERE id = %s
            """,
            (status, datetime.now(timezone.utc), rows_processed,
             message, error_message, json.dumps(details) if details else None, run_id),
        )
    conn.commit()
