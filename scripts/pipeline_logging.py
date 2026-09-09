"""
Shared Supabase logging helper for the scraper, embedder, and (future)
trainer jobs. Each job calls start_run() once at the top of main(), and
finish_run() exactly once before it exits — whether it succeeded, found
nothing to do, or failed.
"""

import json
import os
from datetime import datetime, timezone


def start_run(conn, job_name: str) -> int:
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
        (new_id,) = cur.fetchone()
    conn.commit()
    return new_id


def finish_run(conn, run_id: int, status: str, rows_processed: int = 0,
                message: str = None, error_message: str = None, details: dict = None):
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


def log_training_detail(conn, pipeline_run_id: int, train_start=None, train_end=None,
                         eval_start=None, eval_end=None, metrics: dict = None,
                         hyperparams: dict = None, sanity_passed: bool = None,
                         sanity_details: dict = None, promoted: bool = None,
                         git_commit: str = None):
    """Call once, only from the trainer job, AFTER finish_run() has already
    updated the pipeline_runs header row for this run."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO training_run_detail
               (pipeline_run_id, train_start, train_end, eval_start, eval_end,
                metrics, hyperparams, sanity_passed, sanity_details, promoted, git_commit)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (pipeline_run_id, train_start, train_end, eval_start, eval_end,
             json.dumps(metrics) if metrics else None,
             json.dumps(hyperparams) if hyperparams else None,
             sanity_passed, json.dumps(sanity_details) if sanity_details else None,
             promoted, git_commit),
        )
    conn.commit()
