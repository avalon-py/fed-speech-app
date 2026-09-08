import os
from functools import lru_cache
from supabase import create_client, Client


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    try:
        import streamlit as st
        secrets = st.secrets
    except Exception:
        secrets = {}

    def _get(key: str) -> str:
        if key in secrets:
            return secrets[key]
        if key in os.environ:
            return os.environ[key]
        raise KeyError(
            f"'{key}' not found in st.secrets or os.environ. "
            "On Streamlit Cloud, set it under Settings → Secrets. "
            "Locally, set it in your shell or a .env file."
        )

    url = _get("SUPABASE_URL")
    key = _get("SUPABASE_KEY")
    return create_client(url, key)


def fetch_all_rows(table: str, order_col: str = "date", page_size: int = 1000) -> list[dict]:
    """
    Pages through .range() until a page comes back short of page_size.
    Supabase/PostgREST caps a single response at 1000 rows by default —
    if a table ever grows past that, a plain .select("*") silently
    truncates with no error, which would quietly feed the rolling
    feature engineering bad (incomplete) history. This loop is cheap
    insurance: if the table has fewer than page_size rows, it makes one
    request, sees a short batch, and returns immediately.
    """
    client = get_supabase()
    rows: list[dict] = []
    start = 0
    while True:
        end = start + page_size - 1
        resp = (
            client.table(table)
            .select("*")
            .order(order_col)
            .range(start, end)
            .execute()
        )
        batch = resp.data
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows
