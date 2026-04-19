import os
import re
import hashlib
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect, text

# LangChain
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.chat_models import ChatOllama  # via Ollama
from langchain.agents import create_sql_agent

# ============================
# Utilities
# ============================

def to_table_name(filename: str) -> str:
    """Derive a safe SQLite table name from a filename.
    Example: "Sales-2024 Q1.csv" -> "sales_2024_q1".
    """
    base = re.sub(r"\.[Cc][Ss][Vv]$", "", filename)
    base = re.sub(r"[^0-9a-zA-Z_]+", "_", base)
    base = base.strip("_").lower() or "table1"
    return base


def ensure_unique_table_name(engine, base: str) -> str:
    """Ensure the table name is unique in the SQLite database by appending a counter if needed."""
    insp = inspect(engine)
    name = base
    i = 2
    while insp.has_table(name):
        name = f"{base}_{i}"
        i += 1
    return name


def _files_fingerprint(files: List) -> str:
    """Create a lightweight fingerprint of the uploaded files to avoid rebuilding DB if unchanged."""
    h = hashlib.sha1()
    for f in files:
        name = getattr(f, "name", "unknown")
        pos = f.tell() if hasattr(f, "tell") else 0
        head = f.read(4096)
        if hasattr(f, "seek"):
            f.seek(pos)
        h.update(name.encode("utf-8"))
        h.update(len(head).to_bytes(4, "little"))
        h.update(head)
    return h.hexdigest()


def _apply_performance_pragmas(engine):
    """SQLite speed-ups suitable for ephemeral analysis DBs."""
    with engine.begin() as conn:
        conn.execute(text("PRAGMA temp_store = MEMORY;"))
        conn.execute(text("PRAGMA synchronous = OFF;"))
        conn.execute(text("PRAGMA journal_mode = MEMORY;"))


def _create_engine(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    _apply_performance_pragmas(engine)
    return engine


def load_csvs_to_sqlite(files: List, db_path: str) -> List[Tuple[str, pd.DataFrame]]:
    """Load multiple CSV files into a single SQLite database as separate tables.
    Returns a list of (table_name, df_preview) tuples.
    """
    # Always start fresh: remove old DB file if exists
    if os.path.exists(db_path):
        os.remove(db_path)

    engine = _create_engine(db_path)
    loaded = []
    for f in files:
        df = pd.read_csv(f)
        if hasattr(f, "seek"):
            f.seek(0)
        base = to_table_name(getattr(f, "name", "table.csv"))
        table = ensure_unique_table_name(engine, base)
        df.to_sql(table, con=engine, if_exists="replace", index=False)
        loaded.append((table, df.head(100)))
    return loaded


def create_heuristic_indexes(db_path: str):
    """Create indexes on likely join/filter columns to speed up JOINs and filters."""
    engine = _create_engine(db_path)
    insp = inspect(engine)
    with engine.begin() as conn:
        for t in insp.get_table_names():
            cols = [c["name"] for c in insp.get_columns(t)]
            # Join keys
            for c in cols:
                if c == "id" or c.endswith("_id") or c in ("car_id", "part_id", "model_id"):
                    conn.execute(text(f'CREATE INDEX IF NOT EXISTS idx_{t}_{c} ON "{t}"("{c}")'))
            # Common filters
            for c in ("date", "brand", "category"):
                if c in cols:
                    conn.execute(text(f'CREATE INDEX IF NOT EXISTS idx_{t}_{c} ON "{t}"("{c}")'))


def build_sql_agent(db_path: str, model_name: str = "qwen3:8b"):
    """Build a LangChain SQL agent connected to the SQLite DB and warm it up."""
    db = SQLDatabase.from_uri(
        f"sqlite:///{db_path}",
        sample_rows_in_table_info=3,
    )
    llm = ChatOllama(
        model=model_name,
        temperature=0,
        keep_alive=300,  # keep the model warm for 5 minutes
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,  # hide the long step-by-step logs
        agent_type="zero-shot-react-description",
        handle_parsing_errors=True,
        max_iterations=6,
        early_stopping_method="generate",
    )

    # Warmup (non-blocking best-effort)
    try:
        _ = llm.invoke("ping")
    except Exception:
        pass

    return agent, db


def list_tables_and_columns(db_path: str) -> pd.DataFrame:
    """Return a DataFrame describing tables and columns for quick inspection."""
    engine = _create_engine(db_path)
    insp = inspect(engine)
    rows = []
    for t in insp.get_table_names():
        cols = insp.get_columns(t)
        for c in cols:
            rows.append({
                "table": t,
                "column": c.get("name"),
                "type": str(c.get("type")),
                "nullable": c.get("nullable"),
            })
    return pd.DataFrame(rows)


# ============================
# Streamlit App
# ============================
st.set_page_config(page_title="LangChain SQL Agent for CSVs", layout="wide")
st.title("Chat with your CSVs")

with st.sidebar:
    st.markdown("### 1) Upload CSV files")
    uploaded_files = st.file_uploader(
        "Choose one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    st.markdown("### 2) Model")
    st.info("Using model: **qwen3:8b** (via Ollama)")

    keep_history = st.checkbox("Keep chat history", value=True)

    st.markdown("---")
    st.caption("""
    Try cross-table questions:
    - List tables and their row counts
    - Join sales and products to compute total revenue by category
    - For each user, show the latest purchase date from orders and customers
    """)

# Session state for chat history & DB cache key
if "history" not in st.session_state:
    st.session_state.history = []
if "db_hash" not in st.session_state:
    st.session_state.db_hash = None
if "table_names" not in st.session_state:
    st.session_state.table_names = []

DB_PATH = str(Path.cwd() / "tmp_chat_with_csv.db")

if not uploaded_files:
    st.info("Please upload one or more CSV files to start")
    st.stop()

# Build / reuse the DB only when files change
curr_hash = _files_fingerprint(uploaded_files)
try:
    if st.session_state.db_hash != curr_hash:
        with st.spinner("Loading CSVs into SQLite and building indexes..."):
            loaded = load_csvs_to_sqlite(uploaded_files, DB_PATH)
            create_heuristic_indexes(DB_PATH)
            st.session_state.table_names = [t for t, _ in loaded]
            st.session_state.db_hash = curr_hash
            st.success(f"Loaded {len(loaded)} table(s) into SQLite.")
            tabs = st.tabs([f"{t}" for t, _ in loaded])
            for (t, df_preview), tab in zip(loaded, tabs):
                with tab:
                    st.caption(f"Preview of table `{t}`")
                    st.dataframe(df_preview, use_container_width=True)
    else:
        st.info("Reusing existing database (no file changes detected).")

    # Show schema overview
    with st.expander("Schema overview (tables & columns)", expanded=False):
        schema_df = list_tables_and_columns(DB_PATH)
        st.dataframe(schema_df, use_container_width=True)
except Exception as e:
    st.error(f"Load failed: {e}")
    st.stop()

# Build agent (cached-ish; lightweight, but we could also place in cache_resource if desired)
agent, db = build_sql_agent(DB_PATH, model_name="qwen3:8b")

# Render history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])

# Chat input
user_q = st.chat_input("Ask anything about your data")
if user_q:
    with st.chat_message("user"):
        st.write(user_q)

    system_hint = (
        "You are a helpful SQL data analyst connected to a SQLite database with multiple tables. "
        "When a question requires information from multiple tables, infer reasonable join keys from column names and values, "
        "and perform JOINs as needed. Only use SELECT queries; never use INSERT/UPDATE/DELETE/DROP. "
        "Prefer to select only necessary columns. If unsure about join keys, first query table schemas. "
    )

    final_q = f"{system_hint}\nUser question: {user_q}"

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent.invoke({"input": final_q})
                answer = result.get("output", "(no output)")
            except Exception as e:
                answer = f"Error: {e}"
        st.write(answer)

    if keep_history:
        st.session_state.history.append({"q": user_q, "a": answer})
