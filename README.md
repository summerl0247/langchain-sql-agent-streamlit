# LangChain SQL Agent with Streamlit

A Streamlit app that lets users upload multiple CSV files, load them into SQLite, and ask natural-language questions across tables through a LangChain SQL agent.

## What this project does
- Upload one or more CSV files
- Load them into a temporary SQLite database
- Show table previews and schema overview
- Answer natural-language questions across multiple tables
- Support cross-table reasoning and inferred joins

## Why this project is useful
This project shows how to turn structured data into a user-facing analytics workflow. Instead of writing SQL manually, users can explore relational data with natural language.

## Tech stack
- Python
- Streamlit
- SQLite
- SQLAlchemy
- LangChain
- ChatOllama

## Key features
- Multi-file CSV upload
- Automatic table naming
- Schema inspection
- SQLite-backed querying
- Basic performance tuning with PRAGMA settings and heuristic indexes
- Chat history support

## How to run
1. Install dependencies from `requirements.txt`
2. Start your local model service
3. Run:
   ```bash
   streamlit run app.py
