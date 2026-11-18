# Streamlit Dashboard Demo

This folder provides a deployment entrypoint and synthetic sample data for the comment-analysis dashboard.

## Contents

- `app.py`: thin deployment entrypoint that runs the application from the repository root.
- `sample_reviews.csv`: synthetic example data with no real customer or business records.

## Run locally

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run examples/streamlit-dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501/` and upload `examples/streamlit-dashboard/sample_reviews.csv`. Select `original_text` as the review column.

## Deploy on Streamlit Community Cloud

1. Sign in to <https://share.streamlit.io/> with the GitHub account that can access this repository.
2. Create a new app.
3. Use these settings:
   - Repository: `bbloey/Social-Media-Comment-Analysis`
   - Branch: `main`
   - Main file path: `examples/streamlit-dashboard/app.py`
4. Deploy the app and wait for dependency installation to finish.

The offline dashboard, rule evidence, CSV export, and review workspace do not require an API key.

For optional model features, add the following as root-level secrets in Streamlit Cloud's advanced settings. Never commit real keys to GitHub:

```toml
OPENAI_API_KEY = "your-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-5.6-sol"
OPENAI_API_MODE = "responses"
OPENAI_REASONING_EFFORT = "low"
```

## Data safety

Only synthetic data is included. Do not commit customer reviews, production exports, API keys, `.env` files, or files under `outputs/`.
