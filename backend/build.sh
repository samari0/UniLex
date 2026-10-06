#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python - <<'PY'
from data_loader import load_dataset, build_search_text
from search_semantic import SemanticSearch

df = load_dataset()
SemanticSearch(df, [build_search_text(row) for _, row in df.iterrows()])
PY
