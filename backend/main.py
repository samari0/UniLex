"""
UniLex backend API (Day 2 of the plan).

Endpoints:
  GET  /health                - health check
  GET  /                      - website
  GET  /term/{term_name}      - exact term lookup
  GET  /search                - basic TF-IDF keyword search
  GET  /search/semantic       - SBERT semantic search (natural-language questions)
  POST /extract                - extract known UniLex terms from pasted lecture text
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_loader import load_dataset, build_search_text, to_entry_dict
from search_tfidf import TfidfSearch
from term_extraction import TermExtractor

STATE = {}

# Semantic search is optional at import time: if the SBERT model can't be
# downloaded (e.g. no internet access), the API still runs with TF-IDF only,
# and /search/semantic returns a clear 503 instead of crashing the whole app.
SEMANTIC_ENABLED = os.environ.get("UNILEX_DISABLE_SEMANTIC", "") != "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    df = load_dataset()
    search_texts = [build_search_text(row) for _, row in df.iterrows()]

    STATE["df"] = df
    STATE["tfidf"] = TfidfSearch(df, search_texts)
    STATE["extractor"] = TermExtractor(df)
    STATE["semantic"] = None
    STATE["semantic_error"] = None

    if SEMANTIC_ENABLED:
        try:
            from search_semantic import SemanticSearch
            STATE["semantic"] = SemanticSearch(df, search_texts)
        except Exception as e:
            STATE["semantic_error"] = str(e)

    yield
    STATE.clear()


app = FastAPI(title="UniLex API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    df = STATE.get("df")
    return {
        "status": "ok",
        "terms_loaded": int(len(df)) if df is not None else 0,
        "semantic_search_available": STATE.get("semantic") is not None,
        "semantic_search_error": STATE.get("semantic_error"),
    }


@app.get("/term/{term_name}")
def get_term(term_name: str):
    df = STATE["df"]
    match = df[df["term_lower"] == term_name.strip().lower()]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Term '{term_name}' not found")
    return to_entry_dict(match.iloc[0])


@app.get("/search")
def search(q: str = Query(..., min_length=1), top_k: int = 5):
    df = STATE["df"]
    hits = STATE["tfidf"].search(q, top_k=top_k)
    return {
        "query": q,
        "method": "tfidf",
        "results": [
            {**to_entry_dict(df.iloc[h["index"]]), "score": round(h["score"], 4)}
            for h in hits
        ],
    }


@app.get("/search/semantic")
def search_semantic(q: str = Query(..., min_length=1), top_k: int = 5):
    df = STATE["df"]
    semantic = STATE.get("semantic")
    if semantic is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Semantic search model is unavailable right now "
                f"({STATE.get('semantic_error') or 'model not loaded'}). "
                "Try /search for TF-IDF keyword search instead."
            ),
        )
    hits = semantic.search(q, top_k=top_k)
    return {
        "query": q,
        "method": "sbert",
        "results": [
            {**to_entry_dict(df.iloc[h["index"]]), "score": round(h["score"], 4)}
            for h in hits
        ],
    }


@app.post("/extract")
def extract_terms(req: ExtractRequest):
    df = STATE["df"]
    hits = STATE["extractor"].extract(req.text)
    return {
        "text_length": len(req.text),
        "terms_found": [to_entry_dict(df.iloc[h["index"]]) for h in hits],
    }


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="website")
