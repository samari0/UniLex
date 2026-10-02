#!/usr/bin/env bash
# Render (and any host) runs this before starting the app.
set -e
pip install -r requirements.txt
python -m spacy download en_core_web_sm
