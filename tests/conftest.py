import sys,os
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
@pytest.fixture(scope='session')
def services():
 from data_loader import load_dataset,build_search_text
 from search_tfidf import TfidfSearch
 from term_extraction import TermExtractor
 from search_semantic import SemanticSearch
 df=load_dataset();texts=[build_search_text(r) for _,r in df.iterrows()]
 return {'df':df,'tfidf':TfidfSearch(df,texts),'extractor':TermExtractor(df),'semantic':SemanticSearch(df,texts,str(ROOT/'backend')),'semantic_error':None}
@pytest.fixture(scope='session')
def client(services):
 import main
 from fastapi.testclient import TestClient
 main.STATE.update(services)
 return TestClient(main.app)
