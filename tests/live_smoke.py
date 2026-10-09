"""Read-only checks against the explicitly selected deployed website."""
import json,time,urllib.request,urllib.parse,urllib.error
from pathlib import Path
BASE='https://unilex-samar.onrender.com'
report={'url':BASE,'checks':[]}
def req(path,data=None):
 request=urllib.request.Request(BASE+path,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=120) as r:return json.load(r)
def record(label,actual,passed):
 row={'check':label,'actual':actual,'passed':bool(passed)};report['checks'].append(row);print(json.dumps(row),flush=True)
health=req('/health');record('health',health,health['terms_loaded']==3067 and health['semantic_search_available'])
cases=[('Why does my model perform well on training data but poorly on unseen data?',['Overfitting','Overfitting In Machine Learning']),('Which data structure follows first in first out?',['Queue','A Queue Data Structure']),('What is it called when test information accidentally enters model training?',['Data Leakage','Pipeline Leakage']),('How do I predict a continuous numerical value from input features?',['Regression']),('LIFO',['Stack','A Stack Data Structure','A Stack Class In Java Collections']),('why does lamya call herself a queen?',[]),('xyzabc123',[])]
for q,expected in cases:
 d=req('/search/semantic?'+urllib.parse.urlencode({'q':q}));terms=[x['term'] for x in d['results']];record(q,terms,bool(terms) and terms[0] in expected if expected else not terms)
for endpoint in ['/search','/search/semantic']:
 d=req(endpoint+'?q=');record('empty '+endpoint,d,d['results']==[])
try:req('/term/not_a_real_unilex_term')
except urllib.error.HTTPError as e:record('unknown term',e.code,e.code==404)
for text in ['','I waited in a queue for coffee.','Machine learning uses neural networks. Overfitting can be reduced with regularization.']:
 terms=[x['term'] for x in req('/extract',{'text':text})['terms_found']];record('extract '+text,terms,{'Machine Learning','Overfitting'}<=set(terms) if text.startswith('Machine') else not terms)
terms=[x['term'] for x in req('/search?q=Machine%20Learning')['results']];record('keyword machine learning',terms,bool(terms) and terms[0]=='Machine Learning')
report['tested_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
(Path(__file__).resolve().parents[1]/'docs/live-results.json').write_text(json.dumps(report,indent=2))
assert all(r['passed'] for r in report['checks']),report
