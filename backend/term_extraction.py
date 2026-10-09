"""Dictionary lemma/alias matching, with punctuation and ambiguity guards."""
import re
import pandas as pd
from preprocessing import get_nlp

AMBIGUOUS = {'model','train','class','tree','forest','window','thread','process',
             'stack','queue','attention','token','memory','kernel','array','cloud','network','program'}
TECHNICAL = {'computer','computing','algorithm','data','dataset','software','hardware',
             'neural','learning','learn','machine','code','python','database','cpu','memory',
             'classification','regression','operating','programming','function','variable','ai','ml'}

class TermExtractor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.nlp = get_nlp()
        self.term_to_index, self.surface_to_index = {}, {}
        self.max_term_len = 1
        aliases = []
        for idx, term in enumerate(df['term']):
            term = term.strip()
            aliases.append((term, idx))
            match = re.fullmatch(r'(.+?)\s*\(([A-Z][A-Z0-9-]{1,9})\)', term)
            if match:
                aliases.extend([(match[1],idx),(match[2],idx)])
        aliases.sort(key=lambda item: item[0] != df.iloc[item[1]]['term'])
        for (alias,idx),doc in zip(aliases,self.nlp.pipe([a for a,_ in aliases],disable=['parser','ner'])):
            tokens = [t for t in doc if not t.is_space and not t.is_punct]
            key = tuple(t.lemma_.casefold() for t in tokens)
            surface = tuple(t.text.casefold() for t in tokens)
            if key:
                self.term_to_index.setdefault(key,idx)
                self.surface_to_index.setdefault(surface,idx)
                self.max_term_len=max(self.max_term_len,len(key))

    def extract(self,text: str) -> list[dict]:
        if not isinstance(text,str) or not text.strip():
            return []
        doc=self.nlp(text)
        context=any(t.lemma_.casefold() in TECHNICAL for t in doc)
        segments,current=[],[]
        for t in doc:
            if t.is_space:
                if '\n' in t.text and current:
                    segments.append(current);current=[]
                continue
            if t.is_punct:
                if t.text in {'-','‐','‑'}:
                    continue
                if current:
                    segments.append(current);current=[]
            else:
                current.append(t)
        if current:
            segments.append(current)
        found={}
        total_tokens=sum(map(len,segments))
        for tokens in segments:
            lemmas=[t.lemma_.casefold() for t in tokens]
            occupied=set()
            for size in range(min(self.max_term_len,len(tokens)),0,-1):
                for start in range(len(tokens)-size+1):
                    span=set(range(start,start+size))
                    if occupied & span:
                        continue
                    key=tuple(lemmas[start:start+size])
                    surface=tuple(t.text.casefold() for t in tokens[start:start+size])
                    idx=self.surface_to_index.get(surface,self.term_to_index.get(key))
                    if idx is None:
                        continue
                    if size==1 and total_tokens>1:
                        t=tokens[start]
                        if t.is_stop:
                            continue
                        if key[0] in AMBIGUOUS and (not context or t.pos_ not in {'NOUN','PROPN'}):
                            continue
                    occupied |= span
                    found.setdefault(idx,(size,tokens[start].idx))
        indices=sorted(found,key=lambda idx:(-found[idx][0],found[idx][1]))
        return [{'index':int(idx)} for idx in indices]
