import sys
import json
from pathlib import Path
import sqlite3
import traceback

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
SAMPLE_JSON = DATA_DIR / 'summaries.json'
SAMPLE_JSONL = DATA_DIR / 'summaries.jsonl'
SAMPLE_CSV = DATA_DIR / 'summaries.csv'
SQLITE_DB = DATA_DIR / 'triplets_smoke.db'

result = {"checks": []}

# check python
result['python'] = sys.executable

# check optional libraries
try:
    import pandas as pd
    result['pandas'] = True
except Exception:
    result['pandas'] = False

try:
    import sentence_transformers
    result['sentence_transformers'] = True
except Exception:
    result['sentence_transformers'] = False

try:
    import faiss
    result['faiss'] = True
except Exception:
    result['faiss'] = False

# load sample data if present
rows = []
try:
    if SAMPLE_JSON.exists():
        with open(SAMPLE_JSON, 'r', encoding='utf-8') as f:
            obj = json.load(f)
            if isinstance(obj, list):
                for i,entry in enumerate(obj[:200]):
                    s = entry.get('subject')
                    p = entry.get('predicate')
                    o = entry.get('object')
                    rows.append({'subject':s,'predicate':p,'object':o,'row_id':i,'source':str(SAMPLE_JSON)})
    elif SAMPLE_JSONL.exists():
        with open(SAMPLE_JSONL, 'r', encoding='utf-8') as f:
            for i,line in enumerate(f):
                if i>199: break
                if not line.strip(): continue
                obj = json.loads(line)
                rows.append({'subject':obj.get('subject'),'predicate':obj.get('predicate'),'object':obj.get('object'),'row_id':i,'source':str(SAMPLE_JSONL)})
    elif SAMPLE_CSV.exists():
        try:
            import pandas as pd
            df = pd.read_csv(SAMPLE_CSV)
            for i,row in df.head(200).iterrows():
                rows.append({'subject':row.get('subject'),'predicate':row.get('predicate'),'object':row.get('object'),'row_id':int(i),'source':str(SAMPLE_CSV)})
        except Exception:
            pass
    else:
        result['checks'].append(('data_file', False, 'No sample data file found (summaries.json/jsonl/csv)'))

    result['loaded_rows'] = len(rows)
    result['checks'].append(('load_data', len(rows)>0, f'loaded {len(rows)} rows'))
except Exception as e:
    result['checks'].append(('load_data_error', False, str(e)))
    traceback.print_exc()

# basic validation
bad = 0
for r in rows:
    if not (isinstance(r.get('subject'), str) and isinstance(r.get('predicate'), str) and isinstance(r.get('object'), str)):
        bad += 1
result['checks'].append(('validate_types', bad==0, f'{bad} invalid rows'))

# build inverted index
inv = {'s':{}, 'p':{}, 'o':{}}
for r in rows:
    rid = r['row_id']
    s = (r.get('subject') or '').strip()
    p = (r.get('predicate') or '').strip()
    o = (r.get('object') or '').strip()
    inv['s'].setdefault(s, []).append(rid)
    inv['p'].setdefault(p, []).append(rid)
    inv['o'].setdefault(o, []).append(rid)

result['checks'].append(('build_inverted_index', True, f'subject_keys={len(inv["s"])}, predicate_keys={len(inv["p"])}, object_keys={len(inv["o"]) }'))

# sqlite insert and simple query
try:
    conn = sqlite3.connect(str(SQLITE_DB))
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS triplets (id INTEGER PRIMARY KEY, subject TEXT, predicate TEXT, object TEXT, source TEXT)''')
    conn.commit()
    cur.execute('DELETE FROM triplets')
    conn.commit()
    for r in rows[:100]:
        cur.execute('INSERT INTO triplets (id,subject,predicate,object,source) VALUES (?,?,?,?,?)', (int(r['row_id']), r.get('subject'), r.get('predicate'), r.get('object'), r.get('source')))
    conn.commit()
    cur.execute('SELECT COUNT(*) FROM triplets')
    cnt = cur.fetchone()[0]
    result['checks'].append(('sqlite_insert', cnt==min(100,len(rows)), f'inserted={cnt}'))
    # run a sample pattern query if any subject contains a word
    sample_q = None
    for s in inv['s'].keys():
        if s and ' ' in s:
            sample_q = s.split()[0]
            break
    if sample_q:
        cur.execute('SELECT id,subject,predicate,object FROM triplets WHERE subject LIKE ? LIMIT 5', (f'%{sample_q}%',))
        res = cur.fetchall()
        result['checks'].append(('sqlite_query', True, f'pattern={sample_q}, rows={len(res)}'))
    else:
        result['checks'].append(('sqlite_query', True, 'no suitable sample subject found'))
except Exception as e:
    result['checks'].append(('sqlite_error', False, str(e)))
    traceback.print_exc()

# final summary
print('\nSMOKE VALIDATION SUMMARY')
print('Python:', result['python'])
print('Pandas available:', result.get('pandas'))
print('SentenceTransformers available:', result.get('sentence_transformers'))
print('FAISS available:', result.get('faiss'))
print('Loaded rows:', result.get('loaded_rows',0))
print('\nChecks:')
for c in result['checks']:
    ok = 'PASS' if c[1] else 'FAIL'
    print(f'- {c[0]}: {ok} - {c[2]}')

if any(not c[1] for c in result['checks']):
    sys.exit(2)
else:
    sys.exit(0)
