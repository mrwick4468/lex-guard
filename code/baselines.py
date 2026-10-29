# Task A: PAN12 predator identification (official split, user level, F0.5 as in PAN 2012).
# Task C: cross-corpus recall on Koutli predators not seen in PAN12, against PAN12 test non-predators.
import glob, json, re, hashlib, os, time
import numpy as np, xml.etree.ElementTree as ET
from collections import defaultdict
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score, average_precision_score

def find(pat):
    hits = glob.glob(f'/kaggle/input/**/{pat}', recursive=True); assert hits, pat; return hits[0]
norm = lambda s: re.sub(r'[^a-z0-9 ]', '', (s or '').lower()).strip()
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()

def load(xml_path, pred_path):
    preds = {l.strip() for l in open(pred_path) if l.strip()}
    docs = defaultdict(list)
    for _, c in ET.iterparse(xml_path):
        if c.tag != 'conversation': continue
        for m in c.iter('message'):
            docs[m.findtext('author')].append(m.findtext('text') or '')
        c.clear()
    users = sorted(docs)
    return users, [' \n '.join(docs[u]) for u in users], np.array([u in preds for u in users], int), docs

t0 = time.time()
tr_u, tr_x, tr_y, tr_docs = load(find('*training-corpus-2012-05-01.xml'), find('*training-corpus-predators-2012-05-01.txt'))
te_u, te_x, te_y, te_docs = load(find('*test-corpus-2012-05-17.xml'), find('*groundtruth-problem1.txt'))
print('users train', len(tr_u), tr_y.sum(), 'test', len(te_u), te_y.sum(), f'{time.time()-t0:.0f}s')

# Koutli predators not in PAN12 (>=20% of long lines shared with any PAN12 predator = duplicate).
pan_lines = {md5(norm(t)) for d, y, u in ((tr_docs, tr_y, tr_u), (te_docs, te_y, te_u))
             for uu, yy in zip(u, y) if yy for t in d[uu] if len(norm(t)) > 25}
kx, kname = [], []
for f in sorted(glob.glob('/kaggle/input/**/predators/**/*.txt', recursive=True)):
    lines = [l.strip() for l in open(f, errors='ignore') if l.strip()]
    h = {md5(norm(l)) for l in lines if len(norm(l)) > 25}
    if len(h & pan_lines) / max(len(h), 1) < 0.2:
        kx.append(' \n '.join(lines)); kname.append(os.path.basename(f))
off = {os.path.basename(f): f for f in glob.glob('/kaggle/input/**/officers/**/*.txt', recursive=True)}
ox = [' \n '.join(l.strip() for l in open(off[n.replace('predator_', 'officer_', 1)], errors='ignore') if l.strip())
      for n in kname if n.replace('predator_', 'officer_', 1) in off]
kpair = np.array([n.replace('predator_', 'officer_', 1) in off for n in kname])
print('koutli unseen', len(kx), 'with decoy file', kpair.sum())

word = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=300000, ngram_range=(1, 2))
char = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=300000, analyzer='char_wb', ngram_range=(2, 5))
Xtr = hstack([word.fit_transform(tr_x), char.fit_transform(tr_x)]).tocsr()
Xte = hstack([word.transform(te_x), char.transform(te_x)]).tocsr()
Xk = hstack([word.transform(kx), char.transform(kx)]).tocsr()
Xo = hstack([word.transform(ox), char.transform(ox)]).tocsr()
neg = te_y == 0

def evaluate(name, clf, score):
    # Threshold: maximise F0.5 on out-of-fold training scores (no test peeking).
    oof = cross_val_predict(clf, Xtr, tr_y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method=score)
    grid = np.quantile(oof, np.linspace(0.95, 0.99999, 400))
    thr = max(grid, key=lambda t: fbeta_score(tr_y, oof >= t, beta=0.5))
    clf.fit(Xtr, tr_y)
    s_te, s_k, s_o = getattr(clf, score)(Xte), getattr(clf, score)(Xk), getattr(clf, score)(Xo)
    if s_te.ndim == 2: s_te, s_k = s_te[:, 1], s_k[:, 1]
    p = s_te >= thr
    taskC_s = np.r_[s_k, s_te[neg]]; taskC_y = np.r_[np.ones(len(s_k)), np.zeros(neg.sum())]
    r = dict(model=name, threshold=float(thr),
             A_precision=precision_score(te_y, p), A_recall=recall_score(te_y, p), A_F05=fbeta_score(te_y, p, beta=0.5),
             A_F1=fbeta_score(te_y, p, beta=1), A_AUPRC=average_precision_score(te_y, s_te), A_ROCAUC=roc_auc_score(te_y, s_te),
             C_recall=float((s_k >= thr).mean()), C_ROCAUC=roc_auc_score(taskC_y, taskC_s),
             C_AUPRC=average_precision_score(taskC_y, taskC_s))
    hs, hy = np.r_[s_k[kpair], s_o], np.r_[np.ones(kpair.sum()), np.zeros(len(s_o))]
    r.update(Chard_n_pairs=int(kpair.sum()), Chard_recall=float((s_k[kpair] >= thr).mean()), Chard_decoy_FPR=float((s_o >= thr).mean()),
             Chard_precision=precision_score(hy, hs >= thr, zero_division=0), Chard_F05=fbeta_score(hy, hs >= thr, beta=0.5),
             Chard_ROCAUC=roc_auc_score(hy, hs))
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), f'{time.time()-t0:.0f}s')
    np.save(f'/kaggle/working/scores_{name}.npy', dict(te=s_te, k=s_k, o=s_o, tr_oof=oof))
    return r

res = [evaluate('tfidf_logreg', LogisticRegression(C=10, max_iter=3000, class_weight='balanced'), 'decision_function'),
       evaluate('tfidf_linsvm', LinearSVC(C=0.5, class_weight='balanced'), 'decision_function')]
json.dump(dict(n_train_users=len(tr_u), n_train_pred=int(tr_y.sum()), n_test_users=len(te_u), n_test_pred=int(te_y.sum()),
               n_koutli_unseen=len(kx), results=res), open('/kaggle/working/baselines.json', 'w'), indent=1)
json.dump(kname, open('/kaggle/working/koutli_unseen.json', 'w'))
