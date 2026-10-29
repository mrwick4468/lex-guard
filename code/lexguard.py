# LEX-GUARD: explainable grooming detection with guaranteed false-accusation control and element-level legal reasoning.
# One run = one LLM. FULL=True runs every experiment (primary model); FULL=False runs detection-only (model comparison).
#   E1 detection (PAN12 train candidates = calibration, PAN12 test candidates = evaluation), P(grooming) from first-token logprobs
#   E2 conformal three-way decision: flag / human review / clear, calibrated on PAN12 train only
#   E3 cross-corpus hard negatives: Koutli unseen predators vs the decoys of the same chats
#   E4 early warning: detection on message prefixes                                   (FULL)
#   E5 explanation faithfulness: remove / keep cited lines vs random vs TF-IDF lines   (FULL)
#   E6 legal RAG: statutory query rewrite + hybrid retrieval, RAG vs no-RAG citations  (FULL)
#   E7 offence-element reasoning + verifier, cross-jurisdiction coverage               (FULL)
import glob, json, os, re, subprocess, sys, time
from collections import defaultdict

MODEL = 'Qwen/Qwen3-8B'
FULL = True
SMOKE = False
RESUME = False  # True: reuse the LLM outputs of a previous full run (kernel source) for E1, E3, E4, E5; rerun E6, E7
N_TR, N_TE = (20, 20) if SMOKE else (300, 400)
OUT = '/kaggle/working'
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vllm'], capture_output=True)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchaudio'], capture_output=True)
import numpy as np, xml.etree.ElementTree as ET
from vllm import LLM, SamplingParams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, precision_score, recall_score

def find(pat):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pat}', recursive=True)); assert hits, pat; return hits[0]

# ---------------- data ----------------
def load(xml_path):
    convs, uc = {}, defaultdict(list)
    for _, c in ET.iterparse(xml_path):
        if c.tag != 'conversation': continue
        msgs = [(m.get('line'), m.findtext('author'), m.findtext('text') or '') for m in c.iter('message')]
        convs[c.get('id')] = msgs
        for a in {m[1] for m in msgs}: uc[a].append(c.get('id'))
        c.clear()
    return convs, uc
cv_tr, uc_tr = load(find('*training-corpus-2012-05-01.xml'))
cv_te, uc_te = load(find('*test-corpus-2012-05-17.xml'))
u_tr, u_te = sorted(uc_tr), sorted(uc_te)
P_tr = {l.strip() for l in open(find('*training-corpus-predators-2012-05-01.txt')) if l.strip()}
P_te = {l.strip() for l in open(find('*groundtruth-problem1.txt')) if l.strip()}
gt_lines = {'{}:{}'.format(*l.split()[:2]) for l in open(find('*groundtruth-problem2.txt')) if len(l.split()) >= 2}
S = np.load(find('scores_tfidf_linsvm.npy'), allow_pickle=True).item()
assert len(S['tr_oof']) == len(u_tr) and len(S['te']) == len(u_te)
s1_tr, s1_te = dict(zip(u_tr, S['tr_oof'])), dict(zip(u_te, S['te']))
cand_tr = sorted(u_tr, key=lambda u: -s1_tr[u])[:N_TR]
cand_te = sorted(u_te, key=lambda u: -s1_te[u])[:N_TE]
log('stage-1 recall@N train', sum(u in P_tr for u in cand_tr) / len(P_tr), 'test', sum(u in P_te for u in cand_te) / len(P_te))
legal = [json.loads(l) for l in open(find('legal_corpus.jsonl'))]
refmap = json.load(open(find('reference_map.json')))['behaviours']
ELEM = {k: v for k, v in json.load(open(find('elements.json'))).items() if not k.startswith('_')}

# ---------------- LLM ----------------
llm = LLM(MODEL, dtype='half', tensor_parallel_size=2, max_model_len=8192, gpu_memory_utilization=0.9, seed=0)
tok = llm.get_tokenizer()
log('model loaded', MODEL)
SYS = ('You assist child-protection researchers and law-enforcement analysts. You review chat logs from research corpora '
       '(PAN 2012, Perverted Justice) in which adult volunteers posed as children to identify offenders. Your work is analytical. '
       'Never reproduce explicit content; refer to chat lines only by their ID.')

def chat(user, prefix=''):
    msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user}]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:  # templates without a system role
        p = tok.apply_chat_template([{'role': 'user', 'content': SYS + '\n\n' + user}], tokenize=False, add_generation_prompt=True)
    return p + prefix

def run(prompts, max_new, prefix=''):
    """Greedy generation; returns (text including prefix, P(true) of the first generated token when prefix asks for a boolean)."""
    out = llm.generate([chat(p, prefix) for p in prompts], SamplingParams(temperature=0, max_tokens=max_new, logprobs=20), use_tqdm=False)
    res = []
    for o in out:
        g = o.outputs[0]; pt = pf = 0.0
        if g.logprobs:
            for lp in g.logprobs[0].values():
                t = (lp.decoded_token or '').strip().lower()
                if t and 'true'.startswith(t): pt += np.exp(lp.logprob)
                elif t and 'false'.startswith(t): pf += np.exp(lp.logprob)
        res.append((prefix + g.text, pt / (pt + pf) if pt + pf > 0 else np.nan))
    return res

def run_json(prompts, schema, max_new):
    """Schema-constrained decoding (vLLM structured outputs); returns parsed dicts."""
    try:
        from vllm.sampling_params import StructuredOutputsParams
        sp = SamplingParams(temperature=0, max_tokens=max_new, structured_outputs=StructuredOutputsParams(json=schema))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        sp = SamplingParams(temperature=0, max_tokens=max_new, guided_decoding=GuidedDecodingParams(json=schema))
    return [parse(o.outputs[0].text) or {} for o in llm.generate([chat(p) for p in prompts], sp, use_tqdm=False)]

def parse(s):
    m = re.search(r'\{.*\}', s, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except json.JSONDecodeError:
        try: return json.loads(m.group(0)[:m.group(0).rfind('}') + 1])
        except json.JSONDecodeError: return None
REFUSAL = re.compile(r"(i can(no|')t|i am unable|i'm unable|not able to (help|assist)|cannot (help|assist|comply))", re.I)

def clip(lines, budget=3000):
    out, n = [], 0
    for l in lines:
        n += len(tok.encode(l, add_special_tokens=False))
        if n > budget: break
        out.append(l)
    return out

def transcript(u, convs, uc, drop=frozenset(), keep=None, max_msgs=None):
    cs = sorted(uc[u], key=lambda c: -sum(a == u for _, a, _ in convs[c]))
    alias, lines = {u: 'TARGET'}, []
    for c in cs:
        for k, (ln, a, t) in enumerate(convs[c]):
            if max_msgs is not None and k >= max_msgs: break
            lid = f'{c}:{ln}'
            if lid in drop or (keep is not None and lid not in keep): continue
            alias.setdefault(a, f'OTHER{len(alias)}')
            lines.append(f'[{lid}] {alias[a]}: {t[:300]}')
        if max_msgs is not None: break  # early-warning prefixes use the main conversation only
    return clip(lines)

STAGES = 'friendship_forming, relationship_forming, risk_assessment, exclusivity, sexual, meeting_planning'
SPEC = ('Behaviour labels:\n' + '\n'.join(f'- {k}: {v["desc"]}' for k, v in refmap.items()) +
        f"\nGrooming stages (O'Connell 2003): {STAGES}.\n")
def detect_prompt(lines, who='TARGET'):
    return (f'Decide whether {who} is grooming a minor for sexual purposes.\n{SPEC}\nTranscript:\n' + '\n'.join(lines) +
            '\n\nReturn JSON only, starting with the decision: {"grooming": true|false, "role": "predator|child_or_decoy|bystander", '
            '"stages": [...], "behaviours": [...], "evidence_lines": ["ID", ...] (at most 10 IDs exactly as shown without brackets, '
            f'written by {who}), "rationale": "at most 40 words, no explicit content"}}')
DPFX = '{"grooming": '

def detect(users, convs, uc, **kw):
    outs = run([detect_prompt(transcript(u, convs, uc, **kw)) for u in users], 500, DPFX)
    rows = []
    for u, (txt, p) in zip(users, outs):
        j = parse(txt) or {}
        rows.append(dict(user=u, p=float(np.nan_to_num(p, nan=0.5)), grooming=j.get('grooming'), role=j.get('role'), stages=j.get('stages') or [],
                         behaviours=[b for b in (j.get('behaviours') or []) if b in refmap],
                         evidence=[str(x).strip('[] ') for x in (j.get('evidence_lines') or [])][:10],
                         rationale=j.get('rationale'), parsed=bool(j), refusal=bool(REFUSAL.search(txt)), raw=txt[:1500]))
    return rows

def prf(y, pred):
    return dict(P=precision_score(y, pred, zero_division=0), R=recall_score(y, pred, zero_division=0),
                F05=fbeta_score(y, pred, beta=0.5, zero_division=0), F1=fbeta_score(y, pred, beta=1, zero_division=0), n_flag=int(sum(pred)))
R = {'model': MODEL, 'N_tr': N_TR, 'N_te': N_TE,
     'stage1_recall_tr': sum(u in P_tr for u in cand_tr) / len(P_tr), 'stage1_recall_te': sum(u in P_te for u in cand_te) / len(P_te)}
def save():
    json.dump(R, open(f'{OUT}/metrics.json', 'w'), indent=1, default=float)

# ---------------- E1 detection ----------------
if RESUME:
    Dj = json.load(open(find('detection.json'))); D_tr, D_te = Dj['train'], Dj['test']
    assert [r['user'] for r in D_tr] == cand_tr and [r['user'] for r in D_te] == cand_te
    PREV = json.load(open(find('metrics.json')))
else:
    D_tr = detect(cand_tr, cv_tr, uc_tr); D_te = detect(cand_te, cv_te, uc_te)
json.dump(dict(train=D_tr, test=D_te), open(f'{OUT}/detection.json', 'w'))
for name, D, PP in (('train', D_tr, P_tr), ('test', D_te, P_te)):
    y = np.array([r['user'] in PP for r in D]); p = np.nan_to_num(np.array([r['p'] for r in D]), nan=0.5)
    R[f'E1_{name}'] = dict(parse_rate=np.mean([r['parsed'] for r in D]), refusal_rate=np.mean([r['refusal'] for r in D]),
                           cand_ROCAUC=roc_auc_score(y, p), cand_AUPRC=average_precision_score(y, p))
# Population-level PAN12 metrics on the full test set (non-candidates are cleared by stage 1).
yte = np.array([u in P_te for u in u_te]); pte = {r['user']: r['p'] for r in D_te}
for thr in (0.5, 0.9):
    R[f'E1_test_full_p{thr}'] = prf(yte, np.array([pte.get(u, 0) >= thr for u in u_te]))
# Stacked score (stage-1 + LLM), fitted on train candidates only.
def feats(D, s1): return np.c_[[s1[r['user']] for r in D], np.nan_to_num([r['p'] for r in D], nan=0.5)]
stk = LogisticRegression(C=1.0).fit(feats(D_tr, s1_tr), [r['user'] in P_tr for r in D_tr])
for r, q in zip(D_te, stk.predict_proba(feats(D_te, s1_te))[:, 1]): r['q'] = float(q)
for r, q in zip(D_tr, stk.predict_proba(feats(D_tr, s1_tr))[:, 1]): r['q'] = float(q)
qte = {r['user']: r['q'] for r in D_te}
R['E1_test_full_stacked_q0.5'] = prf(yte, np.array([qte.get(u, 0) >= 0.5 for u in u_te]))
R['E1_test_stacked_cand_ROCAUC'] = roc_auc_score([r['user'] in P_te for r in D_te], [r['q'] for r in D_te])
log('E1', json.dumps({k: v for k, v in R.items() if k.startswith('E1')}, default=float)); save()

# ---------------- E2 conformal three-way decision ----------------
def thresholds(cal, PP, n_missed, alpha, beta, key):
    """Split-conformal thresholds. Predators missed by stage 1 enter the positive calibration set with score 0."""
    neg = np.sort([r[key] for r in cal if r['user'] not in PP])
    pos = np.sort([r[key] for r in cal if r['user'] in PP] + [0.0] * n_missed)
    n0, n1 = len(neg), len(pos)
    k = int(np.ceil((n0 + 1) * (1 - alpha)))           # P(score_innocent >= t_flag) <= alpha
    t_flag = neg[k - 1] + 1e-9 if k <= n0 else np.inf
    m = int(np.floor((n1 + 1) * beta))                  # P(score_predator < t_clear) <= beta
    t_clear = pos[m - 1] if m >= 1 else 0.0
    return t_flag, min(t_clear, t_flag)
def decide(ev, PP, n_missed, t_flag, t_clear, key):
    """Three-way decisions on candidates ev; stage-1 misses are cleared, non-candidate innocents are cleared."""
    dec = {r['user']: 'flag' if r[key] >= t_flag else ('clear' if r[key] < t_clear else 'review') for r in ev}
    inn = [dec[r['user']] for r in ev if r['user'] not in PP]
    pred = [dec[r['user']] for r in ev if r['user'] in PP] + ['clear'] * n_missed
    flags = [r['user'] in PP for r in ev if dec[r['user']] == 'flag']
    return dict(false_flag_rate_innocent_candidates=float(np.mean([x == 'flag' for x in inn])),
                miss_rate_predators=float(np.mean([x == 'clear' for x in pred])),
                review_rate_candidates=float(np.mean([x == 'review' for x in dec.values()])),
                n_flag=len(flags), flag_precision=float(np.mean(flags)) if flags else np.nan,
                n_review=sum(x == 'review' for x in dec.values()), predators_flagged=float(np.mean([x == 'flag' for x in pred])))
miss_tr = len(P_tr) - sum(r['user'] in P_tr for r in D_tr); miss_te = len(P_te) - sum(r['user'] in P_te for r in D_te)
AB = [(a, b) for a in (0.01, 0.05, 0.10) for b in (0.05, 0.10, 0.20)]
# (i) calibrated on PAN12 train candidates, evaluated on test candidates (tests robustness to train->test shift)
R['E2'] = []
for key in ('q', 'p'):
    for a, b in AB:
        tf, tc = thresholds(D_tr, P_tr, miss_tr, a, b, key)
        R['E2'].append(dict(alpha=a, beta=b, t_flag=tf, t_clear=tc, score=key) | decide(D_te, P_te, miss_te, tf, tc, key))
# (ii) exchangeable split conformal: 50 random halves of the test candidates, calibrate on one half, evaluate on the other
rng2 = np.random.RandomState(0); half = len(D_te) // 2
splits = [rng2.permutation(len(D_te)) for _ in range(50)]
R['E2_split'] = []
for key in ('q', 'p'):
    for a, b in AB:
        runs = []
        for idx in splits:
            cal, ev = [D_te[i] for i in idx[:half]], [D_te[i] for i in idx[half:]]
            tf, tc = thresholds(cal, P_te, miss_te // 2, a, b, key)
            runs.append(decide(ev, P_te, miss_te - miss_te // 2, tf, tc, key))
        agg = {m: float(np.nanmean([x[m] for x in runs])) for m in runs[0]}
        R['E2_split'].append(dict(alpha=a, beta=b, score=key, n_splits=len(runs), **agg,
                                  ffr_violation_rate=float(np.mean([x['false_flag_rate_innocent_candidates'] > a for x in runs])),
                                  miss_violation_rate=float(np.mean([x['miss_rate_predators'] > b for x in runs]))))
log('E2', json.dumps([x for x in R['E2_split'] if x['score'] == 'q' and x['beta'] == 0.10], default=float)); save()
# Operating point for E3, E4, E6, E7: stacked score q, train-calibrated, alpha=0.05, beta=0.10.
T_FLAG, T_CLEAR = thresholds(D_tr, P_tr, miss_tr, 0.05, 0.10, 'q')
T_FLAG = T_FLAG if np.isfinite(T_FLAG) else 0.5  # too few calibration negatives (smoke runs)
R['T_FLAG'], R['T_CLEAR'] = T_FLAG, T_CLEAR
def q_of(s1, p): return stk.predict_proba(np.c_[s1, np.nan_to_num(p, nan=0.5)])[:, 1]

# ---------------- E3 Koutli hard negatives ----------------
unseen = json.load(open(find('koutli_unseen.json')))
pf = {os.path.basename(f): f for f in glob.glob('/kaggle/input/**/predators/**/*.txt', recursive=True)}
of = {os.path.basename(f): f for f in glob.glob('/kaggle/input/**/officers/**/*.txt', recursive=True)}
pairs = [(n, n.replace('predator_', 'officer_', 1)) for n in unseen if n.replace('predator_', 'officer_', 1) in of]
if SMOKE: pairs = pairs[:10]
def side(f): return clip([f'[L{i}] SPEAKER: {l.strip()[:300]}' for i, l in enumerate(open(f, errors='ignore')) if l.strip()], 2500)
who = "SPEAKER (only this speaker's side of a one-to-one chat is shown)"
if RESUME:
    kj = json.load(open(find('koutli.json'))); assert [x['name'] for x in kj] == [a for a, _ in pairs] + [b for _, b in pairs]
    ko = [('', x['p']) for x in kj]
else:
    ko = run([detect_prompt(side(pf[a]), who) for a, _ in pairs] + [detect_prompt(side(of[b]), who) for _, b in pairs], 300, DPFX)
kp = np.nan_to_num([p for _, p in ko], nan=0.5); ky = np.r_[np.ones(len(pairs)), np.zeros(len(pairs))]
# stage-1 scores of the same files (baselines kernel: k = all unseen predators, o = decoys of the paired ones, same order)
inpair = [n.replace('predator_', 'officer_', 1) in of for n in unseen]
ks1 = np.r_[np.asarray(S['k'])[inpair], np.asarray(S['o'])]
kq = q_of(ks1, kp) if len(ks1) == len(kp) else np.full(len(kp), np.nan)  # smoke runs truncate pairs
R['E3'] = dict(pairs=len(pairs), ROCAUC=roc_auc_score(ky, kp), AUPRC=average_precision_score(ky, kp),
               at_p05=prf(ky, kp >= 0.5) | {'decoy_FPR': float((kp[ky == 0] >= 0.5).mean())},
               at_conformal=prf(ky, kp >= T_FLAG) | {'decoy_FPR': float((kp[ky == 0] >= T_FLAG).mean()), 't': T_FLAG},
               stage1_ROCAUC=roc_auc_score(ky, ks1) if len(ks1) == len(kp) else np.nan,
               stacked_ROCAUC=roc_auc_score(ky, kq) if np.isfinite(kq).all() else np.nan,
               at_stacked=(prf(ky, kq >= T_FLAG) | {'decoy_FPR': float((kq[ky == 0] >= T_FLAG).mean()), 't': T_FLAG}) if np.isfinite(kq).all() else {},
               refusal_rate=float(np.mean([bool(REFUSAL.search(t)) for t, _ in ko])))
json.dump([dict(name=n, label=int(l), p=float(p)) for n, l, p in zip([a for a, _ in pairs] + [b for _, b in pairs], ky, kp)],
          open(f'{OUT}/koutli.json', 'w'))
json.dump(dict(q=kq.tolist(), s1=ks1.tolist()), open(f'{OUT}/koutli_stacked.json', 'w'))
log('E3', json.dumps(R['E3'], default=float)); save()
if not FULL:
    log('done (detection-only)'); sys.exit(0)

# ---------------- E4 early warning ----------------
ew_pos = [r['user'] for r in D_te if r['user'] in P_te]
ew_neg = [r['user'] for r in sorted(D_te, key=lambda r: -s1_te[r['user']]) if r['user'] not in P_te][:len(ew_pos)]
CUTS = [5, 10, 20, 40, 80, 160]
users = ew_pos + ew_neg
if RESUME:
    ewj = json.load(open(find('early_warning.json'))); ew = {u: {int(k): v for k, v in ewj[u].items()} for u in users}
else:
    ew = defaultdict(dict)
    for k in CUTS:
        for u, (_, p) in zip(users, run([detect_prompt(transcript(u, cv_te, uc_te, max_msgs=k)) for u in users], 1, DPFX)):
            ew[u][k] = float(p)
# Stage-1 model refit exactly as in the baselines kernel (word 1-2 + char_wb 2-5 TF-IDF, LinearSVC C=0.5), scored on each prefix.
from scipy.sparse import hstack
docs1 = defaultdict(list)
for c, msgs in cv_tr.items():
    for _, a_, t in msgs: docs1[a_].append(t)
tr_x = [' \n '.join(docs1[u]) for u in u_tr]; del docs1
w1 = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=300000, ngram_range=(1, 2))
c1 = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=300000, analyzer='char_wb', ngram_range=(2, 5))
svm1 = LinearSVC(C=0.5, class_weight='balanced').fit(hstack([w1.fit_transform(tr_x), c1.fit_transform(tr_x)]).tocsr(), [u in P_tr for u in u_tr])
del tr_x
def s1_prefix(u, k):
    c = max(uc_te[u], key=lambda c: sum(a_ == u for _, a_, _ in cv_te[c]))  # main conversation, as in transcript()
    return ' \n '.join(t for _, a_, t in cv_te[c][:k] if a_ == u)
chk = [s1_prefix(u, 10**9) for u in users[:50]]  # sanity: refit model scores full-history docs on the baselines' scale
log('stage-1 refit corr with baselines on 50 users (main conversation only)', np.corrcoef(svm1.decision_function(hstack([w1.transform(chk), c1.transform(chk)]).tocsr()), [s1_te[u] for u in users[:50]])[0, 1])
ewq = {u: {} for u in users}
for k in CUTS:
    xs = [s1_prefix(u, k) for u in users]
    s1k = svm1.decision_function(hstack([w1.transform(xs), c1.transform(xs)]).tocsr())
    for u, qq in zip(users, q_of(s1k, [ew[u][k] for u in users])): ewq[u][k] = float(qq)
yy = np.array([1] * len(ew_pos) + [0] * len(ew_neg))
R['E4'] = {k: dict(ROCAUC_p=roc_auc_score(yy, np.nan_to_num([ew[u][k] for u in users], nan=0.5)),
                   ROCAUC_q=roc_auc_score(yy, [ewq[u][k] for u in users]),
                   TPR=float(np.mean([ewq[u][k] >= T_FLAG for u in ew_pos])), FPR=float(np.mean([ewq[u][k] >= T_FLAG for u in ew_neg])))
           for k in CUTS}
first = [next((k for k in CUTS if ewq[u][k] >= T_FLAG), None) for u in ew_pos]
R['E4_first_alert_msgs'] = dict(detected=float(np.mean([f is not None for f in first])),
                                median=float(np.median([f for f in first if f is not None])) if any(first) else np.nan)
json.dump(ewq, open(f'{OUT}/early_warning_stacked.json', 'w'))
json.dump(ew, open(f'{OUT}/early_warning.json', 'w'))
log('E4', json.dumps(R['E4'], default=float)); save()

# ---------------- E5 faithfulness ----------------
if RESUME:
    R['E5'], R['E5_evidence_vs_gt'] = PREV['E5'], PREV['E5_evidence_vs_gt']
else:
    # Line-level TF-IDF SVM (word features) as a model-agnostic attribution baseline.
    docs_tr = defaultdict(list)
    for c, msgs in cv_tr.items():
        for _, a, t in msgs: docs_tr[a].append(t)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200000, ngram_range=(1, 2))
    svm = LinearSVC(C=0.5, class_weight='balanced').fit(vec.fit_transform([' \n '.join(docs_tr[u]) for u in u_tr]), [u in P_tr for u in u_tr])
    rng = np.random.RandomState(0)
    tp = [r for r in D_te if r['user'] in P_te and r['p'] >= 0.5 and r['evidence']]
    variants = defaultdict(list)
    for r in tp:
        lines = transcript(r['user'], cv_te, uc_te)
        tgt = [l for l in lines if '] TARGET:' in l]
        ids = [l[1:].split(']')[0] for l in tgt]
        cited = [e for e in r['evidence'] if e in set(ids)]
        if not cited: continue
        k = len(cited)
        sc = svm.decision_function(vec.transform([l.split('TARGET: ', 1)[1] for l in tgt]))
        variants['remove_cited'].append(dict(drop=frozenset(cited)))
        variants['remove_random'].append(dict(drop=frozenset(rng.choice(ids, k, replace=False))))
        variants['remove_tfidf'].append(dict(drop=frozenset(np.array(ids)[np.argsort(-sc)[:k]])))
        variants['keep_cited'].append(dict(keep=frozenset(cited)))
        variants['keep_random'].append(dict(keep=frozenset(rng.choice(ids, k, replace=False))))
        variants['_users'].append(r)
    users_f = variants.pop('_users')
    fa = {}
    for v, kws in variants.items():
        ps = [p for _, p in run([detect_prompt(transcript(r['user'], cv_te, uc_te, **kw)) for r, kw in zip(users_f, kws)], 1, DPFX)]
        fa[v] = np.nan_to_num(ps, nan=0.5)
    base = np.array([r['p'] for r in users_f])
    R['E5'] = dict(n=len(users_f), p_full=float(base.mean()),
                   **{f'drop_{v}': float((base - fa[v]).mean()) for v in fa},
                   **{f'flip_{v}': float((fa[v] < 0.5).mean()) for v in fa})
    ev_hit = [len(set(r['evidence']) & gt_lines) / len(r['evidence']) for r in users_f]
    shown = [len({l[1:].split(']')[0] for l in transcript(r['user'], cv_te, uc_te)} & gt_lines) for r in users_f]
    R['E5_evidence_vs_gt'] = dict(precision=float(np.mean(ev_hit)), users_with_gt_in_context=int(sum(s > 0 for s in shown)))
    log('E5', json.dumps(R['E5'], default=float), json.dumps(R['E5_evidence_vs_gt'])); save()

# ---------------- E6 legal RAG ----------------
from transformers import AutoTokenizer, AutoModel
import torch
etok = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5'); emod = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5').eval()
def embed(texts, q=False):
    if q: texts = ['Represent this sentence for searching relevant passages: ' + t for t in texts]
    e = etok(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad(): v = emod(**e).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(v, dim=-1).numpy()
pas, owner = [], []
for i, d in enumerate(legal):
    buf = ''
    for part in re.split(r'(?<=[.;:])\s+(?=\(?[a-z0-9]{1,3}\)|\d+\.)', d['text']):
        if len(buf) + len(part) > 700 and buf: pas.append(buf); owner.append(i); buf = ''
        buf += ' ' + part
    if buf: pas.append(buf); owner.append(i)
ptext = [f'{legal[o]["instrument"]} {legal[o]["unit"]} {legal[o]["title"]}. {p}' for o, p in zip(owner, pas)]
owner = np.array(owner); Lm = np.concatenate([embed(ptext[i:i + 32]) for i in range(0, len(ptext), 32)])
tfv = TfidfVectorizer(sublinear_tf=True, stop_words='english').fit(ptext); Tm = tfv.transform(ptext)
JUR = {'EU': 'European Union', 'US': 'United States', 'VN': 'Vietnam'}
def retrieve(queries, k=2, hybrid=True):
    got = {}
    for qt in queries:
        d = Lm @ embed([qt], q=True)[0]; s = (Tm @ tfv.transform([qt]).T).toarray().ravel()
        score = d / d.max() + s / max(s.max(), 1e-9) if hybrid else d
        best = {}
        for pi in np.argsort(-score): best.setdefault(owner[pi], score[pi])
        for j in JUR.values():
            for i in [i for i in sorted(best, key=lambda i: -best[i]) if legal[i]['jurisdiction'] == j][:k]: got[legal[i]['id']] = legal[i]
    return list(got.values())
flag = [r for r in D_te if r['q'] >= T_FLAG and r['behaviours']]
rw = [parse(t) or {} for t, _ in run([
    'Rewrite each observed behaviour below as the elements of a possible criminal offence, in neutral statutory language '
    '(who, what act, towards whom, by what means, for what purpose). Do not name any statute, article or country.\n'
    + '\n'.join(f'- {b}: {refmap[b]["desc"]}' for b in r['behaviours']) + f'\nContext: {r["rationale"]}\n'
    'Return JSON only: {"queries": ["one sentence per behaviour", ...]}' for r in flag], 300)]
plainq = [['Online conduct by an adult towards a child: ' + refmap[b]['desc'] for b in r['behaviours']] for r in flag]
queries = [[str(q) for q in (w.get('queries') or [])] or pq for w, pq in zip(rw, plainq)]
ret = {'rewrite_hybrid': [retrieve(q) for q in queries], 'rewrite_dense': [retrieve(q, hybrid=False) for q in queries],
       'plain_hybrid': [retrieve(q) for q in plainq], 'plain_dense': [retrieve(q, hybrid=False) for q in plainq]}
def facts(r): return (f'Behaviours observed: {", ".join(r["behaviours"])}. Stages: {", ".join(r["stages"])}. '
                      f'Analyst rationale: {r["rationale"]}. The other party presented as a child aged 13.')
def legal_prompt(r, prov):
    if prov:
        head = 'Provisions:\n' + '\n\n'.join(f'[{d["id"]}] {d["instrument"]} {d["unit"]} - {d["title"]}: {d["text"][:1200]}' for d in prov) + '\n\n'
        rule = 'Cite ONLY provision IDs from the list above, e.g. US-18-2422.'
    else:
        head, rule = '', 'Cite provisions as "<instrument> <article or section>", e.g. "18 U.S.C. 2422" or "Directive 2011/93/EU Article 6".'
    return (head + f'Facts:\n{facts(r)}\n\nFor each jurisdiction (EU = European Union, US = United States federal law, VN = Vietnam Criminal Code) '
            f'identify the provisions this conduct may engage and explain element by element. If none applies directly, say so. {rule} '
            'This is decision support for analysts, not legal advice.\nReturn JSON only: {"EU": {"provisions": [...], "reasoning": "..."}, "US": {...}, "VN": {...}}')
rag = [parse(t) for t, _ in run([legal_prompt(r, p) for r, p in zip(flag, ret['rewrite_hybrid'])], 700)]
norag = [parse(t) for t, _ in run([legal_prompt(r, None) for r in flag], 700)]
ID_RE = {'EU': r'2011/93[^0-9]*?Art(?:icle|\.)?\s*(\d+)', 'US': r'(?:U\.?S\.?C\.?|USC)[^0-9]*(\d{4}[A-Z]?)', 'VN': r'Art(?:icle|\.)?\s*(\d{2,3})'}
PFX = {'EU': 'EU-2011/93-', 'US': 'US-18-', 'VN': 'VN-CC2015-'}
ids = {d['id'] for d in legal}
def cited(j, mode):
    out = set()
    for jur in JUR:
        blk = (j or {}).get(jur)
        for c in (blk.get('provisions') or []) if isinstance(blk, dict) else []:
            c = str(c).strip('[] ')
            if mode == 'rag': out.add(c)
            else:
                m = re.search(ID_RE[jur], c); out.add(PFX[jur] + m.group(1) if m else f'UNMAPPED:{jur}:{c}')
    return out
agg = defaultdict(list)
for i, r in enumerate(flag):
    ref = {x for b in r['behaviours'] for x in refmap[b]['provisions']}
    for name, rr in ret.items():
        if ref: agg[f'retrieval_recall_{name}'].append(len(ref & {d['id'] for d in rr[i]}) / len(ref))
    got = {d['id'] for d in ret['rewrite_hybrid'][i]}; ca, cb = cited(rag[i], 'rag'), cited(norag[i], 'norag')
    agg['n_retrieved'].append(len(got)); agg['rag_parse'].append(rag[i] is not None); agg['norag_parse'].append(norag[i] is not None)
    agg['rag_cite_outside_context'].append(len(ca - got) / max(len(ca), 1))
    agg['rag_cite_nonexistent'].append(len(ca - ids) / max(len(ca), 1))
    agg['norag_cite_outside_corpus'].append(len(cb - ids) / max(len(cb), 1))
    if ref:
        for nm, cc in (('rag', ca), ('norag', cb)):
            agg[f'{nm}_cite_precision'].append(len(cc & ref) / max(len(cc), 1)); agg[f'{nm}_cite_recall'].append(len(cc & ref) / len(ref))
R['E6'] = {k: float(np.mean(v)) for k, v in agg.items()} | {'n': len(flag)}
json.dump([dict(user=r['user'], label=r['user'] in P_te, behaviours=r['behaviours'], queries=q, retrieved=[d['id'] for d in p],
                rag=a, norag=b) for r, q, p, a, b in zip(flag, queries, ret['rewrite_hybrid'], rag, norag)],
          open(f'{OUT}/legal_rag.json', 'w'), default=list)
log('E6', json.dumps(R['E6'])); save()

# ---------------- E7 offence elements + verifier ----------------
def elem_prompt(lines, off):
    e = ELEM[off]
    return (f'Offence: {e["name"]} ({e["provision"]}).\nAssess each element against the transcript. Judge age on the age the other party '
            'presented (attempt doctrine); an age the other party states in the chat counts as the presented age, and online chat counts as '
            'the internet / a facility of interstate commerce / information and communication technology. Mark "met" only if specific lines '
            'show it, and cite at most 3 line IDs per element.\n'
            + '\n'.join(f'- {k}: {d}' for k, d in e['elements']) + '\n\nTranscript:\n' + '\n'.join(lines) +
            '\n\nReturn JSON with one entry per element key: {"<element key>": {"status": "met|not_met|unknown", "evidence": ["ID", ...]}}')
def elem_schema(off):
    item = {'type': 'object', 'properties': {'status': {'type': 'string', 'enum': ['met', 'not_met', 'unknown']},
            'evidence': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 3}}, 'required': ['status', 'evidence']}
    keys = [k for k, _ in ELEM[off]['elements']]
    return {'type': 'object', 'properties': {k: item for k in keys}, 'required': keys, 'additionalProperties': False}
assess = [(r, off) for r in flag for off in ELEM]
el = [None] * len(assess)
for off in ELEM:  # one schema per offence
    idx = [i for i, (_, o) in enumerate(assess) if o == off]
    for i, j in zip(idx, run_json([elem_prompt(transcript(assess[i][0]['user'], cv_te, uc_te), off) for i in idx], elem_schema(off), 700)):
        el[i] = j
R['E7_parse_rate'] = float(np.mean([bool(j) for j in el]))
# Verifier: each "met" element is re-judged from its cited lines alone.
line_text = {}
for c, msgs in cv_te.items():
    for ln, a, t in msgs: line_text[f'{c}:{ln}'] = (a, t[:300])
vq, vk = [], []
for i, ((r, off), j) in enumerate(zip(assess, el)):
    for key, desc in ELEM[off]['elements']:
        v = j.get(key) if isinstance(j.get(key), dict) else {}
        evid = [str(x).strip('[] ') for x in (v.get('evidence') or []) if str(x).strip('[] ') in line_text][:6]
        if v.get('status') == 'met' and evid:
            txt = '\n'.join(f'[{x}] {"TARGET" if line_text[x][0] == r["user"] else "OTHER"}: {line_text[x][1]}' for x in evid)
            vq.append(f'Element: {desc}\nCited lines:\n{txt}\n\nConventions: age is the age the other party presented (attempt doctrine); '
                      'an age the other party states counts as the presented age; online chat counts as the internet / a facility of interstate '
                      'commerce / information and communication technology.\nDo these lines, on their own, support that the element is met by TARGET? '
                      'Return JSON only: {"supported": true|false}')
            vk.append((i, key))
ver = {k: p for k, (_, p) in zip(vk, run(vq, 8, '{"supported": '))}
# Context elements (the actor is an adult; the medium is online) are established by the case record, not by chat lines.
CONTEXT = {'adult', 'adult_18', 'ict', 'interstate_facility'}
rows7 = []
for i, ((r, off), j) in enumerate(zip(assess, el)):
    st, sup = {}, {}
    for key, _ in ELEM[off]['elements']:
        st[key] = (j.get(key) or {}).get('status') if isinstance(j.get(key), dict) else None
        sup[key] = float(ver[(i, key)]) if (i, key) in ver else None  # None: not met, or met without a valid line ID
    met = all(x == 'met' for x in st.values())
    rows7.append(dict(user=r['user'], label=r['user'] in P_te, offence=off, jurisdiction=off.split('-')[0], status=st, support=sup, met=met,
                      verified_strict=met and all((sup[k] or 0) >= 0.5 for k in st),
                      verified=met and all(k in CONTEXT or (sup[k] or 0) >= 0.5 for k in st)))
json.dump(rows7, open(f'{OUT}/elements_assessed.json', 'w'))
pred_users = {r['user'] for r in flag if r['user'] in P_te}
met_el = [(x['offence'], k, x['support'][k]) for x in rows7 if x['label'] for k, v in x['status'].items() if v == 'met']
R['E7'] = dict(n_users=len(flag), n_true_predators=len(pred_users), verifier_calls=len(vq),
               verifier_support_rate=float(np.mean([p >= 0.5 for p in ver.values()])) if ver else np.nan,
               met_without_valid_evidence=float(np.mean([sp is None for _, _, sp in met_el])) if met_el else np.nan,
               element_support={f'{o}:{k}': float(np.mean([(sp or 0) >= 0.5 for oo, kk, sp in met_el if (oo, kk) == (o, k)]))
                                for o in ELEM for k, _ in ELEM[o]['elements'] if any((oo, kk) == (o, k) for oo, kk, _ in met_el)},
               offence_rate_met={o: float(np.mean([x['met'] for x in rows7 if x['offence'] == o and x['label']])) for o in ELEM},
               offence_rate_verified={o: float(np.mean([x['verified'] for x in rows7 if x['offence'] == o and x['label']])) for o in ELEM},
               offence_rate_verified_strict={o: float(np.mean([x['verified_strict'] for x in rows7 if x['offence'] == o and x['label']])) for o in ELEM},
               innocent_offence_rate_verified={o: float(np.mean([x['verified'] for x in rows7 if x['offence'] == o and not x['label']]))
                                               for o in ELEM if any(not x['label'] for x in rows7)},
               jurisdiction_any_verified={j: float(np.mean([any(x['verified'] for x in rows7 if x['user'] == u and x['jurisdiction'] == j)
                                                            for u in pred_users])) for j in JUR})
log('E7', json.dumps(R['E7'], default=float)); save()
log('done')
