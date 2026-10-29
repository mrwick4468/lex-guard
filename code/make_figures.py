"""Builds the manuscript figures, tables and numbers.json from the Kaggle result files.
  ../kaggle/eda/out              corpus statistics (kernel imtihanswe7/pan12-koutli-eda)
  ../kaggle/baselines/out        TF-IDF baselines (kernel imtihanswe7/grooming-baselines)
  ../kaggle/lexguard_v3/out      LEX-GUARD with Qwen3-8B, all experiments (kernel imtihanswe7/lexguard-qwen3-8b-v3)
  ../kaggle/cmp_*/out            detection-only runs of the comparison LLMs
  ../legal                       legal corpus, behaviour-to-provision map, offence elements
Run:  ../.venv/bin/python make_figures.py"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
R = lambda *p: os.path.join(ROOT, *p)
FIG, TAB = os.path.join(HERE, 'figures'), os.path.join(HERE, 'tables')
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e3', 'grid.linewidth': 0.6, 'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42})
OK = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442', '#000000']
INK, GREY = '#1d1d1b', '#8c8c88'
GREEN, AMBER, RED = '#2e9e6b', '#e8a33d', '#d1495b'
NUM = {}


def save(fig, name):
    fig.savefig(os.path.join(FIG, f'{name}.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(FIG, f'{name}.png'), dpi=300, bbox_inches='tight'); plt.close(fig)
def tag(ax, s): ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=9.5, ha='right', va='bottom')
def f3(v): return '--' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.3f}'
def pct(v, d=1): return f'{100 * v:.{d}f}'
def f05(p, r): return 1.25 * p * r / (0.25 * p + r) if p + r else 0.0


# ------------------------------------------------------------------ load
eda = json.load(open(R('kaggle/eda/out/eda.json')))
base = {r['model']: r for r in json.load(open(R('kaggle/baselines/out/baselines.json')))['results']}
MODELS = [('Qwen3-8B', 'kaggle/lexguard_v3/out'), ('Qwen2.5-7B', 'kaggle/cmp_qwen25-7b/out'), ('Granite-3.3-8B', 'kaggle/cmp_granite33-8b/out'),
          ('Mistral-7B-v0.3', 'kaggle/cmp_mistral-7b/out'), ('Phi-4-mini', 'kaggle/cmp_phi4-mini/out')]
M = {n: json.load(open(R(p, 'metrics.json'))) for n, p in MODELS}
Q = M['Qwen3-8B']
kou = json.load(open(R('kaggle/lexguard_v3/out/koutli.json'))); kst = json.load(open(R('kaggle/lexguard_v3/out/koutli_stacked.json')))
el = json.load(open(R('kaggle/lexguard_v3/out/elements_assessed.json')))
ELEM = {k: v for k, v in json.load(open(R('legal/elements.json'))).items() if not k.startswith('_')}
refmap = json.load(open(R('legal/reference_map.json')))['behaviours']
corpus = [json.loads(l) for l in open(R('legal/legal_corpus.jsonl'))]
def split(m, a, b=0.10, key='q'): return next(x for x in m['E2_split'] if x['score'] == key and x['alpha'] == a and x['beta'] == b)
def trcal(m, a, b=0.10, key='q'): return next(x for x in m['E2'] if x['score'] == key and x['alpha'] == a and x['beta'] == b)

# ------------------------------------------------------------------ numbers
NUM['eda'] = eda
NUM['n_train_users'], NUM['n_test_users'] = 97689, 218702
NUM['corpus_units'] = len(corpus); NUM['corpus_by_jur'] = {j: sum(d['jurisdiction'] == j for d in corpus) for j in ('European Union', 'United States', 'Vietnam')}
NUM['baselines'] = {k: {kk: v[kk] for kk in ('A_precision', 'A_recall', 'A_F05', 'A_ROCAUC', 'Chard_recall', 'Chard_decoy_FPR', 'Chard_F05', 'Chard_ROCAUC')} for k, v in base.items()}
op = trcal(Q, 0.05)
NUM['operating_point'] = dict(T_FLAG=Q['T_FLAG'], T_CLEAR=Q['T_CLEAR'], flag_P=op['flag_precision'], flag_R=op['predators_flagged'], flag_F05=f05(op['flag_precision'], op['predators_flagged']),
                              n_flag=op['n_flag'], review_rate=op['review_rate_candidates'], miss=op['miss_rate_predators'])
NUM['split'] = {n: {a: split(M[n], a) for a in (0.01, 0.05, 0.10)} for n in M}
NUM['E3'] = Q['E3']; NUM['E4'] = Q['E4']; NUM['E5'] = Q['E5']; NUM['E5_gt'] = Q['E5_evidence_vs_gt']; NUM['E6'] = Q['E6']; NUM['E7'] = {k: v for k, v in Q['E7'].items()}
pred_rows = [x for x in el if x['label']]
NUM['E7_users'] = len({x['user'] for x in el}); NUM['E7_pred_users'] = len({x['user'] for x in pred_rows})
claims = [(x['offence'], k, x['support'][k]) for x in pred_rows for k, v in x['status'].items() if v == 'met']
NUM['E7_met_claims'] = len(claims); NUM['E7_met_claims_supported'] = sum((s or 0) >= 0.5 for _, _, s in claims)
NUM['E7_claims_no_evidence'] = sum(s is None for _, _, s in claims)
# offences where every evidential (non-context) element is supported
CONTEXT = {'adult', 'adult_18', 'ict', 'interstate_facility'}
NUM['E7_partial_offence'] = {o: float(np.mean([any((x['support'][k] or 0) >= 0.5 for k in x['status'] if k not in CONTEXT) for x in pred_rows if x['offence'] == o])) for o in ELEM}

# ------------------------------------------------------------------ Figure 1: architecture
def chat_stack(ax, x, y, w, h, n=6, hl=()):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc='white', ec=GREY, lw=0.9))
    for i in range(n):
        left = i % 2 == 0; bw = w * (0.55 + 0.15 * ((i * 7) % 3) / 2)
        by = y + h - 0.09 - (i + 1) * (h - 0.12) / n
        bx = x + 0.06 if left else x + w - 0.06 - bw
        fc = '#fde2e4' if i in hl else ('#e8eef6' if left else '#eef6ea')
        ec = RED if i in hl else 'none'
        ax.add_patch(FancyBboxPatch((bx, by), bw, (h - 0.12) / n * 0.7, boxstyle='round,pad=0.01,rounding_size=0.04', fc=fc, ec=ec, lw=0.8))
def doc(ax, x, y, w, h, col, label):
    ax.add_patch(Polygon([[x, y], [x + w, y], [x + w, y + h - 0.12], [x + w - 0.12, y + h], [x, y + h]], closed=True, fc='white', ec=col, lw=1.1))
    for i in range(4): ax.plot([x + 0.07, x + w - 0.08], [y + h - 0.3 - i * 0.14] * 2, color='#cfcfca', lw=0.8)
    ax.text(x + w / 2, y + 0.1, label, ha='center', va='bottom', fontsize=6.6, color=col, fontweight='bold')
def stack3d(ax, x, y, n, w, h, col):
    for i in range(n):
        dx = i * 0.11
        ax.add_patch(Polygon([[x + dx, y + dx * 0.4], [x + dx + w, y + dx * 0.4], [x + dx + w + 0.12, y + dx * 0.4 + h], [x + dx + 0.12, y + dx * 0.4 + h]],
                             closed=True, fc=col, ec='white', lw=0.8, alpha=0.55 + 0.45 * i / n))
def arrow(ax, a, b, rad=-0.25, col=INK, ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=11, lw=1.2, color=col, linestyle=ls))
def badge(ax, x, y, s, col=INK):
    ax.add_patch(Circle((x, y), 0.17, fc=col, ec='none')); ax.text(x, y, s, ha='center', va='center', color='white', fontsize=8, fontweight='bold')

fig = plt.figure(figsize=(16, 6.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 6.4); ax.axis('off')
# 1 chats
badge(ax, 0.35, 6.0, '1'); ax.text(0.6, 6.0, 'Chat logs', va='center', fontsize=10, fontweight='bold')
chat_stack(ax, 0.3, 3.55, 1.7, 2.1, hl=(2, 4)); chat_stack(ax, 0.45, 3.4, 1.7, 2.1, hl=(3,))
ax.text(1.3, 3.05, 'PAN12: 97,689 train /\n218,702 test users', ha='center', va='top', fontsize=6.8)
ax.text(1.3, 2.45, 'Perverted Justice (Koutli):\n409 offender / decoy pairs', ha='center', va='top', fontsize=6.8, color=OK[4])
# 2 stage-1 screen
badge(ax, 2.65, 6.0, '2'); ax.text(2.9, 6.0, 'Screen', va='center', fontsize=10, fontweight='bold')
ax.add_patch(Polygon([[2.55, 5.5], [4.05, 5.5], [3.55, 4.25], [3.55, 3.6], [3.05, 3.6], [3.05, 4.25]], closed=True, fc='#dce9f5', ec=OK[0], lw=1.1))
for i, yy in enumerate(np.linspace(5.3, 4.5, 4)):
    for xx in np.linspace(2.8 + 0.1 * i, 3.8 - 0.1 * i, 6 - i): ax.add_patch(Circle((xx, yy), 0.045, fc=RED if (i + int(xx * 10)) % 5 == 0 else GREY, ec='none'))
ax.text(3.3, 3.35, 'TF-IDF + linear SVM\ntop 400 users\n(85% of offenders kept)', ha='center', va='top', fontsize=6.8)
# 3 LLM
badge(ax, 4.75, 6.0, '3'); ax.text(5.0, 6.0, 'LLM analyst', va='center', fontsize=10, fontweight='bold')
stack3d(ax, 4.55, 3.85, 6, 1.0, 0.95, OK[0]); ax.text(5.35, 3.55, 'open LLM (Qwen3-8B)\nschema-constrained JSON', ha='center', va='top', fontsize=6.8)
card = FancyBboxPatch((4.45, 1.0), 1.85, 1.95, boxstyle='round,pad=0.02,rounding_size=0.08', fc='#fbfbf8', ec=GREY, lw=0.9); ax.add_patch(card)
for i, (k, v, c) in enumerate([('p(grooming)', '0.97', RED), ('stages', 'trust, sexual', INK), ('behaviours', 'requests images', INK), ('evidence', '[c12:41] [c12:57]', RED), ('role', 'offender', INK)]):
    ax.text(4.55, 2.75 - i * 0.36, k, fontsize=6.5, color=GREY, va='center'); ax.text(6.2, 2.75 - i * 0.36, v, fontsize=6.5, color=c, va='center', ha='right', fontweight='bold')
arrow(ax, (5.4, 3.3), (5.4, 3.0), rad=0)
# 4 three-way decision
badge(ax, 7.0, 6.0, '4'); ax.text(7.25, 6.0, 'Calibrated decision', va='center', fontsize=10, fontweight='bold')
xs = np.linspace(0, 1, 200); dens_n = np.exp(-((xs - 0.2) / 0.13) ** 2); dens_p = 0.8 * np.exp(-((xs - 0.85) / 0.1) ** 2)
X0, Y0, W, H = 6.95, 4.1, 2.4, 1.3
ax.fill_between(X0 + xs * W, Y0, Y0 + dens_n * H * 0.8, color=GREY, alpha=0.35, lw=0); ax.fill_between(X0 + xs * W, Y0, Y0 + dens_p * H * 0.8, color=RED, alpha=0.45, lw=0)
for t, lab in ((0.28, r'$t_{clear}$'), (0.735, r'$t_{flag}$')):
    ax.plot([X0 + t * W] * 2, [Y0 - 0.05, Y0 + H], color=INK, lw=1, ls='--'); ax.text(X0 + t * W, Y0 + H + 0.05, lab, ha='center', fontsize=7.5)
for a, b, c, s in ((0, 0.28, GREEN, 'clear'), (0.28, 0.735, AMBER, 'human\nreview'), (0.735, 1, RED, 'flag')):
    ax.add_patch(Rectangle((X0 + a * W, Y0 - 0.45), (b - a) * W, 0.38, fc=c, ec='white', lw=1)); ax.text(X0 + (a + b) / 2 * W, Y0 - 0.26, s, ha='center', va='center', fontsize=6.5, color='white', fontweight='bold')
ax.text(X0 + W / 2, Y0 - 0.6, 'stacked score $q$ = f(SVM, LLM)\nconformal: P(flag | innocent) $\\leq\\alpha$', ha='center', va='top', fontsize=6.8)
# 5 legal RAG
badge(ax, 10.0, 6.0, '5'); ax.text(10.25, 6.0, 'Legal retrieval', va='center', fontsize=10, fontweight='bold')
ax.add_patch(FancyBboxPatch((9.95, 4.85), 1.55, 0.6, boxstyle='round,pad=0.02,rounding_size=0.08', fc='#fff4df', ec=OK[1], lw=1))
ax.text(10.72, 5.15, 'behaviour → statutory\nlanguage (rewrite)', ha='center', va='center', fontsize=6.5)
for i, (lab, col) in enumerate((('EU 2011/93', OK[0]), ('18 U.S.C.', OK[3]), ('VN Crim. Code', OK[2]))):
    doc(ax, 9.95 + i * 0.62, 3.55, 0.52, 0.95, col, {'EU 2011/93': 'EU', '18 U.S.C.': 'US', 'VN Crim. Code': 'VN'}[lab])
ax.text(10.85, 3.35, '43 provisions, 3 jurisdictions\ndense + sparse retrieval', ha='center', va='top', fontsize=6.8)
arrow(ax, (10.72, 4.82), (10.72, 4.55), rad=0)
# 6 element reasoning + verifier
badge(ax, 12.3, 6.0, '6'); ax.text(12.55, 6.0, 'Offence elements', va='center', fontsize=10, fontweight='bold')
ax.add_patch(FancyBboxPatch((12.25, 3.45), 1.9, 2.15, boxstyle='round,pad=0.02,rounding_size=0.08', fc='#fbfbf8', ec=GREY, lw=0.9))
ax.text(12.35, 5.42, 'EU Art. 6(2)', fontsize=7, fontweight='bold', color=OK[0])
for i, (k, s, c) in enumerate((('adult', '✓', GREEN), ('ICT', '✓', GREEN), ('presented age', '✓', GREEN), ('solicits images', '?', AMBER))):
    ax.text(12.4, 5.1 - i * 0.36, k, fontsize=6.6, va='center'); ax.text(14.0, 5.1 - i * 0.36, s, fontsize=8.5, va='center', ha='right', color=c, fontweight='bold')
ax.add_patch(Circle((14.65, 4.3), 0.28, fc='white', ec=INK, lw=1.4)); ax.plot([14.85, 15.1], [4.1, 3.85], color=INK, lw=2.2)
ax.text(14.75, 3.6, 'verifier:\ncited lines\nonly', ha='center', va='top', fontsize=6.5)
ax.text(13.2, 3.25, 'decision support for\nanalysts, not a verdict', ha='center', va='top', fontsize=6.8, color=GREY, style='italic')
# arrows between stages
arrow(ax, (2.2, 4.6), (2.6, 4.9)); arrow(ax, (4.0, 4.4), (4.5, 4.4), rad=0.15); arrow(ax, (6.3, 4.6), (6.9, 4.7), rad=-0.2)
arrow(ax, (9.4, 4.9), (9.9, 5.15), rad=-0.2); arrow(ax, (11.55, 4.0), (12.2, 4.3), rad=0.2); arrow(ax, (14.15, 4.6), (14.4, 4.45), rad=0)
ax.text(8.15, 2.55, 'flagged + review cases', ha='center', fontsize=6.5, color=GREY)
arrow(ax, (8.15, 3.0), (9.9, 3.9), rad=0.35, col=GREY, ls='--')
# evaluation strip
ax.add_patch(FancyBboxPatch((0.3, 0.12), 15.4, 0.72, boxstyle='round,pad=0.02,rounding_size=0.1', fc='#f4f4f1', ec='none'))
ax.text(0.5, 0.48, 'Evaluated', fontsize=8.5, fontweight='bold', va='center')
items = ['detection vs. PAN12 ground truth', 'false-flag control (5 LLMs)', 'offender vs. police decoy', 'early warning', 'explanation faithfulness', 'citation validity', 'element verification']
for i, s in enumerate(items):
    xx = 1.75 + i * 2.0; ax.add_patch(Circle((xx, 0.48), 0.07, fc=(OK[:6] + [RED])[i], ec='none')); ax.text(xx + 0.13, 0.48, s, fontsize=6.6, va='center')
save(fig, 'fig1_architecture')

# ------------------------------------------------------------------ Figure 2: funnel + decisions (Qwen3)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={'width_ratios': [1.25, 1]})
ax = axs[0]; tag(ax, 'A')
n_pred = 254; cand = 400; cand_pred = round(Q['stage1_recall_te'] * n_pred)
stages = [('All test users', NUM['n_test_users'], n_pred), ('Stage-1 candidates', cand, cand_pred),
          ('Flag', op['n_flag'], round(op['predators_flagged'] * n_pred)), ]
ys = np.arange(3)[::-1]
ax.barh(ys, [np.log10(s[1]) for s in stages], color=['#cfd8e3', OK[0], RED], height=0.6)
for y, (lab, n, p) in zip(ys, stages):
    ax.text(np.log10(n) + 0.08 if lab != 'All test users' else 0.08, y, f'{lab}: {n:,} users, {p} offenders', va='center', fontsize=7, color=INK, fontweight='bold')
ax.set_yticks([]); ax.set_xlim(0, 7.2); ax.set_xlabel('users (log$_{10}$)'); ax.set_title('PAN12 test set: screening funnel (Qwen3-8B)', loc='left')
ax = axs[1]; tag(ax, 'B')
inn = 400 - cand_pred
fl = op['n_flag']; fl_tp = round(op['flag_precision'] * fl); fl_fp = fl - fl_tp
rv = round(op['review_rate_candidates'] * 400)
cats = ['flag', 'review', 'clear']
pos = [fl_tp, round((1 - op['predators_flagged'] - op['miss_rate_predators']) * n_pred), cand_pred - fl_tp - round((1 - op['predators_flagged'] - op['miss_rate_predators']) * n_pred)]
neg = [fl_fp, rv - pos[1], inn - fl_fp - (rv - pos[1])]
ax.bar(cats, pos, color=RED, label='offenders'); ax.bar(cats, neg, bottom=pos, color=GREY, alpha=0.6, label='other users')
for i in range(3): ax.text(i, pos[i] + neg[i] + 4, f'{pos[i]} / {neg[i]}', ha='center', fontsize=6.8)
ax.set_ylabel('candidates'); ax.set_ylim(0, 195); ax.set_title(r'Three-way decision ($\alpha$=0.05, $\beta$=0.10)', loc='left'); ax.legend(frameon=False, loc='upper left', ncol=2)
NUM['decision_counts'] = dict(pos=pos, neg=neg)
fig.tight_layout(); save(fig, 'fig2_funnel')

# ------------------------------------------------------------------ Figure 3: models + conformal validity
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
names = [n for n, _ in MODELS]; short = ['Qwen3', 'Qwen2.5', 'Granite', 'Mistral', 'Phi-4-mini']
ax = axs[0]; tag(ax, 'A'); x = np.arange(len(names))
ax.bar(x - 0.2, [M[n]['E1_test_full_p0.5']['F05'] for n in names], 0.4, color='#b9c7d6', label='LLM alone (p≥0.5)')
ax.bar(x + 0.2, [M[n]['E1_test_full_stacked_q0.5']['F05'] for n in names], 0.4, color=OK[0], label='SVM + LLM (q≥0.5)')
ax.axhline(base['tfidf_linsvm']['A_F05'], color=INK, ls='--', lw=0.9); ax.text(4.45, base['tfidf_linsvm']['A_F05'] + 0.012, 'SVM alone', fontsize=6.5, ha='right')
ax.set_xticks(x); ax.set_xticklabels(short, rotation=35, ha='right'); ax.set_ylim(0.4, 0.95); ax.set_ylabel('F$_{0.5}$, full PAN12 test set'); ax.legend(frameon=False, fontsize=6.3, loc='upper left', ncol=1)
ax.set_title('Detection', loc='left')
ax = axs[1]; tag(ax, 'B')
al = [0.01, 0.05, 0.10]
ax.plot([0, 0.11], [0, 0.11], color=INK, lw=0.8, ls=':')
for i, n in enumerate(names):
    ax.plot(al, [split(M[n], a)['false_flag_rate_innocent_candidates'] for a in al], 'o-', ms=3.5, lw=1, color=OK[i], label=short[i])
ax.plot(al, [trcal(Q, a)['false_flag_rate_innocent_candidates'] for a in al], 's--', ms=3.5, lw=1, color=OK[0], mfc='white', label='Qwen3, train-calibrated')
ax.set_xlabel(r'target false-flag rate $\alpha$'); ax.set_ylabel('observed false-flag rate'); ax.set_xticks(al); ax.legend(frameon=False, fontsize=5.8)
ax.set_title('False-flag control', loc='left')
ax = axs[2]; tag(ax, 'C')
for i, n in enumerate(names):
    s = split(M[n], 0.05); ax.scatter(s['review_rate_candidates'], s['predators_flagged'], s=40, color=OK[i]); ax.annotate(short[i], (s['review_rate_candidates'], s['predators_flagged']), xytext=[(5, 2), (5, 2), (-38, 8), (-14, -12), (6, 6)][i], textcoords='offset points', fontsize=6.5)
ax.set_xlabel('share of candidates sent to review'); ax.set_ylabel('share of offenders auto-flagged'); ax.set_title(r'Cost of a weaker LLM ($\alpha$=0.05)', loc='left')
ax.set_xlim(0.55, 1.0); ax.set_ylim(0, 0.65)
fig.tight_layout(); save(fig, 'fig3_models')

# ------------------------------------------------------------------ Figure 4: decoys
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
kp = np.array([x['p'] for x in kou]); ky = np.array([x['label'] for x in kou]); kq = np.array(kst['q'])
bins = np.linspace(0, 1, 21)
for ax, v, t, lab in ((axs[0], kp, 'A', 'LLM probability p'), (axs[1], kq, 'B', 'stacked score q')):
    tag(ax, t); ax.hist(v[ky == 1], bins, color=RED, alpha=0.65, label='offender side'); ax.hist(v[ky == 0], bins, color=OK[0], alpha=0.45, label='decoy side')
    ax.set_xlabel(lab); ax.set_ylabel('chat sides'); ax.set_yscale('log'); ax.legend(frameon=False, loc='upper center')
axs[1].axvline(Q['T_FLAG'], color=INK, ls='--', lw=0.9); axs[1].text(Q['T_FLAG'] - 0.02, 300, '$t_{flag}$', ha='right', fontsize=7)
axs[0].set_title('LLM alone (Qwen3-8B)', loc='left'); axs[1].set_title('SVM + LLM', loc='left')
ax = axs[2]; tag(ax, 'C')
ax.bar(x - 0.2, [100 * M[n]['E3']['at_p05']['decoy_FPR'] for n in names], 0.4, color='#d9c2d4', label='LLM alone, p≥0.5')
ax.bar(x + 0.2, [100 * M[n]['E3']['at_stacked']['decoy_FPR'] for n in names], 0.4, color=OK[0], label='SVM + LLM, $t_{flag}$')
for i, n in enumerate(names): ax.text(i + 0.2, 100 * M[n]['E3']['at_stacked']['decoy_FPR'] + 2, f"{100 * M[n]['E3']['at_stacked']['decoy_FPR']:.1f}", ha='center', fontsize=5.8, rotation=90)
ax.set_xticks(x); ax.set_xticklabels(short, rotation=35, ha='right'); ax.set_ylabel('decoys flagged (%)'); ax.set_ylim(0, 125); ax.legend(frameon=False, fontsize=6.3, loc='upper left', ncol=2)
ax.set_title('Decoy false positives', loc='left')
fig.tight_layout(); save(fig, 'fig4_decoys')

# ------------------------------------------------------------------ Figure 5: early warning + faithfulness
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={'width_ratios': [1, 1.3]})
ax = axs[0]; tag(ax, 'A'); cuts = [5, 10, 20, 40, 80, 160]
ax.plot(cuts, [Q['E4'][str(c)]['ROCAUC_p'] for c in cuts], 'o-', color=OK[0], ms=4)
ax.axhline(0.5, color=GREY, ls=':', lw=0.8); ax.set_xscale('log'); ax.set_xticks(cuts); ax.set_xticklabels(cuts); ax.minorticks_off()
ax.set_xlabel('messages seen'); ax.set_ylabel('ROC-AUC'); ax.set_ylim(0.45, 0.85); ax.set_title('Early warning (Qwen3-8B)', loc='left')
ax = axs[1]; tag(ax, 'B')
var = [('remove_cited', 'remove cited'), ('remove_tfidf', 'remove TF-IDF top'), ('remove_random', 'remove random'), ('keep_cited', 'keep only cited'), ('keep_random', 'keep only random')]
E5 = Q['E5']; yy = np.arange(len(var))[::-1]
cols = [OK[3], OK[1], GREY, OK[2], GREY]
ax.barh(yy, [100 * E5[f'flip_{v}'] for v, _ in var], color=cols, height=0.6)
for y, (v, _) in zip(yy, var): ax.text(100 * E5[f'flip_{v}'] + 0.8, y, f'{100 * E5[f"flip_{v}"]:.1f}%  (p drop {E5[f"drop_{v}"]:.3f})', va='center', fontsize=6.6)
ax.set_yticks(yy); ax.set_yticklabels([l for _, l in var]); ax.set_xlim(0, 48); ax.set_xlabel('decisions flipped to "not grooming" (%)')
ax.set_title(f'Faithfulness of cited lines (n={E5["n"]} offenders)', loc='left')
fig.tight_layout(); save(fig, 'fig5_warning_faithfulness')

# ------------------------------------------------------------------ Figure 6: legal retrieval + citations
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
E6 = Q['E6']; ax = axs[0]; tag(ax, 'A')
rv_ = [('plain_dense', 'plain\ndense'), ('plain_hybrid', 'plain\nhybrid'), ('rewrite_hybrid', 'rewrite\nhybrid'), ('rewrite_dense', 'rewrite\ndense')]
ax.bar(range(4), [E6[f'retrieval_recall_{k}'] for k, _ in rv_], color=[GREY, GREY, OK[1], OK[1]])
for i, (k, _) in enumerate(rv_): ax.text(i, E6[f'retrieval_recall_{k}'] + 0.015, f'{E6[f"retrieval_recall_{k}"]:.2f}', ha='center', fontsize=7)
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in rv_]); ax.set_ylim(0, 1); ax.set_ylabel('recall of reference provisions'); ax.set_title(f'Retrieval (n={E6["n"]} flagged users)', loc='left')
ax = axs[1]; tag(ax, 'B')
mets = [('outside allowed set', E6['rag_cite_outside_context'], E6['norag_cite_outside_corpus']), ('precision', E6['rag_cite_precision'], E6['norag_cite_precision']), ('recall', E6['rag_cite_recall'], E6['norag_cite_recall'])]
xx = np.arange(3)
ax.bar(xx - 0.2, [m[1] for m in mets], 0.4, color=OK[0], label='with retrieval'); ax.bar(xx + 0.2, [m[2] for m in mets], 0.4, color='#b9c7d6', label='without retrieval')
for i, m in enumerate(mets): ax.text(i - 0.2, m[1] + 0.015, f'{m[1]:.2f}', ha='center', fontsize=6.6); ax.text(i + 0.2, m[2] + 0.015, f'{m[2]:.2f}', ha='center', fontsize=6.6)
ax.set_xticks(xx); ax.set_xticklabels([m[0] for m in mets]); ax.set_ylim(0, 0.85); ax.set_ylabel('share of cited provisions'); ax.legend(frameon=False); ax.set_title('Citations in the legal analysis', loc='left')
fig.tight_layout(); save(fig, 'fig6_legal_rag')

# ------------------------------------------------------------------ Figure 7: offence elements
OFF = list(ELEM)
OL = {'EU-2011/93-6(1)': 'EU 6(1) meet', 'EU-2011/93-6(2)': 'EU 6(2) images', 'EU-2011/93-3(2)': 'EU 3(2) witness', 'US-18-2422(b)': 'US §2422(b)',
      'US-18-1470': 'US §1470', 'US-18-2251(a)': 'US §2251(a)', 'VN-CC2015-147': 'VN Art. 147', 'VN-CC2015-145': 'VN Art. 145'}
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={'width_ratios': [1, 1.55]})
ax = axs[0]; tag(ax, 'A'); yy = np.arange(len(OFF))[::-1]
E7 = Q['E7']
ax.barh(yy + 0.2, [100 * E7['offence_rate_met'][o] for o in OFF], 0.38, color='#b9c7d6', label='all elements claimed met')
ax.barh(yy - 0.2, [100 * NUM['E7_partial_offence'][o] for o in OFF], 0.38, color=OK[2], label='≥1 evidential element verified')
ax.barh(yy - 0.2, [100 * E7['offence_rate_verified'][o] for o in OFF], 0.38, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([OL[o] for o in OFF]); ax.set_xlabel('offenders (%)'); ax.set_xlim(0, 60); ax.legend(frameon=False, fontsize=6.2, loc='upper right')
ax.set_title(f'Offence level (n={NUM["E7_pred_users"]})', loc='left')
ax = axs[1]; tag(ax, 'B')
rows = []
for o in OFF:
    for k, _ in ELEM[o]['elements']:
        c = [s for oo, kk, s in claims if (oo, kk) == (o, k)]
        rows.append((OL[o], k, len(c), np.mean([(s or 0) >= 0.5 for s in c]) if c else np.nan, np.mean([s is None for s in c]) if c else np.nan))
yy = np.arange(len(rows))[::-1]
ax.barh(yy, [100 * (r[3] if r[3] == r[3] else 0) for r in rows], color=[GREY if r[1] in CONTEXT else OK[2] for r in rows], height=0.7)
ax.barh(yy, [-100 * (r[4] if r[4] == r[4] else 0) for r in rows], color=OK[3], height=0.7, alpha=0.7)
ax.set_yticks(yy); ax.set_yticklabels([f'{r[0]}: {r[1].replace("_", " ")} ({r[2]})' for r in rows], fontsize=5.6)
ax.axvline(0, color=INK, lw=0.6); ax.set_xlim(-102, 60); ax.set_xticks([-100, -50, 0, 50]); ax.set_xticklabels(['100', '50', '0', '50'])
ax.set_xlabel('← no valid line cited (%)      verified (%) →'); ax.set_title('Element level: claims marked "met" (count)', loc='left')
fig.tight_layout(); save(fig, 'fig7_elements')

# ------------------------------------------------------------------ per-user reconstruction (labels from the PAN12 ground truth, scores from the Kaggle outputs)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from collections import Counter
LBL = R('datasets/pan12_labels')
P_te = {l.strip() for l in open(os.path.join(LBL, 'pan12-sexual-predator-identification-groundtruth-problem1.txt')) if l.strip()}
P_tr = {l.strip() for l in open(os.path.join(LBL, 'pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt')) if l.strip()}
SV = np.load(R('kaggle/baselines/out/scores_tfidf_linsvm.npy'), allow_pickle=True).item()
def rebuild(path):
    """Candidates are the top-N stage-1 users, so their SVM scores are the top-N sorted scores; the stacker is refitted exactly as in the kernel."""
    D = json.load(open(R(path, 'detection.json'))); tr, te = D['train'], D['test']
    s1tr, s1te = np.sort(SV['tr_oof'])[::-1][:len(tr)], np.sort(SV['te'])[::-1][:len(te)]
    ptr, pte = np.nan_to_num([r['p'] for r in tr], nan=0.5), np.nan_to_num([r['p'] for r in te], nan=0.5)
    ytr, yte = np.array([r['user'] in P_tr for r in tr]), np.array([r['user'] in P_te for r in te])
    q = LogisticRegression(C=1.0).fit(np.c_[s1tr, ptr], ytr).predict_proba(np.c_[s1te, pte])[:, 1]
    return dict(te=te, s1=s1te, p=pte, q=q, y=yte)
RB = {n: rebuild(p) for n, p in MODELS}
for n in names: assert abs(roc_auc_score(RB[n]['y'], RB[n]['q']) - M[n]['E1_test_stacked_cand_ROCAUC']) < 1e-6, n   # exact reconstruction
NUM['cand_auc_s1'] = roc_auc_score(RB['Qwen3-8B']['y'], RB['Qwen3-8B']['s1'])

# ------------------------------------------------------------------ Figure: score space, ROC, calibration (Qwen3)
rb = RB['Qwen3-8B']; y = rb['y']
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6))
ax = axs[0]; tag(ax, 'A')
dec = np.where(rb['q'] >= Q['T_FLAG'], 2, np.where(rb['q'] < Q['T_CLEAR'], 0, 1))
for lab, mk, c in ((0, 'o', GREY), (1, '^', RED)):
    m_ = y == lab; ax.scatter(rb['s1'][m_], rb['p'][m_], s=9, marker=mk, c=c, alpha=0.6, lw=0, label=['other users', 'offenders'][lab])
gx, gy = np.meshgrid(np.linspace(rb['s1'].min(), rb['s1'].max(), 200), np.linspace(0, 1, 200))
stk_q = LogisticRegression(C=1.0).fit(np.c_[np.sort(SV['tr_oof'])[::-1][:300], np.nan_to_num([r['p'] for r in json.load(open(R('kaggle/lexguard_v3/out/detection.json')))['train']], nan=0.5)],
                                      [r['user'] in P_tr for r in json.load(open(R('kaggle/lexguard_v3/out/detection.json')))['train']])
gq = stk_q.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1].reshape(gx.shape)
ax.contourf(gx, gy, gq, levels=[0, Q['T_CLEAR'], Q['T_FLAG'], 1], colors=[GREEN, AMBER, RED], alpha=0.12)
ax.contour(gx, gy, gq, levels=[Q['T_CLEAR'], Q['T_FLAG']], colors=[INK, INK], linewidths=0.8, linestyles=['--', '-'])
ax.set_xlabel('stage-1 SVM score'); ax.set_ylabel('LLM probability p'); ax.legend(frameon=True, framealpha=0.9, loc=(0.03, 0.42), fontsize=6.3, markerscale=2); ax.set_title('Score space: clear / review / flag', loc='left')
ax = axs[1]; tag(ax, 'B')
for v, lab, c in ((rb['s1'], 'SVM', GREY), (rb['p'], 'LLM p', OK[1]), (rb['q'], 'stacked q', OK[0])):
    f_, t_, _ = roc_curve(y, v); ax.plot(f_, t_, color=c, lw=1.3, label=f'{lab} ({roc_auc_score(y, v):.3f})')
ax.plot([0, 1], [0, 1], ':', color=GREY, lw=0.8); ax.set_xlabel('false-positive rate'); ax.set_ylabel('true-positive rate'); ax.legend(frameon=False, loc='lower right'); ax.set_title('ROC among 400 candidates', loc='left')
ax = axs[2]; tag(ax, 'C')
bins_ = np.linspace(0, 1, 11)
for v, lab, c in ((rb['p'], 'LLM p', OK[1]), (rb['q'], 'stacked q', OK[0])):
    idx = np.clip(np.digitize(v, bins_) - 1, 0, 9); xs_, ys_, ns_ = [], [], []
    for b in range(10):
        m_ = idx == b
        if m_.sum() >= 5: xs_.append(v[m_].mean()); ys_.append(y[m_].mean()); ns_.append(m_.sum())
    ax.plot(xs_, ys_, 'o-', color=c, ms=3.5, lw=1.2, label=lab)
ax.plot([0, 1], [0, 1], ':', color=GREY, lw=0.8); ax.set_xlabel('predicted probability'); ax.set_ylabel('observed share of offenders'); ax.legend(frameon=False, loc='upper left'); ax.set_title('Calibration among candidates', loc='left')
NUM['p_extreme'] = float(np.mean((rb['p'] < 0.05) | (rb['p'] > 0.95)))
fig.tight_layout(); save(fig, 'fig_scores')

# ------------------------------------------------------------------ Figure: stages, behaviours and roles (Qwen3)
te = rb['te']
STG = ['friendship_forming', 'relationship_forming', 'risk_assessment', 'exclusivity', 'sexual', 'meeting_planning']
BEH = list(refmap)
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={'width_ratios': [1.1, 1.25, 0.8]})
for ax, keys, field, t, ttl in ((axs[0], STG, 'stages', 'A', "Grooming stages (O'Connell)"), (axs[1], BEH, 'behaviours', 'B', 'Behaviours')):
    tag(ax, t); yy = np.arange(len(keys))[::-1]
    po = [100 * np.mean([k in r[field] for r, l in zip(te, y) if l]) for k in keys]; ne = [100 * np.mean([k in r[field] for r, l in zip(te, y) if not l]) for k in keys]
    ax.barh(yy + 0.2, po, 0.38, color=RED, label='offenders'); ax.barh(yy - 0.2, ne, 0.38, color=GREY, alpha=0.7, label='other candidates')
    ax.set_yticks(yy); ax.set_yticklabels([k.replace('_', ' ') for k in keys], fontsize=6.6); ax.set_xlabel('candidates labelled (%)'); ax.set_xlim(0, 100); ax.set_title(ttl, loc='left')
    NUM[f'{field}_rates'] = {k: (a, b) for k, a, b in zip(keys, po, ne)}
axs[1].legend(frameon=False, loc='center right', fontsize=6.3)
ax = axs[2]; tag(ax, 'C')
ROLES = ['predator', 'child_or_decoy', 'bystander', None]
po = [np.mean([r['role'] == k for r, l in zip(te, y) if l]) * 100 for k in ROLES]; ne = [np.mean([r['role'] == k for r, l in zip(te, y) if not l]) * 100 for k in ROLES]
xx = np.arange(2); bot = np.zeros(2)
for k, a, b, c in zip(ROLES, po, ne, [RED, OK[4], OK[5], '#dddddd']):
    ax.bar(xx, [a, b], bottom=bot, color=c, label={'predator': 'offender', 'child_or_decoy': 'child / decoy', 'bystander': 'bystander', None: 'not parsed'}[k]); bot += [a, b]
ax.set_xticks(xx); ax.set_xticklabels(['offenders', 'other'], fontsize=7); ax.set_ylabel('candidates (%)'); ax.set_title('Role assigned', loc='left'); ax.legend(frameon=False, fontsize=5.8, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
NUM['role_rates'] = {str(k): (a, b) for k, a, b in zip(ROLES, po, ne)}
fig.tight_layout(); save(fig, 'fig_stages')

# ------------------------------------------------------------------ Figure: conformal grid (Qwen3, stacked q)
fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.2))
AL, BE = [0.01, 0.05, 0.10], [0.05, 0.10, 0.20]
mets = [('false_flag_rate_innocent_candidates', 'false-flag rate'), ('predators_flagged', 'offenders flagged'), ('review_rate_candidates', 'review rate')]
for row, (src, rl) in enumerate(((lambda a, b: split(Q, a, b), 'split conformal'), (lambda a, b: trcal(Q, a, b), 'train-calibrated'))):
    for col, (k, ttl) in enumerate(mets):
        ax = axs[row, col]; Z = np.array([[src(a, b)[k] for a in AL] for b in BE])
        ax.imshow(Z, cmap='Reds' if col == 0 else ('Blues' if col == 1 else 'Oranges'), vmin=0, vmax=max(0.25, Z.max()) if col == 0 else 1, aspect='auto')
        for i in range(3):
            for j in range(3):
                bad = col == 0 and Z[i, j] > AL[j] + 1e-9
                ax.text(j, i, f'{100 * Z[i, j]:.1f}', ha='center', va='center', fontsize=7, color='white' if bad else INK, fontweight='bold' if bad else 'normal')
        ax.set_xticks(range(3)); ax.set_xticklabels([f'{int(a * 100)}%' for a in AL]); ax.set_yticks(range(3)); ax.set_yticklabels([f'{int(b * 100)}%' for b in BE]); ax.grid(False)
        if row == 1: ax.set_xlabel(r'target $\alpha$')
        if col == 0: ax.set_ylabel(f'{rl}\n' + r'target $\beta$')
        if row == 0: ax.set_title(ttl + ' (%)', loc='left')
fig.tight_layout(); save(fig, 'fig_conformal_grid')

# ------------------------------------------------------------------ Figure: ROC of every LLM, PAN12 candidates and Perverted Justice pairs
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.6))
ax = axs[0]; tag(ax, 'A')
for i, n in enumerate(names):
    f_, t_, _ = roc_curve(RB[n]['y'], RB[n]['p']); ax.plot(f_, t_, color=OK[i], lw=1.1, label=f'{short[i]} ({roc_auc_score(RB[n]["y"], RB[n]["p"]):.2f})')
f_, t_, _ = roc_curve(RB['Qwen3-8B']['y'], RB['Qwen3-8B']['s1']); ax.plot(f_, t_, color=INK, lw=1, ls='--', label=f'SVM ({NUM["cand_auc_s1"]:.2f})')
ax.set_title('PAN12 candidates, LLM alone', loc='left')
ax = axs[1]; tag(ax, 'B')
KO = {}
for i, (n, p_) in enumerate(MODELS):
    kj = json.load(open(R(p_, 'koutli.json'))); ks = json.load(open(R(p_, 'koutli_stacked.json')))
    KO[n] = (np.array([x['label'] for x in kj]), np.array([x['p'] for x in kj]), np.array(ks['q']), np.array(ks['s1']))
    f_, t_, _ = roc_curve(KO[n][0], KO[n][1]); ax.plot(f_, t_, color=OK[i], lw=1.1, label=f'{short[i]} ({roc_auc_score(KO[n][0], KO[n][1]):.2f})')
ky_, _, _, ks1 = KO['Qwen3-8B']; f_, t_, _ = roc_curve(ky_, ks1); ax.plot(f_, t_, color=INK, lw=1, ls='--', label=f'SVM ({roc_auc_score(ky_, ks1):.2f})')
ax.set_title('Offender vs. decoy, LLM alone', loc='left')
ax = axs[2]; tag(ax, 'C')
for i, n in enumerate(names):
    f_, t_, _ = roc_curve(RB[n]['y'], RB[n]['q']); ax.plot(f_, t_, color=OK[i], lw=1.1, label=f'{short[i]} ({roc_auc_score(RB[n]["y"], RB[n]["q"]):.2f})')
ax.plot(*roc_curve(RB['Qwen3-8B']['y'], RB['Qwen3-8B']['s1'])[:2], color=INK, lw=1, ls='--', label=f'SVM ({NUM["cand_auc_s1"]:.2f})')
ax.set_title('PAN12 candidates, SVM + LLM', loc='left')
for ax in axs:
    ax.plot([0, 1], [0, 1], ':', color=GREY, lw=0.8); ax.set_xlabel('false-positive rate'); ax.legend(frameon=False, fontsize=5.6, loc='lower right')
axs[0].set_ylabel('true-positive rate')
NUM['koutli_auc_p'] = {n: roc_auc_score(KO[n][0], KO[n][1]) for n in names}
fig.tight_layout(); save(fig, 'fig_roc_models')

# ------------------------------------------------------------------ Figure: which provisions were retrieved and cited
lr_ = json.load(open(R('kaggle/lexguard_v3/out/legal_rag.json')))
ids_order = [d['id'] for d in corpus]
def cites(block, mode):
    out = []
    for jur in ('EU', 'US', 'VN'):
        b = (block or {}).get(jur)
        for c in (b.get('provisions') or []) if isinstance(b, dict) else []:
            out.append(str(c).strip('[] '))
    return out
cnt_ret = Counter(i for x in lr_ for i in x['retrieved']); cnt_rag = Counter(i for x in lr_ for i in set(cites(x['rag'], 'rag')))
refset = Counter(i for x in lr_ for b in x['behaviours'] for i in refmap[b]['provisions'])
top = [i for i in ids_order if cnt_ret[i] or cnt_rag[i]]
top = sorted(top, key=lambda i: -(cnt_ret[i] + cnt_rag[i]))[:22]
fig, ax = plt.subplots(figsize=(7.4, 3.3)); tag(ax, '')
xx = np.arange(len(top)); n_ = len(lr_)
ax.bar(xx - 0.2, [100 * cnt_ret[i] / n_ for i in top], 0.4, color='#cfd8e3', label='retrieved')
ax.bar(xx + 0.2, [100 * cnt_rag[i] / n_ for i in top], 0.4, color=OK[0], label='cited in the analysis (with retrieval)')
for k, i in enumerate(top):
    if refset[i]: ax.text(k, max(cnt_ret[i], cnt_rag[i]) / n_ * 100 + 2, '★', ha='center', fontsize=8, color=OK[1])
lab_ = {i: i.replace('EU-2011/93-', 'EU Art. ').replace('US-18-', 'US §').replace('VN-CC2015-', 'VN Art. ') for i in top}
ax.set_xticks(xx); ax.set_xticklabels([lab_[i] for i in top], rotation=60, ha='right', fontsize=6.5); ax.set_ylabel('flagged users (%)'); ax.set_ylim(0, 115)
ax.legend(frameon=False, loc='upper right'); ax.set_title(f'Provisions retrieved and cited for {n_} flagged users (★ = in the reference map)', loc='left')
NUM['cite_top'] = {i: (cnt_ret[i], cnt_rag[i]) for i in top}
fig.tight_layout(); save(fig, 'fig_citations')

# ------------------------------------------------------------------ Figure: element status and verifier scores
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={'width_ratios': [1.4, 1]})
ax = axs[0]; tag(ax, 'A'); yy = np.arange(len(OFF))[::-1]
ST = ['met', 'not_met', 'unknown', None]; SC = [OK[2], OK[3], AMBER, '#dddddd']
left = np.zeros(len(OFF))
for s_, c in zip(ST, SC):
    v = np.array([100 * np.mean([st == s_ for x in pred_rows if x['offence'] == o for st in x['status'].values()]) for o in OFF])
    ax.barh(yy, v, left=left, color=c, label=s_.replace('_', ' ') if s_ else 'missing'); left += v
ax.set_yticks(yy); ax.set_yticklabels([OL[o] for o in OFF]); ax.set_xlabel('element assessments (%)'); ax.set_xlim(0, 100)
ax.legend(frameon=False, ncol=4, fontsize=6.3, loc='upper center', bbox_to_anchor=(0.5, -0.2)); ax.set_title('Element status claimed by the LLM', loc='left')
ax = axs[1]; tag(ax, 'B')
sp = np.array([s for _, _, s in claims if s is not None])
ax.hist(sp, bins=np.linspace(0, 1, 21), color=OK[2], alpha=0.8); ax.axvline(0.5, color=INK, ls='--', lw=0.9)
ax.set_yscale('log'); ax.set_xlabel('verifier P(supported)'); ax.set_ylabel('element claims'); ax.set_title(f'Verifier scores (n={len(sp)})', loc='left')
NUM['verifier_extreme'] = float(np.mean((sp < 0.05) | (sp > 0.95)))
fig.tight_layout(); save(fig, 'fig_element_status')

# ------------------------------------------------------------------ tables
T = []
T.append(r'''\begin{tabular}{lrrrr}
\toprule
Corpus & Conversations & Messages & Users & Offenders \\
\midrule
PAN12 training & %s & %s & %s & %d \\
PAN12 test & %s & %s & %s & %d \\
Perverted Justice (Koutli), all & %s & -- & -- & %d \\
\quad not in PAN12 (used here) & -- & -- & -- & %d \\
\quad with the decoy's side available & %d pairs & -- & -- & %d \\
\bottomrule
\end{tabular}''' % (f"{eda['pan12_train']['conversations']:,}", f"{eda['pan12_train']['messages']:,}", '97,689', eda['pan12_train']['predators'],
                    f"{eda['pan12_test']['conversations']:,}", f"{eda['pan12_test']['messages']:,}", '218,702', eda['pan12_test']['predators'],
                    '622', eda['koutli']['predators'], eda['koutli']['unseen'], Q['E3']['pairs'], Q['E3']['pairs']))
open(os.path.join(TAB, 'tab_data.tex'), 'w').write(T[-1])

lr, sv = base['tfidf_logreg'], base['tfidf_linsvm']
def prow(name, d, auc='--'): return f"{name} & {f3(d['P'])} & {f3(d['R'])} & {f3(d['F05'])} & {auc} \\\\"
tm = [r'\begin{tabular}{lcccc}', r'\toprule', r'System & Precision & Recall & F$_{0.5}$ & ROC-AUC \\', r'\midrule',
      r'\multicolumn{5}{l}{\textit{Single-stage baselines (this work)}} \\',
      f"TF-IDF + logistic regression & {f3(lr['A_precision'])} & {f3(lr['A_recall'])} & {f3(lr['A_F05'])} & {f3(lr['A_ROCAUC'])} \\\\",
      f"TF-IDF + linear SVM (stage 1) & {f3(sv['A_precision'])} & {f3(sv['A_recall'])} & {f3(sv['A_F05'])} & {f3(sv['A_ROCAUC'])} \\\\",
      r'\multicolumn{5}{l}{\textit{LEX-GUARD, stage 2 on the 400 stage-1 candidates (Qwen3-8B)}} \\',
      prow('LLM alone, $p\\geq0.5$', Q['E1_test_full_p0.5']), prow('LLM alone, $p\\geq0.9$', Q['E1_test_full_p0.9']),
      prow('Stacked SVM + LLM, $q\\geq0.5$', Q['E1_test_full_stacked_q0.5']),
      f"Three-way decision, flags only ($\\alpha$=0.05) & {f3(op['flag_precision'])} & {f3(op['predators_flagged'])} & {f3(NUM['operating_point']['flag_F05'])} & -- \\\\",
      r'\multicolumn{5}{l}{\textit{Best PAN 2012 submission (reported)}} \\',
      r'Two-step lexical classifier \citep{VillatoroTello_2012} & -- & -- & 0.935 & -- \\', r'\bottomrule', r'\end{tabular}']
open(os.path.join(TAB, 'tab_main.tex'), 'w').write('\n'.join(tm))

tc = [r'\begin{tabular}{lccccccc}', r'\toprule',
      r' & Valid & \multicolumn{2}{c}{F$_{0.5}$ (full test set)} & \multicolumn{4}{c}{Three-way decision, $\alpha$=0.05 (split conformal, 50 splits)} \\',
      r'\cmidrule(lr){3-4}\cmidrule(lr){5-8}',
      r'LLM & JSON & LLM alone & SVM + LLM & false-flag rate & flag precision & offenders flagged & review rate \\', r'\midrule']
for n in names:
    m = M[n]; s = split(m, 0.05)
    tc.append(f"{n} & {pct(m['E1_test']['parse_rate'], 0)}\\% & {f3(m['E1_test_full_p0.5']['F05'])} & {f3(m['E1_test_full_stacked_q0.5']['F05'])} & "
              f"{pct(s['false_flag_rate_innocent_candidates'])}\\% & {pct(s['flag_precision'])}\\% & {pct(s['predators_flagged'])}\\% & {pct(s['review_rate_candidates'])}\\% \\\\")
tc += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(TAB, 'tab_models.tex'), 'w').write('\n'.join(tc))

td = [r'\begin{tabular}{lccccc}', r'\toprule', r' & \multicolumn{2}{c}{LLM alone ($p\geq0.5$)} & \multicolumn{3}{c}{SVM + LLM ($q\geq t_{flag}$)} \\',
      r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}', r'LLM & ROC-AUC & decoys flagged & offenders flagged & decoys flagged & F$_{0.5}$ \\', r'\midrule',
      f"SVM alone (stage 1) & {f3(sv['Chard_ROCAUC'])} & -- & {pct(sv['Chard_recall'])}\\% & {pct(sv['Chard_decoy_FPR'])}\\% & {f3(sv['Chard_F05'])} \\\\"]
for n in names:
    e = M[n]['E3']
    td.append(f"{n} & {f3(e['ROCAUC'])} & {pct(e['at_p05']['decoy_FPR'])}\\% & {pct(e['at_stacked']['R'])}\\% & {pct(e['at_stacked']['decoy_FPR'])}\\% & {f3(e['at_stacked']['F05'])} \\\\")
td += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(TAB, 'tab_decoys.tex'), 'w').write('\n'.join(td))

json.dump(NUM, open(os.path.join(HERE, 'numbers.json'), 'w'), indent=1, default=float)
print('figures + tables written'); print(json.dumps({k: NUM[k] for k in ('cand_auc_s1', 'p_extreme', 'verifier_extreme', 'koutli_auc_p', 'role_rates', 'stages_rates', 'behaviours_rates', 'operating_point', 'decision_counts', 'E7_met_claims', 'E7_met_claims_supported', 'E7_claims_no_evidence', 'E7_partial_offence', 'corpus_by_jur')}, indent=1, default=float))
