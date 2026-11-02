# PAN12 + Koutli PJ: corpus stats and overlap check (text only).
import glob, json, re, hashlib, os
import xml.etree.ElementTree as ET
from collections import Counter

def find(pat):
    hits = glob.glob(f'/kaggle/input/**/{pat}', recursive=True)
    assert hits, pat
    return hits[0]

norm = lambda s: re.sub(r'[^a-z0-9 ]', '', (s or '').lower()).strip()
h = lambda s: hashlib.md5(s.encode()).hexdigest()

def load(xml_path, pred_path):
    preds = {l.strip() for l in open(pred_path) if l.strip()}
    convs, pred_lines, n_msg = 0, set(), 0
    conv_pred = Counter()
    for _, c in ET.iterparse(xml_path):
        if c.tag != 'conversation': continue
        convs += 1
        has = False
        for m in c.iter('message'):
            n_msg += 1
            a, t = m.findtext('author'), norm(m.findtext('text'))
            if a in preds:
                has = True
                if len(t) > 25: pred_lines.add(h(t))
        conv_pred[has] += 1
        c.clear()
    return dict(conversations=convs, messages=n_msg, predators=len(preds),
                conversations_with_predator=conv_pred[True]), pred_lines

out = {}
out['pan12_train'], tr_lines = load(find('*training-corpus-2012-05-01.xml'), find('*training-corpus-predators-2012-05-01.txt'))
out['pan12_test'], te_lines = load(find('*test-corpus-2012-05-17.xml'), find('*groundtruth-problem1.txt'))
pan_lines = tr_lines | te_lines

overlap, sizes = [], []
for f in sorted(glob.glob('/kaggle/input/**/predators/**/*.txt', recursive=True)):
    lines = [norm(l) for l in open(f, errors='ignore')]
    lines = {h(l) for l in lines if len(l) > 25}
    sizes.append(len(lines))
    frac = len(lines & pan_lines) / max(len(lines), 1)
    overlap.append((os.path.basename(f), round(frac, 3)))
dup = [n for n, fr in overlap if fr >= 0.2]
out['koutli'] = dict(predators=len(overlap), median_long_lines=sorted(sizes)[len(sizes)//2],
                     overlapping_with_pan12=len(dup), unseen=len(overlap) - len(dup))
json.dump(out, open('/kaggle/working/eda.json', 'w'), indent=1)
json.dump(dup, open('/kaggle/working/koutli_overlap.json', 'w'))
print(json.dumps(out, indent=1))
