"""Parse raw legal sources into one JSONL row per article/section for retrieval."""
import html, json, re, sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / 'datasets' / 'legal_raw'
OUT = Path(__file__).resolve().parent / 'legal_corpus.jsonl'
clean = lambda s: re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s))).strip()
rows = []

# European Union Directive 2011/93/EU: <p class="oj-ti-art">Article N</p><p class="oj-sti-art">Title</p> ...
eu = (RAW / 'eu_2011_93.html').read_text(encoding='utf8')
parts = re.split(r'<p[^>]*class="oj-ti-art"[^>]*>', eu)[1:]
for p in parts:
    num = clean(p.split('</p>', 1)[0])
    title = clean(re.search(r'class="oj-sti-art"[^>]*>(.*?)</p>', p, re.S).group(1))
    body = clean(p.split('</p>', 2)[2].split('<p class="oj-doc-end"')[0])
    if title in ('Transposition', 'Entry into force', 'Addressees'):  # final provisions, no legal substance
        continue
    rows.append(dict(id=f'EU-2011/93-{num.split()[-1]}', jurisdiction='European Union',
                     instrument='Directive 2011/93/EU', unit=num, title=title, text=body,
                     url='https://eur-lex.europa.eu/eli/dir/2011/93/oj/eng'))

# United States Code, Title 18 (Cornell LII): text of tab_default_1, minus legislative history.
for f in sorted(RAW.glob('us_18_*.html')):
    t = f.read_text(encoding='utf8')
    sec = f.stem.split('_')[-1]
    title = clean(re.search(r'<title>(.*?)</title>', t, re.S).group(1)).split('|')[0]
    body = clean(t[t.find('tab_default_1'):].split('>', 1)[1])
    body = re.sub(r'^U\.S\. Code (?:Notes )?prev \| next ', '', body)
    body = re.split(r' \((?:[A-Z][a-z]+\.? \d{1,2}, \d{4}|Added Pub\. L\.)', body)[0]
    rows.append(dict(id=f'US-18-{sec}', jurisdiction='United States', instrument='18 U.S.C.',
                     unit=f'§ {sec}', title=re.sub(r'^18 U\.S\. Code § \S+ - ', '', title).strip(),
                     text=body, url=f'https://www.law.cornell.edu/uscode/text/18/{sec}'))

# Vietnam Criminal Code 100/2015/QH13 (English translation, original 2015 text): Articles 141-147, 151.
vn = (RAW / 'vn_penal_2015.txt').read_text(encoding='utf8')
arts = re.split(r'\n\s*(?=Article \d+\. )', vn)
for a in arts:
    m = re.match(r'Article (\d+)\. (.*)', a)
    if m and int(m.group(1)) in (14, 15, 141, 142, 143, 144, 145, 146, 147, 151):
        head, _, body = a.partition('\n')
        if head.rstrip().endswith('aged'):  # title wraps onto the next line
            nxt, _, body = body.partition('\n'); head += ' ' + nxt.strip()
        body = re.split(r'\n\s*(?:Section|Chapter) [IVXLC\d]+', body)[0]
        rows.append(dict(id=f'VN-CC2015-{m.group(1)}', jurisdiction='Vietnam',
                         instrument='Criminal Code No. 100/2015/QH13', unit=f'Article {m.group(1)}',
                         title=re.sub(r'^Article \d+\. ', '', head).strip(), text=re.sub(r'\s+', ' ', body).strip(),
                         url='https://apolatlegal.com/wp-content/uploads/2023/12/Criminal-Code-2015.pdf'))

# Article 14 as amended by Law No. 12/2017/QH14 (English translation, english.luatvietnam.vn); the 2015 list is superseded.
for r in rows:
    if r['id'] == 'VN-CC2015-14':
        r.update(instrument='Criminal Code No. 100/2015/QH13 as amended by Law No. 12/2017/QH14', text=(
            '1. Preparation for a crime means finding, preparing tools, equipment, or other conditions for the crime, or establishing, '
            'joining a group of criminals, except in the circumstances specified in Article 109, Point a Clause 2 Article 113, or Point a '
            'Clause 2 Article 299 hereof. 2. A person who prepares for any of the crimes specified in Articles 108, 109, 110, 111, 112, 113, '
            '114, 115, 116, 117, 118, 119, 120, 121, 123, 134, 168, 169, 207, 299, 300, 301, 302, 303, 324 hereof shall bear criminal '
            'responsibility. 3. A person from 14 to under 16 years of age who prepares for any of the crimes specified in Article 123 and '
            'Article 168 hereof has to bear criminal responsibility.'),
            url='https://english.luatvietnam.vn/law-no-12-2017-qh14-dated-june-20-2017-of-the-national-assembly-on-amending-the-criminal-code-no-100-2015-qh13-115503-doc1.html')

with OUT.open('w') as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')

# Self-check: expected units present and non-trivial.
ids = {r['id'] for r in rows}
assert {'EU-2011/93-6', 'US-18-2422', 'VN-CC2015-146'} <= ids, ids
assert all(len(r['text']) > 120 for r in rows), [r['id'] for r in rows if len(r['text']) <= 150]
print(len(rows), 'units ->', OUT)
