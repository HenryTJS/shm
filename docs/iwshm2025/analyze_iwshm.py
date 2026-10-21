# -*- coding: utf-8 -*-
"""IWSHM 2025 体裁基线：从 39 篇选读论文里抽可比的结构与证据特征。

判据（全部写死，可逐条复核）
----------------------------
公开数据集   文本中出现 "publicly available dataset" / "open dataset" /
             "open-source dataset" / "benchmark dataset" / "data available at" /
             "data set is available" / 已知公开集名（NASA、CWRU、XJTU-SY、ReMAP、
             PHM 20xx、SIMA、LANL、Z24、Open Guided Waves、IWSHM、MFPT 等）
在线或因果   "real-time" / "real time" / "online" / "on-line" / "causal" /
             "streaming" / "in-situ" / "in situ"
跨结构泛化   "transfer learn" / "domain adapt" / "generaliz" / "generalis" /
             "cross-structure" / "across structures" / "unseen" /
             "population-based" / "cross-domain" / "different specimen" /
             "different structures"
限制声明     全文出现 "limitation"（不看章节位置，只看有没有）
背景对比     全文出现 "baseline" / "benchmark" / "compared with" / "compared to"

结构量
------
页数、词数、图数（去重后的 Fig./Figure 编号）、表数（Table 编号）、
参考文献数（末尾匹配 ^[n] 或 ^n. 的条目数）

用法
    python analyze_iwshm.py
产物
    体裁基线明细.csv   每篇一行
    体裁基线.md        汇总表 + 逐篇表
"""
import csv
import os
import re
import statistics as st
import sys

import pdfplumber

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
PDFDIR = os.path.join(HERE, 'pdfs')

PUBLIC_PHRASE = [
    'publicly available dataset', 'public dataset', 'public data set',
    'open dataset', 'open-source dataset', 'open source dataset',
    'benchmark dataset', 'benchmark data set', 'data available at',
    'dataset is available', 'data set is available', 'available in a public',
    'make the dataset', 'shared dataset', 'released dataset',
]
# 已知公开集名：只有与 dataset / data set / benchmark 在 100 字符内共现才算
DATASET_NAME = ['nasa', 'cwru', 'xjtu-sy', 'remap', 'phm 20', 'phm20', 'lanl',
                'los alamos', 'z24', 'open guided waves', 'mfpt', 'i-40 bridge']
ONLINE = ['real-time', 'real time', 'online monitoring', 'on-line monitoring',
          'online detection', 'online estimation', 'causal', 'streaming',
          'in-service', 'near real-time', 'online damage']
CROSS = ['transfer learn', 'domain adapt', 'generaliz', 'generalis',
         'cross-structure', 'across structures', 'unseen', 'population-based',
         'cross-domain', 'different specimen', 'different structures',
         'multiple structures', 'different panels', 'across specimens',
         'other structures']
BASE = ['baseline', 'benchmark', 'compared with', 'compared to', 'comparison with']


def near(t, name, anchors=('dataset', 'data set', 'benchmark'), win=100):
    for m in re.finditer(re.escape(name), t):
        seg = t[max(0, m.start() - win):m.end() + win]
        if any(a in seg for a in anchors):
            return True
    return False


def flags(text):
    t = text.lower()
    hits = {k: [] for k in ('public', 'online', 'cross', 'limit')}
    for k in PUBLIC_PHRASE:
        if k in t:
            hits['public'].append(k)
    for k in DATASET_NAME:
        if near(t, k):
            hits['public'].append(k)
    for k in ONLINE:
        if k in t:
            hits['online'].append(k)
    for k in CROSS:
        if k in t:
            hits['cross'].append(k)
    if 'limitation' in t:
        hits['limit'].append('limitation')
    return {
        'public': int(bool(hits['public'])),
        'online': int(bool(hits['online'])),
        'cross': int(bool(hits['cross'])),
        'limit': int(bool(hits['limit'])),
        'base': int(any(k in t for k in BASE)),
        'match': ' | '.join('%s:%s' % (k, ','.join(v[:2]))
                            for k, v in hits.items() if v),
    }


def counts(text):
    figs = {int(m) for m in re.findall(r'(?:Fig\.|Figure)\s*(\d{1,2})', text)}
    tabs = {int(m) for m in re.findall(r'Table\s*(\d{1,2})', text)}
    refs = len(re.findall(r'^\s*\[\d{1,3}\]', text, re.M))
    if refs < 5:
        refs = len(re.findall(r'^\s*\d{1,3}\.\s+[A-Z]', text, re.M))
    words = len(re.findall(r"[A-Za-z][A-Za-z\-']+", text))
    return len(figs), len(tabs), refs, words


def main():
    files = sorted(f for f in os.listdir(PDFDIR) if f.endswith('.pdf'))
    rows = []
    for i, f in enumerate(files, 1):
        path = os.path.join(PDFDIR, f)
        try:
            with pdfplumber.open(path) as pdf:
                pages = len(pdf.pages)
                text = '\n'.join((p.extract_text() or '') for p in pdf.pages)
        except Exception as e:
            print('  [%2d/%d] %-18s 解析失败 %s' % (i, len(files), f, type(e).__name__))
            continue
        fl = flags(text)
        nfig, ntab, nref, nw = counts(text)
        title = ''
        first = text.strip().split('\n')[0] if text.strip() else ''
        title = re.sub(r'\s+', ' ', first)[:100]
        rows.append(dict(pdf=f, pages=pages, words=nw, figs=nfig, tabs=ntab,
                         refs=nref, public=fl['public'], online=fl['online'],
                         cross=fl['cross'], limit=fl['limit'], base=fl['base'],
                         match=fl['match'], title=title))
        print('  [%2d/%d] %-18s 页%2d 词%5d 图%2d 表%2d 参考%2d | 公开%d 在线%d 跨结构%d 限制%d'
              % (i, len(files), f, pages, nw, nfig, ntab, nref,
                 fl['public'], fl['online'], fl['cross'], fl['limit']))

    with open(os.path.join(HERE, '体裁基线明细.csv'), 'w', newline='',
              encoding='utf-8-sig') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    n = len(rows)
    pct = lambda k: 100.0 * sum(r[k] for r in rows) / n
    med = lambda k: st.median(r[k] for r in rows)

    lines = ['# IWSHM 2025 体裁基线（%d 篇选读）\n' % n,
             '判据见 `analyze_iwshm.py` 头部，全部为文本关键词匹配，可逐条复核。\n',
             '## 1. 汇总\n',
             '| 指标 | 数值 |', '| --- | --- |',
             '| 论文数 | %d |' % n,
             '| 页数中位（范围） | %g（%d 至 %d） |'
             % (med('pages'), min(r['pages'] for r in rows), max(r['pages'] for r in rows)),
             '| 词数中位 | %g |' % med('words'),
             '| 图数中位 | %g |' % med('figs'),
             '| 表数中位 | %g |' % med('tabs'),
             '| 参考文献中位 | %g |' % med('refs'),
             '| 提到公开数据集 | %d 篇（%.0f%%） |' % (sum(r['public'] for r in rows), pct('public')),
             '| 出现在线/实时/因果 | %d 篇（%.0f%%） |' % (sum(r['online'] for r in rows), pct('online')),
             '| 出现跨结构/迁移/泛化 | %d 篇（%.0f%%） |' % (sum(r['cross'] for r in rows), pct('cross')),
             '| 出现 limitation | %d 篇（%.0f%%） |' % (sum(r['limit'] for r in rows), pct('limit')),
             '| 出现 baseline/对比 | %d 篇（%.0f%%） |' % (sum(r['base'] for r in rows), pct('base')),
             '', '## 2. 逐篇\n',
             '| 文件 | 页 | 词 | 图 | 表 | 参考 | 公开集 | 在线 | 跨结构 | 限制 | 对比 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in sorted(rows, key=lambda x: x['pdf']):
        mark = lambda v: 'Y' if v else '-'
        lines.append('| %s | %d | %d | %d | %d | %d | %s | %s | %s | %s | %s |'
                     % (r['pdf'].replace('.pdf', ''), r['pages'], r['words'], r['figs'],
                        r['tabs'], r['refs'], mark(r['public']), mark(r['online']),
                        mark(r['cross']), mark(r['limit']), mark(r['base'])))
    lines.append('\n## 3. 命中关键词（可逐条复核）\n')
    for r in sorted(rows, key=lambda x: x['pdf']):
        if r.get('match'):
            lines.append('- **%s**：%s' % (r['pdf'].replace('.pdf', ''), r['match']))
    open(os.path.join(HERE, '体裁基线.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n%d 篇完成；汇总 -> 体裁基线.md' % n)
    print('公开集 %d%% | 在线 %d%% | 跨结构 %d%% | 限制 %d%% | 对比 %d%%'
          % (pct('public'), pct('online'), pct('cross'), pct('limit'), pct('base')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
