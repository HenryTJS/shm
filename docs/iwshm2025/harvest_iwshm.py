# -*- coding: utf-8 -*-
"""IWSHM 2025 论文集：建索引、按主题筛选、下载选中的 PDF。

数据源
------
NDT.net 的 IWSHM 2025 会议论文集，全部开放获取：
    https://www.ndt.net/search/docs.php?MainSource=374&issue=1&rppoffset=0
每页 20 条，`rppoffset` 每 20 递增；论文 PDF 直链形如
    /article/iwshm2025/papers/IWSHM_<n>.pdf

两阶段
------
    python harvest_iwshm.py --index                 # 只建目录 → index.csv
    python harvest_iwshm.py --select --download     # 按主题筛选并下载 → pdfs/

筛选口径（--select）
--------------------
与本文选题（跨机构可比性、归一化寿命轴、无标度指标、AE 与导波、复合材料板）
相关的关键词分三组，命中「主题组 2 个以上」或「主题组 1 个 + 对象组 1 个」才入选：

  主题组  normalisation / normalization / scale / dimensionless / comparable /
          comparability / cross-institution / multi-institution / transfer /
          generalisation / generalization / domain shift / population-based /
          benchmark / open dataset / public dataset / reproducibility
  对象组  stiffness / composite / laminate / stiffened / panel / disbond /
          delamination / impact / fatigue / crack
  模态组  acoustic emission / guided wave / lamb wave / ultrasonic /
          fibre optic / fiber optic / strain / piezoelectric / transducer

下载行为
--------
PDF 直链在 ndt.net 上，一般可直接取得。脚本检查 `%PDF` 文件头，
失败原因记入 download_log.csv；已存在且大于 20 KB 的文件跳过。
"""
import argparse
import csv
import os
import re
import sys
import time

import requests

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = 'https://www.ndt.net'
LIST = BASE + '/search/docs.php?MainSource=374&issue=1&rppoffset=%d'
INDEX = os.path.join(HERE, 'index.csv')
PDFDIR = os.path.join(HERE, 'pdfs')
LOG = os.path.join(HERE, 'download_log.csv')
HDR = {'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/126.0 Safari/537.36')}
PAUSE = 0.5

THEME = ['normalis', 'normaliz', 'scale', 'dimensionless', 'comparab',
         'cross-institution', 'multi-institution', 'transfer', 'generalis',
         'generaliz', 'domain shift', 'population-based', 'benchmark',
         'open dataset', 'public dataset', 'reproduc']
OBJECT = ['stiffness', 'composite', 'laminate', 'stiffened', 'panel', 'disbond',
          'delamination', 'impact', 'fatigue', 'crack']
MODAL = ['acoustic emission', 'guided wave', 'lamb wave', 'ultrasonic',
         'fibre optic', 'fiber optic', 'strain', 'piezoelectric', 'transducer']


def get(url, stream=False):
    return requests.get(url, headers=HDR, timeout=30, stream=stream)


def harvest_index():
    rows, seen, offset = [], set(), 0
    while offset <= 2000:
        try:
            r = get(LIST % offset)
        except requests.RequestException as e:
            print('  第 %d 条起取失败：%s' % (offset, e))
            break
        if r.status_code != 200:
            print('  HTTP %d，停止' % r.status_code)
            break
        html = r.text
        # 以「详情页链接」为切点分块，每块里找标题与 PDF
        chunks = re.split(r'(?=<a href="/search/docs\.php\?id=\d+")', html)
        new = 0
        for ch in chunks:
            m = re.search(r'href="/search/docs\.php\?id=(\d+)"[^>]*>(.*?)</a>', ch, re.S)
            p = re.search(r'href="(/article/iwshm2025/papers/IWSHM_\d+\.pdf)"', ch)
            if not (m and p):
                continue
            did, title = m.group(1), re.sub(r'<[^>]+>', '', m.group(2))
            title = re.sub(r'\s+', ' ', title).replace('&amp;', '&').strip()
            pdf = p.group(1)
            if pdf in seen or len(title) < 12:
                continue
            seen.add(pdf)
            rows.append({'doc_id': did, 'title': title,
                         'pdf': BASE + pdf, 'file': ''})
            new += 1
        print('  offset %4d：本页新增 %2d，累计 %3d' % (offset, new, len(rows)))
        if new == 0:
            break
        offset += 20
        time.sleep(PAUSE)

    with open(INDEX, 'w', newline='', encoding='utf-8-sig') as fh:
        wr = csv.DictWriter(fh, fieldnames=['doc_id', 'title', 'pdf', 'file'])
        wr.writeheader()
        wr.writerows(rows)
    print('索引 -> %s（%d 篇）' % (INDEX, len(rows)))
    return rows


def score(title):
    t = title.lower()
    th = sum(1 for k in THEME if k in t)
    ob = sum(1 for k in OBJECT if k in t)
    mo = sum(1 for k in MODAL if k in t)
    return th, ob, mo


def select(rows):
    hits = []
    for r in rows:
        th, ob, mo = score(r['title'])
        # 规则：主题 >=2；或 主题 1 且（对象或模态）>=1
        if th >= 2 or (th >= 1 and (ob >= 1 or mo >= 1)):
            r['hits'] = '%d/%d/%d' % (th, ob, mo)
            hits.append(r)
    hits.sort(key=lambda r: (-int(r['hits'].split('/')[0]),
                             -int(r['hits'].split('/')[1]),
                             r['title']))
    return hits


def safe_name(r):
    n = re.search(r'IWSHM_(\d+)\.pdf', r['pdf'])
    return 'IWSHM_%s.pdf' % (n.group(1) if n else r['doc_id'])


def download(rows):
    os.makedirs(PDFDIR, exist_ok=True)
    log = []
    for i, r in enumerate(rows, 1):
        path = os.path.join(PDFDIR, safe_name(r))
        note = ''
        if os.path.exists(path) and os.path.getsize(path) > 20000:
            note = 'exists'
        else:
            try:
                with get(r['pdf'], stream=True) as resp:
                    if resp.status_code != 200:
                        note = 'HTTP %d' % resp.status_code
                    else:
                        buf = bytearray()
                        for c in resp.iter_content(65536):
                            buf += c
                            if len(buf) > 60 * 1024 * 1024:
                                note = 'too-large'
                                break
                        if not note:
                            if bytes(buf[:5]).startswith(b'%PDF'):
                                open(path, 'wb').write(buf)
                                note = 'ok'
                            else:
                                note = 'not-pdf'
            except requests.RequestException as e:
                note = 'err:%s' % type(e).__name__
        r['file'] = os.path.relpath(path, HERE) if note in ('ok', 'exists') else ''
        print('  %2d/%d %-22s %s' % (i, len(rows), safe_name(r), note))
        log.append({'pdf': r['pdf'], 'note': note, 'file': r['file']})
        time.sleep(PAUSE)
    with open(LOG, 'w', newline='', encoding='utf-8-sig') as fh:
        wr = csv.DictWriter(fh, fieldnames=['pdf', 'note', 'file'])
        wr.writeheader()
        wr.writerows(log)
    ok = sum(1 for x in log if x['file'])
    print('\n下载成功 %d / %d' % (ok, len(rows)))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', action='store_true')
    ap.add_argument('--select', action='store_true')
    ap.add_argument('--download', action='store_true')
    a = ap.parse_args()
    if not (a.index or a.select):
        ap.print_help()
        return 0

    rows = None
    if a.index:
        rows = harvest_index()
    if a.select:
        if rows is None:
            rows = list(csv.DictReader(open(INDEX, encoding='utf-8-sig')))
        hits = select(rows)
        print('\n筛选命中 %d / %d 篇' % (len(hits), len(rows)))
        for r in hits:
            print('  [%s] %s' % (r.get('hits', '?'), r['title'][:96]))
        if a.download:
            print()
            download(hits)
        else:
            with open(os.path.join(HERE, 'selected.csv'), 'w',
                      newline='', encoding='utf-8-sig') as fh:
                wr = csv.DictWriter(fh, fieldnames=['doc_id', 'title', 'pdf', 'hits'])
                wr.writeheader()
                for r in hits:
                    wr.writerow({k: r.get(k, '') for k in wr.fieldnames})
            print('\n清单 -> selected.csv（未下载，加 --download 才会下）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
