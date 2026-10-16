#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自查：**正文内部**的重复 —— 章节两两的 8-gram 复用率 + 单节内的自我重复。

与 `check_duplication.py` 的分工：
  · `check_duplication.py` 查的是「我们有没有抄论文集里的别人」（外部）；
  · 本脚本查的是「我们有没有把同一件事在论文里写两遍」（内部）。

存在的理由：2026-10-16 发现 Conclusions 是摘要的加长版（8.9% / 4.3% 双向复用，
其中「produced no alarm for 40% … factor of 3.3」等是逐字重复）。这类问题
逐字查重查不出（不是抄别人），肉眼读也容易漏，只有自己跟自己比才看得见。

用法
----
    python check_repetition.py                    # 默认 n=8，阈值 2%
    python check_repetition.py --n 6 --min 0.03   # 更细的 n-gram、更严的阈值

输出
----
  1. 章节两两复用率（只报 >= 阈值的），并列出共用的 8-gram
  2. 单节内部的自我重复
  3. 各节词数与占正文比例

判读经验（本文实测）
--------------------
  · Abstract ↔ Conclusions 是最高危的一对：两个章节结构天然平行，最容易写成一样。
    健康值 < 1%；> 3% 就该重写结论。
  · 章节之间 2% 上下、且共用片段是「定义性短语」或「数据集描述」，属正常。
  · §4 Discussion ↔ §5 Conclusions 有 2% 左右也正常（结论要综合讨论）。
  · 单节内部的重复（同一事实在相邻两段各说一遍）一律是错的，要删一处。
"""
from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parent / 'docs' / '论文正文.md'
HEAD = re.compile(r'^#{1,3} (.+)$')


def load() -> tuple[list[str], dict[str, list[str]]]:
    raw = DOC.read_text(encoding='utf-8')
    raw = re.sub(r'<!--.*?-->', '', raw, flags=re.S)      # 排版位置标记不算正文
    en = raw[raw.index('# Title'):raw.index('## Data availability')]

    order: list[str] = []
    buf: dict[str, list[str]] = {}
    cur = None
    for line in en.splitlines():
        m = HEAD.match(line)
        if m:
            cur = m.group(1).strip()[:24]
            if cur not in buf:
                buf[cur] = []
                order.append(cur)
            continue
        if cur:
            buf[cur].extend(re.findall(r"[A-Za-z][A-Za-z\-']*", line.lower()))
    return order, buf


def grams(words: list[str], n: int) -> set[str]:
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8, help='n-gram 长度，默认 8')
    ap.add_argument('--min', type=float, default=0.02, help='复用率阈值，默认 0.02')
    ap.add_argument('--min-words', type=int, default=40,
                    help='短于这么多词的节不参与比对')
    a = ap.parse_args()

    order, buf = load()
    G = {k: grams(buf[k], a.n) for k in order}

    print('=' * 74)
    print('章节两两复用率 >= %.1f%%（n=%d）' % (100 * a.min, a.n))
    print('=' * 74)
    hits = []
    for b, x in itertools.permutations(order, 2):
        if len(buf[b]) < a.min_words or len(buf[x]) < a.min_words:
            continue
        f = len(G[b] & G[x]) / len(G[b])
        if f >= a.min:
            hits.append((f, b, x))
    if not hits:
        print('  (无)')
    for f, b, x in sorted(hits, reverse=True):
        print('\n  %-24s <- %-24s %5.1f%%' % (b, x, 100 * f))
        for g in sorted(G[b] & G[x]):
            print('       · ' + g)

    print('\n' + '=' * 74)
    print('单节内部的自我重复（一律应删掉一处）')
    print('=' * 74)
    found = False
    for k in order:
        v = buf[k]
        if len(v) < max(a.min_words, 60):   # 过短的节（如作者单位）不算
            continue
        pos: dict[str, list[int]] = {}
        for i in range(len(v) - a.n + 1):
            pos.setdefault(' '.join(v[i:i + a.n]), []).append(i)
        for g, p in sorted(pos.items()):
            if len(p) > 1:
                print('  §%s : %s' % (k, g))
                found = True
    if not found:
        print('  (无)')

    print('\n' + '=' * 74)
    print('各章词数与占正文比例（子节已并入所属章）')
    print('=' * 74)
    chap: dict[str, int] = {}
    for k in order:
        m = re.match(r'^(\d)\.', k)
        if m:
            chap[m.group(1)] = chap.get(m.group(1), 0) + len(buf[k])
    total = sum(chap.values()) or 1
    names = {'1': '1. Introduction', '2': '2. Protocol', '3': '3. Results',
             '4': '4. Discussion', '5': '5. Conclusions'}
    for d in sorted(chap):
        print('  %-24s %5d  %5.1f%%'
              % (names.get(d, d), chap[d], 100 * chap[d] / total))
    print('  %-24s %5d' % ('合计（1..5 章）', total))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
