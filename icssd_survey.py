"""扫描 ICSSD 2025 议程集，抽每篇论文的标题页与摘要，用于分析「创新点是怎么构成的」。

（2026-10-14：议程集 PDF 已归档至 `docs/refs/`，全文提取缓存为 `icssd2025_fulltext.txt`。
 正文查重请用 `check_duplication.py`。）

用法:
    python icssd_survey.py             # 打印逐页首行 + 疑似标题页的摘要
    python icssd_survey.py dump        # 把每篇的标题页文本写到 docs/icssd_papers.txt
"""
import os
import re
import sys

import pdfplumber

sys.stdout.reconfigure(encoding='utf-8')

PDF = os.path.join('docs', 'refs', '9781643686301.pdf')
OUT = os.path.join('docs', 'icssd_papers.txt')


def main():
    dump = len(sys.argv) > 1 and sys.argv[1] == 'dump'
    with pdfplumber.open(PDF) as pdf:
        n = len(pdf.pages)
        print('总页数 %d' % n)
        pages = []
        for i, pg in enumerate(pdf.pages):
            t = pg.extract_text() or ''
            pages.append(t)
            if i % 40 == 0:
                print('  ... %d/%d' % (i, n))

    # 每页首行（运行页眉通常含论文标题）
    print('\n=== 逐页首行（用于定位论文边界）===')
    for i, t in enumerate(pages):
        ls = [ln.strip() for ln in t.splitlines() if ln.strip()]
        head = ls[0] if ls else '(空)'
        print('%4d | %s' % (i + 1, head[:110]))

    if not dump:
        return

    # 假定：首行与上一页不同 => 新论文的起始页
    starts = []
    prev = None
    for i, t in enumerate(pages):
        ls = [ln.strip() for ln in t.splitlines() if ln.strip()]
        head = ls[0] if ls else ''
        if head and head != prev:
            starts.append(i)
        prev = head

    with open(OUT, 'w', encoding='utf-8') as fh:
        for j, i in enumerate(starts):
            end = starts[j + 1] if j + 1 < len(starts) else len(pages)
            body = '\n'.join(pages[i:min(i + 3, end)])
            fh.write('\n' + '=' * 100 + '\n')
            fh.write('### 候选论文 %d：起始页 %d（至 %d）\n' % (j + 1, i + 1, end))
            fh.write('=' * 100 + '\n')
            fh.write(body[:6000] + '\n')
    print('\n共 %d 个候选起点，已写 %s' % (len(starts), OUT))


if __name__ == '__main__':
    main()
