"""自查：正文与论文集全文的 8-gram 重叠 + 波浪号 / 零宽字符。"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DOC = Path(r'd:\shm\docs\论文正文.md')
REF = Path(r'd:\shm\icssd2025_fulltext.txt')

body = DOC.read_text(encoding='utf-8')
ref = REF.read_text(encoding='utf-8', errors='replace')

# 格式自查
tilde = body.count('~')
zw = sum(1 for ch in body if 0x200B <= ord(ch) <= 0x200F or ord(ch) == 0xFEFF)
print('FILE: lines=%d  tilde=%d  zerowidth=%d' % (body.count('\n') + 1, tilde, zw))

# 英文正文区（Title 到 Acknowledgment）
en = body[body.index('## Title'):body.index('## Acknowledgment')]

def ngrams(text, n=8):
    w = re.findall(r"[A-Za-z][A-Za-z\-']*", text.lower())
    return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}

a, b = ngrams(en), ngrams(ref)
inter = a & b
print('our 8-grams=%d   ref 8-grams=%d   overlap=%d' % (len(a), len(b), len(inter)))

# 只看重叠里较长的（12-gram）
a12, b12 = ngrams(en, 12), ngrams(ref, 12)
i12 = a12 & b12
print('12-gram overlap=%d' % len(i12))
for x in sorted(i12)[:30]:
    print('   [12] ' + x)

print('--- sample of 8-gram overlaps (first 25) ---')
for x in sorted(inter)[:25]:
    print('   ' + x)

# 词数
for name, s, e in [('Abstract', '## Abstract', '## Keywords'),
                   ('Introduction', '## 1. Introduction', '## 2. The Cross'),
                   ('Conclusions', '## 5. Conclusions', '## Acknowledgment')]:
    seg = body[body.index(s) + len(s):body.index(e)]
    seg = seg.replace('**', '')
    print('%s words=%d' % (name, len(re.findall(r"[A-Za-z][A-Za-z\-']*", seg))))
