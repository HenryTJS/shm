"""自查：正文与论文集全文的 8-gram 重叠 + 波浪号 / 零宽字符。"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DOC = Path(r'd:\shm\docs\论文正文.md')
CAP = Path(r'd:\shm\docs\论文图注表注.md')   # 图注/表注不入正文（会议体例：排在图下表上）
REF = Path(r'd:\shm\icssd2025_fulltext.txt')

body = DOC.read_text(encoding='utf-8')
ref = REF.read_text(encoding='utf-8', errors='replace')

# 正文里的 `<!-- ... -->` 是排版位置标记（图/表插在哪、短标题），不算正文
body = re.sub(r'<!--.*?-->', '', body, flags=re.S)

# 图注/表注：只取 `## 图注` 到 `## 压缩备选` 之间 —— 文件头说明与压缩备选是工作稿，
# 不印进论文集，不能计入词数与查重。
cap_all = CAP.read_text(encoding='utf-8') if CAP.exists() else ''
cap = ''
if cap_all:
    _s = cap_all.index('## 图注')
    _e = cap_all.index('## 压缩备选') if '## 压缩备选' in cap_all else len(cap_all)
    cap = cap_all[_s:_e]

# 格式自查（两份都要查）
def fmt(t):
    tilde = t.count('~')
    zw = sum(1 for ch in t if 0x200B <= ord(ch) <= 0x200F or ord(ch) == 0xFEFF)
    return t.count('\n') + 1, tilde, zw

for _name, _t in (('正文', body), ('图注表注（计入部分）', cap)):
    _n, _tilde, _zw = fmt(_t)
    print('%s: lines=%d  tilde=%d  zerowidth=%d' % (_name, _n, _tilde, _zw))

# 会印进论文集的英文区 = 正文（Title 到 Acknowledgment）+ 图注/表注
en = body[body.index('# Title'):body.index('## Acknowledgment')]
if cap:
    en += '\n' + cap

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
def count(seg):
    seg = seg.replace('**', '')
    return (len(re.findall(r"[A-Za-z][A-Za-z\-']*", seg)), len(seg.split()))

for name, s, e, src in [('Abstract', '## Abstract', '## Keywords', body),
                        ('Introduction', '## 1. Introduction', '## 2. The Cross', body),
                        ('Conclusions', '## 5. Conclusions', '## Data availability', body),
                        ('Captions', None, None, cap)]:
    seg = src if s is None else src[src.index(s) + len(s):src.index(e)]
    latin, split = count(seg)
    print('%s words: latin=%d  whitespace=%d' % (name, latin, split))

# 正文本身（Title 到 Conclusions 结束，不含参考文献与图注/表注）
front = body[body.index('# Title'):body.index('## Data availability')]
latin, split = count(front)
print('Manuscript (title to conclusions): latin=%d  whitespace=%d' % (latin, split))
