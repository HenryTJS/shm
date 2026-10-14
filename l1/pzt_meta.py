"""解析各组 PZT 的 structure_info.m：损伤数据库（位置/尺寸/类别）与 PZT 坐标。

只读。
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

R = 'E:\\l1'
GROUPS = ['L1-03', 'L1-04', 'L1-05', 'L1-06', 'L1-09', 'L1-13', 'L1-14', 'L1-22',
          'L1-23', 'L1-24', 'L1-25', 'L1-27', 'L1-29', 'L1-30', 'L1-31', 'L1-34',
          'L1-35', 'L1-36', 'L1-41', 'L1-44']

ROW = re.compile(
    r"'([^']*)'\s+'([^']*)'\s+\[\s*([-\d.eE]+|NaN)\s+([-\d.eE]+|NaN)\s*\]\s+"
    r"\[\s*([-\d.eE]+|NaN)\s+([-\d.eE]+|NaN)\s*\]\s+([-\d.eE]+|NaN)\s+"
    r"([-\d.eE]+|NaN)\s+'([^']*)'")


def pzt_dirs(g):
    d = os.path.join(R, g)
    if not os.path.isdir(d):
        return []
    return [e.path for e in sorted(os.scandir(d), key=lambda x: x.name)
            if e.is_dir() and 'PZT' in e.name
            and ReAL_STATE(e.path)]


def ReAL_STATE(p):
    return any(e.is_dir() and e.name.startswith('State_') for e in os.scandir(p))


print('%-8s %-9s %5s %5s %6s %8s  %s' % ('组', 'PZT', '状态', '有位置', '尺寸非NaN', '类别分布', '坐标(mm)'))
print('-' * 108)

summary = {}
for g in GROUPS:
    dirs = pzt_dirs(g)
    if not dirs:
        print('%-8s  （无 PZT）' % g)
        continue
    ndir = 0
    for d in dirs:
        p = os.path.join(d, 'structure_info.m')
        if not os.path.exists(p):
            print('%-8s %-9s  缺 structure_info.m' % (g, os.path.basename(d)))
            continue
        t = open(p, encoding='utf-8', errors='replace').read()
        # PZT 坐标
        m = re.search(r'PZT_info_geo\s*=\s*(.{0,400}?)\];', t, re.S)
        ncoord = 0
        if m:
            nums = re.findall(r'-?\d+\.?\d*', m.group(1))
            ncoord = len(nums) // 2
        rows = ROW.findall(t)
        nstate = len({r[0] for r in rows})
        npos = sum(1 for r in rows if r[2] != 'NaN' or r[3] != 'NaN')
        nsize = sum(1 for r in rows if r[6] != 'NaN')
        cats = {}
        for r in rows:
            cats[r[8]] = cats.get(r[8], 0) + 1
        cat_s = ' '.join('%s=%d' % (k, v) for k, v in sorted(cats.items()))
        print('%-8s %-9s %5d %5d %6d  %-28s %d 个' % (
            g, os.path.basename(d), nstate, npos, nsize, cat_s[:28], ncoord))
        ndir += 1
        summary.setdefault(g, []).append((os.path.basename(d), nstate, npos, nsize, cats))
    print()

print('=' * 108)
print('汇总：各组的状态总数与是否有损伤位置')
for g in GROUPS:
    if g not in summary:
        continue
    tot = sum(s[1] for s in summary[g])
    npos = sum(s[2] for s in summary[g])
    nsz = sum(s[3] for s in summary[g])
    allcats = {}
    for s in summary[g]:
        for k, v in s[4].items():
            allcats[k] = allcats.get(k, 0) + v
    print('  %-8s PZT目录 %d 个 · 状态合计 %3d · 有几何位置 %3d · 有尺寸 %3d · 类别 %s'
          % (g, len(summary[g]), tot, npos, nsz, allcats))
