# -*- coding: utf-8 -*-
"""看板数据包结构校验。

为什么需要它
------------
`export_dashboard.py --ds l1v3` 生成的是 JS（`window.SHM_DATA['L1-xx'] = {...};` ），
字段名错一个、数组长度差一个，前端都不会报错，只会**静默画歪或空白**。
本脚本把每条 `window.SHM_DATA` 反解成 JSON，逐项校验：

  1. 必备字段是否齐全（缺了前端读不到）
  2. 各波形数组长度是否都等于 `nfr`（差一个就会整条曲线错位）
  3. `fo` 的每个通道长度是否等于 `nfr`；`foCols` 与 `fo` 的键是否一致
  4. 取值域是否满足前端约定：`D/risk/eae/est` 在 0 至 1（前端按固定 0 至 1 轴画）
  5. 索引文件 `index_l1v3.js` 的 groups 是否为对象数组（前端用 g.gid）

用法
----
    python l1/check_dashboard_pkg.py                 # 默认查 l1v3
    python l1/check_dashboard_pkg.py --ds l1v2       # 查别的数据集
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DEFAULT_DIR = os.path.join(PROJ, 'dashboard', 'data')

SERIES_KEYS = ('D', 'risk', 'eae', 'est', 'lv', 'st', 'ael', 'aen', 't')
UNIT_KEYS = ('D', 'risk', 'eae', 'est')          # 前端按 0 至 1 轴绘制


def load_js_obj(path):
    """从 `window.X[a]={<json>};` 或 `window.X={...};` 中取出 JSON 对象。

    ⚠️ 不能用「最后一个 = 之后到最后一个 }」这种粗办法：
    文件里可能有多处赋值（如 `SHM_DATA = SHM_DATA || {};`），
    且 JSON 内部的字符串可能含 `=` / `}`。这里改为
    「最后一个 `= {` 起点 + 括号配平找闭合」。
    """
    with open(path, encoding='utf-8') as f:
        s = f.read()
    m = None
    for mm in re.finditer(r'=\s*\{', s):
        m = mm
    if m is None:
        raise ValueError('未找到 `= {` 形式的赋值：%s' % path)
    j = s.index('{', m.start())
    depth, end, instr, esc = 0, None, False, False
    for idx in range(j, len(s)):
        ch = s[idx]
        if instr:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end is None:
        raise ValueError('对象未闭合：%s' % path)
    return json.loads(s[j:end + 1])


def check_group(path, expect_nfr=None):
    pkg = load_js_obj(path)
    errs, warns = [], []
    nfr = pkg.get('nfr')
    if not isinstance(nfr, int) or nfr <= 0:
        errs.append('nfr 非法: %r' % nfr)
        return pkg, errs, warns
    if expect_nfr is not None and nfr != expect_nfr:
        warns.append('nfr=%d 与索引里的 %d 不一致' % (nfr, expect_nfr))

    for k in SERIES_KEYS:
        if k not in pkg:
            errs.append('缺字段 %s' % k)
        elif len(pkg[k]) != nfr:
            errs.append('%s 长度 %d != nfr %d' % (k, len(pkg[k]), nfr))

    for k in ('meta', 'warn', 'chans', 'foCols', 'fo'):
        if k not in pkg:
            errs.append('缺字段 %s' % k)

    fo = pkg.get('fo') or {}
    cols = pkg.get('foCols') or []
    if sorted(fo.keys()) != sorted(cols):
        errs.append('fo 的键与 foCols 不一致')
    for k, v in fo.items():
        if len(v) != nfr:
            errs.append('fo[%s] 长度 %d != nfr %d' % (k, len(v), nfr))

    for k in UNIT_KEYS:
        v = pkg.get(k)
        if not v:
            continue
        # ⚠️ 这些通道是以 ×1000 编码存储的（前端 dec(x,1000) 解码），
        #    必须先解码再判值域 —— 否则 1000 会被误判为超过 1.0。
        lo, hi = min(v) / 1000.0, max(v) / 1000.0
        if lo < -1e-9 or hi > 1.0 + 1e-9:
            errs.append('%s 超出 0 至 1（实测 %.4f 至 %.4f）'
                        '—— 前端按固定 0 至 1 轴绘制，会画出界外' % (k, lo, hi))

    ax = pkg.get('ax') or {}
    if 'ael' not in ax or 'rate' not in ax:
        errs.append('ax 缺 ael/rate')
    elif not (isinstance(ax['ael'], list) and len(ax['ael']) == 2):
        errs.append('ax.ael 应为 [lo, hi]')

    ch = pkg.get('chans') or []
    if not ch:
        errs.append('chans 为空（前端用不了通道健康表）')
    return pkg, errs, warns


def main():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')     # 否则 GBK 控制台报编码错
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='l1v3')
    ap.add_argument('--dir', default=DEFAULT_DIR)
    a = ap.parse_args()

    idxp = os.path.join(a.dir, 'index_%s.js' % a.ds)
    if not os.path.exists(idxp):
        print('缺索引文件', idxp)
        return 1
    ds = load_js_obj(idxp)
    print('数据集 %s：%s' % (ds.get('id'), ds.get('name')))
    print('path=%s  组数=%d' % (ds.get('path'), len(ds.get('groups') or [])))
    if not ds.get('groups') or not isinstance(ds['groups'][0], dict) \
            or 'gid' not in ds['groups'][0]:
        print('❌ index 的 groups 必须是含 gid 的对象数组（前端 fillGroupSel 用 g.gid）')
        return 1
    if not ds.get('path'):
        print('⚠️ index 缺 path，前端会用默认 data/')
    print()

    bad = 0
    for g in ds['groups']:
        gid = g['gid']
        p = os.path.join(a.dir, gid + '.js')
        if not os.path.exists(p):
            print('❌ %-7s 缺数据包 %s' % (gid, p))
            bad += 1
            continue
        pkg, errs, warns = check_group(p, g.get('nfr'))
        tag = '✅' if not errs else '❌'
        print('%s %-7s nfr=%5d fo=%2d通道 lvmax=%d warn=%d'
              % (tag, gid, pkg.get('nfr', -1), len(pkg.get('foCols') or []),
                 max(pkg.get('lv') or [0]), len(pkg.get('warn') or [])))
        for e in errs:
            print('     ✗', e)
            bad += 1
        for w in warns:
            print('     ⚠', w)
    print('\n%s' % ('全部通过' if not bad else '共 %d 处问题' % bad))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
