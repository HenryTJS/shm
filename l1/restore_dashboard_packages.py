# -*- coding: utf-8 -*-
"""重建 L1 看板数据包（**离线复评口径**）。

为什么需要这个脚本
------------------
`l1/export_dashboard.py`（三批统一入口）在 2026-10-07「L1 在线口径下线」时被归档，
磁盘上已无 `.py` 源码；但 `l1/__pycache__/` 里仍留有当年打包逻辑的字节码：

  export_dashboard_l1.cpython-312.pyc      → 数据集 `l1`    C1 恒幅+FBG+DFOS（4 组）
  export_dashboard_l1_v2.cpython-312.pyc   → 数据集 `l1v2`  C2 恒幅+DFOS（9 组）
  export_dashboard_l1_v3.cpython-312.pyc   → 数据集 `l1v3`  C3 变幅 + C4 谱载（14 组）

本脚本**原样回放这些字节码**，因此看板包与当年发布的同构（字段、编码、文案都不变），
只补两处当年由已删模块提供、现在没人给的东西（见 `_recover_shim/`）：

  * 点级循环轴 `cyc` 与点级应变 `strain`（`evaluate_l1_degree*.py` 已归档）
  * C1 包的帧级 AE 事件数 `aen` 与 `meta.aeEvents`（同上 ⇒ 这里用 HI 的分箱计数回填）

**主曲线始终是离线复评 HI**（`evaluate_l1_hi_ae.hi_of_group()`，`HI = unity01(累积 AE)`），
不是任何在线 D(t)：L1 的在线口径已证伪下线，不要因为能看到逐帧回放就当成在线能力。

用法
----
    python l1/restore_dashboard_packages.py                  # 三批全部重建
    python l1/restore_dashboard_packages.py --batch c1       # 只重建 C1
    python l1/restore_dashboard_packages.py --batch c2 c3c4
    python l1/restore_dashboard_packages.py --dry-run        # 只检查字节码可用性

产物
----
    dashboard/data/{L1-xx}.js
    dashboard/data/index_l1.js  |  index_l1v2.js  |  index_l1v3.js
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import marshal
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
DASH = os.path.join(PROJ, 'dashboard', 'data')
SHIM = os.path.join(HERE, '_recover_shim')

BATCHES = {
    'c1': ('export_dashboard_l1',
           ['L1-03', 'L1-04', 'L1-05', 'L1-09']),
    'c2': ('export_dashboard_l1_v2',
           ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54',
            'L1-55', 'L1-56', 'L1-59', 'L1-60']),
    'c3c4': ('export_dashboard_l1_v3', []),          # 空 = 导出器自带清单
}
ORDER = ['c1', 'c2', 'c3c4']


# ------------------------------------------------------------------ 字节码回放
def _pyc_path(name):
    tag = f'cpython-{sys.version_info.major}{sys.version_info.minor}'
    return os.path.join(HERE, '__pycache__', f'{name}.{tag}.pyc')


def load_archived(name):
    """把归档的 .pyc 当模块执行（`__file__` 指回其原 .py 路径，导出器靠它定位数据）。"""
    p = _pyc_path(name)
    if not os.path.exists(p):
        raise SystemExit(f'找不到归档字节码: {p}')
    blob = open(p, 'rb').read()
    if blob[:4] != importlib.util.MAGIC_NUMBER:
        raise SystemExit(
            f'{p} 与当前解释器不匹配（需要 Python '
            f'{sys.version_info.major}.{sys.version_info.minor}）。'
            f'请用对应版本的解释器运行本脚本。')
    mod = types.ModuleType(name)
    mod.__file__ = os.path.join(HERE, name + '.py')
    mod.__name__ = name
    mod.__package__ = None
    code = marshal.loads(blob[16:])
    exec(code, mod.__dict__)
    return mod


def _cached_hi():
    """让 `evaluate_l1_hi_ae.hi_of_group` 每组只算一次（C1 的 AE 表最大 777 MB）。"""
    import evaluate_l1_hi_ae as h
    cache = {}
    orig = h.hi_of_group

    def wrapped(gid, *a, **k):
        if gid not in cache:
            cache[gid] = orig(gid, *a, **k)
        return cache[gid]

    h.hi_of_group = wrapped
    return cache


# ------------------------------------------------------------------ C1 补字段
def patch_c1(path, hi, frame_dt):
    """回填 `aen`（帧级 AE 事件数）与 `meta.aeEvents`，并给出 AE 面板纵轴上限。

    归档导出器这两项取自已删的 `evaluate_l1_degree.run_group()`（点级 AE 峰值），
    回放时只能给 NaN ⇒ `aen` 全 0、健康表 AE 事件数 0。这里用 HI 的
    500-cycle 分箱计数按**均匀分布**铺回 50-cycle 帧栅格（不改变总事件数）。
    """
    import numpy as np
    s = open(path, encoding='utf-8').read()
    head = 'window.SHM_DATA['
    i = s.index(head) + len(head)
    j = s.index(']=', i) + 2
    d = json.loads(s[j:].rstrip().rstrip(';'))

    nfr = int(d['nfr'])
    hit_bin = np.asarray(hi['hit_bin'], float)
    cum = np.concatenate([[0.0], np.cumsum(hit_bin)])
    bin_edge = np.arange(hit_bin.size + 1) * TMP_BIN_CYC
    cf = np.interp(np.arange(nfr + 1) * float(frame_dt), bin_edge, cum)
    aen = np.maximum(0, np.diff(np.round(cf)).astype(int))

    d['aen'] = aen.tolist()
    d['meta']['aeEvents'] = int(cum[-1])
    rate = float(np.percentile(aen, 95)) if aen.size else 1.0
    d['ax'] = {'ael': d.get('ax', {}).get('ael', [-1.0, 1.0]),
               'rate': max(1.0, round(rate * 1.2, 1))}
    d['labels']['ae'] = '声发射 · 事件率（离线复评）'

    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('window.SHM_DATA=window.SHM_DATA||{};')
        fh.write(f'window.SHM_DATA[{json.dumps(d["gid"], ensure_ascii=False)}]=')
        fh.write(json.dumps(d, ensure_ascii=False) + ';')
    return int(cum[-1]), int(aen.max()) if aen.size else 0


TMP_BIN_CYC = 500          # evaluate_l1_hi_ae.CYCS_BIN


# 这三组的 AE 采集窗口远短于全寿命（跨度 0.83 d / 12 min / 29 min），
# 600 s 帧下只有 13 / 2 / 5 帧，被打包器的「≥ 20 帧」门槛刷掉。
# 处理方式：按更细的帧长重建帧表（帧长写进包里的 `labels.ae`，不冒充当 600 s）。
SHORT_AE = {'L1-30': 60.0, 'L1-36': 60.0, 'L1-34': 30.0}


def _read_pkg(path):
    s = open(path, encoding='utf-8').read()
    head = 'window.SHM_DATA['
    i = s.index(head) + len(head)
    j = s.index(']=', i) + 2
    return json.loads(s[j:].rstrip().rstrip(';'))


def _write_pkg(path, d):
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('window.SHM_DATA=window.SHM_DATA||{};')
        fh.write(f'window.SHM_DATA[{json.dumps(d["gid"], ensure_ascii=False)}]=')
        fh.write(json.dumps(d, ensure_ascii=False) + ';')


def tidy(path, frame_s=None):
    """收尾：一次读、一次写，把归档导出器遗留的文案/多余层改干净。

    四件事（都是“删”，不是“加”）：
      1. 删论文参考层 `hiF` / `meta.hiF85` / `hiLeadPt` / `refs`（C1；其它批为空，不触发）
      2. 台架去掉 `, 无 FBG`（C2：“无 FBG”属通道信息，通道表里 `n=0` 已表达）
      3. 短 AE 窗口组（`SHORT_AE`）把硬编码的“600 s 帧”改成真实帧长 + 脚注声明窗口
      4. 无 FBG 有效通道的组（`nFo=0`）把 FBG 面板文案改成“无有效通道”    并顺带把数值数组编码成 base64（前端 `hydrateAll()` 解码）。返回（KB, 改动列表）。
    """
    from pkgcodec import encode_package
    d = _read_pkg(path)
    meta, labels, chg = d.get('meta') or {}, d.get('labels') or {}, []

    if 'hiF' in d or 'hiF85' in meta or 'hiLeadPt' in meta or meta.get('refs'):
        d.pop('hiF', None)
        for k in ('hiF85', 'hiLeadPt', 'refs'):
            meta.pop(k, None)
        chg.append('论文参考层')

    rig = d.get('rig')
    if isinstance(rig, str):
        new = rig.replace(', 无 FBG', '').replace('，无 FBG', '').replace('(无 FBG)', '').strip()
        if new != rig:
            d['rig'] = new
            chg.append('台架: ' + new)

    span = meta.get('span_h')
    if frame_s and span:
        old, new, hit = '600 s 帧', f'{frame_s:g} s 帧', False
        for key, val in list(labels.items()):
            if isinstance(val, str) and old in val:
                labels[key] = val.replace(old, new)
                hit = True
        for c in d.get('chans', []):
            if isinstance(c.get('mode'), str) and old in c['mode']:
                c['mode'] = c['mode'].replace(old, new)
                hit = True
        note = (f"⚠ 本组 AE 仅 {int(d.get('nfr', 0))} 帧（{frame_s:g} s 帧），覆盖试验时间 {span:g} h、"
                f"有效加载 {meta.get('load_h', 0):g} h，FBG 有效窗口 {meta.get('fbgWin', 0)} 个："
                f"曲线只代表 AE 窗口内的 HI_hit 形状，不代表全寿命进程，"
                f"勿与 C1/C2 的全寿命曲线并列解读。")
        foot = labels.get('foot', '').split(' ⚠')[0] + ' ' + note      # 幂等
        if labels.get('foot') != foot:
            labels['foot'] = foot
            hit = True
        if int(meta.get('nFo') or 1) == 0 and 'fo' in labels:
            labels['fo'] = '光纤光栅 · 无有效通道（原始仅 2 个瞬时快照，未构成剖面序列）'
            hit = True
        if hit:
            chg.append(f'帧长 {frame_s:g} s / 窗口声明')

    encode_package(d)
    _write_pkg(path, d)
    return round(os.path.getsize(path) / 1024, 1), chg


def prepare_short():
    """按 `SHORT_AE` 的帧长重建 AE 帧表与循环轴（原始 600 s 表备份为 `*.bak600s`）。"""
    import subprocess
    res = os.path.join(HERE, 'results')
    for g, fs in SHORT_AE.items():
        for pre in ('_l1ae_frames', '_l1cyc'):
            src = os.path.join(res, f'{pre}_{g}.npz')
            for bak in (src + '.bak600s', src + '.600s.bak'):
                if os.path.exists(bak):
                    continue
                if os.path.exists(src):
                    os.replace(src, bak)
            out = os.path.join(res, f'{pre}_{g}.npz')
            if os.path.exists(out):
                continue
            print(f'  [{g}] 重建 {os.path.basename(out)} @ {fs:g} s ...', end=' ', flush=True)
            script = 'ae_frames.py' if pre == '_l1ae_frames' else 'ae_cycle.py'
            r = subprocess.run([sys.executable, os.path.join(HERE, script),
                                g, '--frame-s', str(int(fs))],
                               cwd=PROJ, capture_output=True, text=True,
                               encoding='utf-8', errors='replace')
            if r.returncode != 0:
                raise SystemExit(f'{script} 失败:\n{r.stdout}\n{r.stderr}')
            print('完成')


def compact_file(path):
    """只做编码压缩（数值数组 → base64），不改文案；返回体积 KB。"""
    return tidy(path)[0]


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', nargs='*', default=None, choices=ORDER,
                    help='要重建的批次（默认全部）')
    ap.add_argument('--out', default=DASH, help='输出目录（默认 dashboard/data）')
    ap.add_argument('--dry-run', action='store_true', help='只检查字节码可加载')
    a = ap.parse_args()

    sys.path.insert(0, SHIM)
    sys.path.insert(0, PROJ)
    sys.path.insert(0, os.path.join(PROJ, 'dashboard'))   # pkgcodec.py
    sys.path.insert(0, HERE)
    os.makedirs(a.out, exist_ok=True)

    batches = a.batch or ORDER
    if a.dry_run:
        for b in batches:
            name, groups = BATCHES[b]
            load_archived(name)
            print(f'  [{b}] {name}.pyc 可加载 · {len(groups) or "自带"} 组')
        return 0

    hi_cache = _cached_hi()
    for b in batches:
        name, groups = BATCHES[b]
        mod = load_archived(name)
        if b == 'c3c4':
            print('--- 短 AE 窗口组（C4）的帧表预处理 ---')
            os.chdir(HERE)
            prepare_short()
        argv = [name] + (groups or []) + ['--out', a.out]
        sys.argv = argv
        print(f'\n########## 回放 {name}.pyc ##########')
        os.chdir(HERE)
        mod.main()

        if b == 'c1':
            print('--- 回填 C1 的帧级 AE 事件数（归档模块缺失项）---')
            for g in groups:
                fp = os.path.join(a.out, f'{g}.js')
                if not os.path.exists(fp):
                    print(f'  [{g}] 缺数据包 → 跳过')
                    continue
                hi = hi_cache.get(g)
                if hi is None:
                    print(f'  [{g}] 无 HI → 跳过')
                    continue
                tot, mx = patch_c1(fp, hi, 50.0)
                print(f'  {g}  AE 事件 {tot:>9d}  峰值/帧 {mx}')

        # 单遍收尾：删论文参考层 / 台架后缀 / 帧长文案 / 编码压缩（见 tidy()）
        import glob
        for fp in sorted(glob.glob(os.path.join(a.out, 'L1-*.js'))):
            g = os.path.basename(fp)[:-3]
            kb, chg = tidy(fp, SHORT_AE.get(g))
            print(f'  [收尾] {g:9s} {kb:6.0f} KB  {"、".join(chg) or "无改动"}')

    print('\n完成。看板入口在 dashboard/index.html（data/index_c1..c4.js）。')
    # 数据包 → C1–C4 四个数据集清单（分组真源是 datasets.L1_CAMPAIGNS）
    try:
        import build_campaign_index
        build_campaign_index.main()
    except Exception as e:  # noqa: BLE001
        print(f'  [warn] 重建 C1–C4 清单失败：{e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
