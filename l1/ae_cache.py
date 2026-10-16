#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐 hit AE 数据的二进制缓存 —— 把 9.5 GB 的 SQLite 固化一次，之后毫秒级读取。

问题
----
`ae_io.read_hits()` 每次调用要跑三件重活：

1. `plan()` → `part_stats()`：对整张表 `SELECT COUNT(*)` 与 `MIN/MAX(Time)`
   （两次全表扫描，只为拿首末时间与段数）；
2. `SELECT Time, Chan, Amp FROM ae_data WHERE SetType=2 ORDER BY Time`
   （再扫一遍，还要排序）；
3. `np.asarray(rows, np.float64)`：把几百万个 Python tuple **逐行**转成数组。

实测 9 组的 `.pridb` 合计 **9.54 GB**（最大 `L156.pridb` 2.66 GB），
所以每跑一次定位脚本都要等几分钟 —— 反复排查根本不可行。

为什么不是 CSV
--------------
- 数据集自带的 `{gid}声发射.csv` 是**按时间聚合**的表
  （`time, n_hits, energy, amplitude`），**没有 `Chan` 列，做不了 TDOA 定位**；
- 真转成逐 hit CSV 会是 **3 GB 以上的纯文本**，解析要逐字符转浮点，
  比读 SQLite 更慢，而且磁盘占用翻几倍。文本格式不是加速方案。

方案
----
一次性把 `(Time, Chan, Amp)` 固化成**裸 `.npy`**，读取走 `np.load(mmap_mode='r')`：

| 项目 | 每次读 SQLite | 读 npy 缓存 |
| --- | --- | --- |
| 磁盘占用 | 9.54 GB | 每 hit 24 字节 |
| 单组读取耗时 | 数十秒到数分钟 | 毫秒级（只建内存映射） |
| 内存峰值 | 数 GB | 近乎零（按需分页） |

`plan()` 的结果（段数 / mode / 平移量 / 原因）一并缓存成 `.json`，
连 `COUNT(*)` 与 `MIN/MAX` 都不用再跑。

失效判定
--------
缓存与源文件强绑定：`{gid}.json` 里记录每段 `.pridb` 的
（文件名, 字节数, mtime）。任一不符即视为失效，自动回落 SQLite 并重建。

用法
    python l1/ae_cache.py --build                 # 全量构建（首次，慢）
    python l1/ae_cache.py --build --groups L1-49,L1-50
    python l1/ae_cache.py --info                  # 查看缓存状态与体积
    python l1/ae_cache.py --clear                 # 删除全部缓存

环境变量 `AE_NO_CACHE=1` 可强制绕过缓存（构建时内部即用它防止自噬）。
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'cache')
CACHE_COLS = ('Time', 'Chan', 'Amp')
ENV_OFF = 'AE_NO_CACHE'

# 默认构建这 9 组（`ae_locate.GROUPS`，即 C2 批里几何/声速可用者）
GROUPS = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54',
          'L1-55', 'L1-56', 'L1-59', 'L1-60']


def _paths(gid):
    return (os.path.join(CACHE_DIR, gid + '.npy'),
            os.path.join(CACHE_DIR, gid + '.json'))


def _sig(gid):
    """源文件指纹：[(文件名, 字节数, mtime), ...]（**只 glob，不查库**）。"""
    import ae_io
    out = []
    for fp in ae_io.pridb_parts(gid):
        st = os.stat(fp)
        out.append([os.path.basename(fp), int(st.st_size), int(st.st_mtime)])
    return out


def load(gid):
    """命中则返回 (mmap 数组, plan dict)，否则 None。"""
    if os.environ.get(ENV_OFF):
        return None
    dp, mp = _paths(gid)
    if not (os.path.exists(dp) and os.path.exists(mp)):
        return None
    try:
        with open(mp, encoding='utf-8') as fh:
            meta = json.load(fh)
    except Exception:                                   # noqa: BLE001
        return None
    if meta.get('sig') != _sig(gid):
        return None
    try:
        a = np.load(dp, mmap_mode='r')
    except Exception:                                   # noqa: BLE001
        return None
    return a, meta['plan']


def save(gid, a, p):
    """原子写入：先写 `.tmp.npy` 再 rename，避免半截文件被当成有效缓存。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    dp, mp = _paths(gid)
    tmp = dp + '.tmp.npy'
    np.save(tmp, a)
    os.replace(tmp, dp)
    with open(mp, 'w', encoding='utf-8') as fh:
        json.dump({'gid': gid, 'n': int(a.shape[0]),
                   'cols': list(CACHE_COLS),
                   'built': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'sig': _sig(gid), 'plan': p}, fh, ensure_ascii=False, indent=1)


def build(gid, verbose=True):
    """从 SQLite 读一遍并落盘。返回命中数（0 = 跳过/失败）。"""
    import ae_io
    dp, _mp = _paths(gid)
    t0 = time.time()
    os.environ[ENV_OFF] = '1'                           # 防自噬（构建时绕过缓存）
    try:
        a, p = ae_io._read_hits_sqlite(gid, cols=CACHE_COLS, verbose=verbose)
    finally:
        os.environ.pop(ENV_OFF, None)
    t1 = time.time()
    # 单段组的 plan 里 n 是体积估算值，用真实命中数覆盖后再落盘
    if len(p.get('parts', [])) == 1 and p['parts'][0].get('approx'):
        p['parts'][0]['n'] = int(a.shape[0])
        p['parts'][0]['approx'] = False
    save(gid, a, p)
    t2 = time.time()
    mb = a.shape[0] * a.shape[1] * 8 / 1e6
    if verbose:
        print('  %-7s %9d hits  读库 %6.1fs  写盘 %5.1fs  缓存 %7.1f MB'
              % (gid, a.shape[0], t1 - t0, t2 - t1, mb))
    return int(a.shape[0])


def info():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, '*.npy')))
    if not files:
        print('缓存目录为空：%s' % CACHE_DIR)
        return
    tot = 0
    print('%-8s %12s %10s %7s  %s' % ('gid', 'hits', 'MB', '有效', '构建时间'))
    for f in files:
        gid = os.path.basename(f)[:-4]
        _dp, mp = _paths(gid)
        ok, built = '-', '-'
        try:
            with open(mp, encoding='utf-8') as fh:
                meta = json.load(fh)
            ok = '是' if meta.get('sig') == _sig(gid) else '否（源已变）'
            built = meta.get('built', '-')
        except Exception:                               # noqa: BLE001
            ok = '否（无 json）'
        n = int(os.path.getsize(f)) // 24
        mb = os.path.getsize(f) / 1e6
        tot += os.path.getsize(f)
        print('%-8s %12s %10.1f %7s  %s' % (gid, format(n, ','), mb, ok, built))
    print('合计 %.2f GB' % (tot / 1e9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true', help='构建缓存')
    ap.add_argument('--info', action='store_true', help='查看缓存状态')
    ap.add_argument('--clear', action='store_true', help='删除全部缓存')
    ap.add_argument('--groups', default=','.join(GROUPS))
    a = ap.parse_args()
    gids = [g.strip() for g in a.groups.split(',') if g.strip()]

    if a.clear:
        n = 0
        for f in glob.glob(os.path.join(CACHE_DIR, '*.npy')) + \
                glob.glob(os.path.join(CACHE_DIR, '*.json')):
            os.remove(f)
            n += 1
        print('已删除 %d 个缓存文件' % n)
        return
    if a.info or not a.build:
        info()
        return

    print('构建二进制缓存 —— 只有首次慢，之后所有定位脚本都是毫秒级。')
    t0 = time.time()
    done = []
    for gid in gids:
        try:
            n = build(gid)
            if n:
                done.append(gid)
        except Exception as e:                          # noqa: BLE001
            print('  %-7s 失败：%s: %s' % (gid, type(e).__name__, e))
    print('完成 %d 组，用时 %.0f s' % (len(done), time.time() - t0))
    info()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
