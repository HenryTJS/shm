"""PZT 管线耗时基准：定位瓶颈在哪一段（目录扫描 / loadmat / 滤波 / 互相关）。

只读，不写任何产物。
"""
import importlib.util
import os
import sys
import time

import numpy as np
import scipy.io as sio

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PI = _load('pzt_index', os.path.join(HERE, 'pzt_index.py'))
PC = _load('pzt_crosslab', os.path.join(HERE, 'pzt_crosslab.py'))

GID = sys.argv[1] if len(sys.argv) > 1 else 'L1-44'
FREQ = '250kHz_5cycles'


def t(fn, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
    return (time.perf_counter() - t0) / n, r


# ---- 1. 目录扫描 + structure_info 解析
dt_scan, st = t(lambda: PC.dam_order(GID))
print('%-34s %8.1f ms   （%d 个 DAM 状态）' % ('① dam_order（扫目录+解析）', dt_scan * 1e3, len(st)))

# ---- 2. 单个 .mat 的 loadmat
gdir = os.path.join(HERE, GID)
s0 = st[len(st) // 2]
p0 = os.path.join(gdir, s0[1], s0[2], FREQ, 'Actionneur1', 'measured_data_rep_1.mat')
dt_load, _ = t(lambda: sio.loadmat(p0), 50)
sz = os.path.getsize(p0)
print('%-34s %8.1f ms   （文件 %.0f KB）' % ('② loadmat 单个 .mat（×50 平均）', dt_load * 1e3, sz / 1024))

# ---- 3. 一个状态的完整 signals()
ba = PI.bandpass(250.0)
dt_sig, sig3 = t(lambda: PC.signals(gdir, s0[1], s0[2], ba), 5)
sig = sig3[0]
print('%-34s %8.1f ms   （含 8 次 loadmat + 56 次滤波）' % ('③ signals() 一个状态（×5 平均）', dt_sig * 1e3))

# ---- 4. 分解：8 次 loadmat vs 56 次滤波
def only_load():
    for k in range(1, 9):
        sio.loadmat(os.path.join(gdir, s0[1], s0[2], FREQ, 'Actionneur%d' % k,
                                 'measured_data_rep_1.mat'))


dt_8, _ = t(only_load, 5)


def only_exists():
    for k in range(1, 9):
        os.path.exists(os.path.join(gdir, s0[1], s0[2], FREQ, 'Actionneur%d' % k,
                                    'measured_data_rep_1.mat'))


dt_e, _ = t(only_exists, 20)
tr = sio.loadmat(p0)['Time_Response']
x = tr[:, 2]


def only_filt():
    for _ in range(56):
        PI.prep(x, ba)


dt_f, _ = t(only_filt, 5)
print('%-34s %8.1f ms' % ('   其中：8 次 loadmat', dt_8 * 1e3))
print('%-34s %8.1f ms   <== 逐次 stat，每次 %.1f ms'
      % ('   其中：8 次 os.path.exists', dt_e * 1e3, dt_e / 8 * 1e3))
print('%-34s %8.1f ms' % ('   其中：56 次滤波', dt_f * 1e3))
print('%-34s %8.1f ms' % ('   未归属（signals 其余开销）',
                          (dt_sig - dt_8 - dt_f) * 1e3))

# ---- 5. 互相关
ref = sig[(1, 2)]
def only_xc():
    for _ in range(56):
        PC.delay_us(sig[(1, 2)], ref)


dt_x, _ = t(only_xc, 5)
print('%-34s %8.3f ms' % ('④ delay_us ×56', dt_x * 1e3))

# ---- 6. 外推
N_STATES = 1972
per = dt_sig + dt_x
print()
print('单状态合计 ≈ %.1f ms  ⇒  %d 个状态（全量 17 组）≈ %.1f 分钟'
      % (per * 1e3, N_STATES, per * N_STATES / 60))
print('若去掉逐次 os.path.exists（改用 try/except 直接打开）：')
print('    单状态 ≈ %.1f ms  ⇒  全量 ≈ %.1f 分钟（提速 %.1f×）'
      % ((per - dt_e) * 1e3, (per - dt_e) * N_STATES / 60, per / max(per - dt_e, 1e-9)))
