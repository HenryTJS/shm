# -*- coding: utf-8 -*-
"""L1 新组（C3 变幅 / C4 谱载）AE **帧级聚合** —— 把 1.96 亿条 hit 压成可分析的帧表。

为什么需要它
------------
`l1/ae_dta.py` 能把 `.DTA` 解成逐条 hit，但新组单卷可达 4400 万条，
逐条落盘既慢又占空间（十几 GB）。本模块**流式**扫一遍，只保留**帧级统计**：
默认每 600 s 一帧，全 14 组合计约 3 千行 → 几十 KB，可直接进分析与看板。

帧指标（全部存**绝对量**，归一化留到分析阶段做，保持口径灵活）
------------------------------------------------------------
计数类：n / n_ch1 / n_ch2
幅值类：amp_mean amp_p50 amp_p90 amp_max amp_min / frac_ge80 frac_ge90 / a95_a50
分布类：b_value（Gutenberg-Richter 最大似然，dB 口径）/ amp_entropy（1 dB 分箱）
能量类：ener_sum（ABS-ENERGY, aJ）ener_per_hit / sig_sum sig_per_hit（SIG STRENGTH, pV·s）
时间类：dur_p50 dur_mean（µs）
机制类：ra_p50（= RISE/AMP）/ af_p50（= COUN/DURATION×1000, kHz）
        —— 与 `l1/ae_raf.py` 的老组口径一致（老组 AMP 是 µV 要转 dB，本格式 AMP 直接是 dB）

用法
----
    python l1/ae_frames.py --groups L1-25 L1-31        # 指定组
    python l1/ae_frames.py                             # 全部有 .DTA 的组
    python l1/ae_frames.py --frame-s 300               # 改帧长
输出：results/_l1ae_frames_{gid}.npz（含 frame 级数组 + 元信息）
"""

import collections
import datetime as _dt
import glob
import io
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ae_dta import iter_messages, _rtot, CHID_NBYTES      # noqa: E402

RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)

FRAME_S = 600.0
A_THRESHOLD = 62.0            # 幅值阈值下限（dB），b 值基线；L1-41 为 67，组内自动取实测最小值
BINS = np.arange(60.0, 101.0, 1.0)     # 熵的分箱

# AE 时钟相对 FBG 日历的实测平移（天）：C4 谱载批慢 8 天，C3 变幅批同步
CLOCK_SHIFT = {'L1-06': 0, 'L1-13': 0, 'L1-14': 0, 'L1-24': 0,
               'L1-25': 8, 'L1-27': 8, 'L1-29': 8, 'L1-30': 8, 'L1-31': 8,
               'L1-34': 8, 'L1-35': 8, 'L1-36': 8, 'L1-41': 8, 'L1-44': 8}

# 只需这几个特征；kind: u8/u16/i32/f32
_WANT = {'RISE': 1, 'COUN': 3, 'DURATION': 5, 'AMP': 6,
         'SIG STRENGTH': 20, 'ABS-ENERGY': 21}
_KIND = {1: 'u16', 3: 'u16', 5: 'i32', 6: 'u8', 20: 'i32', 21: 'f32'}
_SCALE = {1: 1.0, 3: 1.0, 5: 1.0, 6: 1.0, 20: 3.05, 21: 9.31e-4}


def _plan(chid):
    """把 CHID 列表编译成 [(名字, 字节偏移, 字节数, kind, 标定)]，并给出记录总字节数。

    注意：偏移是**累加**出来的，所以必须能跳过**全部**特征（含我们不要的）。
    """
    out, off = [], 7                                  # 7 = RTOT(6) + CID(1)
    for c in chid:
        nb = CHID_NBYTES.get(c)
        if nb is None:                                # 未知特征无法跳过 → 放弃
            return None, None
        key = next((k for k, v in _WANT.items() if v == c), None)
        if key:
            out.append((key, off, nb, _KIND[c], _SCALE[c]))
        off += nb
    return out, off


def _frame_metrics(buf, cnt):
    """把一帧内收集的原始值算成指标。buf 是 dict of list，cnt = [ch1, ch2]。"""
    amp = np.asarray(buf['AMP'], dtype=np.float64)
    n = len(amp)
    if n == 0:
        return None
    m = {'n': n, 'n_ch1': cnt[0], 'n_ch2': cnt[1]}
    m['amp_mean'] = float(amp.mean())
    m['amp_p50'] = float(np.median(amp))
    m['amp_p90'] = float(np.percentile(amp, 90))
    m['amp_max'] = float(amp.max())
    m['amp_min'] = float(amp.min())
    m['a95_a50'] = float(np.percentile(amp, 95) - m['amp_p50'])
    m['frac_ge80'] = float((amp >= 80).mean())
    m['frac_ge90'] = float((amp >= 90).mean())

    # Gutenberg-Richter b 值（dB 口径）：b = 20 / (ln10 * (A_mean - A0))
    a0 = max(float(amp.min()), A_THRESHOLD)
    denom = np.log(10.0) * (amp.mean() - a0)
    m['b_value'] = float(20.0 / denom) if denom > 1e-9 else np.nan

    # 幅值熵（1 dB 分箱，归一化到 log2(档数)）
    h, _ = np.histogram(amp, bins=BINS)
    p = h[h > 0] / float(h.sum())
    m['amp_entropy'] = float(-(p * np.log2(p)).sum())

    ener = np.asarray(buf['ABS-ENERGY'], dtype=np.float64)
    sig = np.asarray(buf['SIG STRENGTH'], dtype=np.float64)
    dur = np.asarray(buf['DURATION'], dtype=np.float64)
    m['ener_sum'] = float(ener.sum())
    m['ener_per_hit'] = float(ener.mean())
    m['sig_sum'] = float(sig.sum())
    m['sig_per_hit'] = float(sig.mean())
    m['dur_p50'] = float(np.median(dur))
    m['dur_mean'] = float(dur.mean())
    ok = dur > 0
    m['af_p50'] = float(np.median(np.asarray(buf['COUN'], float)[ok]
                                 / dur[ok] * 1000.0)) if ok.any() else np.nan
    m['ra_p50'] = float(np.median(np.asarray(buf['RISE'], float) / amp))
    return m


def scan_file(path, frame_s=FRAME_S, plan=None, t_offset=0.0):
    """流式扫一个 `.DTA`，返回 (帧数组字典, 元信息)。帧按 t_offset + RTOT 对齐。"""
    from ae_dta import DtaMeta
    meta = DtaMeta()
    frames = collections.OrderedDict()
    buf = None
    cnt = [0, 0]
    cur = None

    def newbuf():
        return collections.defaultdict(list)

    with open(path, 'rb') as fh:
        for _off, LEN, mid, pay in iter_messages(fh):
            meta.msg_count[mid] += 1
            if mid == 42:
                body = pay[2:]
                L = len(body)
                q = 0
                while q + 3 <= L:
                    (LS,) = struct.unpack_from('<H', body, q)
                    if LS <= 0 or q + LS + 2 > L:
                        break
                    s = body[q + 2]
                    if s == 5:
                        meta.chid = list(struct.unpack_from('<%dB' % body[q + 3], body, q + 4))
                        plan = _plan(meta.chid)[0]
                    elif s == 173 and body[q + 3] == 42:
                        try:
                            f2 = struct.unpack_from('<BBBBxHBH2xHHHhHH', body, q + 4)
                            meta.hardware[f2[5]] = (f2[7] * 1000.0, f2[10], f2[12])
                        except struct.error:
                            pass
                    q += LS + 2
            elif mid == 99:
                try:
                    meta.start_time = _dt.datetime.strptime(
                        pay.decode('latin-1').strip('\x00').strip(), '%a %b %d %H:%M:%S %Y')
                except ValueError:
                    pass
            elif mid == 8:
                # 续文件：载荷前 8 字节 = 续接时间；其后的头消息已由 iter_messages 递归产出
                pass
            elif mid == 1:
                if not plan:
                    continue
                t = t_offset + _rtot(pay[0:6])
                fidx = int(t // frame_s)
                if fidx != cur:
                    if buf is not None and len(buf['AMP']):
                        mm = _frame_metrics(buf, cnt)
                        if mm is not None:
                            mm['frame'] = cur
                            frames[cur] = mm
                    buf, cur, cnt = newbuf(), fidx, [0, 0]
                ci = pay[6]
                if ci == 1:
                    cnt[0] += 1
                elif ci == 2:
                    cnt[1] += 1
                for key, o, nb, kind, sc in plan:
                    if kind == 'u8':
                        v = pay[o]
                    elif kind == 'u16':
                        v = struct.unpack_from('<H', pay, o)[0]
                    elif kind == 'i32':
                        v = struct.unpack_from('<i', pay, o)[0] * sc
                    else:
                        v = struct.unpack_from('<f', pay, o)[0] * sc
                    buf[key].append(v)
    if buf is not None and len(buf['AMP']):
        mm = _frame_metrics(buf, cnt)
        if mm is not None:
            mm['frame'] = cur
            frames[cur] = mm
    return frames, meta


def group_frames(gid, root=None, frame_s=FRAME_S, verbose=True):
    """把一组的全部卷按**日历时间**拼成一条帧序列。

    每卷的相对时间从 0 重新计时，因此用「该卷采集起始时刻 + RTOT」当绝对坐标，
    再加上实测的 AE→FBG 钟差（C4 谱载批 8 天）。这样跨卷、跨天自动对齐。
    """
    root = root or _data_l1()
    shift = CLOCK_SHIFT.get(gid, 0)
    allf = {}
    files = sorted(glob.glob(os.path.join(root, gid, 'AE', '*.DTA')))
    for f in files:
        # 先取起始时刻（只读头部若干条消息）
        _, meta0 = scan_file_head(f)
        base = meta0.start_time
        off = 0.0
        if base is not None:
            base = base + _dt.timedelta(days=shift)
            off = base.timestamp()
        else:                                        # 没有起始时刻就只能按文件名序拼接
            off = 1e9 * (files.index(f) + 1)
        fr, meta = scan_file(f, frame_s=frame_s, t_offset=off)
        for k, v in fr.items():
            v['file'] = os.path.basename(f)
            allf[k] = v
        if verbose:
            print('   %-16s 帧 %5d  起始 %s' % (os.path.basename(f), len(fr),
                                            base.strftime('%Y-%m-%d %H:%M') if base else '—'))
    if not allf:
        return None
    keys = sorted(allf)
    cols = ['n', 'n_ch1', 'n_ch2', 'amp_mean', 'amp_p50', 'amp_p90', 'amp_max',
            'amp_min', 'a95_a50', 'frac_ge80', 'frac_ge90', 'b_value',
            'amp_entropy', 'ener_sum', 'ener_per_hit', 'sig_sum', 'sig_per_hit',
            'dur_p50', 'dur_mean', 'af_p50', 'ra_p50']
    out = {'frame': np.asarray(keys, dtype=np.int64),
           't_day': np.asarray(keys, dtype=np.float64) * frame_s / 86400.0,
           'src': np.asarray([allf[k]['file'] for k in keys])}
    for c in cols:
        out[c] = np.asarray([allf[k][c] for k in keys], dtype=np.float64)
    out['frame_s'] = frame_s
    return out


def scan_file_head(path, max_msgs=400):
    """只读文件头（取起始时刻与硬件设置），用于拼时间轴前的轻量探测。"""
    from ae_dta import DtaMeta
    meta = DtaMeta()
    with open(path, 'rb') as fh:
        for _o, LEN, mid, pay in iter_messages(fh, max_msgs=max_msgs):
            if mid == 99:
                try:
                    meta.start_time = _dt.datetime.strptime(
                        pay.decode('latin-1').strip('\x00').strip(),
                        '%a %b %d %H:%M:%S %Y')
                except ValueError:
                    pass
            elif mid == 42:
                body = pay[2:]
                L, q = len(body), 0
                while q + 3 <= L:
                    (LS,) = struct.unpack_from('<H', body, q)
                    if LS <= 0 or q + LS + 2 > L:
                        break
                    s = body[q + 2]
                    if s == 5:
                        meta.chid = list(struct.unpack_from('<%dB' % body[q + 3], body, q + 4))
                    elif s == 173 and body[q + 3] == 42:
                        try:
                            f2 = struct.unpack_from('<BBBBxHBH2xHHHhHH', body, q + 4)
                            meta.hardware[f2[5]] = (f2[7] * 1000.0, f2[10], f2[12])
                        except struct.error:
                            pass
                    q += LS + 2
            if meta.start_time is not None and meta.chid:
                break
    return None, meta


def _data_l1():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        from shm import paths
        p = os.path.join(paths.data_root(), 'l1')
        if os.path.isdir(p):
            return p
    except Exception:                                     # noqa: BLE001
        pass
    return HERE


def groups_with_dta(root=None):
    root = root or _data_l1()
    return sorted(d for d in os.listdir(root)
                  if d.startswith('L1-')
                  and glob.glob(os.path.join(root, d, 'AE', '*.DTA')))


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    frame_s = FRAME_S
    tag = ''
    gids = []
    i = 0
    while i < len(argv):
        if argv[i] == '--frame-s':
            frame_s = float(argv[i + 1])
            i += 2
        elif argv[i] == '--tag':
            tag = argv[i + 1]
            i += 2
        elif argv[i] == '--groups':
            gids.extend(argv[i + 1:])
            break
        else:
            gids.append(argv[i])
            i += 1
    gids = gids or groups_with_dta()
    for gid in gids:
        print('=' * 78)
        print('## %s  帧长 %.0f s' % (gid, frame_s))
        d = group_frames(gid, frame_s=frame_s)
        if d is None:
            print('   无 .DTA，跳过')
            continue
        out = os.path.join(RES, '_l1ae_frames%s_%s.npz' % (tag, gid))
        np.savez_compressed(out, **d)
        span = (d['t_day'][-1] - d['t_day'][0]) if len(d['t_day']) else 0
        print('   帧数 %d  跨度 %.1f 天  总 hit %d  → %s'
              % (len(d['n']), span, int(d['n'].sum()), os.path.basename(out)))


if __name__ == '__main__':
    main(sys.argv[1:])
