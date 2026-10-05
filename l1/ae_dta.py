# -*- coding: utf-8 -*-
"""自研 Mistras / Physical Acoustics **AEwin `.DTA`** 解析器（纯 struct + numpy，无第三方依赖）。

格式来源
--------
官方：*Mistras User's Manual* **Appendix II** 描述了 `.DTA` 的二进制结构。
可复现的公开实现（用于交叉验证，本脚本不依赖它们）：
  · d-cogswell/MistrasDTA（Python，MIT，PyPI `MistrasDTA`）
  · sayginer/Mistras-AE-MATLAB-Library（MATLAB，.DTA + .WFS）
  · SchneiderLiu/MISTRAS-Standard-Data-Toolbox（Python GUI 封装）

文件结构（TLV 消息流，直到 EOF）
--------------------------------
    每条消息：  uint16 LE  LEN          # 长度字段
                uint8      ID           # 消息类型
                [uint8     extra]       # 仅当 40 <= ID <= 49 时多 1 字节
                payload                 # 共 LEN 字节（不含开头 2 字节 LEN 自身）

    即 **消息总字节数 = LEN + 2**。

关键消息 ID
-----------
    1    AE hit / event       → 相对时间(6B) + 通道(1B) + 各特征(按 ID 42 定义)
    7    用户注释 / 试件标签（ASCII）
    8    续文件消息
    41   ASCII 产品定义（系统名 + 软件版本）
    42   硬件设置（子记录：5=事件特征集定义、23=增益、173/42=采样率与时延…）
    99   测试起始时刻（ASCII `%a %b %d %H:%M:%S %Y\\n`）
    128/129/130  开始 / 停止 / 暂停
    173  数字 AE 波形（int16 采样）

用法
----
    python l1/ae_dta.py info  <文件> [<文件> ...]   # 消息结构 + 元信息
    python l1/ae_dta.py hits  <组号> [...]          # 逐卷 hit 统计（数据根下 l1/）
    python l1/ae_dta.py check <文件>                # 与 MistrasDTA 对拍（若已安装）
"""

import collections
import datetime as _dt
import glob
import io
import os
import struct
import sys

import numpy as np

# ------------------------------------------------------------------ 规范表
CHID_TO_STR = {
    1: 'RISE', 2: 'PCNTS', 3: 'COUN', 4: 'ENER', 5: 'DURATION', 6: 'AMP',
    8: 'ASL', 10: 'THR', 13: 'A-FRQ', 17: 'RMS', 18: 'R-FRQ', 19: 'I-FRQ',
    20: 'SIG STRENGTH', 21: 'ABS-ENERGY', 23: 'FRQ-C', 24: 'P-FRQ'}

CHID_NBYTES = {
    1: 2, 2: 2, 3: 2, 4: 2, 5: 4, 6: 1, 8: 1, 10: 1, 13: 2, 17: 2, 18: 2,
    19: 2, 20: 4, 21: 4, 23: 2, 24: 2}

# 特征的中文名（写表用）
CHID_CN = {
    'RISE': '上升时间', 'PCNTS': '振铃计数', 'COUN': '持续计数', 'ENER': '能量',
    'DURATION': '持续时间', 'AMP': '幅值', 'ASL': '平均信号电平', 'THR': '阈值',
    'A-FRQ': '平均频率', 'RMS': 'RMS电压', 'R-FRQ': '谐振频率', 'I-FRQ': '初始频率',
    'SIG STRENGTH': '信号强度', 'ABS-ENERGY': '绝对能量', 'FRQ-C': '质心频率',
    'P-FRQ': '峰值频率'}

ID_NAME = {1: 'AE hit', 7: '试件标签', 8: '续文件', 41: '产品定义',
           42: '硬件设置', 99: '起始时刻', 128: '开始', 129: '停止', 130: '暂停',
           173: '波形'}


def _rtot(b):
    """6 字节 → 相对时间（秒）。单位 0.25 µs。"""
    i1, i2 = struct.unpack('<IH', b)
    return (i1 + 2 ** 32 * i2) * 0.25e-6


def _decode(v, tag):
    """按特征名做单位标定（与官方实现一致）。"""
    if tag == 'RMS':
        return v / 5000.0
    if tag == 'SIG STRENGTH':
        return v * 3.05
    if tag == 'ABS-ENERGY':
        return v * 9.31e-4
    return v


class DtaMeta(object):
    """一个 `.DTA` 文件的元信息。"""

    def __init__(self):
        self.start_time = None          # datetime（本地挂钟，即采集软件所在机器的时钟）
        self.labels = []                # ID 7 试件标签
        self.product = ''               # ID 41 产品定义
        self.chid = []                  # hit 记录里包含的特征（顺序即字节顺序）
        self.hardware = {}              # CH -> (采样率 Hz, 前置时延 µs)
        self.gain = {}                  # CH -> 增益 dB
        self.msg_count = collections.Counter()
        self.n_hit = 0
        self.n_waveform = 0
        self.t_min = None
        self.t_max = None
        self.unknown_ids = collections.Counter()
        self.truncated = False

    def as_dict(self):
        return {'起始时刻': self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else '',
                '产品定义': self.product.strip(),
                '试件标签': ' / '.join(self.labels),
                '特征列表': ' '.join('%s(%d)' % (CHID_TO_STR.get(c, '?'), c)
                                 for c in self.chid),
                '硬件(通道: 采样率kHz/时延us)': '; '.join(
                    '%s: %.0f/%.1f' % (c, s / 1000.0, d) for c, (s, d) in sorted(self.hardware.items())),
                '增益dB': '; '.join('%s: %s' % kv for kv in sorted(self.gain.items())),
                'hit 数': self.n_hit, '波形数': self.n_waveform,
                '时间跨度s': (round(self.t_max - self.t_min, 1)
                          if self.t_min is not None else None),
                '消息类型': ' '.join('%s×%d' % (ID_NAME.get(k, k), v)
                                 for k, v in sorted(self.msg_count.items()))}


def iter_stream(fh, base_off=0, max_msgs=None):
    """在字节流上遍历消息：产出 (offset, LEN, ID, payload_bytes)。

    ID 8「续文件」的载荷 = [8 字节续接时间] + **一整套内嵌头消息序列**
    （产品定义 / 硬件设置等都会再写一遍），因此对这段字节递归解析。
    """
    off = base_off
    n = 0
    while True:
        head = fh.read(2)
        if len(head) < 2:
            return
        (LEN,) = struct.unpack('<H', head)
        idb = fh.read(1)
        if len(idb) < 1:
            return
        (mid,) = struct.unpack('<B', idb)
        rest = LEN - 1
        if 40 <= mid <= 49:                       # 40-49 多一个字节
            fh.read(1)
            rest -= 1
        if rest < 0:
            return
        payload = fh.read(rest)
        if len(payload) < rest:
            return
        yield off, LEN, mid, payload
        if mid == 8 and len(payload) > 8:         # 解析内嵌的头消息序列
            for m in iter_stream(io.BytesIO(payload[8:]), off + 3 + 8):
                yield m
        off += LEN + 2
        n += 1
        if max_msgs is not None and n >= max_msgs:
            return


def iter_messages(fh, max_msgs=None):
    """同 `iter_stream`（入口默认从偏移 0 开始）。"""
    return iter_stream(fh, 0, max_msgs)


def read_dta(path, want_waveform=False, waveform_limit=0):
    """解析 `.DTA`。

    返回 (hits, meta)：
      hits  — dict：'t'（相对秒）、'CH'，以及各特征（numpy 数组）；
      meta  — DtaMeta；
      waveform — 若 want_waveform，附在 meta.waveforms = [(t, CH, V 电压数组), ...]（最多 waveform_limit 条，0 表示不限）
    """
    meta = DtaMeta()
    cols = collections.defaultdict(list)
    t_list, ch_list = [], []
    wl = []

    with open(path, 'rb') as fh:
        for _off, LEN, mid, payload in iter_messages(fh):
            meta.msg_count[mid] += 1

            if mid == 1:                                    # ---- AE hit
                if not meta.chid:
                    continue                                # 未读到特征定义，无法切分
                t = _rtot(payload[0:6])
                cid = payload[6]
                p = 7
                t_list.append(t)
                ch_list.append(cid)
                for c in meta.chid:
                    nb = CHID_NBYTES[c]
                    raw = payload[p:p + nb]
                    p += nb
                    if len(raw) < nb:
                        break
                    if c in (5, 20):                        # 有符号 int32
                        v = struct.unpack('<i', raw)[0]
                    elif c == 21:                           # float32
                        v = struct.unpack('<f', raw)[0]
                    elif nb == 1:
                        v = raw[0]
                    else:
                        v = struct.unpack('<H', raw)[0]
                    cols[CHID_TO_STR[c]].append(_decode(v, CHID_TO_STR[c]))
                meta.n_hit += 1
                meta.t_min = t if meta.t_min is None else min(meta.t_min, t)
                meta.t_max = t if meta.t_max is None else max(meta.t_max, t)

            elif mid == 41:                                 # ---- 产品定义
                meta.product = payload[2:].decode('latin-1').replace('\x00', ' ').strip()

            elif mid == 7:                                  # ---- 试件标签
                meta.labels.append(payload.decode('latin-1').strip('\x00').strip())

            elif mid == 99:                                 # ---- 起始时刻
                txt = payload.decode('latin-1').strip('\x00').strip()
                try:
                    meta.start_time = _dt.datetime.strptime(txt, '%a %b %d %H:%M:%S %Y')
                except ValueError:
                    meta.start_time = None

            elif mid == 42:                                 # ---- 硬件设置
                # 结构：MVERN(2) + 子记录序列；子记录 = [LSUB(2)][SUBID(1)][内容]
                # 子记录总字节数 = LSUB + 2（LSUB 只数「SUBID + 内容」）
                body = payload[2:]
                L = len(body)
                q = 0
                while q + 3 <= L:
                    (LSUB,) = struct.unpack_from('<H', body, q)
                    if LSUB <= 0 or q + LSUB + 2 > L:
                        break
                    sub = body[q + 2]                       # SUBID
                    if sub == 5:                            # 事件特征集定义
                        nch = body[q + 3]
                        meta.chid = list(struct.unpack_from('<%dB' % nch, body, q + 4))
                    elif sub == 23:                         # Set Gain
                        meta.gain[body[q + 3]] = body[q + 4]
                    elif sub == 173 and body[q + 3] == 42:  # 173,42 硬件设置
                        try:
                            # 解包顺序：MVERN b2 ADT SETS pad SLEN CHID HLK pad
                            #            SRATE TMODE TSRC TDLY MXIN THRD
                            f = struct.unpack_from('<BBBBxHBH2xHHHhHH', body, q + 4)
                            meta.hardware[f[5]] = (f[7] * 1000.0, f[10])
                        except struct.error:
                            pass
                    q += LSUB + 2

            elif mid == 173:                                # ---- 波形
                meta.n_waveform += 1
                if want_waveform and (waveform_limit == 0 or len(wl) < waveform_limit):
                    try:
                        t = _rtot(payload[1:7])
                        cid, _alb = payload[7], payload[8]
                        g = meta.gain.get(cid, 0)
                        scale = 10.0 / (10 ** (g / 20.0) * 32768.0)
                        s = np.frombuffer(payload[9:], dtype='<i2').astype(np.float64) * scale
                        wl.append((t, cid, s))
                    except Exception:                       # noqa: BLE001
                        pass

            else:
                meta.unknown_ids[mid] += 1
                if mid not in (8, 128, 129, 130):
                    meta.unknown_ids[mid] += 0

    hits = {'t': np.asarray(t_list, dtype=np.float64),
            'CH': np.asarray(ch_list, dtype=np.int16)}
    for k, v in cols.items():
        hits[k] = np.asarray(v, dtype=np.float64)
    if want_waveform:
        meta.waveforms = wl
    return hits, meta


def scan_dta(path, max_amp_sample=200000):
    """流式扫描 `.DTA`，**不保存逐条 hit**，只回聚合统计（用于批量体检）。

    返回 dict：hit 数、时间跨度、通道分布、幅值分位（采样）、每小时事件率、元信息。
    """
    meta = DtaMeta()
    ch = collections.Counter()
    amp = []
    hourly = collections.Counter()
    n = 0
    step = 1
    with open(path, 'rb') as fh:
        for _off, LEN, mid, payload in iter_messages(fh):
            meta.msg_count[mid] += 1
            if mid == 1:
                if not meta.chid:
                    continue
                t = _rtot(payload[0:6])
                ch[payload[6]] += 1
                n += 1
                meta.t_min = t if meta.t_min is None else min(meta.t_min, t)
                meta.t_max = t if meta.t_max is None else max(meta.t_max, t)
                hourly[int(t // 3600)] += 1
                # 幅值：CHID 顺序里找 AMP，按需采样
                if len(amp) < max_amp_sample:
                    p = 7
                    for c in meta.chid:
                        nb = CHID_NBYTES[c]
                        if c == 6:                            # AMP
                            amp.append(payload[p])
                            break
                        p += nb
                else:
                    step += 1
                    if step % 97 == 0 and len(amp) < max_amp_sample:
                        pass
            elif mid == 41:
                meta.product = payload[2:].decode('latin-1').replace('\x00', ' ').strip()
            elif mid == 7:
                meta.labels.append(payload.decode('latin-1').strip('\x00').strip())
            elif mid == 99:
                try:
                    meta.start_time = _dt.datetime.strptime(
                        payload.decode('latin-1').strip('\x00').strip(),
                        '%a %b %d %H:%M:%S %Y')
                except ValueError:
                    meta.start_time = None
            elif mid == 42:
                body = payload[2:]
                L, q = len(body), 0
                while q + 3 <= L:
                    (LSUB,) = struct.unpack_from('<H', body, q)
                    if LSUB <= 0 or q + LSUB + 2 > L:
                        break
                    sub = body[q + 2]
                    if sub == 5:
                        meta.chid = list(struct.unpack_from('<%dB' % body[q + 3], body, q + 4))
                    elif sub == 23:
                        meta.gain[body[q + 3]] = body[q + 4]
                    elif sub == 173 and body[q + 3] == 42:
                        try:
                            f = struct.unpack_from('<BBBBxHBH2xHHHhHH', body, q + 4)
                            meta.hardware[f[5]] = (f[7] * 1000.0, f[10])
                        except struct.error:
                            pass
                    q += LSUB + 2
            elif mid == 173:
                meta.n_waveform += 1

    meta.n_hit = n
    a = np.asarray(amp, dtype=float)
    hr = np.asarray(sorted(hourly.values()), dtype=float) if hourly else np.asarray([])
    return {'hit': n,
            'span_h': round((meta.t_max - meta.t_min) / 3600.0, 2) if n else 0.0,
            't_start': round(meta.t_min, 1) if n else None,
            'ch': dict(ch),
            'amp_min': float(a.min()) if len(a) else None,
            'amp_p50': float(np.median(a)) if len(a) else None,
            'amp_p90': float(np.percentile(a, 90)) if len(a) else None,
            'hr_med': float(np.median(hr)) if len(hr) else None,
            'hr_max': float(hr.max()) if len(hr) else None,
            'hourly': dict(sorted(hourly.items())),
            'meta': meta}


def cmd_stats(root, gids, per_file=True):
    import time
    for gid in gids:
        print('=' * 96)
        print('## %s' % gid)
        for f in sorted(glob.glob(os.path.join(root, gid, 'AE', '*'))):
            if not os.path.isfile(f):
                continue
            t0 = time.time()
            r = scan_dta(f)
            m = r['meta']
            print('   %-16s %8.1f MB  hit %9d  跨度 %7.2f h  t %s→%s  通道 %s'
                  % (os.path.basename(f), os.path.getsize(f) / 1048576, r['hit'],
                     r['span_h'], r['t_start'],
                     round(m.t_max, 1) if m.t_max is not None else None, r['ch']))
            print('        幅值: 最小 %s 中位 %s P90 %s dB   每小时命中数: 中位 %s 峰值 %s   耗时 %.1fs'
                  % (r['amp_min'], r['amp_p50'], r['amp_p90'], r['hr_med'], r['hr_max'],
                     time.time() - t0))


def cmd_export(root, gids, out_csv):
    """把各组 `.DTA` 的体检统计写成 CSV（零依赖，用标准库 csv）。

    ⚠️ **不要往这个 CSV 里放机器相关的列**（如解析耗时）：它是交付物，
    还会被 `survey_groups.py` 并入 `L1数据记录.xlsx`。2026-10-05 实测：
    原有一列 `解析耗时s`（wall-clock 秒，含文件缓存冷/热差异）会让产物
    **不可逐字节复现**（连跑两次 MD5 不同）。已移除；耗时仍打到控制台。
    """
    import csv
    import time
    if not gids:
        gids = sorted(d for d in os.listdir(root)
                      if d.startswith('L1-') and os.path.isdir(os.path.join(root, d)))
    cols = ['组号', '文件', '大小MB', 'hit数', '时间跨度h', 't_min_s', 't_max_s',
            '通道1', '通道2', '幅值最小', '幅值中位', '幅值P90', '每小时中位',
            '每小时峰值', '起始时刻', '采样率kHz', '波形数']
    rows = []
    for gid in gids:
        for f in sorted(glob.glob(os.path.join(root, gid, 'AE', '*.DTA'))):
            t0 = time.time()
            r = scan_dta(f)
            m = r['meta']
            srate = '/'.join('%.0f' % (v[0] / 1000.0) for _c, v in sorted(m.hardware.items()))
            rows.append([gid, os.path.basename(f), round(os.path.getsize(f) / 1048576, 1),
                         r['hit'], r['span_h'], r['t_start'],
                         round(m.t_max, 1) if m.t_max is not None else '',
                         r['ch'].get(1, 0), r['ch'].get(2, 0),
                         r['amp_min'], r['amp_p50'], r['amp_p90'], r['hr_med'], r['hr_max'],
                         m.start_time.strftime('%Y-%m-%d %H:%M:%S') if m.start_time else '',
                         srate, m.n_waveform])
            print('   %-16s hit %9d  %.1fs' % (os.path.basename(f), r['hit'], time.time() - t0))
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    print('已写出 -> %s（%d 行）' % (out_csv, len(rows)))


# ------------------------------------------------------------------ CLI
def _data_l1():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from shm import paths
        p = os.path.join(paths.data_root(), 'l1')
        if os.path.isdir(p):
            return p
    except Exception:                                       # noqa: BLE001
        pass
    return here


def cmd_info(paths):
    for p in paths:
        print('=' * 78)
        print('##', p)
        _, meta = read_dta(p)
        d = {'文件大小MB': round(os.path.getsize(p) / 1048576, 1)}
        d.update(meta.as_dict())
        for k, v in d.items():
            print('   %-24s %s' % (k, v))
        if meta.unknown_ids:
            print('   %-24s %s' % ('未处理 ID', dict(meta.unknown_ids)))


def cmd_msgs(path, n=40):
    """逐条列出消息头，用于核对结构与字节偏移。"""
    print('## 消息序列（前 %d 条）: 偏移, LEN, ID, 名称, 载荷前 40 字节' % n)
    with open(path, 'rb') as fh:
        for i, (off, LEN, mid, payload) in enumerate(iter_messages(fh, max_msgs=n)):
            txt = payload[:40].decode('latin-1').replace('\x00', '.').replace('\n', '\\n')
            print('   %6d  LEN=%5d  总长=%5d  ID=%-4d %-8s %s'
                  % (off, LEN, LEN + 2, mid, ID_NAME.get(mid, '?'), txt))


def cmd_hits(root, gids):
    for gid in gids:
        print('=' * 78)
        print('##', gid)
        for f in sorted(glob.glob(os.path.join(root, gid, 'AE', '*'))):
            if not os.path.isfile(f):
                continue
            hits, meta = read_dta(f)
            t = hits['t']
            span = (t.max() - t.min()) / 3600.0 if len(t) else 0.0
            ch = collections.Counter(hits['CH'].tolist())
            amp = hits.get('AMP')
            print('   %-16s %8.1f MB  hit %7d  跨度 %6.2f h  通道 %s  幅值中位 %s dB'
                  % (os.path.basename(f), os.path.getsize(f) / 1048576, meta.n_hit, span,
                     dict(ch),
                     ('%.1f' % np.median(amp)) if amp is not None and len(amp) else '—'))


def cmd_check(path):
    """与 MistrasDTA 对拍（若已安装）。"""
    import MistrasDTA
    import logging
    logging.disable(logging.CRITICAL)
    mine, meta = read_dta(path)
    ref, _ = MistrasDTA.read_bin(path, skip_wfm=True)
    print('文件:', path)
    print('  自研 hit 数 = %d   参考实现 = %d   %s'
          % (meta.n_hit, 0 if ref is None else len(ref),
             '一致' if ref is not None and len(ref) == meta.n_hit else '不一致'))
    if ref is not None and len(ref):
        dt = np.abs(ref['SSSSSSSS.mmmuuun'] - mine['t']).max()
        dch = int(np.abs(ref['CH'].astype(int) - mine['CH'].astype(int)).max())
        print('  相对时间最大偏差 = %.3e s   通道最大偏差 = %d' % (dt, dch))
        for k in ('AMP', 'PCNTS', 'DURATION', 'SIG STRENGTH', 'ABS-ENERGY', 'RISE'):
            if k in ref.dtype.names and k in mine:
                dd = np.abs(np.asarray(ref[k], dtype=float) - mine[k]).max()
                print('  %-14s 最大偏差 = %.3e' % (k, dd))


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if not argv:
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == 'info':
        cmd_info(argv[1:])
    elif cmd == 'msgs':
        cmd_msgs(argv[1], int(argv[2]) if len(argv) > 2 else 40)
    elif cmd == 'hits':
        cmd_hits(_data_l1(), argv[1:])
    elif cmd == 'stats':
        cmd_stats(_data_l1(), argv[1:])
    elif cmd == 'export':
        out = argv[1] if len(argv) > 1 else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'AE特征概览.csv')
        cmd_export(_data_l1(), argv[2:], out)
    elif cmd == 'check':
        cmd_check(argv[1])
    else:
        print('未知子命令:', cmd)
        print(__doc__)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
