# -*- coding: utf-8 -*-
"""探测 Vallen DiSP-4 `.DTA` 文件内部结构（只读，绝不写数据）。

分两个子命令：

  1) 结构探测
       python l1/probe_dta.py head  <某个.DTA>
     打印 544 字节文件头（ASCII 串 + 逐字段 int32/float32 猜测视图），
     再在数据区做「定长记录」扫描：找出使某个 4 字节字段单调不减的步长。

  2) 分卷关系验证
       python l1/probe_dta.py split <组号>        # 例如 L1-27
     把无后缀大文件的数据区与带 `_1.._n` 后缀的分卷数据区逐一比对，
     判断分卷是「同一份数据的切块」还是「各段独立采集」。

   python l1/probe_dta.py files <组号> ...      # 列出组的 AE 文件与时间
   python l1/probe_dta.py scan  <组号>          # 对每段做轻量结构统计

不依赖任何第三方库。
"""

import glob
import hashlib
import os
import re
import struct
import sys

HDR = 544              # 由 L1-27 的字节关系推出：5 段合计 = 单文件 + 4×HDR
CHUNK = 1 << 20


# --------------------------------------------------------------- 路径
def data_l1():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.dirname(here), here):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from shm import paths
        root = os.path.join(paths.data_root(), 'l1')
        if os.path.isdir(root):
            return root
    except Exception:                                  # noqa: BLE001
        pass
    return here


def ae_files(root, gid):
    d = os.path.join(root, gid, 'AE')
    fs = [f for f in glob.glob(os.path.join(d, '*')) if os.path.isfile(f)]
    return sorted(fs, key=lambda f: (os.path.splitext(os.path.basename(f))[0].count('_'),
                                     os.path.basename(f)))


# --------------------------------------------------------------- 基础工具
def strings(b, minlen=5):
    out, cur, start = [], b'', 0
    for i, ch in enumerate(b):
        if 32 <= ch < 127:
            if not cur:
                start = i
            cur += bytes([ch])
        else:
            if len(cur) >= minlen:
                out.append((start, cur.decode('ascii')))
            cur = b''
    if len(cur) >= minlen:
        out.append((start, cur.decode('ascii')))
    return out


def read_at(path, off, n):
    with open(path, 'rb') as fh:
        fh.seek(off)
        return fh.read(n)


def md5_of_region(path, off, length):
    h = hashlib.md5()
    left = length
    with open(path, 'rb') as fh:
        fh.seek(off)
        while left > 0:
            b = fh.read(min(CHUNK, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest(), length - left


def md5_skip_head(path, skip=HDR):
    """对 [skip:] 全长求 md5（分卷数据区拼接也用它）。"""
    return md5_of_region(path, skip, os.path.getsize(path) - skip)[0]


# --------------------------------------------------------------- 1) 头部
def cmd_head(path):
    sz = os.path.getsize(path)
    print('文件:', path)
    print('大小: %d 字节 (%.1f MB)' % (sz, sz / 1048576))
    head = read_at(path, 0, HDR)
    print('\n--- 头部 %d 字节里的 ASCII 串 ---' % HDR)
    for off, s in strings(head, 4):
        print('  %5d  %s' % (off, s))

    print('\n--- 头部按 4 字节小端解读（前 16 组）---')
    print('  off   hex         int32        uint32      float32')
    for i in range(0, min(HDR, 256) - 3, 4):
        b = head[i:i + 4]
        i_, u_, f_ = struct.unpack('<i', b)[0], struct.unpack('<I', b)[0], struct.unpack('<f', b)[0]
        f_txt = ('%g' % f_) if (abs(f_) > 1e-6 and abs(f_) < 1e9) else '-'
        print('  %4d  %-10s  %-11d %-11d %s' % (i, b.hex(), i_, u_, f_txt))

    print('\n--- 数据区（跳过前 %d 字节）头部 32 字节 ---' % HDR)
    dat = read_at(path, HDR, 32)
    print('  hex :', dat.hex())
    print('  ints:', [struct.unpack_from('<I', dat, k)[0] for k in range(0, 32, 4)])
    return sz


# --------------------------------------------------------------- 2) 记录扫描
def cmd_scan_stride(path, span=1 << 21):
    """在数据区前 span 字节里找「等步长、某 4 字节字段单调不减」的候选。"""
    sz = os.path.getsize(path)
    n = min(span, sz - HDR)
    buf = read_at(path, HDR, n)
    # 补齐到 4 的倍数
    n4 = (n // 4) * 4
    vals = struct.unpack('<%dI' % (n4 // 4), buf[:n4])

    print('扫描 %d 字节（%d 个 uint32）' % (n, len(vals)))
    hits = []
    for stride4 in range(1, 129):               # 步长 = stride4*4 字节，4B~512B
        if stride4 * 4 > n4 // 4:
            break
        for off in range(0, min(stride4, 16)):
            v = vals[off::stride4]
            if len(v) < 200:
                continue
            # 单调不减 + 不能全相同 + 量级合理
            if all(v[k] <= v[k + 1] for k in range(len(v) - 1)) and v[0] != v[-1]:
                hits.append((stride4 * 4, off * 4, len(v), v[0], v[-1], v[-1] - v[0]))
    hits.sort(key=lambda h: (-h[2], h[0]))
    if not hits:
        print('未发现任何「等步长单调不减」字段 → 记录很可能是变长的（含波形）')
    else:
        print('候选（步长, 字段偏移, 条数, 首值, 末值, 增量）:')
        for h in hits[:12]:
            print('   步长 %4dB  偏移 %3dB  条数 %6d  首 %d  末 %d  增量 %d' % h)
    return hits


# --------------------------------------------------------------- 3) 分卷验证
def _parts_for(root, gid):
    fs = ae_files(root, gid)
    main = [f for f in fs if re.fullmatch(re.escape(gid) + r'\.DTA', os.path.basename(f), re.I)]
    parts = [f for f in fs if re.fullmatch(re.escape(gid) + r'_\d+\.DTA', os.path.basename(f), re.I)]
    return (main[0] if main else None), parts


def cmd_split(root, gid):
    main, parts = _parts_for(root, gid)
    if not main:
        print('该组没有无后缀大文件，跳过与分卷的比对')
        return
    sm = os.path.getsize(main)
    print('单文件 %s  %.1f MB' % (os.path.basename(main), sm / 1048576))
    total_data = 0
    for p in parts:
        print('  分卷 %s  %.1f MB' % (os.path.basename(p), os.path.getsize(p) / 1048576))
        total_data += os.path.getsize(p) - HDR
    print('分卷数据区合计 = %d 字节；单文件数据区 = %d 字节；差 = %d'
          % (total_data, sm - HDR, total_data - (sm - HDR)))

    print('\n[判定] 单文件数据区 md5 vs 分卷数据区按序拼接 md5')
    h1 = md5_skip_head(main, HDR)
    h2 = hashlib.md5()
    for p in parts:
        left = os.path.getsize(p) - HDR
        with open(p, 'rb') as fh:
            fh.seek(HDR)
            while left > 0:
                b = fh.read(min(CHUNK, left))
                if not b:
                    break
                h2.update(b)
                left -= len(b)
    print('   单文件  ', h1)
    print('   拼分卷  ', h2.hexdigest())
    print('   →', '完全一致：分卷就是这份数据的等长切块（内容重复，用其一即可）'
          if h1 == h2.hexdigest() else '不一致：分卷与单文件并非同一份数据')

    if h1 == h2.hexdigest():
        print('\n[提示] 逐段偏移对照（分卷 i 的数据区 == 单文件数据区第 off..off+len 段）成立')
        off = HDR
        for p in parts:
            d = os.path.getsize(p) - HDR
            print('   %-14s 对应单文件数据区 [%d, %d)' % (os.path.basename(p), off - HDR, off - HDR + d))
            off += d


# --------------------------------------------------------------- 4) 文件清单
def cmd_diff(root, gid, max_report=30):
    """逐字节比对「单文件数据区」与「分卷数据区按序拼接」，报告差异位置与间隔。"""
    main, parts = _parts_for(root, gid)
    if not main or not parts:
        print('缺少单文件或分卷，无法比对')
        return
    diffs = []
    gap_stats = {}
    total_diff = 0
    last = None
    off = HDR                              # 单文件数据区内的相对偏移
    fh1 = open(main, 'rb')
    fh1.seek(HDR)
    for p in parts:
        left = os.path.getsize(p) - HDR
        fh2 = open(p, 'rb')
        fh2.seek(HDR)
        base = off
        while left > 0:
            n = min(CHUNK, left)
            a, b = fh1.read(n), fh2.read(n)
            if a != b:
                for i in range(n):
                    if a[i] != b[i]:
                        pos = base + i - HDR
                        total_diff += 1
                        if len(diffs) < max_report:
                            diffs.append((pos, a[i], b[i]))
                        if last is not None:
                            gap_stats[pos - last] = gap_stats.get(pos - last, 0) + 1
                        last = pos
            base += n
            left -= n
            off += n
        fh2.close()
    fh1.close()
    print('单文件 %s   差异字节总数 = %d' % (os.path.basename(main), total_diff))
    print('\n前 %d 个差异（数据区相对偏移, 单文件值, 分卷值）:' % max_report)
    for pos, x, y in diffs:
        print('   off %12d  %3d -> %3d   (0x%08x 处)' % (pos, x, y, pos))
    top = sorted(gap_stats.items(), key=lambda kv: -kv[1])[:10]
    if top:
        print('\n差异间隔分布（间隔 → 出现次数，前 10）:')
        for g, c in top:
            print('   %8d 字节  × %d' % (g, c))
    return total_diff


# --------------------------------------------------------------- 5) 文件清单
def cmd_files(root, gids):
    for gid in gids:
        print('\n== %s ==' % gid)
        for f in ae_files(root, gid):
            st = os.stat(f)
            print('  %-18s %12d B  %7.1f MB  写入 %s' % (
                os.path.basename(f), st.st_size, st.st_size / 1048576,
                __import__('datetime').datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')))


# --------------------------------------------------------------- 6) 加载场次时间线
_DATE_RE = re.compile(r'([A-Z][a-z]{2} [A-Z][a-z]{2} [ \d]\d \d{2}:\d{2}:\d{2} \d{4})')


def dta_start_time(path):
    """从 DTA 头部 ASCII 区取「采集起始时刻」字符串。"""
    head = read_at(path, 0, 200)
    m = _DATE_RE.search(head.decode('latin-1'))
    return m.group(1) if m else ''


def fbg_stamps(gid_dir):
    """FBG 目录里 Sensors.YYYYMMDDHHMMSS.txt 的时间戳列表。"""
    out = []
    for f in glob.glob(os.path.join(gid_dir, 'FBG', '*.txt')):
        m = re.search(r'(\d{14})', os.path.basename(f))
        if m:
            out.append(m.group(1))
    return sorted(out)


def _day(s):
    return '%s-%s-%s' % (s[0:4], s[4:6], s[6:8])


def cmd_sessions(root, gids):
    import datetime
    for gid in gids:
        gd = os.path.join(root, gid)
        print('\n' + '=' * 74)
        print('## %s' % gid)
        print('  AE 文件（头部内嵌采集起始时刻 / 写入时间）:')
        for f in ae_files(root, gid):
            st = os.stat(f)
            mt = datetime.datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M')
            print('    %-16s %9.1f MB   头部起始 %-24s  写入 %s'
                  % (os.path.basename(f), st.st_size / 1048576, dta_start_time(f), mt))
        st = fbg_stamps(gd)
        if not st:
            print('  FBG: 无')
            continue
        days = {}
        for s in st:
            days[_day(s)] = days.get(_day(s), 0) + 1
        print('  FBG 分块 %d 个  首 %s  末 %s' % (len(st), st[0], st[-1]))
        print('  按日计数（日期: 块数）:')
        line = []
        for d in sorted(days):
            line.append('%s:%d' % (d, days[d]))
        for i in range(0, len(line), 8):
            print('    ' + '  '.join(line[i:i + 8]))


# --------------------------------------------------------------- main
def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    root = data_l1()
    if not argv:
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == 'head':
        for p in argv[1:]:
            cmd_head(p)
            print()
    elif cmd == 'split':
        for gid in argv[1:]:
            print('\n' + '=' * 70)
            print('## 分卷验证', gid)
            cmd_split(root, gid)
    elif cmd == 'files':
        cmd_files(root, argv[1:])
    elif cmd == 'diff':
        for gid in argv[1:]:
            print('\n' + '=' * 70)
            print('## 逐字节比对', gid)
            cmd_diff(root, gid)
    elif cmd == 'sessions':
        cmd_sessions(root, argv[1:])
    elif cmd == 'scan':
        for gid in argv[1:]:
            for f in ae_files(root, gid):
                if os.path.getsize(f) < 200_000:
                    continue
                print('\n== %s / %s ==' % (gid, os.path.basename(f)))
                cmd_scan_stride(f)
    else:
        print('未知子命令:', cmd)
        print(__doc__)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
