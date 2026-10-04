#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1 数据集梳理 —— 融入主代码前的清点，产出 `l1/L1数据记录.xlsx`

做三件事：
  1) 扫描数据根 `l1/` 下每一组的数据资产（模态 / 格式 / 文件数 / 体积 / 时间跨度）
  2) 解析数据根根目录的 6 个 **campaign 级 PDF**：
        Impact_Locations.pdf                  恒幅一批 冲击位置
        Damage locations.pdf                  恒幅二批 冲击/脱粘位置
        tables of cycles variable.pdf          变幅组 载荷级别与各级循环
        Damage locations variable.pdf          变幅组 冲击/脱粘位置
        table of specimen cycles to failure.pdf 谱载组 载荷级别与各级循环
        Damage locations spectrum.pdf          谱载组 冲击/脱粘位置
  3) 汇总成 8 个 sheet 的 xlsx（试件总表 / 批次定义 / 数据资产 / 采集场次与时钟 /
     接入注意 / FBG采集覆盖 / 长空隙清单 / AE特征概览）

为什么单独成脚本（而不是直接接入 `l1/` 现有管线）：
  · **新增 14 组没有组内 `L1-xx.pdf`** —— 老组的 n_f / 脚空间段靠它，新的只能从 campaign PDF 取
  · **AE 是 PAC/Mistras AEwin 的 `.DTA`**（不是老组的 Vallen `.pridb`）—— 格式见 Mistras
    手册 Appendix II；`vallenae.io` 读不了，但仓库内已自研 `l1/ae_dta.py` 解析（与公开
    实现 MistrasDTA 逐字段一致，详见 `l1/ae_dta.py` 头部注释）
  · **FBG 是 Micron Optics sm130 的 `Sensors.<时间戳>.txt`**（ENLIGHT 导出），
    10 个应变通道（R1-R5 / L1-L5）+ 10 个波长通道；**采样率 = 1000 / Data Interleave**，
    变幅组 5 Hz、谱载组 10 Hz —— **两批不一致**
  · 新增组**只有 AE + FBG**，没有 LUNA(DFOS) 与 PZT 数据

用法：
    python l1/survey_groups.py            # 扫描并写出 l1/L1数据记录.xlsx
    python l1/survey_groups.py --print    # 只打印，不写文件
"""
import collections
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_XLSX = os.path.join(HERE, 'L1数据记录.xlsx')

if HERE not in sys.path:
    sys.path.insert(0, HERE)
try:                                   # 头部时间 / FBG 时间戳解析复用探测脚本里的实现
    from probe_dta import dta_start_time, fbg_stamps
except Exception:                      # noqa: BLE001
    dta_start_time = fbg_stamps = None

# 4 个 campaign 的成员（来源见上：C3/C4 由 PDF 表推出；C1/C2 沿用既有分组）
C1_CA1 = ['L1-03', 'L1-04', 'L1-05', 'L1-09']                          # Broer 2021，恒幅
C2_CA2 = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54',
          'L1-55', 'L1-56', 'L1-59', 'L1-60']                            # 恒幅二批
C3_VA = ['L1-06', 'L1-13', 'L1-14', 'L1-24']                            # 变幅（本次新增）
C4_SP = ['L1-25', 'L1-27', 'L1-29', 'L1-30', 'L1-31', 'L1-34',
         'L1-35', 'L1-36', 'L1-41', 'L1-44']                             # 谱载（本次新增）
NEW = set(C3_VA) | set(C4_SP)
CAMPAIGN = {}
for _g in C1_CA1:
    CAMPAIGN[_g] = 'C1 恒幅一批'
for _g in C2_CA2:
    CAMPAIGN[_g] = 'C2 恒幅二批'
for _g in C3_VA:
    CAMPAIGN[_g] = 'C3 变幅 VA'
for _g in C4_SP:
    CAMPAIGN[_g] = 'C4 谱载'


# --------------------------------------------------------------- 路径
def data_dir():
    """数据根下的 l1 目录：优先 paths.json 的 data_root（外接盘），否则仓库内 l1/。"""
    try:
        sys.path.insert(0, REPO)
        from shm import paths                      # noqa: PLC0415
        r = paths.data_root()
        if r and os.path.isdir(os.path.join(r, 'l1')):
            return os.path.join(r, 'l1')
    except Exception:                              # noqa: BLE001
        pass
    return HERE


# --------------------------------------------------------------- PDF 文本
def pdf_text(fp):
    import pdfplumber
    with pdfplumber.open(fp) as pdf:
        return '\n'.join((p.extract_text() or '') for p in pdf.pages)


def _num(s):
    return float(s.replace(',', '').replace('~', ''))


# --------------------------------------------------------------- 解析 6 个 PDF
def parse_impact_locations(txt):
    """Impact_Locations.pdf  →  {gid: (x, y, '冲击'/'脱粘')}"""
    out = {}
    for m in re.finditer(r'(L1-\d{2})\s*\n\s*([\d.]+)\s*mm\s*\n\s*([\d.]+)\s*mm\s*\n\s*(\w[\w ]*)',
                         txt):
        kind = '脱粘' if 'disbond' in m.group(4).lower() else '冲击'
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)), kind)
    return out


def _coords(block):
    """在文本块里找 'X(mm) Y(mm)' 后的坐标，连同其前面的标签。"""
    res = []
    for m in re.finditer(r'X\(mm\)\s*Y\(mm\)\s*\n\s*([\d.]+)\s+([\d.\-]+)', block):
        head = block[:m.start()]
        kind = '脱粘下缘' if 'Lower edge of disbond' in head[-240:] else '冲击中心'
        res.append((kind, float(m.group(1)), m.group(2)))
    # 只有 Y 的情况（如 L1-22 的脱粘位置）
    for m in re.finditer(r'Lower edge of disbond\s*\n\s*Y\(mm\)\s*\n\s*([\d.\-]+)', block):
        res.append(('脱粘下缘(Y)', None, m.group(1)))
    return res


def parse_damage_locations(txt):
    """Damage locations*.pdf → {gid: {'冲击中心': 'x, y', '脱粘下缘': 'x, y', '传感器编号': …}}

    三份 PDF 的组标题行格式不同（`L1 49 (stiffener side)` / `L1 06`），
    统一按行首 `L1 NN` 切块。
    """
    out = {}
    parts = re.split(r'(?m)^(L1 \d{2}.*)$', txt)
    for i in range(1, len(parts), 2):
        gid = 'L1-' + re.search(r'L1 (\d{2})', parts[i]).group(1)
        body = parts[i + 1] if i + 1 < len(parts) else ''
        rec = {}
        for kind, x, y in _coords(body):
            rec[kind] = ('%s, %s' % (x, y)).replace('None, ', '')
        # 传感器编号方向：正常为 L5 R5 … L1 R1；出现 "L1 R5" 说明该组编号被翻转
        if 'L1 R5' in body.replace('\n', ' '):
            rec['传感器编号'] = '反向（L5R1…L1R5）'
        out[gid] = rec
    return out


def parse_cycles_variable(txt):
    """tables of cycles variable.pdf → {gid: {'能量': …, '级别': [(min,max,cycles),…], '总循环': n}}"""
    out = {}
    blocks = re.split(r'\n(?=L1-\d{2}\s)', txt)
    for b in blocks:
        m = re.match(r'(L1-\d{2})\s+(.*)', b)
        if not m:
            continue
        gid, rest = m.group(1), b
        head = rest.split('\n')[0]
        e = re.search(r'([\d.]+)\s*J', head)
        lv = [(float(a), float(bb), int(_num(c)))
              for a, bb, c in re.findall(r'(-[\d.]+)\s*kN\s+(-[\d.]+)\s*kN\s+([\d,]+)', rest)]
        nums = re.findall(r'^\s*([\d,]{3,})\s*$', rest, re.M)
        total = int(_num(nums[-1])) if nums else sum(x[2] for x in lv)
        out[gid] = {
            '能量': ('%s J' % e.group(1)) if e else (re.search(r'([\d.]+) mm', head).group(1) + ' mm(预置脱粘)'
                                                   if re.search(r'([\d.]+) mm', head) else '—'),
            '级别': lv,
            '总循环': total,
        }
    return out


def parse_cycles_failure(txt):
    """table of specimen cycles to failure.pdf → 同上"""
    out = {}
    blocks = re.split(r'\n(?=L1-\d{2}\b)', txt)
    for b in blocks:
        m = re.match(r'(L1-\d{2})\s*(.*)', b)
        if not m:
            continue
        gid = m.group(1)
        head = b.split('\n')[0]
        e = re.search(r'([\d.~]+)\s*J', head)
        mD = re.search(r'D\s*\(?\s*(\d+)\s*x\s*(\d+)\s*\)?', b)
        note = ('预置脱粘 D%sx%s' % mD.groups()) if mD else ''
        lv = []
        for a, bb, tag, c in re.findall(
                r'(-[\d.]+)\s+to\s+(-[\d.]+)(?:\s*\(([^)]*)\))?\s+([\d,~]+)', b):
            lv.append((float(a), float(bb), int(_num(c)), (tag or '').strip()))
        t = re.search(r'Total:\s*([\d,]+)', b)
        total = int(_num(t.group(1))) if t else sum(x[2] for x in lv)
        out[gid] = {
            '能量': ('%s J' % e.group(1)) if e else '—',
            '级别': lv,
            '总循环': total,
            '附加': note,
        }
    return out


# --------------------------------------------------------------- 数据资产
def scan_assets(gdir):
    rows = []
    for sub in sorted(os.listdir(gdir)):
        p = os.path.join(gdir, sub)
        if not os.path.isdir(p):
            continue
        fs = [f for f in glob.glob(os.path.join(p, '**', '*'), recursive=True) if os.path.isfile(f)]
        if not fs:
            continue
        ext = collections.Counter(os.path.splitext(f)[1].lower() for f in fs)
        mb = sum(os.path.getsize(f) for f in fs) / 1048576
        rows.append({'模态目录': sub, '文件数': len(fs), '体积MB': round(mb, 1),
                     '格式': ', '.join('%s×%d' % (k or '(无)', v) for k, v in ext.most_common())})
    # 组根目录的派生文件（老组才有：L1-xx*.csv / 组内 PDF）
    top = [f for f in os.listdir(gdir) if os.path.isfile(os.path.join(gdir, f))]
    if top:
        ext = collections.Counter(os.path.splitext(f)[1].lower() or '(无)' for f in top)
        rows.append({'模态目录': '(组根)', '文件数': len(top), '体积MB':
                     round(sum(os.path.getsize(os.path.join(gdir, f)) for f in top) / 1048576, 1),
                     '格式': ', '.join('%s×%d' % (k, v) for k, v in ext.most_common())})
    return rows


def fbg_rate(gdir):
    """从 FBG 头部的 Data Interleave 推采样率：1000 / interleave。"""
    d = os.path.join(gdir, 'FBG')
    fs = sorted(glob.glob(os.path.join(d, '*.txt')))
    if not fs:
        return None, None, None
    with open(fs[0], encoding='utf-8', errors='replace') as fh:
        head = ''.join(next(fh) for _ in range(30))
    m = re.search(r'Data Interleave:\s*(\d+)', head)
    rate = (1000 // int(m.group(1))) if m else None
    return rate, len(fs), (os.path.basename(fs[0]), os.path.basename(fs[-1]))


# ------------------------------------- FBG 采集覆盖与加载状态（实测 2026-10-02）
# 新组 FBG 是**突发式**采集：每个文件只有 20.0 s 数据（200 行 x 0.1 s），
# 文件间隔 420 s（谱载部分组）或 240 s ⇒ 时间占空仅约 5%，**不是连续记录**。
# 但「文件级覆盖率」在活跃期是完整的（实测 98.6% 至 100.1%）。
# 跨度里的长空隙经 AE 交叉验证为**试验暂停**（区间内 AE 事件近零），不是丢数据。
_FRAME_S = 600.0
_GAP_H = 2.0                      # 认定为「长空隙」的门槛（小时）
_ROWS_PER_FILE = 200.0            # 每个突发文件的典型数据行数（260 行总长 − 60 行表头）


def _fbg_file_times(gdir):
    import datetime as _d
    fs = sorted(glob.glob(os.path.join(gdir, 'FBG', '*.txt')))
    tt = []
    for f in fs:
        m = re.search(r'(\d{14})', os.path.basename(f))
        if not m:
            continue
        s = m.group(1)
        try:
            tt.append(_d.datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]),
                                  int(s[8:10]), int(s[10:12]),
                                  int(s[12:14])).timestamp())
        except ValueError:
            continue
    return np.asarray(sorted(tt), dtype=float)


def fbg_coverage(gdir, rate=None):
    """FBG 突发式采集的覆盖体检。

    「活跃期覆盖率」= 小间隙（<= 2 h）段内的实有文件数 / 应有 slot 数。
    ⚠️ 不要拿「首末文件之间的整段日历时间」当分母：那会把试验暂停期
    也算成缺数据（曾因此误报 L1-31 覆盖率仅 39%，实际为 100%）。

    有效观测时长 = 文件数 x 每文件记录时长，而每文件记录时长 = 行数 / 采样率。
    因此必须传入 rate（Hz）；缺省按 10 Hz 算（= 20 s）。

    对只有 1 至 2 个大文件的长记录组返回 None（「文件间隔 / 空隙」概念不适用）。
    """
    tt = _fbg_file_times(gdir)
    if len(tt) < 4:
        return None
    g = np.diff(tt)
    small = g[g <= 7200]
    if not len(small):
        return None
    iv = float(np.median(small))
    big = g[g > 7200]
    idx = np.where(g > 7200)[0]
    per_file_s = _ROWS_PER_FILE / float(rate or 10.0)
    return dict(n_files=len(tt), interval_s=iv,
                span_h=float((tt[-1] - tt[0]) / 3600.0),
                n_gap=len(big), gap_h=float(big.sum() / 3600.0),
                act_cov=100.0 * (len(tt) - len(big)) / max(small.sum() / iv, 1.0),
                obs_h=len(tt) * per_file_s / 3600.0,
                per_file_s=per_file_s,
                gaps=[(tt[i], tt[i + 1]) for i in idx])


def resolved_freq(gid):
    """读 `results/_l1cyc_{gid}.npz` 的反解载荷频率与加载时长。

    ⚠️ PDF 均未给出新组载荷频率，这两个数是**反解值**（不是实测）。
    口径：holdgap（零阶保持 + AE 静默守卫），见 `l1/ae_cycle.py`。
    """
    p = os.path.join(HERE, 'results', '_l1cyc_%s.npz' % gid)
    if not os.path.exists(p):
        return None, None, None
    z = np.load(p, allow_pickle=True)
    try:
        return (float(z['f_hz']), float(z['load_h']),
                str(z['fill']) if 'fill' in z.files else None)
    except Exception:                              # noqa: BLE001
        return None, None, None


def ae_in_window(gid, t0, t1):
    """统计时间窗 [t0, t1]（epoch 秒）内的 AE 事件数，及全程事件数。"""
    p = os.path.join(HERE, 'results', '_l1ae_frames_%s.npz' % gid)
    if not os.path.exists(p):
        return None, None
    z = np.load(p, allow_pickle=True)
    td = np.asarray(z['t_day'], dtype=float)
    n = np.asarray(z['n'], dtype=float)
    m = (td >= t0 / 86400.0) & (td <= t1 / 86400.0)
    return float(n[m].sum()), float(n.sum())


# ------------------------------------------------- 采集场次 / 时钟（实测）
# 实测结论（2026-09-30 由文件头 ASCII 起始时刻、文件写入时间、FBG 分块时间三方对齐得出）：
#   C3 变幅批：AE 采集时钟与 FBG 时钟同步（偏差 0 天）
#   C4 谱载批：AE 采集时钟比 FBG 慢整 8 天（AE 全部时间需 +8 天才是 FBG 日历）
# 注意：这里只确定「两组之间的相对平移」，绝对日期仍需数据方确认。
CLOCK_SHIFT_DAYS = {'C3 变幅 VA': 0, 'C4 谱载': 8}


def ae_segments(root, gid):
    """列出该组 AE 采集文件（按文件内嵌起始时刻排序）。

    同一组里：起始时刻连续（后一卷起点≈前一卷终点）说明是「连续采集、按大小/时段切卷」；
    起始时刻相差数天至数周说明是「分次加载、各存一卷」。
    """
    import datetime
    out = []
    d = os.path.join(root, gid, 'AE')
    for f in sorted(glob.glob(os.path.join(d, '*'))) if os.path.isdir(d) else []:
        if not os.path.isfile(f):
            continue
        st = os.stat(f)
        head = dta_start_time(f)
        out.append({'file': os.path.basename(f),
                    'mb': round(st.st_size / 1048576, 1),
                    'head': head,
                    'stamp': _parse_tw(head) or datetime.datetime.min,
                    'mtime': st.st_mtime})
    out.sort(key=lambda r: (r['stamp'], r['file']))

    # 同一启动时刻的多卷：按「写入时间单链聚类」（隙 ≤ 600 s 归为一簇）判定
    #   多文件簇 = 同一次拷贝/切分产生的副本（内容重复，必须去重）
    #   单文件且时间靠后 = DAQ 按卷续写（内容不重复，如 L1-44-2/-3 达到体积上限后另开一卷）
    by_head = collections.defaultdict(list)
    for s in out:
        by_head[s['head']].append(s)
    for s in out:
        s['role'] = ''
    for _head, grp in by_head.items():
        if len(grp) < 2:
            continue
        g = sorted(grp, key=lambda s: s['mtime'])
        clusters = [[g[0]]]
        for s in g[1:]:
            if s['mtime'] - clusters[-1][-1]['mtime'] <= 600:
                clusters[-1].append(s)
            else:
                clusters.append([s])
        for ci, c in enumerate(clusters):
            for s in c:
                s['role'] = 'copy' if len(c) > 1 else ('origin' if ci == 0 else 'cont')
    return out


def fbg_profile(gdir):
    """FBG 记录节奏：连续长记录 / 每 7 分钟 20 秒快照（含按日窗口）。"""
    import datetime
    fs = sorted(glob.glob(os.path.join(gdir, 'FBG', '*.txt')))
    if not fs:
        return {}
    sizes = [os.path.getsize(f) for f in fs]
    stamps = fbg_stamps(gdir)
    days = sorted({datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8])) for s in stamps})
    wins = []
    for d in days:
        if wins and (d - wins[-1][-1]).days <= 2:
            wins[-1].append(d)
        else:
            wins.append([d])
    style = '连续长记录' if max(sizes) > 5 * 1048576 else '每 7 分钟 20 秒快照（全天覆盖，占空约 5%）'
    return {'style': style, 'windows': wins, 'n': len(fs),
            'max_mb': round(max(sizes) / 1048576, 1)}


def _parse_tw(s):
    """'Thu May 28 13:30:57 2020' → datetime。"""
    import datetime
    for fmt in ('%a %b %d %H:%M:%S %Y',):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _fmt_dt(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')


def _shift_txt(head, shift):
    """把 AE 头部时刻按实测钟差平移到 FBG 日历。"""
    if shift is None or not head:
        return ''
    t = _parse_tw(head)
    if t is None:
        return ''
    import datetime as _dt
    return (t + _dt.timedelta(days=shift)).strftime('%Y-%m-%d %H:%M')


def ae_seg_shape(segs):
    """描述 AE 采集段结构：单段 / 同一次加载内多卷 / 分次加载。

    阈值取实测口径：新增谱载组的实际加载是连续的（FBG 全天覆盖可证），
    AE 里 1～2 天的空档是停机夜歇，只有 ≥5 天的空档才是「另一次加载」。
    """
    if not segs:
        return ''
    if len(segs) == 1:
        return '单段'
    ncopy = sum(1 for s in segs if s.get('role') == 'copy')
    if ncopy:
        return '同一采集 + %d 卷副本（须去重，勿重复累计）' % ncopy
    if all(s.get('role') in ('origin', 'cont') for s in segs):
        return '同一采集、DAQ 按卷续写（%d 卷，内容不重复）' % len(segs)
    uniq = sorted({s['stamp'] for s in segs})
    gaps = [(uniq[i + 1] - uniq[i]).total_seconds() / 86400.0 for i in range(len(uniq) - 1)]
    g = max(gaps)
    if g < 1:
        return '同一加载期连续采集（%d 卷，最长停顿 %.1f 小时）' % (len(segs), g * 24)
    if g < 5:
        return '同一加载期多段采集（%d 段，最长停顿 %.1f 天）' % (len(segs), g)
    return '分次加载（%d 段，最长间隔 %.0f 天）' % (len(segs), g)


def ae_seg_note(segs, i):
    """第 i 段（0 基）相对上一段的关系。"""
    if i == 0:
        return '首段'
    role = segs[i].get('role', '')
    if role == 'copy':
        return '与同批写入的分卷互为副本（同一份数据，须去重）'
    if role == 'cont':
        return '与上一卷同一采集启动时刻（DAQ 按卷续写，内容不重复）'
    a, b = segs[i - 1]['stamp'], segs[i]['stamp']
    d = (b - a).total_seconds() / 86400.0
    if abs(d) < 1e-9:
        return '与上一卷同一采集启动时刻'
    if d >= 5:
        return '距上一段 %.0f 天（推断为另一次加载）' % d
    if d >= 1:
        return '距上一段 %.1f 天（同一加载期内的停机夜歇）' % d
    return '与上一段接续（间隔 %.1f 小时）' % (d * 24)


# --------------------------------------------------------------- 主流程
def main(do_print_only=False):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    root = data_dir()
    print('数据根 l1 =', root)
    groups = sorted(d for d in os.listdir(root)
                    if re.fullmatch(r'L1-\d{2}', d) and os.path.isdir(os.path.join(root, d)))
    print('组数 =', len(groups))

    # --- 解析 campaign PDF
    P = {}
    for name in ['Impact_Locations', 'Damage locations', 'Damage locations variable',
                 'Damage locations spectrum', 'tables of cycles variable',
                 'table of specimen cycles to failure']:
        fp = os.path.join(root, name + '.pdf')
        P[name] = pdf_text(fp) if os.path.exists(fp) else ''
        print('  PDF %-38s %s' % (name, 'OK' if P[name] else '缺失'))
    imp1 = parse_impact_locations(P['Impact_Locations'])
    dmg2 = parse_damage_locations(P['Damage locations'])
    cyc_va = parse_cycles_variable(P['tables of cycles variable'])
    dmg_va = parse_damage_locations(P['Damage locations variable'])
    cyc_sp = parse_cycles_failure(P['table of specimen cycles to failure'])
    dmg_sp = parse_damage_locations(P['Damage locations spectrum'])

    # --- 老组 n_f：用组内 PDF 解析器（不硬编码）
    sys.path.insert(0, HERE)
    try:
        import l1_meta
    except Exception:                              # noqa: BLE001
        l1_meta = None

    sheet1, sheet3, sheet5 = [], [], []
    for g in groups:
        gd = os.path.join(root, g)
        camp = CAMPAIGN.get(g, '?')
        rate, nfbg, span = fbg_rate(gd)
        subs = sorted(d for d in os.listdir(gd) if os.path.isdir(os.path.join(gd, d)))
        segs = ae_segments(root, g)
        fbgp = fbg_profile(gd)
        shift = CLOCK_SHIFT_DAYS.get(camp)

        rec = {'组号': g, '状态': '新增' if g in NEW else '既有', '批次': camp,
               '组内PDF': '有' if os.path.exists(os.path.join(gd, g + '.pdf')) else '无',
               'AE': '有' if 'AE' in subs else '无',
               'AE格式': '', 'AE文件数': 0,
               'AE段数': len(segs), 'AE段结构': ae_seg_shape(segs),
               'FBG': '有' if 'FBG' in subs else '无', 'FBG采样率Hz': rate, 'FBG文件数': nfbg,
               'FBG节奏': fbgp.get('style', ''),
               '时钟平移(天)': ('%+d' % shift) if (shift is not None and segs) else '',
               'DFOS(LUNA)': '有' if 'LUNA' in subs else '无',
               'PZT': '有' if 'PZT' in subs else '无',
               '冲击能量': '', '载荷级别(kN)': '', '各级循环': '', 'n_f(总循环)': '', 'n_f来源': '',
               '冲击/损伤位置(蒙皮侧)': '', '备注': ''}

        # 载荷 / 循环 / 位置
        if g in cyc_va:
            c = cyc_va[g]
            rec['冲击能量'] = c['能量']
            rec['载荷级别(kN)'] = ' → '.join('%g/%g' % (x[0], x[1]) for x in c['级别'])
            rec['各级循环'] = ' + '.join('{:,}'.format(x[2]) for x in c['级别'])
            rec['n_f(总循环)'] = int(c['总循环'])
            rec['n_f来源'] = 'tables of cycles variable.pdf（含 Total 行）'
        if g in cyc_sp:
            c = cyc_sp[g]
            rec['冲击能量'] = c['能量'] if c['能量'] != '—' else '无冲击'
            if c['附加']:
                rec['冲击能量'] = (c['附加'] + '（无冲击）') if c['能量'] == '—' else \
                    (c['能量'] + ' / ' + c['附加'])
            rec['载荷级别(kN)'] = ' → '.join('%g/%g' % (x[0], x[1]) for x in c['级别'])
            rec['各级循环'] = ' + '.join(('{:,}'.format(x[2]) + ('(%s)' % x[3] if x[3] else ''))
                                     for x in c['级别'])
            rec['n_f(总循环)'] = int(c['总循环'])
            rec['n_f来源'] = 'table of specimen cycles to failure.pdf（含 Total 行）'
        # 恒幅两批：PDF 给的是「批次统一」的单一载荷，无分级
        if not rec['载荷级别(kN)'] and camp in ('C1 恒幅一批', 'C2 恒幅二批'):
            rec['冲击能量'] = '10 J（批次统一）'
            rec['载荷级别(kN)'] = '-6.5/-65（批次统一）'
            rec['各级循环'] = '等幅，无分级'
        if g in imp1:
            x, y, kind = imp1[g]
            rec['冲击/损伤位置(蒙皮侧)'] = '%g, %g (%s)' % (x, y, kind)
        for src, tag in ((dmg_va, '蒙皮侧'), (dmg_sp, '蒙皮侧'), (dmg2, '加筋条侧')):
            if g in src and src[g]:
                txt = '; '.join('%s %s' % (k, v) for k, v in src[g].items())
                rec['冲击/损伤位置(蒙皮侧)'] = (rec['冲击/损伤位置(蒙皮侧)'] + ' | ' if rec['冲击/损伤位置(蒙皮侧)'] else '') \
                    + '%s: %s' % (tag, txt)
        # 老组 n_f
        if not rec['n_f(总循环)'] and l1_meta is not None:
            try:
                m = l1_meta.load_meta(g)
                if m.get('n_f'):
                    rec['n_f(总循环)'] = int(m['n_f'])
                    rec['n_f来源'] = '组内 %s.pdf（Applied loads 表）' % g
                if m.get('foot_L'):
                    rec['备注'] = 'FBG 脚空间段 L%s / R%s' % (m['foot_L'], m['foot_R'])
            except Exception:                      # noqa: BLE001
                pass

        # AE 格式与文件数
        ae = os.path.join(gd, 'AE')
        if os.path.isdir(ae):
            aef = [f for f in glob.glob(os.path.join(ae, '**', '*'), recursive=True) if os.path.isfile(f)]
            ext = collections.Counter(os.path.splitext(f)[1].lower() for f in aef)
            rec['AE格式'] = ', '.join('%s×%d' % (k, v) for k, v in ext.most_common())
            rec['AE文件数'] = len(aef)
        if 'LUNA' in subs:
            rec['备注'] = (rec['备注'] + '; ' if rec['备注'] else '') + '含 LUNA DFOS'
        if g in NEW and rate:
            rec['备注'] = (rec['备注'] + '; ' if rec['备注'] else '') + \
                ('FBG 连续长记录' if nfbg and nfbg <= 3 else 'FBG 分块快照(%d 个)' % nfbg)
        sheet1.append(rec)

        for r in scan_assets(gd):
            sheet3.append({'组号': g, '批次': camp, **r})

        for i, s in enumerate(segs, 1):
            sheet5.append({'组号': g, '批次': camp, '类型': 'AE 采集段',
                           '编号/窗口': '%s' % s['file'], '体积MB': s['mb'],
                           'AE 头部起始': s['head'],
                           'AE 写入(结束)': _fmt_dt(s['mtime']),
                           'FBG 日历(平移后)': _shift_txt(s['head'], shift),
                           '说明': ae_seg_note(segs, i - 1)})
        for w in fbgp.get('windows', []):
            sheet5.append({'组号': g, '批次': camp, '类型': 'FBG 窗口',
                           '编号/窗口': '%s → %s（%d 天）' % (w[0], w[-1], len(w)),
                           '体积MB': '', 'AE 头部起始': '', 'AE 写入(结束)': '',
                           'FBG 日历(平移后)': '',
                           '说明': '%s；%d 个分块' % (fbgp.get('style', ''), fbgp.get('n', 0))})

    df1 = pd.DataFrame(sheet1)
    df3 = pd.DataFrame(sheet3)
    df5 = pd.DataFrame(sheet5)
    for c in ('AE文件数', 'FBG文件数'):
        df1[c] = df1[c].astype('Int64')
    df1['n_f(总循环)'] = df1['n_f(总循环)'].astype('Int64')

    sheet2 = pd.DataFrame([
        dict(批次='C1 恒幅一批', 成员=' '.join(C1_CA1), 来源PDF='Impact_Locations.pdf（+ 各组 L1-xx.pdf 给 n_f）',
             加载方式='10 J 冲击成 BVID → 等幅压-压疲劳', 载荷='−6.5 / −65 kN', 频率='2 Hz',
             模态='AE(.pridb) + FBG + LUNA(DFOS)', 特点='论文 Broer 2021 Level1–4 复现对象；唯一同时有 AE+FBG+DFOS 的一批'),
        dict(批次='C2 恒幅二批', 成员=' '.join(C2_CA2), 来源PDF='Damage locations.pdf（+ 各组 L1-xx.pdf 给 n_f）',
             加载方式='10 J 冲击 → 等幅压-压疲劳', 载荷='−6.5 / −65 kN', 频率='2 Hz',
             模态='AE(.pridb) + LUNA(DFOS)；无 FBG', 特点='已判定「不支持跨试件统一阈值的在线预警」：组间 AE 事件率相差 3 倍以上，自适应阈值反而比固定阈值差'),
        dict(批次='C3 变幅 VA（新增）', 成员=' '.join(C3_VA), 来源PDF='tables of cycles variable.pdf + Damage locations variable.pdf',
             加载方式='阶梯变幅：逐级加大载荷，每级跑固定循环数；冲击位置在加筋条脚',
             载荷='−4.0/−40 → −4.5/−45 → −5.0/−50 → −5.5/−55 → −6.0/−60 kN（逐级）',
             频率='PDF 未给；实测反解 1.075 至 1.148 Hz（4 组，口径 holdgap）',
             模态='AE(.DTA, Mistras AEwin) + FBG(sm130, 5 Hz)',
             特点='4 组共用同一套载荷级别 → 与恒幅批的「单一载荷」形成对照；对照组上无 DFOS'),
        dict(批次='C4 谱载（新增）', 成员=' '.join(C4_SP), 来源PDF='table of specimen cycles to failure.pdf + Damage locations spectrum.pdf',
             加载方式='多级块谱，每试件级别各自递增；L1-29 / L1-30 含 pristine（未冲击）段',
             载荷='按试件不同：−45.9 ~ −82.0 kN 之间多级组合',
             频率='PDF 未给；实测反解 1.821 至 2.119 Hz（7 个长测试组，口径 holdgap）',
             模态='AE(.DTA, Mistras AEwin) + FBG(sm130, 10 Hz)',
             特点='冲击能量含 10/12.3/15 J 三档；L1-41(D20x20)、L1-44(D25x20) 为预置脱粘而非冲击；'
                  'n_f 跨 4 个数量级（1,400 ~ 1,580,000）'),
    ])

    sheet4 = pd.DataFrame([
        dict(项目='AE 文件格式（已解决）', 现状='新组是 PAC/Mistras **AEwin `.DTA`**（老组是 Vallen `.pridb`）；官方见 Mistras 手册 Appendix II',
             影响='`vallenae.io` 读不了，但已自研 `l1/ae_dta.py`（与公开实现 MistrasDTA 逐字段偏差 0）',
             建议='直接用 `python l1/ae_dta.py stats|export`；新 14 组共用同一套头：2 通道 / 1 MHz / gain 0 dB / 未存波形'),
        dict(项目='AE 续写卷须递归解析', 现状='L1-44-3 以 ID 8「续文件」消息开头，其载荷内嵌了整套头消息（产品定义、硬件设置）',
             影响='不递归就会丢掉特征表 → 该卷 hit 数归零（已修复，现得 36211634 条）',
             建议='解析器必须处理 ID 8；卷间时间戳无缝衔接可当一致性校验'),
        dict(项目='AE 中 ID 2 消息未解码', 现状='ID 2 是固定 17 字节总长的一类消息，数量约为 hit 的 0.6 至 3 倍',
             影响='不影响 hit 表（ID 1）的完整性，但若它携带参数/状态信息就会遗漏',
             建议='暂不纳入；待查 Mistras 手册 Appendix II 的 ID 2 定义'),
        dict(项目='组内元信息', 现状='新组无 `L1-xx.pdf`；但 `.DTA` 内自带起始时刻/采样率/增益/特征表，n_f 与载荷只能从 campaign PDF 取',
             影响='`l1_meta.load_meta()` 对新组返回 None',
             建议='把 campaign PDF 解析结果固化成本表，或给 l1_meta 加 campaign 分支'),
        dict(项目='FBG 采样率', 现状='变幅组 5 Hz（Interleave 200）· 谱载组 10 Hz（Interleave 100）',
             影响='两批不能直接合并；且载荷频率若为 2 Hz 会欠采样（同 DFOS 的老问题）',
             建议='按批分别读取；先确认实际加载频率'),
        dict(项目='FBG 记录粒度', 现状='变幅组 1–2 个约 17 MB 的连续记录；谱载组上千个约 45 KB 的小文件',
             影响='没有统一的「5000 cycle 一块」锚（老组靠 FBG 块锚做时间对齐）',
             建议='连续记录用时间戳切块；小文件按文件序号当块'),
        dict(项目='缺失模态', 现状='新组只有 AE + FBG，无 LUNA(DFOS)、无 PZT 数据',
             影响='老组那些依赖 DFOS 的证据（`e_strain`、DFOS 块级 HI、空间热图）在新组不可用',
             建议='新组按「AE 单源 + FBG 辅助」设计分析口径'),
        dict(项目='载荷频率', 现状='PDF 均未写频率（老组为 2 Hz）。现已从 AE+FBG 时序反解：变幅批 1.075 至 1.148 Hz；谱载批 1.821 至 2.119 Hz',
             影响='① 影响块定义与预警提前量口径；② 两批相差近一倍，不能合并处理',
             建议='向数据方书面确认（尤其变幅批的 1.1 Hz）；批次间差异是设备还是加载程序尚未知'),
        dict(项目='AE 采集口径', 现状='新组为另一套采集系统（PCI2/DiSP），阈值与增益未知',
             影响='正是「AE 阈值/增益跨 campaign 不同」的又一例，跨批比较须谨慎',
             建议='先统计各组的 AE 事件数与幅值分布做横向体检'),
        dict(项目='AE 时钟平移', 现状='实测：谱载批(C4) 的 AE 文件时钟比 FBG 慢整 8 天；变幅批(C3) 两者同步',
             影响='不做平移，AE 与 FBG 的时间轴差 8 天，跨模态时间对齐必然失败',
             建议='按批平移（C4 的 AE 时间 +8 天）；绝对日期待数据方确认'),
        dict(项目='AE 分卷去重', 现状='L1-27 有 1 个整卷 + 5 个分卷，6 个文件头部启动时刻完全相同（实测数据区字节数相等、仅 4 处共 40 字节不同 → 同一份数据）',
             影响='直接叠加会把 1.18 GB 读成 2.37 GB，事件数翻倍',
             建议='按「同启动时刻 + 写入时间成簇」判定副本；用 probe_dta.py split/diff 验证'),
        dict(项目='AE 分卷续写', 现状='L1-44-2/-3 也共用同一启动时刻，但写入时间相差 4 天（-2 达到 1.9 GB 后 -3 续写）',
             影响='这两种情形不能用同一规则判断，否则会把真正的数据当副本丢掉',
             建议='判据用「数据区是否逐字节可衔接」，不能只看文件名'),
        dict(项目='FBG 采集节奏', 现状='突发式：每文件仅 20.0 s 数据（200 行 x 0.1 s）；文件间隔谱载部分组 420 s、变幅及 L1-30/35/36 为 240 s ⇒ 时间占空仅约 5%',
             影响='① 可当时间轴锚（用来校对 AE 钟差）；② 做不到「整段连续应变谱」；③ 有效观测时长仅 5.7 至 17.2 h，远小于日历跳度',
             建议='时间轴/阶段划分靠 FBG；量化应变趋势只用这 20 s 窗口'),
        dict(项目='FBG 文件级覆盖率', 现状='活跃期（去榅 > 2 h 空隙）文件覆盖率 98.6% 至 100.1%，基本完整；长空隙经 AE 交叉验证为试验暂停',
             影响='曾因分母用错（拿整段日历时间当分母）误判为「覆盖率仅 39%」，已在文档中更正',
             建议='覆盖率分母只用小间隙段；长空隙归因看 AE 事件数'),
        dict(项目='FBG 表头与数据行起点', 现状='典型文件 260 行：0 至 36 设备表头 / 37 至 46 通道标定式（17 列，含数字但不是数据）/ 48 至 57 FBG 定义 / 59 列名行 / 60 至 259 数据',
             影响='⚠️ 不能用「字段数 >= 12」判数据行起点 —— 标定式有 17 列会被误收（曾据此误判解析器丢数据）',
             建议='判据用「前两列能拼出时间戳 + 第 3 至 12 列可转浮点」'),
        dict(项目='加载状态重建口径', 现状='原来的「每帧 ±300 s 内凑到 >= 3 个 FBG 窗才采信，否则赋全局均值」已改为 holdgap（零阶保持 + AE 静默守卫）',
             影响='旧口径下 50% 至 70% 的帧拿到同一个常数，循环轴在那段退化成线性斜坡；L1-31 因此反解出 2.72 Hz（偏高 36%）',
             建议='新口径下 L1-31 为 2.12 Hz（−5.6%），块内一致性大幅改善；旧口径用 ae_cycle.py --fill mean 复现'),
        dict(项目='未提供的试件', 现状='变幅表里的 L1-22 未提供；一批表里的 L1-23 未提供',
             影响='变幅组实际 4/5 组，一批 4/5 组',
             建议='如需完整对照，向数据方补齐'),
        dict(项目='传感器编号方向', 现状='L1-24 的 AE 传感器标注为 L5R1…L1R5（与其余组的 L5R5…L1R1 相反）',
             影响='若沿用「L#/R# 对应左右」的通道语义会左右错位',
             建议='涉及 L1-24 的空间分析单独确认方向'),
        dict(项目='路径归属', 现状='27 组全在 E:\\l1；仓库 l1/ 已建好 27 个 junction（2026-10-04，`tools/relink_data.ps1 -Apply`）',
             影响='仓库内相对路径 l1/L1-xx/ 对全部 27 组都有效（以前只有 13 组）',
             建议='换盘只改 paths.json 的 data_root 再跑 -Apply；'
                  'relink_data.ps1 的通配展开已改为「仓库侧 ∪ 数据盘侧」，盘上新增的组会自动补齐'),
        dict(项目='分析窗长', 现状='剖面分析按 140 行切一个窗 ⇒ 变幅批（5 Hz）= 28.0 s；谱载批（10 Hz）= 14.0 s',
             影响='两批的「一块」物理时长不同，跨批比 e_st 的采样密度会失真（形状漂移本身仍可比）',
             建议='跨批只比趋势不比绝对值；窗长写在此处，避免后人误以为两批都是 28 s'),
    ])

    # ---- sheet6：FBG 采集覆盖与加载状态（实测 2026-10-02）----
    sheet6, sheet7 = [], []
    for g in groups:
        gdir = os.path.join(root, g)
        camp = CAMPAIGN.get(g, '?')
        cov = fbg_coverage(gdir, rate=fbg_rate(gdir)[0])
        f, lh, fill = resolved_freq(g)
        rec = {'组号': g, '批次': camp}
        if cov is None:
            rec.update({'FBG文件数': None, '文件间隔s': None, '跨度h': None,
                        '>2h空隙数': None, '空隙合计h': None,
                        '活跃期覆盖率%': None, '有效观测h': None,
                        '反解载荷频率Hz': (round(f, 3) if f else None),
                        '加载时长h': (round(lh, 1) if lh else None),
                        '记录形态': '大文件（1 至 2 个连续长记录）',
                        '说明': '「文件间隔 / 空隙」概念不适用'})
        else:
            tot = None
            for (a, b) in cov['gaps']:
                ev, tot = ae_in_window(g, a, b)
                sheet7.append(dict(
                    组号=g, 批次=camp,
                    起=_fmt_dt(a), 止=_fmt_dt(b),
                    时长h=round((b - a) / 3600.0, 1),
                    区间内AE事件=ev,
                    占全程AE事件比=(round(100.0 * ev / tot, 3)
                                    if tot else None),
                    判定=('试验暂停（AE 静默）'
                          if (tot and ev / tot <= 0.02) else '需人工确认')))
            rec.update({'FBG文件数': cov['n_files'],
                        '文件间隔s': int(round(cov['interval_s'])),
                        '跨度h': round(cov['span_h'], 1),
                        '>2h空隙数': cov['n_gap'],
                        '空隙合计h': round(cov['gap_h'], 1),
                        '活跃期覆盖率%': round(cov['act_cov'], 1),
                        '有效观测h': round(cov['obs_h'], 1),
                        '反解载荷频率Hz': (round(f, 3) if f else None),
                        '加载时长h': (round(lh, 1) if lh else None),
                        '记录形态': '小文件突发式（每文件 %.0f s 数据）'
                                   % cov['per_file_s'],
                        '说明': ('fill=%s；文件数少，覆盖率估计不可靠'
                                % (fill or '?') if cov['n_files'] < 100
                                else 'fill=%s' % (fill or '?'))})
        sheet6.append(rec)
    df6 = pd.DataFrame(sheet6)
    df7 = pd.DataFrame(sheet7) if sheet7 else pd.DataFrame(
        [dict(组号='(无)', 批次='', 起='', 止='', 时长h=None,
              区间内AE事件=None, 占全程AE事件比=None, 判定='无 >2 h 空隙')])

    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as w:
        df1.to_excel(w, sheet_name='试件总表', index=False)
        sheet2.to_excel(w, sheet_name='批次定义', index=False)
        df3.to_excel(w, sheet_name='数据资产', index=False)
        df5.to_excel(w, sheet_name='采集场次与时钟', index=False)
        sheet4.to_excel(w, sheet_name='接入注意', index=False)
        df6.to_excel(w, sheet_name='FBG采集覆盖', index=False)
        df7.to_excel(w, sheet_name='长空隙清单', index=False)
        # 若已跑过 `python l1/ae_dta.py export`，把逐卷 AE 体检结果一并带进来
        ae_csv = os.path.join(HERE, 'AE特征概览.csv')
        if os.path.exists(ae_csv):
            pd.read_csv(ae_csv).to_excel(w, sheet_name='AE特征概览', index=False)
            print('  并入 AE特征概览.csv')
        else:
            print('  未发现 AE特征概览.csv（可跑 l1/ae_dta.py export 生成）')
    print('\n已写出 ->', OUT_XLSX)

    with pd.option_context('display.width', 220, 'display.max_columns', 50):
        print('\n===== 试件总表 =====')
        print(df1[['组号', '状态', '批次', '组内PDF', 'AE格式', 'AE文件数',
                   'FBG', 'FBG采样率Hz', 'FBG文件数', 'DFOS(LUNA)',
                   '冲击能量', 'n_f(总循环)']].to_string(index=False))
    return df1, sheet2, df3, sheet4


if __name__ == '__main__':
    main('--print' in sys.argv[1:])
