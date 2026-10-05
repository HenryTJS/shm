# -*- coding: utf-8 -*-
"""从**数据集自带的 PDF** 里抽取试件元信息（不依赖任何外部输入）。

背景（2026-10-04 更正）
----------------------
此前认为「AE 传感器坐标与声速不在数据集里、需要向数据方索取」。**这是错的**：
公开数据集虽然无索取渠道，但自带文档里就有 —— 只是分散在两类 PDF 里。

  A. **组内 PDF**（仅老 13 组有）：`E:\\l1\\<gid>\\<gid>.pdf`
     含 AE 传感器坐标表、实测声速（纵向/横向）、传感器布置图、
     ODiSi-B(DFOS) 位置、试件尺寸、以及带逐级循环数的载荷程序。
  B. **根目录汇总 PDF**（`E:\\l1\\*.pdf`，6 份，覆盖全部 27 组）
     · `Damage locations variable.pdf` (C3 五页) / `spectrum.pdf` (C4 十页) /
       `locations.pdf` (C2 九页)：每页一张试件平面图，给出冲击中心或脱粘下缘的
       毫米坐标，含「skin side / stiffener side」观察面标注。
     · `tables of cycles variable.pdf` (C3) / `table of specimen cycles to failure.pdf` (C4)：
       每个试件的**完整分级载荷程序**（每级载荷 + 循环数 + 合计）。
     · `Impact_Locations.pdf`：位图，但**文字层仍可用**，给出 L1-03/04/05/09/23 的位置。

⚠️ 已知不一致（必须随数据一起记录，不能抹平）
  · 组内 PDF 的 AE 传感器表只有 **4 个**探头（与 C1/C2 的 4 通道一致）；
    而根目录 `Damage locations *.pdf` 的平面图标注 `AE sensors` 的是 **10 个**位置
    （`L5R5 L4R4 L3R3 L2R2 L1R1`）—— 与 FBG 的 10 通道数吻合。
    两者归谁尚未定论，**本工具两套都抽、分开存**，不做合并。
  · C3 的 `Damage locations variable.pdf` 页面**没有** skin/stiffener 标注，
    而 C2/C4 页面有 ⇒ 观察面归属对 C3 未知。
  · 位置描述的文字形式逐组不同（见 `impact_truth_check.py` 的结论）。

用法
----
    python l1/pdf_specimen_meta.py            # 解析并打印核对表（不写文件）
    python l1/pdf_specimen_meta.py --write    # 同时写出 CSV
输出：
    results/l1_specimen_meta.csv   长表：gid, field, value, unit, source, page
    results/l1_specimen_meta.txt   人读核对表
"""

import argparse
import csv
import glob
import io
import os
import re
import sys

import numpy as np
import pdfplumber

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)

# 27 个已交付的组；根目录汇总 PDF 还额外覆盖 L1-22 / L1-23（数据未提供）
GROUPS = ['L1-03', 'L1-04', 'L1-05', 'L1-09', 'L1-49', 'L1-50', 'L1-51',
          'L1-52', 'L1-54', 'L1-55', 'L1-56', 'L1-59', 'L1-60',
          'L1-06', 'L1-13', 'L1-14', 'L1-24',
          'L1-25', 'L1-27', 'L1-29', 'L1-30', 'L1-31', 'L1-34', 'L1-35',
          'L1-36', 'L1-41', 'L1-44']

ROOT_LOC_FILES = ['Damage locations variable.pdf', 'Damage locations spectrum.pdf',
                  'Damage locations.pdf']
ROOT_CYC_FILES = ['tables of cycles variable.pdf',
                  'table of specimen cycles to failure.pdf']
IMPACT_FILE = 'Impact_Locations.pdf'

# ⚠️ `l1/results/l1_conditions.csv` 里这 9 行是**手工整理**的产物（含人工填写的
# `impact_loc` 中文描述与 `anomaly`），本工具**逐字保留**它们，只重建其余行。
CURATED_GIDS = {'L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56',
                'L1-59', 'L1-60'}


def data_root():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    try:
        from shm import paths
        r = paths.data_root()
        if r and os.path.isdir(os.path.join(r, 'l1')):
            return os.path.join(r, 'l1')
    except Exception:                                        # noqa: BLE001
        pass
    return os.path.join(REPO, 'l1')


def norm(s):
    """把 PDF 抽出的文本压成单空格形式，便于正则。"""
    return re.sub(r'\s+', ' ', s or '').strip()


def pages_text(fp):
    with pdfplumber.open(fp) as pdf:
        return [p.extract_text() or '' for p in pdf.pages]


# ---------------------------------------------------------------- A. 组内 PDF
def parse_group_pdf(gid, root):
    """老组的组内 PDF → 尺寸 / 声速 / AE 传感器坐标 / 载荷程序 / 冲击描述。"""
    fp = os.path.join(root, gid, '%s.pdf' % gid)
    if not os.path.exists(fp):
        return None
    out = {'source': os.path.join('l1', gid, '%s.pdf' % gid)}
    txt = norm(' '.join(pages_text(fp)))
    out['raw'] = txt

    # 试件尺寸（skin 长 x 宽）：文本形如 "Skin width Top [mm] Middle [mm] Bottom [mm] 165 165 165"
    m = re.search(r'Skin width[^0-9]*([\d.]+)\s+([\d.]+)\s+([\d.]+)', txt)
    if m:
        out['skin_width_mm'] = float(m.group(2))       # 取 Middle
    m = re.search(r'Skin length[^0-9]*([\d.]+)\s+([\d.]+)', txt)
    if m:
        out['skin_length_mm'] = float(m.group(1))

    # 声速。⚠️ C1 批（L1-03/04/05/09）自述为**失效后测量**，需一并标出不可信。
    m = re.search(r'longitudinal[^:]*:\s*([\d.]+)\s*m/s', txt, re.I)
    if m:
        out['vel_longitudinal_ms'] = float(m.group(1))
    m = re.search(r'lateral[^:]*:\s*([\d.]+)\s*m/s', txt, re.I)
    if m:
        out['vel_lateral_ms'] = float(m.group(1))
    mc = re.search(r'performed after specimen failure.{0,120}?useful', txt, re.I)
    if mc:
        out['vel_caveat'] = norm(mc.group(0))

    # AE 传感器坐标表： "Sensor # X-location [mm] Y-location [mm] 1 145 190 2 145 20 ..."
    m = re.search(r'Sensor # X-location \[mm\] Y-location \[mm\](.{0,200}?)'
                  r'(?:\*|\Z)', txt)
    if m:
        nums = re.findall(r'\b(\d+)\s+(\d+)\s+(\d+)\b', m.group(1))
        if nums:
            out['ae_sensors'] = [(int(a), float(b), float(c)) for a, b, c in nums]
    m = re.search(r'as seen from ([a-z\- ]+?)\s*\(([^)]*)\)', txt, re.I)
    if m:
        out['sensor_frame'] = '%s (%s)' % (m.group(1).strip(), m.group(2).strip())

    # 载荷程序： "5000 -6.5 kN -65 kN 2 Hz" 或 "43,702 -6.5 kN -65 kN 2 Hz"
    prog = []
    for mm in re.finditer(r'([\d,]+)\s+(-?[\d.]+)\s*kN\s+(-?[\d.]+)\s*kN\s+'
                          r'([\d.]+)\s*Hz', txt):
        prog.append(dict(cycles=int(mm.group(1).replace(',', '')),
                         min_kn=float(mm.group(2)),
                         max_kn=float(mm.group(3)),
                         freq_hz=float(mm.group(4))))
    if prog:
        out['load_program'] = prog

    # 冲击能量与位置描述
    m = re.search(r'Impact\s+Load\s+([\d.]+)\s*Joules', txt, re.I) or \
        re.search(r'Impact\s+([\d.]+)\s*Joules', txt, re.I)
    if m:
        out['impact_J'] = float(m.group(1))
    m = re.search(r'Location\s*:?\s*(.{0,120}?)(?=-?[\d,]+\s+-?[\d.]+\s*kN|\Z)',
                  txt, re.I)
    if m and m.group(1).strip():
        out['impact_loc_raw'] = norm(m.group(1))
    return out


# ------------------------------------------------- B. 根目录位置 PDF（矢量+位图）
def parse_root_locations(root):
    """根目录 3 份矢量位置 PDF → {gid: dict(kind, x_mm, y_mm, y_hi_mm, side)}。"""
    res = {}
    for name in ROOT_LOC_FILES:
        fp = os.path.join(root, name)
        if not os.path.exists(fp):
            continue
        for i, t in enumerate(pages_text(fp)):
            s = norm(t)
            m = re.search(r'L1[-\s]*(\d{2})\b', s)
            if not m:
                continue
            gid = 'L1-%s' % m.group(1)
            side = None
            ms = re.search(r'\((skin|stiffener)\s*side\)', s, re.I)
            if ms:
                side = ms.group(1).lower()
            kind, x, y, yhi = None, None, None, None
            mi = re.search(r'Impact\s*Center\s*location.*?X\(mm\)\s*Y\(mm\)\s*'
                           r'([\d.]+)\s+([\d.]+)(?:\s*-\s*([\d.]+))?', s, re.I)
            md = re.search(r'Lower\s*edge\s*of\s*disbond.*?X\(mm\)\s*Y\(mm\)\s*'
                           r'([\d.]+)\s+([\d.]+)(?:\s*-\s*([\d.]+))?', s, re.I)
            if mi:
                kind = 'impact_center'
                x = float(mi.group(1))
                y = float(mi.group(2))
                yhi = float(mi.group(3)) if mi.group(3) else None
            elif md:
                kind = 'disbond_lower_edge'
                x = float(md.group(1))
                y = float(md.group(2))
                yhi = float(md.group(3)) if md.group(3) else None
            else:
                # 有些页只有 "Y(mm) 45" 这种残缺描述
                my = re.search(r'Y\(mm\)\s*([\d.]+)', s)
                if my:
                    kind, y = 'disbond_lower_edge', float(my.group(1))
            res.setdefault(gid, dict(kind=kind, x_mm=x, y_mm=y, y_hi_mm=yhi,
                                     side=side, source=name, page=i + 1))
    return res


def parse_impact_locations(root):
    """`Impact_Locations.pdf`（位图，但文字层可用）→ {gid: (x_mm, y_mm, kind)}。"""
    fp = os.path.join(root, IMPACT_FILE)
    if not os.path.exists(fp):
        return {}
    res = {}
    for i, t in enumerate(pages_text(fp)):
        s = norm(t)
        m = re.search(r'L1[-\s]*(\d{2})\b', s)
        if not m:
            continue
        gid = 'L1-%s' % m.group(1)
        kind = 'disbond_location' if 'disbond' in s.lower() else 'impact_location'
        nums = re.findall(r'([\d.]+)\s*mm', s)
        if len(nums) >= 2:
            res[gid] = dict(x_mm=float(nums[0]), y_mm=float(nums[1]), kind=kind,
                            source=IMPACT_FILE, page=i + 1)
    return res


# ------------------------------------------------------- C. 根目录 cycle 表
def parse_root_cycles(root):
    """两张 cycle 表 → {gid: dict(desc, impact, program=[(min,max,cycles)], total)}。"""
    res = {}
    for name in ROOT_CYC_FILES:
        fp = os.path.join(root, name)
        if not os.path.exists(fp):
            continue
        s = norm(' '.join(pages_text(fp)))
        for block in re.split(r'(?=L1-\d\d\b)', s)[1:]:
            m = re.match(r'(L1-\d\d)\b(.*)', block)
            if not m:
                continue
            gid, body = m.group(1), m.group(2)
            prog = []
            # 无括号形式（变幅表）："-4.0 kN -40 kN 10,000"
            for mm in re.finditer(r'(-?[\d.]+)\s*kN\s+(-?[\d.]+)\s*kN\s+([\d,]+)',
                                  body):
                prog.append(dict(min_kn=float(mm.group(1)), max_kn=float(mm.group(2)),
                                 note='', approx=False,
                                 cycles=int(mm.group(3).replace(',', ''))))
            # 括号形式（谱载表）："-50.4 to -78.0 (impact 10 J) 240,000"
            # ⚠️ 括号里**可能含数字**（如 "impact 10 J"），不能用 [^)\d]* 排掉数字。
            for mm in re.finditer(r'(-?[\d.]+)\s+to\s+(-?[\d.]+)\s*'
                                  r'(?:\(([^)]*)\))?\s*(~?[\d][\d,]*)', body):
                prog.append(dict(min_kn=float(mm.group(1)), max_kn=float(mm.group(2)),
                                 note=(mm.group(3) or '').strip(),
                                 approx=mm.group(4).startswith('~'),
                                 cycles=int(mm.group(4).lstrip('~').replace(',', ''))))
            mt = re.search(r'Total:\s*([\d,]+)', body)
            total = int(mt.group(1).replace(',', '')) if mt else None
            mi = re.search(r'^(.*?)(?=-?[\d.]+\s*kN|\Z)', body)
            res[gid] = dict(desc=norm(mi.group(1))[:80] if mi else '',
                            program=prog, total=total,
                            sum_cycles=sum(p['cycles'] for p in prog),
                            source=name,
                            impact=norm(body)[:60])
    return res


def _best_pos(gid, loc, imp):
    """优先用矢量图（带观察面），退化到位图。返回 dict 或 None。"""
    d = loc.get(gid)
    if d and d.get('kind'):
        return d
    e = imp.get(gid)
    return e


def emit_impact_truth(root, gmeta, loc, imp):
    """重建 `l1_impact_truth.csv`：27 组全覆盖 + 溯源。

    在原表基础上**增列**：`kind / x_mm / y_mm / y_hi_mm / side / source / page`。
    原有列保留（`nx_cm` 等只对组内 PDF 有的 13 组有值；`meas_*` 只对 C1/C2 有 DFOS
    定位结果的组有值），因此它是旧表的**超集**。
    """
    cols = ['gid', 'kind', 'x_mm', 'y_mm', 'y_hi_mm', 'side', 'source', 'page',
            'raw_desc', 'nx_cm', 'nx_side', 'ny_cm', 'ny_side',
            'prose_x_mm', 'prose_y_mm', 'flip_x_mm', 'flip_y_mm',
            'meas_cx_mm', 'meas_cy_mm', 'n_good']
    LX, LY = 165.0, 243.0
    rows = []
    for g in GROUPS:
        d = _best_pos(g, loc, imp)
        gm = gmeta.get(g, {})
        raw = gm.get('impact_loc_raw', '')
        nxc = nxs = nyc = nys = None
        mx = re.search(r'([\d.]+)\s*cm\s*from\s*(right|left)\s*edge', raw, re.I)
        my = re.search(r'([\d.]+)\s*cm\s*from\s*(top|bottom)', raw, re.I)
        if mx:
            nxc, nxs = float(mx.group(1)), mx.group(2).lower()
        if my:
            nyc, nys = float(my.group(1)), my.group(2).lower()
        # ① 文字描述推出的坐标：⚠️ 这是**旧脚本的坐标约定**（由 ae_locate 的 IMPACT 反推），
        # 与图纸自己标的 mm 值**逐组不一致**（见 impact_truth_check.py 的结论）。
        # 列名带 prose_ 前缀就是为了防止被当成真值 —— **不要**拿它算定位误差。
        pxk = 10.0 * nxc if (nxc is not None and nxs == 'right') else \
            (LX - 10.0 * nxc if nxc is not None else None)
        pyk = (10.0 * nyc if nys == 'top' else (LY - 10.0 * nyc)) \
            if nyc is not None else None
        # ② 图纸 mm 值 + 180° 旋转 —— **推荐的真值**（列名 flip_*）。
        # 依据：C2 批（locations.pdf，图纸值均为 (115,160)）的实测质心在旋转后
        # 显著更近（L1-49 = 4.1 mm、L1-59 = 15.2 mm），不旋转则对不上。
        # ⚠️ 注意：**哪一帧才对取决于图纸的观察面**，不能当成普适事实；
        # C1 批（impact_location，另一份 PDF）无旋转反而更近，但那批的声速
        # 自述 after-failure 不可信 ⇒ 其绝对位置本身就不可用，不能用来判定约定。
        # 依据与量化见 docs/details.md §21。
        dx_, dy_ = (d or {}).get('x_mm'), (d or {}).get('y_mm')
        fxk = (LX - float(dx_)) if dx_ not in (None, '') else None
        fyk = (LY - float(dy_)) if dy_ not in (None, '') else None
        # 实测质心（仅已有 DFOS 定位 npz 的组）
        cx = cy = ng = None
        fp = os.path.join(RES, '_l1_loc_%s.npz' % g)
        if os.path.exists(fp):
            z = np.load(fp)
            xs, ys, r = z['x'], z['y'], z['rms_us']
            good = r <= 5.0
            if good.sum():
                cx, cy, ng = (float(xs[good].mean()), float(ys[good].mean()),
                              int(good.sum()))
        rows.append(dict(
            gid=g,
            kind=(d or {}).get('kind', ''),
            x_mm=(d or {}).get('x_mm', ''),
            y_mm=(d or {}).get('y_mm', ''),
            y_hi_mm=(d or {}).get('y_hi_mm', ''),
            side=(d or {}).get('side') or '',
            source=(d or {}).get('source', ''),
            page=(d or {}).get('page', ''),
            raw_desc=raw, nx_cm=nxc, nx_side=nxs or '', ny_cm=nyc, ny_side=nys or '',
            prose_x_mm=(round(pxk, 1) if pxk is not None else ''),
            prose_y_mm=(round(pyk, 1) if pyk is not None else ''),
            flip_x_mm=(round(fxk, 1) if fxk is not None else ''),
            flip_y_mm=(round(fyk, 1) if fyk is not None else ''),
            meas_cx_mm=cx, meas_cy_mm=cy, n_good=ng))
    out = os.path.join(RES, 'l1_impact_truth.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    n_pos = sum(1 for r in rows if r['x_mm'] != '')
    print('写出 %s：%d 组，其中有位置坐标的 %d 组' % (out, len(rows), n_pos))


def emit_conditions(root, gmeta, cyc, loc, imp):
    """扩展 `l1_conditions.csv`。

    ⚠️ 原 9 行（C2）**逐字保留**（含手写的 `impact_loc` 与 `anomaly`），
    只**追加**缺失组。无 DFOS 的组，`n_seg / n_ai / seg_cv / seg_x500_over_nf` 留空。
    """
    out = os.path.join(RES, 'l1_conditions.csv')
    cols = ['gid', 'n_f', 'n_load_rows', 'n_load_levels', 'min_load', 'max_load',
            'freq', 'impact_J', 'impact_stiffener', 'impact_loc', 'anomaly',
            'n_seg', 'n_ai', 'seg_cv', 'seg_x500_over_nf', 'site']
    old, have = [], set()
    if os.path.exists(out):
        with open(out, encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                # ⚠️ 只保留**手工整理**的那几行（C2 的 DFOS 批次）——
                # 它们含人工填写的 `impact_loc` 中文描述与 `anomaly`，无法自动重建。
                # 其余行都是本工具生成的，必须每次重建，否则修正改不动旧输出
                # （本工具第一次跑就把上一次的产物当成「原文」了）。
                if r['gid'] in CURATED_GIDS:
                    old.append(r)
                    have.add(r['gid'])
    try:
        from ae_cycle import N_F
    except Exception:                                        # noqa: BLE001
        N_F = {}

    def rev_f(g):
        fp = os.path.join(RES, '_l1cyc_%s.npz' % g)
        if not os.path.exists(fp):
            return ''
        return round(float(np.load(fp, allow_pickle=True)['f_hz']), 4)

    new = []
    for g in GROUPS:
        if g in have:
            continue
        c = cyc.get(g)
        gm = gmeta.get(g, {})
        prog = (c or {}).get('program') or gm.get('load_program') or []
        tot = ((c or {}).get('total') or (c or {}).get('sum_cycles')
               or (sum(p['cycles'] for p in prog) if prog else None))
        d = _best_pos(g, loc, imp) or {}
        # 载荷频率：优先用 PDF 明写的，否则用反解值（C1/C2 的 PDF 均写 2 Hz）
        fr = ''
        if prog:
            fs = sorted({p.get('freq_hz') for p in prog if p.get('freq_hz')})
            if fs:
                fr = fs[0] if len(fs) == 1 else ';'.join('%g' % v for v in fs)
        if not fr:
            fr = rev_f(g)
        # 冲击能量："I 10J" / "I ~10J" / "I 12.3J" / "7.4 J" / "10 J"
        desc = (c or {}).get('desc', '')
        mj = re.search(r'I\s*~?\s*([\d.]+)\s*J', desc) or \
            re.search(r'([\d.]+)\s*J\b', desc)
        ij = (float(mj.group(1)) if mj else gm.get('impact_J', ''))
        # 预置脱粘（非冲击）："D20x20" / "D(25x20)" / "25 mm"
        md = re.search(r'D\s*\(?\s*(\d+\s*x\s*\d+)', desc) or \
            re.search(r'([\d.]+)\s*mm', desc)
        note = ('脱粘 ' + md.group(1).replace(' ', '')) if (md and not mj) else ''
        # ⚠️ impact_stiffener 是原表的「加筋条侧 (left/right)」。
        # 新抽到的是图纸的**观察面**（skin / stiffener side），不是同一回事，
        # 宁缺不填 —— 否则把「skin side」当成「加筋条在 skin 侧」就是错信息。
        new.append(dict(
            gid=g, n_f=N_F.get(g, tot),
            n_load_rows='', n_load_levels=len(prog),
            min_load=min(p['min_kn'] for p in prog) if prog else '',
            max_load=min(p['max_kn'] for p in prog) if prog else '',
            freq=fr,
            impact_J=ij,
            impact_stiffener='',
            impact_loc=(gm.get('impact_loc_raw') or d.get('kind') or ''),
            anomaly='-', n_seg='', n_ai='', seg_cv='', seg_x500_over_nf='',
            site='%s·%s%s' % (d.get('side') or '无标注', d.get('kind', ''),
                             ('·' + note) if note else '')))
    with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(old)
        w.writerows(new)
    print('写出 %s：保留原 %d 行 + 新增 %d 行 = %d 行'
          % (out, len(old), len(new), len(old) + len(new)))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    root = data_root()
    print('数据根目录: %s\n' % root)

    gmeta = {}
    for g in GROUPS:
        d = parse_group_pdf(g, root)
        if d:
            gmeta[g] = d
    print('A. 组内 PDF：%d / %d 组有' % (len(gmeta), len(GROUPS)))
    print('   %-7s %7s %7s %9s %9s %-9s  %s' % (
        '组', 'skinL', 'skinW', '纵向m/s', '横向m/s', '声速可信', 'AE 传感器坐标'))
    for g in GROUPS:
        d = gmeta.get(g)
        if not d:
            print('   %-7s %7s %7s %9s %9s %-9s  %s' % (
                g, '—', '—', '—', '—', '—', '无组内 PDF'))
            continue
        sen = '; '.join('#%d(%.0f,%.0f)' % x for x in d.get('ae_sensors', [])) or '未抽到'
        ok = '否(C1自述)' if 'vel_caveat' in d else '是'
        print('   %-7s %7s %7s %9s %9s %-9s  %s' % (
            g, d.get('skin_length_mm', '—'), d.get('skin_width_mm', '—'),
            d.get('vel_longitudinal_ms', '—'), d.get('vel_lateral_ms', '—'),
            ok, sen))
        if g in ('L1-03', 'L1-04') or d.get('impact_loc_raw'):
            print('           冲击 %s J / %s' % (
                d.get('impact_J', '—'), d.get('impact_loc_raw', '—')[:70]))

    loc = parse_root_locations(root)
    imp = parse_impact_locations(root)
    print('\nB. 根目录位置 PDF：矢量图 %d 组，位图 %d 组' % (len(loc), len(imp)))
    print('   %-7s %-22s %8s %8s %8s %-12s %s' % (
        '组', '种类', 'X(mm)', 'Y(mm)', 'Y高(mm)', '观察面', '来源'))
    for g in GROUPS:
        d = loc.get(g)
        e = imp.get(g)
        if d:
            print('   %-7s %-22s %8s %8s %8s %-12s %s p%d' % (
                g, d['kind'] or '未识别', d['x_mm'], d['y_mm'], d['y_hi_mm'],
                d['side'] or '(无标注)', d['source'][:22], d['page']))
        elif e:
            print('   %-7s %-22s %8s %8s %8s %-12s %s p%d' % (
                g, e['kind'], e['x_mm'], e['y_mm'], '—', '(无标注)',
                e['source'][:22], e['page']))
        else:
            print('   %-7s %-22s %8s %8s %8s %-12s %s' % (
                g, '缺', '—', '—', '—', '—', '—'))

    cyc = parse_root_cycles(root)
    print('\nC. 根目录 cycle 表：%d 组（与 ae_cycle.N_F 交叉核对）' % len(cyc))
    try:
        from ae_cycle import N_F
    except Exception:                                        # noqa: BLE001
        N_F = {}
    for g in sorted(cyc):
        c = cyc[g]
        tot = c['total'] if c['total'] else c['sum_cycles']
        nf = N_F.get(g)
        mark = ''
        if nf is None:
            mark = '  (N_F 无此组)'
        elif tot == nf:
            mark = '  ✓ 与 N_F 一致'
        else:
            mark = '  ✗ N_F=%s 差 %s' % (nf, (tot - nf) if tot else '?')
        lv = ' | '.join('%g~%g x%s' % (p['min_kn'], p['max_kn'], p['cycles'])
                        for p in c['program'])
        print('   %-7s total=%-10s %s%s' % (g, tot, lv, mark))

    if a.write:
        rows = []
        for g in GROUPS:
            d = gmeta.get(g, {})
            for k, v in d.items():
                if k == 'raw':
                    continue
                if k == 'ae_sensors':
                    for sid, x, y in v:
                        rows.append(dict(gid=g, field='ae_sensor_%d_x' % sid,
                                         value=x, unit='mm',
                                         source=d.get('source', ''), page=''))
                        rows.append(dict(gid=g, field='ae_sensor_%d_y' % sid,
                                         value=y, unit='mm',
                                         source=d.get('source', ''), page=''))
                elif k == 'load_program':
                    for j, p in enumerate(v):
                        rows.append(dict(gid=g, field='load_level_%d' % (j + 1),
                                         value='%g~%g kN x%d' % (
                                             p['min_kn'], p['max_kn'], p['cycles']),
                                         unit='', source=d.get('source', ''), page=''))
                else:
                    rows.append(dict(gid=g, field=k, value=v, unit='',
                                     source=d.get('source', ''), page=''))
            for src, key in ((loc, 'loc'), (imp, 'imp')):
                e = src.get(g)
                if not e:
                    continue
                for k, v in e.items():
                    if k == 'source':
                        continue
                    rows.append(dict(gid=g, field='%s_%s' % (key, k), value=v,
                                     unit='mm' if k.endswith(('_mm', '_hi_mm')) else '',
                                     source=e['source'], page=e.get('page', '')))
            c = cyc.get(g)
            if c:
                for j, p in enumerate(c['program']):
                    rows.append(dict(gid=g, field='rootcyc_level_%d' % (j + 1),
                                     value='%g to %g kN x%s%s' % (
                                         p['min_kn'], p['max_kn'], p['cycles'],
                                         (' (%s)' % p['note']) if p['note'] else ''),
                                     unit='', source=c['source'], page=''))
                rows.append(dict(gid=g, field='rootcyc_total', value=c['total'],
                                 unit='cycles', source=c['source'], page=''))
        out = os.path.join(RES, 'l1_specimen_meta.csv')
        with open(out, 'w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=['gid', 'field', 'value', 'unit',
                                               'source', 'page'])
            w.writeheader()
            w.writerows(rows)
        print('\n写出 %s（%d 行）' % (out, len(rows)))
        emit_impact_truth(root, gmeta, loc, imp)
        emit_conditions(root, gmeta, cyc, loc, imp)


if __name__ == '__main__':
    main()
