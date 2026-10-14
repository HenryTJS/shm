# -*- coding: utf-8 -*-
"""从**数据集自带的 PDF** 里抽取试件元信息（不依赖任何外部输入）。

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

# 27 个已交付的组；根目录汇总 PDF 本来就额外覆盖 L1-22 / L1-23
# （2026-10-09：这两个试件的数据已由数据方补齐，故一并纳入 ⇒ docs/details.md §45）
GROUPS = ['L1-03', 'L1-04', 'L1-05', 'L1-09', 'L1-49', 'L1-50', 'L1-51',
          'L1-52', 'L1-54', 'L1-55', 'L1-56', 'L1-59', 'L1-60',
          'L1-06', 'L1-13', 'L1-14', 'L1-24',
          'L1-25', 'L1-27', 'L1-29', 'L1-30', 'L1-31', 'L1-34', 'L1-35',
          'L1-36', 'L1-41', 'L1-44',
          'L1-22', 'L1-23']

ROOT_LOC_FILES = ['Damage locations variable.pdf', 'Damage locations spectrum.pdf',
                  'Damage locations.pdf']
ROOT_CYC_FILES = ['tables of cycles variable.pdf',
                  'table of specimen cycles to failure.pdf']
IMPACT_FILE = 'Impact_Locations.pdf'

# ------------------------------ 人工维护的元数据表（**权威数据源**，2026-10-10 起）
# 代码**默认读这三张表**，不再解析 PDF；只有 `--refresh` 才回到 PDF 重新填充。
# 表里的每一格都可以直接用 Excel 改，改动会在下次跑脚本时生效。
META_DIR = os.path.join(HERE, 'meta')
META_SPECIMENS = os.path.join(META_DIR, 'specimens.csv')
META_LEVELS = os.path.join(META_DIR, 'load_levels.csv')
META_SENSORS = os.path.join(META_DIR, 'ae_sensors.csv')

SPECIMEN_COLS = [
    'gid', 'batch', 'group_pdf',
    'skin_length_mm', 'skin_width_mm', 'vel_longitudinal_ms', 'vel_lateral_ms',
    'vel_caveat', 'sensor_frame', 'impact_J', 'impact_loc_raw',
    'loc_kind', 'loc_x_mm', 'loc_x_hi_mm', 'loc_y_mm', 'loc_y_hi_mm',
    'loc_side', 'loc_src', 'loc_page', 'loc_note',
    'imp_kind', 'imp_x_mm', 'imp_x_hi_mm', 'imp_y_mm', 'imp_y_hi_mm',
    'imp_src', 'imp_page', 'imp_note',
    'cyc_desc', 'cyc_impact', 'cyc_src', 'cyc_total', 'n_f', 'note',
]
LEVEL_COLS = ['gid', 'src', 'level', 'min_kn', 'max_kn', 'freq_hz', 'cycles',
              'approx', 'note']
SENSOR_COLS = ['gid', 'sensor_no', 'x_mm', 'y_mm']

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


# ------------------------------- B0. 位置解析的**人工核对记录表**（不是推断逻辑）
# 数据已交付完毕、不会再变（2026-10-09），所以不写通用识别算法，
# 直接把量出来的值连同来历记在这里，便于复核。
#
# 背景：`Impact_Locations.pdf` 每页只有 `L1-xx` + **两个裸数字** + `Impact location`
# （L1-23 是 `Disbond location`），**没有 X/Y 标签、也没有「距哪一侧」字样**，
# 方向只存在于矢量尺寸线里。2026-10-09 逐组人工核对，确认原解析错了两处。
# 逐组人工核对的经过写在 `l1/meta/specimens.csv` 的 `*_note` 列里；
# 仍需拍板的点见 `docs/PDF解析核查清单.md`（只有 20 余行）。
LOC_OVERRIDE = {
    'L1-05': dict(
        x_mm=50.0, y_mm=80.0,
        pos_note='原文两个裸数字是「80 50」，组内 PDF 写「on stiffener (left '
                 'stiffener). 5 cm from left edge x 8 cm from top」⇒ 横向 50、纵向 80。'
                 '**修正前装反了**（横向被填成 80）。'),
    'L1-09': dict(
        x_mm=82.5, y_mm=100.0,
        pos_note='原文两个裸数字是「100 82.5」，组内 PDF 写「on stiffener (centre). '
                 '8.25 from edge x 10 cm from top right」⇒ 横向 82.5、纵向 100。'
                 '**修正前装反了**（横向被填成 100）。'),
    'L1-22': dict(
        kind='disbond_area', x_mm=82.42, x_hi_mm=118.95,
        y_mm=50.16, y_hi_mm=90.06,
        pos_note='文字层只有「Lower edge of disbond / Y(mm) 45」，**既无 X 也无尺寸**。'
                 '2026-10-09 用 pdfplumber 读矢量图量出：该页独有矩形 '
                 'x 420.72..474.60 pt、top 317.52..378.12 pt；以试件主体轮廓 '
                 '243.36 x 369.12 pt 对 165 x 243 mm 标定 ⇒ 上面这四个数。'
                 '自检：左边界 82.42 落在中轴线 82.50（= 165/2）上，'
                 '右边界 118.95 与加劲肋轮廓右边缘 474.60 pt 完全重合。'
                 '⚠该图**不是等比例**（x/y 比例差 3.0%、端部长度多画 23%、'
                 '文字标注的 45 与量出的下缘 50.16 差 5 mm），故量级精度约 ±5 mm，'
                 '不能排除它是 40 x 40。'),
    'L1-23': dict(
        kind='disbond_area', x_mm=45.0, x_hi_mm=75.0, y_mm=45.0, y_hi_mm=75.0,
        pos_note='原文两个裸数字是「30 45」，**不是坐标**：2026-10-09 人工读图确认 '
                 '30 是正方形脱粘区的**边长**、45 是其右下角顶点距右侧与下侧的距离。'
                 '这里按「矩形范围」写成 x 45..75、y 45..75（距右 / 距下），'
                 '与 L1-22 的表达方式一致。'),
}


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
    """根目录 3 份矢量位置 PDF → {gid: dict(kind, x_mm, x_hi_mm, y_mm, y_hi_mm,
    side, source, page, pos_note)}。

    文字层能直接给出的就取（`Impact Center location` / `Lower edge of disbond`
    后面跟的 `X(mm) Y(mm)` 两列）；文字层残缺的（L1-22 只写了 `Y(mm) 45`）
    用 `LOC_OVERRIDE` 里人工量出的值补齐。
    """
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
            kind, x, xhi, y, yhi = None, None, None, None, None
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
                # 有些页只有 "Y(mm) 45" 这种残缺描述（L1-22）
                my = re.search(r'Y\(mm\)\s*([\d.]+)', s)
                if my:
                    kind, y = 'disbond_lower_edge', float(my.group(1))
            rec = dict(kind=kind, x_mm=x, x_hi_mm=xhi, y_mm=y, y_hi_mm=yhi,
                       side=side, source=name, page=i + 1, pos_note='')
            ov = LOC_OVERRIDE.get(gid)
            if ov:
                rec.update({k: v for k, v in ov.items() if k != 'pos_note'})
                rec['pos_note'] = ov.get('pos_note', '')
            res.setdefault(gid, rec)
    return res


def parse_impact_locations(root, gmeta=None):
    """`Impact_Locations.pdf`（位图，但文字层可用）→ {gid: dict(...)}。

    每页只有 `L1-xx` + **两个裸数字** + `Impact location`（L1-23 是
    `Disbond location`），**没有 X/Y 标签、也没有「距哪一侧」字样**，
    所以「先横后纵」的顺序并不固定。方向 2026-10-09 已逐组人工核对清楚，
    写在 `LOC_OVERRIDE` 里；这里只负责把裸数字读出来，再套覆盖表。

    `gmeta` 参数保留仅为兼容既有调用（不再参与定向推断）。
    """
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
        if len(nums) < 2:
            continue
        rec = dict(kind=kind, source=IMPACT_FILE, page=i + 1, pos_note='',
                   x_mm=float(nums[0]), y_mm=float(nums[1]))
        ov = LOC_OVERRIDE.get(gid)
        if ov:
            rec.update({k: v for k, v in ov.items() if k != 'pos_note'})
            rec['pos_note'] = ov.get('pos_note', '')
        res[gid] = rec
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
            sm = sum(p['cycles'] for p in prog)
            mt = re.search(r'Total:\s*([\d,]+)', body)
            if mt:
                total = int(mt.group(1).replace(',', ''))
            else:
                # ⚠️ 变幅表（`tables of cycles variable.pdf`）**不写 "Total:" 标签**，
                # 只在末级后面留一个**裸数字**当合计（例：L1-06 的 `... 12,300 202,300`）。
                # 2026-10-10 用户指出：原先把这 5 组记成「没有 Total 行」是错的 ——
                # 它们**都有**合计，而且**都与各级之和吻合**（自洽性检验是**通过**的）。
                # 判据取「等于各级之和，且不等于任何一级的循环数」；后者用来排除
                # 「只有一级」的情形（L1-34 / L1-36：唯一那一级就是全部，无合计可谈）。
                lvl = {p['cycles'] for p in prog}
                total = None
                for mm in re.finditer(r'(?<![\d,.])(\d{1,3}(?:,\d{3})+|\d{4,})(?![\d,.])',
                                      body):
                    v = int(mm.group(1).replace(',', ''))
                    if v not in lvl and v == sm:
                        total = v
                        break
            mi = re.search(r'^(.*?)(?=-?[\d.]+\s*kN|\Z)', body)
            res[gid] = dict(desc=norm(mi.group(1))[:80] if mi else '',
                            program=prog, total=total, sum_cycles=sm,
                            source=name,
                            impact=norm(body)[:60])
    return res


# ======================== D. 人工维护的元数据表（读写） ========================
def _f(s):
    """CSV 单元格 → float（空串 → None）。已是数字时原样返回。"""
    if isinstance(s, bool):
        return float(s)
    if isinstance(s, (int, float)):
        return float(s)
    s = (s or '').strip()
    return float(s) if s else None


def _i(s):
    """CSV 单元格 → int（空串 → None）。已是数字时原样返回。"""
    v = _f(s)
    return int(v) if v is not None else None


def _b(s):
    return (s or '').strip().lower() in ('1', 'true', 'yes', 'y', '是')


def _cell(v):
    if v is None or v == '':
        return ''
    if isinstance(v, float) and v == int(v):
        return '%g' % v
    return str(v)


def parse_all_pdfs(root):
    """一次性解析全部 PDF → (gmeta, loc, imp, cyc)。**只给 `--refresh` 用**。"""
    gmeta = {}
    for g in GROUPS:
        d = parse_group_pdf(g, root)
        if d:
            gmeta[g] = d
    loc = parse_root_locations(root)
    imp = parse_impact_locations(root, gmeta)
    cyc = parse_root_cycles(root)
    return gmeta, loc, imp, cyc


def _csv_write(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def _batch_map():
    out = {}
    try:
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from shm.datasets import campaign_members
        for k in ('C1', 'C2', 'C3', 'C4'):
            for g in campaign_members(k):
                out[g] = k
    except Exception:                                        # noqa: BLE001
        pass
    return out


def write_meta_tables(root, gmeta, loc, imp, cyc):
    """把解析结果写成三张人工可编辑的表（**会覆盖**，只在 `--refresh` 时调）。"""
    try:
        from ae_cycle import N_F
    except Exception:                                        # noqa: BLE001
        N_F = {}
    batch = _batch_map()

    rows = []
    for g in GROUPS:
        d = gmeta.get(g) or {}
        e = loc.get(g) or {}
        p = imp.get(g) or {}
        c = cyc.get(g) or {}
        tot = (c.get('total') or c.get('sum_cycles')
               or (sum(x['cycles'] for x in c.get('program') or []) or None))
        rows.append(dict(
            gid=g, batch=batch.get(g, ''), group_pdf=d.get('source', ''),
            skin_length_mm=_cell(d.get('skin_length_mm')),
            skin_width_mm=_cell(d.get('skin_width_mm')),
            vel_longitudinal_ms=_cell(d.get('vel_longitudinal_ms')),
            vel_lateral_ms=_cell(d.get('vel_lateral_ms')),
            vel_caveat=d.get('vel_caveat', ''),
            sensor_frame=d.get('sensor_frame', ''),
            impact_J=_cell(d.get('impact_J')),
            impact_loc_raw=d.get('impact_loc_raw', ''),
            loc_kind=e.get('kind', '') or '',
            loc_x_mm=_cell(e.get('x_mm')), loc_x_hi_mm=_cell(e.get('x_hi_mm')),
            loc_y_mm=_cell(e.get('y_mm')), loc_y_hi_mm=_cell(e.get('y_hi_mm')),
            loc_side=e.get('side') or '', loc_src=e.get('source', ''),
            loc_page=_cell(e.get('page')), loc_note=e.get('pos_note', '') or '',
            imp_kind=p.get('kind', '') or '',
            imp_x_mm=_cell(p.get('x_mm')), imp_x_hi_mm=_cell(p.get('x_hi_mm')),
            imp_y_mm=_cell(p.get('y_mm')), imp_y_hi_mm=_cell(p.get('y_hi_mm')),
            imp_src=p.get('source', ''), imp_page=_cell(p.get('page')),
            imp_note=p.get('pos_note', '') or '',
            cyc_desc=c.get('desc', ''), cyc_impact=c.get('impact', ''),
            cyc_src=c.get('source', ''), cyc_total=_cell(c.get('total')),
            n_f=_cell(N_F.get(g) or tot or ''), note=''))
    _csv_write(META_SPECIMENS, SPECIMEN_COLS, rows)

    lv = []
    for g in GROUPS:
        for j, x in enumerate((gmeta.get(g) or {}).get('load_program') or []):
            lv.append(dict(gid=g, src='group_pdf', level=j + 1,
                           min_kn=_cell(x['min_kn']), max_kn=_cell(x['max_kn']),
                           freq_hz=_cell(x.get('freq_hz')), cycles=x['cycles'],
                           approx='', note=''))
        for j, x in enumerate((cyc.get(g) or {}).get('program') or []):
            lv.append(dict(gid=g, src='root_cycles', level=j + 1,
                           min_kn=_cell(x['min_kn']), max_kn=_cell(x['max_kn']),
                           freq_hz='', cycles=x['cycles'],
                           approx='True' if x.get('approx') else '',
                           note=x.get('note', '')))
    _csv_write(META_LEVELS, LEVEL_COLS, lv)

    sn = []
    for g in GROUPS:
        for sid, x, y in (gmeta.get(g) or {}).get('ae_sensors') or []:
            sn.append(dict(gid=g, sensor_no=sid, x_mm=_cell(x), y_mm=_cell(y)))
    _csv_write(META_SENSORS, SENSOR_COLS, sn)
    print('写出 %s（%d 行）' % (META_SPECIMENS, len(rows)))
    print('写出 %s（%d 行）' % (META_LEVELS, len(lv)))
    print('写出 %s（%d 行）' % (META_SENSORS, len(sn)))
    print('⚠ 以后请直接改这三张表；再跑 `--refresh` 会把它们覆盖。')


def read_meta_tables():
    """读三张人工维护的表 → (gmeta, loc, imp, cyc)，与四个 `parse_*` 同构。"""
    for p in (META_SPECIMENS, META_LEVELS, META_SENSORS):
        if not os.path.exists(p):
            raise SystemExit(
                '缺少元数据表 %s\n先跑一次 `python l1/pdf_specimen_meta.py '
                '--refresh` 从 PDF 生成。' % p)
    gmeta, loc, imp, cyc = {}, {}, {}, {}
    with open(META_SPECIMENS, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            g = r['gid']
            gm = {}
            for k in ('skin_length_mm', 'skin_width_mm', 'vel_longitudinal_ms',
                      'vel_lateral_ms', 'impact_J'):
                if _f(r.get(k)) is not None:
                    gm[k] = _f(r[k])
            for k in ('vel_caveat', 'sensor_frame', 'impact_loc_raw'):
                if (r.get(k) or '').strip():
                    gm[k] = r[k].strip()
            if (r.get('group_pdf') or '').strip():
                gm['source'] = r['group_pdf'].strip()
            if _i(r.get('n_f')) is not None:
                gm['n_f'] = _i(r['n_f'])
            if gm:
                gmeta[g] = gm
            for pre, dst in (('loc', loc), ('imp', imp)):
                e = {'pos_note': (r.get(pre + '_note') or '').strip()}
                if (r.get(pre + '_kind') or '').strip():
                    e['kind'] = r[pre + '_kind'].strip()
                for k in ('x_mm', 'x_hi_mm', 'y_mm', 'y_hi_mm'):
                    if _f(r.get('%s_%s' % (pre, k))) is not None:
                        e[k] = _f(r['%s_%s' % (pre, k)])
                if (r.get(pre + '_side') or '').strip():
                    e['side'] = r[pre + '_side'].strip()
                if (r.get(pre + '_src') or '').strip():
                    e['source'] = r[pre + '_src'].strip()
                if _i(r.get(pre + '_page')) is not None:
                    e['page'] = _i(r[pre + '_page'])
                if e.get('kind') or e.get('source'):
                    # 补齐与 `parse_root_locations` / `parse_impact_locations` 同构的键，
                    # 免得下游用 `d['x_hi_mm']` 这种写法踩 KeyError
                    for k in ('kind', 'x_mm', 'x_hi_mm', 'y_mm', 'y_hi_mm',
                              'side', 'page'):
                        e.setdefault(k, None)
                    e.setdefault('source', '')
                    dst[g] = e
            c = {}
            if (r.get('cyc_desc') or '').strip():
                c['desc'] = r['cyc_desc'].strip()
            if (r.get('cyc_impact') or '').strip():
                c['impact'] = r['cyc_impact'].strip()
            if (r.get('cyc_src') or '').strip():
                c['source'] = r['cyc_src'].strip()
            if _i(r.get('cyc_total')) is not None:
                c['total'] = _i(r['cyc_total'])
            if c:
                c['program'] = []
                cyc[g] = c
    with open(META_LEVELS, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            g = r['gid']
            x = dict(min_kn=_f(r.get('min_kn')), max_kn=_f(r.get('max_kn')),
                     cycles=_i(r.get('cycles')) or 0,
                     note=(r.get('note') or '').strip(),
                     approx=_b(r.get('approx')))
            if (r.get('src') or '').strip() == 'group_pdf':
                if _f(r.get('freq_hz')) is not None:
                    x['freq_hz'] = _f(r['freq_hz'])
                gmeta.setdefault(g, {}).setdefault('load_program', []).append(x)
            else:
                cyc.setdefault(g, {'program': []})['program'].append(x)
    for g, c in cyc.items():
        c['sum_cycles'] = sum(x['cycles'] for x in c.get('program') or [])
        c.setdefault('total', None)
        c.setdefault('desc', '')
        c.setdefault('impact', '')
        c.setdefault('source', '')
        c.setdefault('n_f', None)
    with open(META_SENSORS, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            gmeta.setdefault(r['gid'], {}).setdefault('ae_sensors', []).append(
                (int(float(r['sensor_no'])), _f(r['x_mm']), _f(r['y_mm'])))
    return gmeta, loc, imp, cyc


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
    cols = ['gid', 'kind', 'x_mm', 'y_mm', 'x_hi_mm', 'y_hi_mm', 'side', 'source',
            'page', 'pos_note', 'raw_desc', 'nx_cm', 'nx_side', 'ny_cm', 'ny_side',
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
            x_hi_mm=(d or {}).get('x_hi_mm', ''),
            y_hi_mm=(d or {}).get('y_hi_mm', ''),
            side=(d or {}).get('side') or '',
            source=(d or {}).get('source', ''),
            page=(d or {}).get('page', ''),
            pos_note=(d or {}).get('pos_note', ''),
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
            gid=g, n_f=(_i(gm.get('n_f')) or N_F.get(g) or tot),
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
    ap.add_argument('--write', action='store_true',
                    help='把派生表（l1_impact_truth.csv 等）写到 results/')
    ap.add_argument('--refresh', action='store_true',
                    help='重新从 PDF 解析并**覆盖** l1/meta/ 三张表 —— '
                         '会丢掉人工修改，只在明确要重建时才用')
    a = ap.parse_args()
    root = data_root()
    print('数据根目录: %s\n' % root)

    if a.refresh:
        print('⚠ --refresh：重新解析全部 PDF 并覆盖 l1/meta/ 三张表')
        write_meta_tables(root, *parse_all_pdfs(root))
        print()

    # 唯一的数据来源：人工维护的三张表
    gmeta, loc, imp, cyc = read_meta_tables()
    print('D. 数据来源：`l1/meta/*.csv`（人工维护）—— 不再解析 PDF')
    n_gp = sum(1 for g in GROUPS if (gmeta.get(g) or {}).get('source'))
    print('A. 组内 PDF：%d / %d 组有' % (n_gp, len(GROUPS)))
    print('   %-7s %7s %7s %9s %9s %-9s  %s' % (
        '组', 'skinL', 'skinW', '纵向m/s', '横向m/s', '声速可信', 'AE 传感器坐标'))
    for g in GROUPS:
        d = gmeta.get(g) or {}
        if not d.get('source'):
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

    print('\nB. 根目录位置 PDF：矢量图 %d 组，位图 %d 组' % (len(loc), len(imp)))
    print('   %-7s %-20s %8s %8s %8s %8s %-11s %s' % (
        '组', '种类', 'X(mm)', 'X高(mm)', 'Y(mm)', 'Y高(mm)', '观察面', '来源'))
    for g in GROUPS:
        d = loc.get(g)
        e = imp.get(g)
        if d:
            print('   %-7s %-20s %8s %8s %8s %8s %-11s %s p%d' % (
                g, d.get('kind') or '未识别', d.get('x_mm') or '—',
                d.get('x_hi_mm') or '—', d.get('y_mm') or '—',
                d.get('y_hi_mm') or '—', d.get('side') or '(无标注)',
                (d.get('source') or '')[:22], d.get('page') or 0))
        elif e:
            print('   %-7s %-20s %8s %8s %8s %8s %-11s %s p%d' % (
                g, e.get('kind') or '未识别', e.get('x_mm') or '—',
                e.get('x_hi_mm') or '—', e.get('y_mm') or '—',
                e.get('y_hi_mm') or '—', '(无标注)',
                (e.get('source') or '')[:22], e.get('page') or 0))
        else:
            print('   %-7s %-20s %8s %8s %8s %8s %-11s %s' % (
                g, '缺', '—', '—', '—', '—', '—', '—'))
    print('\n   位置解析备注（非空才列）：')
    for g in GROUPS:
        n = (loc.get(g) or {}).get('pos_note') or (imp.get(g) or {}).get('pos_note')
        if n:
            print('     %-7s %s' % (g, n))

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
