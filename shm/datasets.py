# -*- coding: utf-8 -*-
"""四类数据集的**单一定义源**（single source of truth）—— 一个文件覆盖全部数据。

为什么需要它
------------
原来「这个组属于哪个批次 / 有哪些传感器 / 文件是什么格式 / n_f 是多少 /
AE 时钟要不要平移」散落在至少 6 个地方：

    shm/config.py          DEFAULT_GROUPS（主样本 5 组）+ 常量
    l1/survey_groups.py    C3_VA / C4_SP / CAMPAIGN / CLOCK_SHIFT_DAYS
    l1/ae_cycle.py         N_F（14 组硬编码）+ 数据根走 shm.paths
    l1/ae_hi.py            N_F（又一份 14 组硬编码）
    l1/ae_frames.py        CLOCK_SHIFT（天数）
    l1/l1_meta.py          n_f 靠逐组 PDF 解析（盘不在时静默返回 None）
    l1/results/l1_specimen_meta.csv   已解析好的 29 组元信息（长表）

⇒ 后果：同一事实有多份拷贝，改一处忘一处（本仓已多次踩到 docstring 过期），
  而且盘不在时某些入口**静默**退化。

本模块的立场
------------
- **不新增事实**：能推断的就扫盘推断（传感器/格式/文件数），
  能读表的就读权威表（`l1/results/l1_specimen_meta.csv`），
  只有**表里推不出来的**才写成静态表（C3 与 C4 的归属、时钟平移、采样率）。
- **可自检**：`python -m shm.datasets` 会把静态表与盘上实际情况对一遍，
  不一致就报出来（这是本模块存在的意义）。
- 依赖只有标准库 + `shm.paths`（不 import `l1/*`，保持依赖方向干净）。

数据集
------
    main  : 内部疲劳机主样本 001–027（正式口径 016–020）
    l1    : 公开集 ReMAP/TU-Delft + UPatras，27 个可用组，分 4 个批次
    phmdc : PHM2019 铝搭接件 T1–T8（外部方法验证平台）

用法
----
    python -m shm.datasets                      # 总表 + 自检
    python -m shm.datasets --dataset l1
    from shm.datasets import groups, meta, n_f
    groups('l1')                 # 盘上实际存在的组
    meta('L1-31')['n_f']         # 966000
    meta('L1-31')['campaign']    # 'L1 谱载+FBG'（短别名 C4）
"""

import argparse
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from shm.paths import data_root                                   # noqa: E402

META_CSV = os.path.join(REPO, 'l1', 'results', 'l1_specimen_meta.csv')

# ---------------------------------------------------------------- 数据集定义
DATASETS = {
    'main': dict(
        label='main  主样本（内部疲劳机，016-020）',
        target='main',                       # data_root/<target>/<gid>
        pattern=r'^\d{3}$',
        official=['016', '017', '018', '019', '020'],   # = shm.config.DEFAULT_GROUPS
    ),
    'l1': dict(
        label='l1  ReMAP / TU-Delft L1（29 组；按模态分 4 组，见 L1_CAMPAIGNS）',
        target='l1',
        pattern=r'^L1-\d{2}$',
        official=None,                        # 见 L1_CAMPAIGNS
    ),
    'phmdc': dict(
        label='phmdc  PHM2019 铝搭接件（T1-T8）',
        target='phmdc',
        pattern=r'^T[1-8]$',
        official=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'],
    ),
}

# ------------------------------------------------- L1 模态分组（静态表，扫盘推不出来）
# **为什么这么分组，而不是叫「第几批」**：这四组的差异是**数据本身**的 —— AE 采集格式、有无
# FBG / DFOS、加载谱、AE 时钟差 —— 直接决定代码走哪条分支。「第几批」只是下载/处理的先后，
# 没有任何技术含义（2026-10-06 按用户意见改名）。
# 组成员与 l1/survey_groups.py 的 C1/C2/C3_VA/C4_SP 一致；src 标注该事实的原始出处。
L1_CAMPAIGNS = {
    'L1 恒幅+FBG+DFOS': dict(
        # 2026-10-09：L1-23 的数据由数据方补齐（pridb + FBG + LUNA/DFOS + 组内 PDF），
        # 签名与 C1 完全一致 ⇒ 归入本组。⚠️ 但它**不是单级恒幅**：PDF 的载荷表是
        # -5/-50 kN 至 100,000 循环后**提到 -6/-60 kN**，共 438,000 循环，且中途因
        # 试验机/PZT 故障停机多次（见 l1/results/l1_conditions.csv 的 anomaly 列）。
        members=['L1-03', 'L1-04', 'L1-05', 'L1-09', 'L1-23'], ae='pridb',
        # PZT：**全盘递归搜过 E:\l1，一个 PZT 目录都没有**（2026-10-06 实测）。
        # 旧文档里「新组无 DFOS / PZT」的写法会让人以为老组有 PZT，已更正。
        fbg=True, dfos=True, pzt=False, clock_shift_days=0, src='survey_groups'),
    'L1 恒幅+DFOS': dict(
        members=['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55',
                 'L1-56', 'L1-59', 'L1-60'], ae='pridb',
        fbg=False, dfos=True, pzt=False, clock_shift_days=0, src='survey_groups'),
    'L1 变幅VA+FBG': dict(
        # 2026-10-09：L1-22 的数据由数据方补齐（.DTA + FBG，无 DFOS），签名与 C3 一致。
        # 载荷谱 6 级共 345,000 循环；带**预置脱粘**（Damage locations variable.pdf：
        # Lower edge of disbond，只给了 Y=45 mm，无 X）—— 同 C4 的 L1-41 / L1-44。
        members=['L1-06', 'L1-13', 'L1-14', 'L1-24', 'L1-22'], ae='dta',
        fbg=True, dfos=False, pzt=False, clock_shift_days=0, src='survey_groups',
        fbg_rate_hz=5.0),
    'L1 谱载+FBG': dict(
        members=['L1-25', 'L1-27', 'L1-29', 'L1-30', 'L1-31', 'L1-34',
                 'L1-35', 'L1-36', 'L1-41', 'L1-44'], ae='dta',
        fbg=True, dfos=False, pzt=False, clock_shift_days=8, src='survey_groups',
        fbg_rate_hz=10.0),
}

# 短别名：文档、历史结论与对话里一直用 C1 至 C4 指这四个分组。**继续可用**；
# 新代码建议直接用语义名，或 `ALIAS_OF[key]` 反查。
CAMPAIGN_ALIAS = {
    'C1': 'L1 恒幅+FBG+DFOS',
    'C2': 'L1 恒幅+DFOS',
    'C3': 'L1 变幅VA+FBG',
    'C4': 'L1 谱载+FBG',
}
ALIAS_OF = {v: k for k, v in CAMPAIGN_ALIAS.items()}
# 数据方未提供的组（写在文档里，供自检时解释「27 个而不是 29 个」）
MISSING_GROUPS = {'L1-22': '变幅表里列出但未提供', 'L1-23': '一批表里列出但未提供'}

_CAMPAIGN_OF = {g: c for c, d in L1_CAMPAIGNS.items() for g in d['members']}


# ---------------------------------------------------------------- 基础访问
def root():
    """数据根（来自 paths.json / SHM_DATA_ROOT）。"""
    return data_root()


def dataset_dir(dataset):
    return os.path.join(root(), DATASETS[dataset]['target'])


def groups(dataset=None):
    """盘上**实际存在**的组；不给 dataset 就返回 {dataset: [gid]}。"""
    if dataset is None:
        return {k: groups(k) for k in DATASETS}
    d = DATASETS[dataset]
    base = dataset_dir(dataset)
    if not os.path.isdir(base):
        return []
    pat = re.compile(d['pattern'])
    return sorted(x for x in os.listdir(base)
                  if pat.match(x) and os.path.isdir(os.path.join(base, x)))


def gdir(gid):
    """组的绝对目录。"""
    for k, d in DATASETS.items():
        if re.match(d['pattern'], gid):
            return os.path.join(root(), d['target'], gid)
    return None


def dataset_of(gid):
    for k, d in DATASETS.items():
        if re.match(d['pattern'], gid):
            return k
    return None


# ---------------------------------------------------------------- 元信息表
def meta_table():
    """读 `l1/results/l1_specimen_meta.csv`（长表）→ {gid: {field: value}}。

    该表由 `l1/pdf_specimen_meta.py` 从数据集自带 PDF 解析并人工校对，
    覆盖全部 27 个 L1 组，是本项目**几何 / 载荷程序 / 传感器坐标**的权威源。
    """
    out = {}
    if not os.path.exists(META_CSV):
        return out
    with open(META_CSV, encoding='utf-8-sig', newline='') as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r['gid'], {})[r['field']] = r['value']
    return out


def _n_f_from_table(f):
    """从元信息条目推 n_f（三种表形态，勿混）。

    - **C4 谱载**：`rootcyc_total` 直接给总数。实测 10 组与 `l1/ae_cycle.py` 的
      硬编码 `N_F` **逐组一致**。
    - **C3 变幅**：`rootcyc_total` 是**空值**，只有分级 `rootcyc_level_k` ⇒
      总数 = **各级求和**（L1-06：10000+80000+30000+70000+12300 = **202300**，
      与 `N_F` 一致；若误取「最大级」会得 80000，差 2.5 倍 —— 本模块第一版
      就是这个错，被交叉核对抳出来）。
    - **C1 / C2**：只有 `load_level_*`，形如 `-6.5~-65 kN x152458` ⇒ 取 `x` 后的数。
    """
    v = f.get('rootcyc_total')
    if v and str(v).strip().isdigit():
        return int(v)
    lv = []
    for k, val in sorted(f.items()):
        if k.startswith('rootcyc_level_'):
            for m in re.finditer(r'x\s*([\d,]+)', str(val)):
                lv.append(int(m.group(1).replace(',', '')))
    if lv:
        return sum(lv)                      # 变幅：各级求和
    best = None
    for k, val in f.items():
        if k.startswith('load_level_'):
            for m in re.finditer(r'x\s*([\d,]+)', str(val)):
                n = int(m.group(1).replace(',', ''))
                best = n if best is None else max(best, n)
    return best


def cross_check_nf():
    """与 `l1/ae_cycle.py` 的硬编码 `N_F` 对一遍（迁移期体检用）。

    两边完全一致才说明本模块的推法与既有口径同源。不一致就列出来。
    （依赖方向：l1 → shm，所以这里只能用惰性导入，且只作显式诊断用。）
    """
    try:
        from l1.ae_cycle import N_F as REF
    except Exception as e:                                       # noqa: BLE001
        print('  [跳过] 无法导入 l1.ae_cycle: %s' % e)
        return []
    bad = []
    for g, want in sorted(REF.items()):
        got = n_f(g)
        if got != want:
            bad.append('%s: 元信息表推出 %s，但 ae_cycle.N_F = %s' % (g, got, want))
    print('交叉核对 n_f：%d 组对 %d 组，不一致 %d 条'
          % (len(REF), len(groups('l1')), len(bad)))
    for x in bad:
        print('  [不一致] ' + x)
    return bad


def _scan_files(gid):
    """扫盘推断「有哪些模态 / 什么格式」（不依赖任何静态表）。"""
    d = gdir(gid)
    out = dict(exists=bool(d and os.path.isdir(d)), n_files=0, has={}, files={})
    if not out['exists']:
        return out
    for sub in ('AE', 'FBG', 'LUNA', 'PZT'):
        p = os.path.join(d, sub)
        n = len(glob.glob(os.path.join(p, '*'))) if os.path.isdir(p) else 0
        if n:
            out['has'][sub] = n
            out['n_files'] += n
    for pat, key in (('*光纤.csv', 'FO'), ('*分布式应变.csv', 'DFOS'),
                     ('*声发射.csv', 'AE_csv'), ('*应变.csv', 'STRAIN'),
                     ('*.pdf', 'PDF')):
        hit = sorted(os.path.basename(x) for x in glob.glob(os.path.join(d, pat)))
        if hit:
            out['has'][key] = len(hit)
            out['n_files'] += len(hit)
            out['files'][key] = hit
    ae = os.path.join(d, 'AE')
    if os.path.isdir(ae):
        out['ae_format'] = ('pridb' if glob.glob(os.path.join(ae, '*.pridb'))
                            else 'dta' if glob.glob(os.path.join(ae, '*.DTA'))
                            else None)
    return out


def meta(gid):
    """单个组的**合并视图**：静态批次表 + 权威元信息表 + 盘上实际情况。

    返回 dict(dataset, campaign, dir, n_f, clock_shift_days, fbg_rate_hz,
              ae_format, has, files, geom, load, impact)
    """
    ds = dataset_of(gid)
    scan = _scan_files(gid)
    tab = meta_table().get(gid, {})
    camp = _CAMPAIGN_OF.get(gid)
    cinfo = L1_CAMPAIGNS.get(camp, {}) if camp else {}
    geom = {k: tab[k] for k in ('skin_width_mm', 'skin_length_mm',
                                'sensor_frame') if k in tab}
    for i in (1, 2, 3, 4):
        for ax in ('x', 'y'):
            k = 'ae_sensor_%d_%s' % (i, ax)
            if k in tab:
                geom[k] = tab[k]
    load = {k: v for k, v in tab.items() if k.startswith(('load_level_',
                                                         'rootcyc_'))}
    impact = {k: v for k, v in tab.items()
              if k.startswith('imp_') or k.startswith('loc_')}
    return dict(
        gid=gid, dataset=ds, campaign=camp,
        dataset_label=DATASETS[ds]['label'] if ds else None,
        dir=gdir(gid), exists=scan['exists'], n_files=scan['n_files'],
        n_f=_n_f_from_table(tab),
        clock_shift_days=cinfo.get('clock_shift_days', 0),
        fbg_rate_hz=cinfo.get('fbg_rate_hz'),
        ae_format=scan.get('ae_format') or cinfo.get('ae'),
        has=scan['has'], files=scan['files'],
        geom=geom, load=load, impact=impact,
        declared=dict(ae=cinfo.get('ae'), fbg=cinfo.get('fbg'),
                      dfos=cinfo.get('dfos'), pzt=cinfo.get('pzt')),
        meta_src='l1/results/l1_specimen_meta.csv' if tab else None,
    )


def meta_all():
    """一次把全部 29 组读成 dict（避免每组的 CSV 重读；CSV 小，但扫盘不便宜）。"""
    return {g: meta(g) for g in groups('l1')}


def n_f(gid):
    """失效循环数（取自权威元信息表；没有就返回 None）。"""
    return meta(gid)['n_f']


# ---------------------------------------------------------------- n_f / 批次 API
# n_f 的**兜底常量**：权威源是 `l1/results/l1_specimen_meta.csv`（由 PDF 解析 + 人工校对），
# 但那是生成物（`results/` 可清理）⇒ 这里留一份单拷贝兜底，并用 cross_check_nf 对账。
N_F_FALLBACK = {
    'L1-06': 202300, 'L1-13': 243000, 'L1-14': 217000, 'L1-24': 242000,
    'L1-25': 1580000, 'L1-27': 529000, 'L1-29': 1300000, 'L1-30': 10150,
    'L1-31': 966000, 'L1-34': 1400, 'L1-35': 452000, 'L1-36': 4500,
    'L1-41': 1820000, 'L1-44': 1160000,
    # 2026-10-09 补入：数据此前未提供，但元信息表本来就覆盖它们
    'L1-22': 345000, 'L1-23': 438000,
}

_NF_CACHE = None


def n_f_table():
    """{gid: n_f} —— **全项目唯一的 n_f 表**（原来的 3 份硬编码已并到这里）。

    优先用元信息表推出的值；表缺失/该组缺行则回落到 `N_F_FALLBACK`。
    """
    global _NF_CACHE
    if _NF_CACHE is None:
        tab = {g: _n_f_from_table(f) for g, f in meta_table().items()}
        tab = {g: v for g, v in tab.items() if v}
        tab.update({g: v for g, v in N_F_FALLBACK.items() if g not in tab})
        _NF_CACHE = tab
    return _NF_CACHE


N_F = n_f_table()            #: 兼容旧名（原 `l1/ae_cycle.py: N_F`）


def campaign(name):
    """把 'C1' / 语义名 / 唯一前缀 归一化成 `L1_CAMPAIGNS` 的 key。

    >>> campaign('C4')
    'L1 谱载+FBG'
    """
    if name in L1_CAMPAIGNS:
        return name
    if name in CAMPAIGN_ALIAS:
        return CAMPAIGN_ALIAS[name]
    hit = [k for k in L1_CAMPAIGNS if k.startswith(name)]
    if len(hit) == 1:
        return hit[0]
    raise KeyError('未知 L1 分组 %r（短别名 %s，或 %s）'
                   % (name, list(CAMPAIGN_ALIAS), list(L1_CAMPAIGNS)))


def clock_shift_days(gid):
    """AE 采集时钟相对 FBG 日历的平移（天）。L1 谱载+FBG 为 8，其余为 0。"""
    c = _CAMPAIGN_OF.get(gid)
    return L1_CAMPAIGNS[c]['clock_shift_days'] if c else 0


def campaign_members(name):
    """分组（或短别名）的成员表。`--ds` / `--batch` 这类入口用它取名单。"""
    return list(L1_CAMPAIGNS[campaign(name)]['members'])



# ---------------------------------------------------------------- 自检 / 总表
def self_check(verbose=True):
    """把静态表与盘上实际情况对一遍，返回不一致清单。"""
    bad = []
    present = groups('l1')
    all_declared = sorted(_CAMPAIGN_OF)
    for g in all_declared:
        if g not in present:
            bad.append('L1 静态表声明了 %s，但盘上没有' % g)
    for g in present:
        if g not in _CAMPAIGN_OF:
            bad.append('盘上有 %s，但静态表没有它（新增组？请补 L1_CAMPAIGNS）' % g)
    for g in present:
        m = meta(g)
        if m['n_f'] is None:
            bad.append('%s 缺 n_f（元信息表无 rootcyc_total / load_level_*）' % g)
        if m['clock_shift_days'] and m['dataset'] != 'l1':
            bad.append('%s 非 L1 却配了时钟平移' % g)
        # 静态声明 vs 扫盘
        for key, flag in (('AE', m['declared']['ae']), ('FBG', m['declared']['fbg']),
                          ('LUNA', m['declared']['dfos']), ('PZT', m['declared']['pzt'])):
            if flag is True and key not in m['has']:
                bad.append('%s 静态表声明有 %s，但盘上没有该目录/文件' % (g, key))
    if verbose:
        for x in bad:
            print('  [不一致] ' + x)
        print('自检：%d 条不一致（L1 盘上 %d 组 / 静态表 %d 组）'
              % (len(bad), len(present), len(all_declared)))
    return bad


def print_table(dataset=None):
    print('数据根: %s' % root())
    for k in ([dataset] if dataset else DATASETS):
        d = DATASETS[k]
        gs = groups(k)
        print('\n=== %s  [%s]  %d 组 ===' % (d['label'], k, len(gs)))
        print('  %-9s %-13s %6s %8s %6s %8s  %s'
              % ('组', '批次', 'n_f', '时钟天', '文件', 'AE格式', '盘上模态'))
        for g in gs:
            m = meta(g)
            print('  %-9s %-13s %6s %8d %6d %8s  %s'
                  % (g, m['campaign'] or '—',
                     m['n_f'] if m['n_f'] is not None else '—',
                     m['clock_shift_days'], m['n_files'], m['ae_format'] or '—',
                     ','.join(sorted(m['has'])) or '（空）'))
        if d['official']:
            miss = [g for g in d['official'] if g not in gs]
            if miss:
                print('  ⚠ 正式口径里有 %d 组盘上缺失: %s' % (len(miss), ' '.join(miss)))


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='四类数据集的单一定义源')
    ap.add_argument('--dataset', choices=sorted(DATASETS))
    ap.add_argument('--gid', help='只看某一组的合并元信息')
    ap.add_argument('--cross-nf', action='store_true',
                    help='与 l1/ae_cycle.py 的硬编码 N_F 交叉核对')
    a = ap.parse_args(argv)
    if a.cross_nf:
        cross_check_nf()
        return 0
    if a.gid:
        m = meta(a.gid)
        for k in ('gid', 'dataset_label', 'campaign', 'dir', 'exists', 'n_files',
                  'n_f', 'clock_shift_days', 'fbg_rate_hz', 'ae_format', 'has',
                  'meta_src'):
            print('%-18s %s' % (k, m[k]))
        for grp in ('geom', 'load', 'impact', 'files'):
            print('%s:' % grp)
            for k, v in sorted(m[grp].items()):
                print('    %-20s %s' % (k, v))
        return 0
    print_table(a.dataset)
    if not a.dataset or a.dataset == 'l1':
        print('\n--- L1 自检 ---')
        self_check()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
