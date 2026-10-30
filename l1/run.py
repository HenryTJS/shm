# -*- coding: utf-8 -*-
"""l1 一键脚本 —— ReMAP/TU-Delft L1 公开集的全部任务都在这里跑

L1 只交付**离线复评**与**论文复现**口径（在线预警已于 2026-10-07 下线，见 docs/details.md §38）。

用法::

    python run.py --list                              # 任务矩阵（按主题分组）
    python run.py --task hi-ae                        # 离线复评 HI_AE（C2 主指标）
    python run.py --task paper --dry-run              # 只打印要执行的命令
    python run.py --task paper -- --groups L1-03      # '--' 之后透传给底层脚本
    python run.py --task all                          # 全跑（含训练/全量缓存，先 --dry-run）

本脚本只做调度，不重写算法：每个 task 映射到本目录下的既有脚本，子进程工作目录 = 本目录。
定位相关任务（ae-cache / loc-*）**先跑一次 `ae-cache`**：它把 9 组 .pridb 固化成二进制缓存。
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)                 # paths.py / datasets.py 在仓库根

PY = sys.executable

# task -> 若干条底层命令（脚本路径相对本目录）
TASKS = {
    # ---- 读入与准备 ----
    'prepare':    [['step0.py']],                                  # [恒幅+FBG+DFOS] 原始 → CSV
    'prepare-c2': [['step0.py', '--batch', 'c2']],                 # [恒幅+DFOS] LUNA 段级 + AE 1 s 分箱
    # ---- 论文复现（Broer 2021）----
    'paper':      [['reproduce_broer_l1.py', '--mode', 'all']],    # Level 1 + Level 4
    'paper-l23':  [['reproduce_broer_l23.py', '--mode', 'all']],   # Level 2 + Level 3（L3b 未复现）
    # ---- 离线复评指标 ----
    'dfos':       [['evaluate_l1_dfos.py', '--mode', 'hi']],       # 分布式应变逐块 HI
    'fiber-hi':   [['evaluate_l1_fbg.py', '--mode', 'hi']],        # FBG 块级 HI
    'hi-ae':      [['evaluate_l1_hi_ae.py']],                      # HI_AE（默认 --batch 2）
    'hi-hit':     [['ae_hi.py']],                                  # C3/C4 的 HI_hit
    # ---- AE / FBG 中间层 ----
    'ae-dta':     [['ae_dta.py', 'export']],                       # .DTA 体检概览（交付物）
    'ae-frames':  [['ae_frames.py']],                              # AE 600 s 帧 → npz
    'ae-cycle':   [['ae_cycle.py']],                               # 循环轴 → npz
    'fbg':        [['fbg_tools.py', 'profile']],                   # → results/_l1fbgprof_*.npz
    'fbg-qa':     [['fbg_tools.py', 'coverage']],                  # FBG 覆盖体检（快）
    'fbg-variants': [['fbg_tools.py', 'variants']],                # 10 口径横向对比
    'meta':       [['pdf_specimen_meta.py', '--write'],
                   ['survey_groups.py']],                          # 元信息表 / 记录表
    # ---- 跨机构可比性 ----
    'crosslab-ae':    [['hi_crosslab.py']],                        # AE 跨批次一致性（28 组）
    'crosslab-ae23':  [['hi_l123.py']],                            # 补第 29 组 L1-23（重）
    'crosslab-pzt':   [['pzt_crosslab.py']],                       # 导波 PZT 跨批次（重）
    'pzt-fail':       [['pzt_sensor_fail.py']],                    # 失效路径剔除（需先跑 crosslab-pzt）
    'pzt-causal':     [['pzt_causal.py']],                         # 导波因果（在线）版本核查（重）
    'pzt-locadiff':   [['pzt_locadiff_probe.py']],                 # 尾波空间定位探针（已证伪）
    'snhi-shape':     [['snhi_shape.py']],                         # SNHI 形状不变性前提诊断
    'snhi-rul':       [['snhi_rul.py']],                           # 形状配准作 RUL（已证伪）
    'level-vs-shape': [['level_vs_shape.py']],                     # 绝对阈值 vs SNHI 对照
    'four-modal':     [['four_modal.py']],                         # C1 五组的四模态并行对照（重）
    'crosslab-var':   [['crosslab_variance.py']],                  # 方差分解 η² + 置换零（Table 1/2）
    'thresh-sweep':   [['threshold_sweep.py']],                    # 26 条绝对阈值规则扫描
    'dfos-loc':       [['dfos_peak_migration.py']],                # 分布式应变空间定位（已证伪）
    # ---- AE 事件定位（通道时间偏置标定）----
    'ae-cache':       [['ae_cache.py', '--build']],                # 逐 hit 二进制缓存（一次性）
    'loc-bias-probe': [['loc_bias_probe.py']],                     # 逐通道残差探针
    'loc-vel-calib':  [['loc_vel_calib.py']],                      # 声速标定（结论：拿错了自由度）
    'loc-debias':     [['loc_debias.py']],                         # 通道延迟 δ 标定（全 9 组）
    'loc-drift':      [['loc_debias_drift.py']],                   # δ 是否随寿命漂移
    # ---- 图与表 ----
    'figs':       [['figs.py'], ['four_modal.py']],                # 论文图 1–5
    'fig-routes': [['figs.py', '--only', 'routes']],               # 只重画 Fig. 4
    'fig-instr':  [['figs.py', '--only', 'instrumentation']],      # 试件与仪器图
    'tables-45':  [['make_tables_45.py']],                         # 论文表 3/4 的 markdown
    # ---- 看板 ----
    'dashboard':  [['restore_dashboard_packages.py']],             # 29 组 L1 回放包 + C1–C4 清单
}

# task -> (名称, 说明)。名称带 "(重)" 表示耗时长或需要训练。
DESC = {
    'prepare':  ('[恒幅+FBG+DFOS] 原始 → CSV', 'step0.py（LUNA 逐行 + FBG + AE；--only 只重生一类）'),
    'prepare-c2': ('[恒幅+DFOS] 原始 → CSV', 'step0.py --batch c2（LUNA 段级 + AE 1 s 分箱）'),
    'paper':    ('论文 Level1/4', 'reproduce_broer_l1.py --mode all'),
    'paper-l23': ('论文 Level2/3', 'reproduce_broer_l23.py --mode all（L3b 未复现）'),
    'dfos':     ('分布式应变逐块', 'evaluate_l1_dfos.py --mode hi'),
    'fiber-hi': ('FBG 块级 HI', 'evaluate_l1_fbg.py --mode hi'),
    'hi-ae':    ('离线复评 HI_AE', 'evaluate_l1_hi_ae.py（默认 --batch 2）'),
    'hi-hit':   ('[变幅VA/谱载] HI_hit', 'ae_hi.py（累积 AE 命中数；非因果，只作离线复评）'),
    'ae-dta':   ('[变幅VA/谱载] AE 体检概览 (重)', 'ae_dta.py export → l1/AE特征概览.csv'),
    'ae-frames': ('[变幅VA/谱载] AE 600 s 帧 (重)', 'ae_frames.py → results/_l1ae_frames_*.npz'),
    'ae-cycle': ('[跨分组] 循环轴 (重)', 'ae_cycle.py → results/_l1cyc_*.npz（口径 holdgap）'),
    'fbg':      ('FBG 剖面指标', 'fbg_tools.py profile → results/_l1fbgprof_*.npz'),
    'fbg-qa':   ('FBG 覆盖体检', 'fbg_tools.py coverage（不读数据内容，快）'),
    'fbg-variants': ('FBG 口径对比', 'fbg_tools.py variants（10 口径 + 单组诊断）'),
    'meta':     ('试件元信息/记录表', 'pdf_specimen_meta.py --write + survey_groups.py'),
    'crosslab-ae':    ('跨机构 AE 一致性', 'hi_crosslab.py：28 组放同一寿命分数轴 → 批间/组内'),
    'crosslab-ae23':  ('补第 29 组 L1-23 (重)', 'hi_l123.py：流式读 4.1 GB 事件级 CSV'),
    'crosslab-pzt':   ('跨机构 PZT 一致性 (重)', 'pzt_crosslab.py：17 组 × 56 路径 TOF 漂移'),
    'pzt-fail':       ('传感器失效路径剔除', 'pzt_sensor_fail.py：需先跑 crosslab-pzt'),
    'pzt-causal':     ('导波因果版本核查 (重)', 'pzt_causal.py：在线（因果）版本是否可行'),
    'pzt-locadiff':   ('尾波空间定位探针', 'pzt_locadiff_probe.py：与椭圆敏感核对比（已证伪）'),
    'snhi-shape':     ('SNHI 形状不变性诊断', 'snhi_shape.py：算法前提检验'),
    'snhi-rul':       ('形状配准作 RUL（已证伪）', 'snhi_rul.py：同批留一 92.8% vs 零信息 23.6%'),
    'level-vs-shape': ('绝对阈值 vs SNHI 对照', 'level_vs_shape.py：15 组，两方法产出同一量'),
    'four-modal':     ('四模态并行对照', 'four_modal.py：C1 五组 AE/FBG/DFOS/PZT 同轴对比'),
    'crosslab-var':   ('方差分解（Table 1/2）', 'crosslab_variance.py：按 campaign / university 的 η² 与置换零'),
    'thresh-sweep':   ('26 条阈值规则扫描', 'threshold_sweep.py：四族规则'),
    'dfos-loc':       ('分布式应变空间定位（已证伪）', 'dfos_peak_migration.py：一致率 7/11'),
    'ae-cache':       ('AE 二进制缓存 (一次性)', 'ae_cache.py --build：9.5 GB 库 → npy'),
    'loc-bias-probe': ('定位通道残差探针', 'loc_bias_probe.py：残差与通道强相关 → 偏置主导'),
    'loc-vel-calib':  ('定位声速标定 (已证伪)', 'loc_vel_calib.py：三组最优全部落在搜索边界'),
    'loc-debias':     ('定位通道去偏', 'loc_debias.py：显式建 δ_k，9 组残差 12.07→1.19 等'),
    'loc-drift':      ('δ 随寿命漂移检验', 'loc_debias_drift.py：分窗标定 + 随机基线 + 空间分层'),
    'figs':           ('论文图 1–5', 'figs.py（四张）+ four_modal.py → l1/figures/'),
    'fig-routes':     ('论文图 Fig. 4', 'figs.py --only routes → l1/figures/fig_four_routes.png/.pdf'),
    'fig-instr':      ('试件与仪器图', 'figs.py --only instrumentation → l1/figures/fig_instrumentation.png/.pdf'),
    'tables-45':      ('论文表 3/4', 'make_tables_45.py → 含 η² 与置换 p 的 markdown 草稿'),
    'dashboard':      ('看板回放包（29 组）', 'restore_dashboard_packages.py → dashboard/data/ + C1–C4 清单'),
}


def print_matrix():
    print('l1 任务矩阵：\n')
    for t in TASKS:
        name, note = DESC.get(t, ('', ''))
        print(f'   {t:<16}{name:<24}{note}')
    print('\n用法: python run.py --task <任务> [<任务> ...] [--dry-run] [-- <透传参数>]')
    print('标 (重) 的任务耗时长或需要训练；`--task all` 会把这些一起跑。')


def resolve(tasks):
    cmds, seen = [], set()
    for task in tasks:
        if task == 'all':
            groups = list(TASKS.values())
        elif task in TASKS:
            groups = [TASKS[task]]
        else:
            print(f'[提示] l1 无任务 "{task}"。可用任务: {", ".join(TASKS)} (或 all)')
            return None
        for g in groups:
            for c in g:
                key = tuple(c)
                if key not in seen:
                    seen.add(key)
                    cmds.append(c)
    return cmds


def main():
    ap = argparse.ArgumentParser(description='l1 —— ReMAP L1 一键脚本')
    ap.add_argument('--task', nargs='+', help='任务名（见 --list），可给多个，或 all')
    ap.add_argument('--list', action='store_true', help='打印任务矩阵')
    ap.add_argument('--paths', action='store_true', help='打印数据路径映射（paths.json）并自检')
    ap.add_argument('--dry-run', action='store_true', help='只打印命令，不执行')
    a, extra = ap.parse_known_args()
    if '--' in extra:
        extra = extra[extra.index('--') + 1:]

    if a.paths:
        import paths
        paths.report()
        return

    if a.list or not a.task:
        print_matrix()
        if not a.list:
            ap.print_help()
        return

    cmds = resolve(a.task)
    if cmds is None:
        return
    for c in cmds:
        script = os.path.join(HERE, c[0])
        full = [PY, script] + c[1:] + extra
        print(f'\n>>> [l1/{"/".join(a.task)}] ' + ' '.join(c + extra))
        if a.dry_run:
            continue
        r = subprocess.run(full, cwd=HERE)
        if r.returncode != 0:
            print(f'[警告] 命令返回码 {r.returncode}: {" ".join(c)}')


if __name__ == '__main__':
    main()
