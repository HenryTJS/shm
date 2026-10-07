# -*- coding: utf-8 -*-
"""统一入口 —— 在四类数据集间自由切换并执行对应任务

数据集:
  main  : 内部疲劳机主样本 016-020 (服役期渐进损伤)
  l1    : 公开集 ReMAP/TU-Delft L1 —— **共 27 组**，按**模态/格式/加载谱**分四组：
          L1 恒幅+FBG+DFOS（短别名 C1）/ L1 恒幅+DFOS（C2）/ L1 变幅VA+FBG（C3）/ L1 谱载+FBG（C4）
  phmdc : PHM 2020 DC 轴承 (T1 至 T8 退化外推)

用法:
  python run.py --list                      # 查看任务矩阵（按数据集分组）
  python run.py --dataset main   --task degree
  python run.py --dataset l1     --task paper           # L1 只保留离线复评 / 论文复现口径
  python run.py --dataset phmdc  --task step1
  python run.py --dataset l1     --task all             # 全部任务
  python run.py --dataset l1     --task all --dry-run   # 只打印要跑的命令
  python run.py --dataset l1     --task paper -- --groups L1-03   # '--' 之后透传给底层脚本

说明: 本脚本是"任务调度器", 不重写算法; 每类数据集的任务映射到既有脚本（见 --list）。
      `--dataset l1` **只保留离线复评与论文复现口径** —— L1 不支持跨试件统一阈值的在线预警，
      相关脚本与看板包已于 2026-10-07 移除（见 docs/details.md §38）。
      `--task all` 会把标 (重) 的任务（训练、全量重建缓存）一起跑，先想清楚。
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# ============ 任务矩阵 ============
# 每个 task -> 若干条底层命令(不带 python 前缀); 'all' 为该数据集全部任务。
# 每类数据集的简介（--list 用）
DATASETS = {
    'main': '内部疲劳机主样本 016-020（服役期渐进损伤）',
    'l1': 'ReMAP/TU-Delft L1 公开集（27 组；四组 = 恒幅+FBG+DFOS / 恒幅+DFOS / 变幅VA+FBG / 谱载+FBG）',
    'phmdc': 'PHM 2020 DC 轴承（T1 至 T8，退化外推）',
}

L1_DEGREE = None   # 已移除：L1 在线 D(t) 于 2026-10-07 随在线口径一并下线（见 details.md §38）

TASKS = {
    'main': {
        'prepare': [['main/pipeline.py', 'prepare', 'align']],
        'labels':  [['main/pipeline.py', 'prepare', 'weaklabels']],
        'degree':  [['main/pipeline.py', 'evaluate', 'degree']],
        'warning': [['main/pipeline.py', 'evaluate', 'warning']],
        'curves':  [['main/pipeline.py', 'evaluate', 'curves']],
        'paper':   [['main/pipeline.py', 'evaluate', 'paper']],
        'robust':  [['main/pipeline.py', 'robust', 'sens'],
                    ['main/pipeline.py', 'robust', 'loso'],
                    ['main/pipeline.py', 'robust', 'ablation'],
                    ['main/pipeline.py', 'robust', 'stats']],
        # ---- 验证层（合并入口 main/verify.py；子命令名 = 数据集名）----
        'baselines':   [['main/verify.py', 'baselines']],     # vs 经典/朴素/监督 ML 基线
        'loso-cv':     [['main/verify.py', 'loso-cv']],       # 留一试件交叉验证
        'label-audit': [['main/verify.py', 'labels']],        # 标签审计 + 剔除规则事前化
        'ae-cols':     [['main/verify.py', 'ae-cols']],       # AE 多列逐列对照
        'grade':       [['main/verify.py', 'grade']],         # 级别层异源分级（刚度闸门）
        'fusion':      [['main/verify.py', 'fusion']],        # 源融合对照（单源/二源/三源）
        'candidates':  [['main/verify.py', 'candidates']],    # 候选试件体检（需先跑 grade）
        'verify':  [['main/verify.py', 'baselines'],          # 一键跑完验证层（grade 在 candidates 前）
                    ['main/verify.py', 'loso-cv'],
                    ['main/verify.py', 'labels'],
                    ['main/verify.py', 'ae-cols'],
                    ['main/verify.py', 'grade'],
                    ['main/verify.py', 'fusion'],
                    ['main/verify.py', 'candidates']],
        'dashboard': [['main/export_dashboard.py']],          # → dashboard/data/
    },
    'l1': {
        # ---- L1 恒幅+FBG+DFOS（L1-03/04/05/09，有 FBG）----
        'prepare':  [['l1/step0.py']],                       # 原始 .pridb/.txt → CSV
        'paper':    [['l1/reproduce_broer_l1.py', '--mode', 'all']],  # 论文 Level1 + Level4
        'paper-l23': [['l1/reproduce_broer_l23.py', '--mode', 'all']],  # 论文 Level2 + Level3
        'dfos':     [['l1/evaluate_l1_dfos.py', '--mode', 'hi']],     # 分布式应变逐块（离线块级 HI）
        'fiber-hi': [['l1/evaluate_l1_fbg.py', '--mode', 'hi']],      # FBG(光纤)块级 HI（离线）
        # ---- L1 恒幅+DFOS（L1-49 至 L1-60，无 FBG）----
        'prepare-c2': [['l1/step0.py', '--batch', 'c2']],             # LUNA 段级 + AE 1s 分箱
        # ---- L1 变幅VA+FBG + L1 谱载+FBG（AE .DTA + FBG）----
        'ae-dta':     [['l1/ae_dta.py', 'export']],                    # .DTA 体检概览（交付物）
        'ae-frames':  [['l1/ae_frames.py']],                          # AE 600 s 帧 → npz
        'ae-cycle':   [['l1/ae_cycle.py']],                           # 循环轴 → npz
        # ---- 跨分组 ----
        'hi-ae':    [['l1/evaluate_l1_hi_ae.py']],                     # 离线复评 HI_AE(默认 --batch 2)
        'hi-hit':   [['l1/ae_hi.py']],                                 # C3/C4 的 HI_hit
        'fbg':      [['l1/fbg_tools.py', 'profile']],                  # → _l1fbgprof_*.npz
        'fbg-qa':   [['l1/fbg_tools.py', 'coverage']],                 # FBG 覆盖体检
        'fbg-variants': [['l1/fbg_tools.py', 'variants']],             # 10 口径横向对比
        'meta':     [['l1/pdf_specimen_meta.py', '--write'],
                     ['l1/survey_groups.py']],                         # 元信息表 / 记录表
    },
    'phmdc': {
        'step0': [['phmdc/step0.py']],                                # → labels.csv / index.csv
        'step1': [['phmdc/step1.py'],                                 # 特征 + 可行性 + 基线自洽
                  ['phmdc/step1_model.py']],                          # LOSO 8 折 + 对照臂
        'step2': [['phmdc/step2.py', '--chans', '2']],                # 1D 波形 CNN（训练）
        'step3': [['phmdc/step3_extrap.py']],                         # 自检 + T7/T8 外推
        'step4': [['phmdc/step4_perm.py', '--n-perm', '300',
                   '--n-perm-global', '150', '--scheme', 'both']],    # 置换检验
        'step5': [['phmdc/step5_uncertainty.py']],                    # t 区间 / bootstrap
    },
}

# 每个数据集的 task -> (名称, 说明)。名称带 "(重)" 表示耗时长或需要训练。
DESC = {
    'main': {
        'prepare':  ('数据准备/预处理', 'pipeline.py prepare align：多源对齐'),
        'labels':   ('弱标签/失效锚', 'pipeline.py prepare weaklabels：b2/b3 弱标签'),
        'degree':   ('连续损伤度 D(t)+分级', 'pipeline.py evaluate degree：D 达阈/单调'),
        'warning':  ('预警 onset/分级', 'pipeline.py evaluate warning：A-预警'),
        'curves':   ('D(t) 曲线出图', 'pipeline.py evaluate curves：5 组曲线'),
        'paper':    ('论文图表', 'pipeline.py evaluate paper：四联图+流程图'),
        'robust':   ('稳健性/统计', 'pipeline.py robust sens/loso/ablation/stats'),
        'baselines':   ('基线对照', 'verify.py baselines：本项目 vs 经典/朴素/监督 ML'),
        'loso-cv':     ('留一试件交叉验证', 'verify.py loso-cv：19 配置 × 5 折'),
        'label-audit': ('标签审计', 'verify.py labels：b2 来源/敏感性/剔除规则/统计口径'),
        'ae-cols':     ('AE 多列对照', 'verify.py ae-cols：形状/比值列逐列（9 组）'),
        'grade':       ('级别层异源分级', 'verify.py grade：刚度损失闸门'),
        'fusion':      ('源融合对照', 'verify.py fusion：5 组 × 9 配置'),
        'candidates':  ('候选试件体检', 'verify.py candidates：015-027（需先跑 grade）'),
        'verify':      ('【验证层全部】', 'verify.py 七个子命令按序'),
        'dashboard': ('看板数据导出', 'export_dashboard.py → dashboard/data/'),
    },
    'l1': {
        'prepare':  ('[恒幅+FBG+DFOS] 原始 → CSV', 'step0.py（LUNA 逐行 + FBG + AE；--only 只重生一类）'),
        'prepare-c2': ('[恒幅+DFOS] 原始 → CSV', 'step0.py --batch c2（LUNA 段级 + AE 1 s 分箱）'),
        'paper':    ('论文 Level1/4', 'reproduce_broer_l1.py --mode all'),
        'paper-l23': ('论文 Level2/3', 'reproduce_broer_l23.py --mode all（L3b 未复现）'),
        'dfos':     ('分布式应变逐块', 'evaluate_l1_dfos.py --mode hi'),
        'fiber-hi': ('FBG 块级 HI', 'evaluate_l1_fbg.py --mode hi'),
        'hi-ae':    ('离线复评 HI_AE', 'evaluate_l1_hi_ae.py（默认 --batch 2；见 §7 同名覆盖警告）'),
        'hi-hit':   ('[变幅VA/谱载] HI_hit', 'ae_hi.py（累积 AE 命中数；非因果，只作离线复评）'),
        'ae-dta':   ('[变幅VA/谱载] AE 体检概览 (重)', 'ae_dta.py export → l1/AE特征概览.csv（交付物）'),
        'ae-frames': ('[变幅VA/谱载] AE 600 s 帧 (重)', 'ae_frames.py → results/_l1ae_frames_*.npz'),
        'ae-cycle': ('[跨分组] 循环轴 (重)', 'ae_cycle.py → results/_l1cyc_*.npz（口径 holdgap）'),
        'fbg':      ('FBG 剖面指标', 'fbg_tools.py profile → results/_l1fbgprof_*.npz'),
        'fbg-qa':   ('FBG 覆盖体检', 'fbg_tools.py coverage（不读数据内容，快）'),
        'fbg-variants': ('FBG 口径对比', 'fbg_tools.py variants（10 口径 + 单组诊断）'),
        'meta':     ('试件元信息/记录表', 'pdf_specimen_meta.py --write + survey_groups.py'),
    },
    'phmdc': {
        'step0': ('标签/索引', 'step0.py → labels.csv / index.csv（幂等，约 20 s）'),
        'step1': ('特征 + LOSO', 'step1.py + step1_model.py（32 特征，8 折 + 10 对照臂）'),
        'step2': ('1D 波形 CNN (重)', 'step2.py --chans 2（默认 5 seeds × 200 epochs）'),
        'step3': ('自检 + T7/T8 外推', 'step3_extrap.py（约 10 s；T8 只能作方法演示）'),
        'step4': ('置换检验 (重)', 'step4_perm.py --n-perm 300 --scheme both'),
        'step5': ('t 区间 / bootstrap', 'step5_uncertainty.py（约 10 s）'),
    },
}


def print_matrix():
    print('任务矩阵 (dataset × task)：\n')
    for ds, tasks in TASKS.items():
        print(f'== {ds} —— {DATASETS[ds]}')
        for t in tasks:
            name, note = DESC.get(ds, {}).get(t, ('', ''))
            print(f'   {t:<15}{name:<24}{note}')
        print()
    print('用法: python run.py --dataset <数据集> --task <任务> [--dry-run] [-- <透传参数>]')
    print('标 (重) 的任务耗时长或需要训练；`--task all` 会把这些一起跑。')


def resolve(dataset, task):
    ds = TASKS.get(dataset)
    if ds is None:
        print(f'[错误] 未知数据集: {dataset} (可选: {", ".join(TASKS)})')
        return None
    if task == 'all':
        cmds, seen = [], set()
        for v in ds.values():
            for c in v:
                key = tuple(c)
                if key not in seen:
                    seen.add(key); cmds.append(c)
        return cmds
    if task not in ds:
        print(f'[提示] 数据集 "{dataset}" 无任务 "{task}"。'
              f' 可用任务: {", ".join(ds)} (或 all)')
        return None
    return ds[task]


def main():
    ap = argparse.ArgumentParser(description='多源损伤度 D(t) —— 数据集统一入口')
    ap.add_argument('--dataset', choices=list(TASKS),
                    help='数据源: ' + ' | '.join(TASKS))
    ap.add_argument('--task', help='任务名 (见 --list), 或 all')
    ap.add_argument('--list', action='store_true', help='打印任务矩阵')
    ap.add_argument('--paths', action='store_true', help='打印数据集路径映射（paths.json）并自检')
    ap.add_argument('--dry-run', action='store_true', help='只打印将执行的命令')
    a, extra = ap.parse_known_args()
    # '--' 之后的内容透传
    if '--' in extra:
        extra = extra[extra.index('--') + 1:]
    elif extra and extra[0] == '--':
        extra = extra[1:]

    if a.paths:
        import shm.paths as _paths
        _paths.report()
        return

    if a.list or not a.dataset or not a.task:
        print_matrix()
        if not (a.list or (a.dataset and a.task)):
            ap.print_help()
        return

    cmds = resolve(a.dataset, a.task)
    if cmds is None:
        return
    for c in cmds:
        script = os.path.join(ROOT, c[0])
        full = [PY, script] + c[1:] + extra
        print(f'\n>>> [{a.dataset}/{a.task}] ' + ' '.join(c + extra))
        if a.dry_run:
            continue
        r = subprocess.run(full, cwd=os.path.dirname(script))
        if r.returncode != 0:
            print(f'[警告] 命令返回码 {r.returncode}: {" ".join(c)}')


if __name__ == '__main__':
    main()
