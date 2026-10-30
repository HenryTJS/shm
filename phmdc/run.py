# -*- coding: utf-8 -*-
"""phmdc 一键脚本 —— PHM2019 铝搭接件的全部任务都在这里跑

用法::

    python run.py --list                       # 任务矩阵
    python run.py --task step0                 # 数据规整 → labels.csv / index.csv
    python run.py --task step4 step5           # 多个任务顺序跑
    python run.py --task step2 --dry-run       # 只打印要执行的命令

本脚本只做调度，不重写算法：每个 task 映射到本目录下的 stepN 脚本，子进程工作目录 = 本目录。
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
    'step0': [['step0.py']],
    'step1': [['step1.py'], ['step1_model.py']],
    'step2': [['step2.py', '--chans', '2']],
    'step3': [['step3_extrap.py']],
    'step4': [['step4_perm.py', '--n-perm', '300', '--n-perm-global', '150', '--scheme', 'both']],
    'step5': [['step5_uncertainty.py']],
}

# task -> (名称, 说明)。名称带 "(重)" 表示耗时长或需要训练。
DESC = {
    'step0': ('标签/索引', 'step0.py → labels.csv / index.csv（幂等，约 20 s）'),
    'step1': ('特征 + LOSO', 'step1.py + step1_model.py（32 特征，8 折 + 10 对照臂）'),
    'step2': ('1D 波形 CNN (重)', 'step2.py --chans 2（默认 5 seeds × 200 epochs，需 torch）'),
    'step3': ('自检 + T7/T8 外推', 'step3_extrap.py（约 10 s；T8 只能作方法演示）'),
    'step4': ('置换检验 (重)', 'step4_perm.py --n-perm 300 --scheme both'),
    'step5': ('t 区间 / bootstrap', 'step5_uncertainty.py（约 10 s）'),
}


def print_matrix():
    print('phmdc 任务矩阵：\n')
    for t in TASKS:
        name, note = DESC.get(t, ('', ''))
        print(f'   {t:<10}{name:<22}{note}')
    print('\n用法: python run.py --task <任务> [--dry-run] [-- <透传参数>]')
    print('标 (重) 的任务耗时长或需要训练；`--task all` 会把这些一起跑。')


def resolve(tasks):
    cmds, seen = [], set()
    for task in tasks:
        if task == 'all':
            groups = list(TASKS.values())
        elif task in TASKS:
            groups = [TASKS[task]]
        else:
            print(f'[提示] phmdc 无任务 "{task}"。可用任务: {", ".join(TASKS)} (或 all)')
            return None
        for g in groups:
            for c in g:
                key = tuple(c)
                if key not in seen:
                    seen.add(key)
                    cmds.append(c)
    return cmds


def main():
    ap = argparse.ArgumentParser(description='phmdc —— PHM2019 一键脚本')
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
        print(f'\n>>> [phmdc/{"/".join(a.task)}] ' + ' '.join(c + extra))
        if a.dry_run:
            continue
        r = subprocess.run(full, cwd=HERE)
        if r.returncode != 0:
            print(f'[警告] 命令返回码 {r.returncode}: {" ".join(c)}')


if __name__ == '__main__':
    main()
