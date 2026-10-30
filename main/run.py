# -*- coding: utf-8 -*-
"""main 一键脚本 —— 主样本（内部疲劳机）的全部任务都在这里跑

用法::

    python run.py --list                          # 任务矩阵
    python run.py --task cdio                     # 预测示例流
    python run.py --task dashboard                # 重出看板数据包（24 组）
    python run.py --task dashboard --dry-run      # 只打印要执行的命令
    python run.py --task cdio -- --every 200      # '--' 之后透传给底层脚本

本脚本只做调度，不重写算法：每个 task 映射到 `cdio_baseline/` 下的既有脚本
（相对本目录），子进程的工作目录 = 脚本所在目录。
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
    # 在线寿命百分比预测示例（示例流 200 个窗，每 5000 行输出一次）
    'cdio':      [['cdio_baseline/live_predictor.py',
                   '--input', 'example_stream.csv', '--every', '5000']],
    # 看板数据包：先逐组留一预测，再由对齐数据生成 24 组回放包
    'dashboard': [['cdio_baseline/export_dashboard.py',
                   '--exclude', '001,002,015'],
                  ['cdio_baseline/build_dashboard_all.py', '--write']],
}

# task -> (名称, 说明)
DESC = {
    'cdio':      ('在线进度预测（示例流）', 'live_predictor.py --input example_stream.csv'),
    'dashboard': ('看板数据包（24 组）', 'export_dashboard.py + build_dashboard_all.py --write → dashboard/data/'),
}


def print_matrix():
    print('main 任务矩阵：\n')
    for t in TASKS:
        name, note = DESC.get(t, ('', ''))
        print(f'   {t:<12}{name:<22}{note}')
    print('\n用法: python run.py --task <任务> [<任务> ...] [--dry-run] [-- <透传参数>]')
    print('`--task all` 会按上面的顺序全部跑一遍。')


def resolve(tasks):
    cmds, seen = [], set()
    for task in tasks:
        if task == 'all':
            groups = list(TASKS.values())
        elif task in TASKS:
            groups = [TASKS[task]]
        else:
            print(f'[提示] main 无任务 "{task}"。可用任务: {", ".join(TASKS)} (或 all)')
            return None
        for g in groups:
            for c in g:
                key = tuple(c)
                if key not in seen:
                    seen.add(key)
                    cmds.append(c)
    return cmds


def main():
    ap = argparse.ArgumentParser(description='main —— 主样本一键脚本')
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
        print(f'\n>>> [main/{"/".join(a.task)}] ' + ' '.join(c + extra))
        if a.dry_run:
            continue
        r = subprocess.run(full, cwd=os.path.dirname(script))
        if r.returncode != 0:
            print(f'[警告] 命令返回码 {r.returncode}: {" ".join(c)}')


if __name__ == '__main__':
    main()
