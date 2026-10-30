# -*- coding: utf-8 -*-
"""morpho 一键脚本 —— MORPHO FOD7（H2020，发动机风扇叶片前缘 FOD 板）的全部任务都在这里跑

用法::

    python run.py --list                    # 任务矩阵
    python run.py --task prepare            # 全部模态解包成 CSV（约 4 分钟，读满 7.8 GB）
    python run.py --task index impedance    # 只做指定几类
    python run.py --task check              # 只体检数据在不在、结构对不对
    python run.py --task prepare --limit-active 2   # 主动导波每作动器只取 2 次重复（冒烟）

本脚本只做调度：真正的解包逻辑在 `prepare.py`，子进程工作目录 = 本目录。
产物写在 `data_root/morpho/processed/`（默认 E:\\morpho\\processed）。
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
    'prepare':   [['prepare.py']],                          # 五类模态全解包
    'index':     [['prepare.py', '--only', 'index']],        # 组树 / 工况表 / PZT 坐标
    'impedance': [['prepare.py', '--only', 'impedance']],    # 电化学阻抗（健康 vs 失效）
    'fbg':       [['prepare.py', '--only', 'fbg']],          # 16 路 FBG 应变
    'mts':       [['prepare.py', '--only', 'mts']],          # 试验机载荷-位移
    'impact':    [['prepare.py', '--only', 'impact']],       # 冲击响应 29 通道
    'active':    [['prepare.py', '--only', 'active']],       # 主动导波 8000 条信号
    'wave50':    [['prepare50k.py', '--waves']],             # 50 kHz 派生集（5 块板）→ CSV
    'analyze':   [['analyze.py']],                           # 跨试件分析（读 wave50 产物）
    'reproduce': [['reproduce.py']],                         # 复现官方论文 A/B/C/D 四项
    'check':     [['prepare.py', '--dry-run']],              # 只列结构，不落盘
}

# task -> (名称, 说明)
DESC = {
    'prepare':   ('全部解包（推荐）', 'prepare.py：五类模态一次做完，约 4 分钟'),
    'index':     ('索引三表', '树 / 工况表 / PZT 坐标 → 01_inventory/'),
    'impedance': ('电化学阻抗', '|Z| / Re(Z) / phase 宽表 + 健康↔失效 delta'),
    'fbg':       ('FBG 应变', '21 工况 × 16 路；summary 含 MTS 加载段统计'),
    'mts':       ('试验机数据', '21 工况：载荷（N/kN）+ 位移 + 循环数'),
    'impact':    ('冲击响应 (重)', '29 通道 × 1e7 点，1 MHz；实测只有 8 个独立通道'),
    'active':    ('主动导波 (重)', '8000 条信号 → 22.4 万行特征'),
    'wave50':    ('50 kHz 派生集（5 板）', 'prepare50k.py --waves → 11_wave50/，约 90 s'),
    'analyze':   ('跨试件分析', 'analyze.py：数据可用性 / 单调性 / 级内 vs 级跃迁 / LOSO'),
    'reproduce': ('复现官方论文', 'reproduce.py：刚度 / FBG-HI / 导波三状态 / 导波 vs 刚度'),
    'check':     ('数据体检', '只打印 HDF5 顶层结构与组数，不写文件'),
}


def print_matrix():
    print('morpho 任务矩阵：\n')
    for t in TASKS:
        name, note = DESC.get(t, ('', ''))
        print(f'   {t:<11}{name:<18}{note}')
    print('\n用法: python run.py --task <任务> [<任务> ...] [--dry-run] [-- <透传参数>]')
    print('`--task all` = prepare → reproduce → wave50 → analyze；标 (重) 的耗时以分钟计。')


def resolve(tasks):
    cmds, seen = [], set()
    ALL = ('prepare', 'reproduce', 'wave50', 'analyze')   # all 不含 check（只体检）
    for task in tasks:
        if task == 'all':
            groups = [TASKS[t] for t in ALL if t in TASKS]
        elif task in TASKS:
            groups = [TASKS[task]]
        else:
            print(f'[提示] morpho 无任务 "{task}"。可用任务: {", ".join(TASKS)} (或 all)')
            return None
        for g in groups:
            for c in g:
                key = tuple(c)
                if key not in seen:
                    seen.add(key)
                    cmds.append(c)
    return cmds


def main():
    ap = argparse.ArgumentParser(description='morpho —— MORPHO FOD7 一键脚本')
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
        print(f'\n>>> [morpho/{"/".join(a.task)}] ' + ' '.join(c + extra))
        if a.dry_run:
            continue
        r = subprocess.run(full, cwd=HERE)
        if r.returncode != 0:
            print(f'[警告] 命令返回码 {r.returncode}: {" ".join(c)}')


if __name__ == '__main__':
    main()
