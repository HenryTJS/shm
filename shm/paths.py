#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据路径解析 —— 读仓库根的 `paths.json`（全项目唯一的路径入口）。

背景（2026-09-28）：数据集已搬到外接盘，仓库里用 **目录联接（junction）** 指过去，
所以 `main/`、`l1/`、`phmdc/` 下的脚本仍按相对路径工作，**无需改动**。
本模块供"需要知道数据根在哪"的新代码 / 诊断脚本使用，避免各自拼绝对路径。

优先级：环境变量 ``SHM_DATA_ROOT``  >  ``paths.json`` 的 ``data_root``

用法::

    from shm import paths
    paths.data_root()                      # 外接盘数据根，如 'E:\\shm_data'
    paths.items()                          # paths.json 的 items
    paths.link_paths(paths.items()[0])     # 展开通配后，仓库内的目录列表
    paths.status()                         # 逐条状态（仓库侧 / 数据侧）
    paths.repo_rel(ROOT)                   # 写进产物用：相对仓库根、不含机器绝对路径

命令行自检::

    python shm/paths.py
    python shm/paths.py --json
"""
from __future__ import annotations

import fnmatch
import glob
import json
import os
import stat as _stat
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(PROJ, 'paths.json')
ENV_KEY = 'SHM_DATA_ROOT'
_REPARSE = getattr(_stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)


def load(path=None):
    """读取 paths.json；文件缺失时返回最小结构（不抛异常）。"""
    fp = path or CONFIG
    if not os.path.exists(fp):
        return {'data_root': '', 'items': []}
    with open(fp, encoding='utf-8') as f:
        cfg = json.load(f)
    cfg.setdefault('items', [])
    return cfg


def data_root(path=None):
    """数据根目录。裸盘符会补成 ``E:\\``（否则 join 会得到盘符相对路径 ``E:l1``）。"""
    env = os.environ.get(ENV_KEY)
    root = os.path.abspath(env) if env else (load(path).get('data_root') or '')
    root = root.rstrip('\\/')
    if len(root) == 2 and root[1] == ':':
        root += os.sep
    return root


def repo_rel(path, base=None):
    """把路径渲染成【相对仓库根】的形式，供写进产物（报告 / CSV / 日志）使用。

    目的：产物里**不出现机器相关的绝对路径**（如 ``D:\\lixiang\\...``），
    换机器 / 换盘后仍可读，也才适合随仓库提交。

    - 一律用正斜杠 ``/``，跨平台一致；
    - 路径在仓库内 -> 返回相对路径，如 ``main/results/loso_cv.csv``；
    - 路径在仓库外，或与仓库不同盘符 -> 原样返回绝对路径（无法相对化）。
    """
    p = os.path.abspath(path)
    root = os.path.abspath(base) if base else PROJ
    try:
        rel = os.path.relpath(p, root)
    except ValueError:              # Windows：跨盘符无法相对化（如仓库在 D:、数据在 E:）
        return p.replace(os.sep, '/')
    if rel == '.' or rel.startswith('..'):
        return p.replace(os.sep, '/')
    return rel.replace(os.sep, '/')


def items(path=None):
    """paths.json 的 items 列表。"""
    return load(path).get('items', [])


def link_paths(item):
    """把 item['link'] 展开成仓库内的绝对路径列表（支持 * ? [0-9]，只回目录）。

    ⚠️ 必须取【**仓库侧 ∪ 数据盘侧**】的并集，只看仓库侧是不够的：
    新组总是**先在数据盘上出现**，那时它还没有联接（仓库侧不存在）⇒
    报告里看不见它，建链也永远建不上。
    （2026-10-09 实测：新加入的 L1-22 / L1-23 因为这条缺失而被完全漏掉。）
    数据侧只收**文件名能匹配该条通配符**的目录，避免把 `main/cache` 这类
    非试件目录也当成试件。
    """
    pat = item.get('link') or ''
    if not pat:
        return []
    full = os.path.join(PROJ, pat.replace('/', os.sep))
    if not any(ch in pat for ch in '*?['):
        return [full]
    out = [p for p in glob.glob(full) if os.path.isdir(p)]
    root, tgt = data_root(), (item.get('target') or '')
    side = os.path.join(root, tgt.replace('/', os.sep)) if (root and tgt) else ''
    if side and os.path.isdir(side):
        rel = pat.replace('/', os.sep)
        parent, base_pat = os.path.dirname(rel), os.path.basename(rel)
        for name in sorted(os.listdir(side)):
            if not fnmatch.fnmatchcase(name, base_pat):
                continue
            if not os.path.isdir(os.path.join(side, name)):
                continue
            out.append(os.path.join(PROJ, parent, name))
    return sorted(set(out))


def is_link(path):
    """是否为目录联接 / 符号链接（Windows 下看 reparse point 属性）。"""
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return bool(getattr(st, 'st_file_attributes', 0) & _REPARSE)


def link_target(path):
    """读取联接指向的目标；不是联接则返回空串。"""
    try:
        return os.readlink(path)
    except OSError:
        return ''


def target_of(item, link_dir):
    """算出某个仓库目录对应的外接盘目标路径。

    规则：link 含通配符 → target 视为【容器目录】，目标 = data_root/target/<目录名>；
          link 不含通配符 → target 视为【精确目标】，目标 = data_root/target。
    """
    tgt = (item.get('target') or '').replace('/', os.sep)
    root = data_root()
    base = os.path.join(root, tgt) if root else tgt
    if any(ch in (item.get('link') or '') for ch in '*?['):
        base = os.path.join(base, os.path.basename(link_dir.rstrip('\\/')))
    return base


def apply(path=None, dry_run=False):
    """把「数据盘已就绪但仓库侧还没联接」的目录建成目录联接（junction）。

    目录联接（`mklink /J`）**不需要管理员权限**，且不占 D 盘空间。
    安全约束：
      · 只处理仓库侧**不存在**且数据侧**存在**的条目；
      · 仓库侧已是联接/实体目录一律**不动**（绝不覆盖已有数据）；
      · 默认 `dry_run=True` 只列出打算做什么。
    返回 (已建, 跳过, 失败) 三个列表。
    """
    made, skipped, failed = [], [], []
    for it in items(path):
        tgt_root = (it.get('target') or '').replace('/', os.sep)
        for lp in link_paths(it):
            tp = target_of(it, lp)
            if os.path.exists(lp):
                skipped.append((lp, '仓库侧已存在'))
                continue
            if not os.path.isdir(tp):
                skipped.append((lp, '数据侧无此目录'))
                continue
            if dry_run:
                made.append((lp, tp))
                continue
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            r = subprocess.run(['cmd', '/c', 'mklink', '/J',
                                lp.replace('/', os.sep), tp.replace('/', os.sep)],
                               capture_output=True, text=True)
            if r.returncode == 0:
                made.append((lp, tp))
            else:
                failed.append((lp, (r.stdout + r.stderr).strip()))
    return made, skipped, failed


def status(path=None):
    """逐条映射的当前状态（供报告 / 自检）。"""
    out = []
    for it in items(path):
        for lp in link_paths(it):
            tp = target_of(it, lp)
            if is_link(lp):
                state = 'junction'
            elif os.path.isdir(lp):
                state = 'real'
            else:
                state = 'missing'
            out.append({
                'name': it.get('name', ''),
                'link': os.path.relpath(lp, PROJ),
                'target': tp,
                'state': state,
                'target_ok': os.path.isdir(tp),
            })
    return out


def as_dict(path=None):
    """机器可读的整体视图（给 --json 或其它脚本用）。"""
    return {
        'data_root': data_root(path),
        'config': CONFIG,
        'items': status(path),
    }


def report(path=None):
    """打印人读映射表，返回 status()。"""
    rows = status(path)
    print('数据根  : %s' % (data_root() or '(未配置)'))
    print('配置文件: %s' % CONFIG)
    if not rows:
        print('  （paths.json 的 items 为空或未匹配到目录）')
        return rows
    w = max(len(r['link']) for r in rows)
    fmt = '{0:<' + str(w) + '}  {1:<9}  {2:<6}  {3}'
    print()
    print(fmt.format('仓库内路径', '仓库侧', '数据侧', '外接盘目标'))
    mark = {'junction': '已联接', 'real': '实体目录', 'missing': '不存在'}
    for r in rows:
        print(fmt.format(r['link'], mark[r['state']],
                         'OK' if r['target_ok'] else '缺', r['target']))
    n_link = sum(1 for r in rows if r['state'] == 'junction')
    n_real = sum(1 for r in rows if r['state'] == 'real')
    print()
    print('合计 %d 条：已联接 %d · 仍是本地实体目录 %d · 不存在 %d'
          % (len(rows), n_link, n_real, len(rows) - n_link - n_real))
    if n_real:
        print('  → 还有实体目录未搬迁（数据本体应在外接盘）。'
              '先把数据放到 %s，再在 paths.json 的 items 里登记。' % (data_root() or 'data_root'))
    return rows


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    argv = sys.argv[1:]
    if '--json' in argv:
        print(json.dumps(as_dict(), ensure_ascii=False, indent=2))
    elif '--apply' in argv or '-apply' in argv:
        report()
        made, skipped, failed = apply(dry_run=False)
        print()
        if made:
            print('已建立联接 %d 条：' % len(made))
            for lp, tp in made:
                print('  %s  ->  %s' % (os.path.relpath(lp, PROJ), tp))
        if failed:
            print('⚠️ 建链失败 %d 条：' % len(failed))
            for lp, err in failed:
                print('  %s : %s' % (os.path.relpath(lp, PROJ), err))
        if not made and not failed:
            print('没有需要新建的联接。')
        print('  （跳过 %d 条：仓库侧已存在或数据侧缺）' % len(skipped))
    else:
        report()
