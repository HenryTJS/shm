# -*- coding: utf-8 -*-
"""把 `dashboard/data/L1-*.js` 按 `datasets.L1_CAMPAIGNS` 归成 **C1–C4 四个数据集清单**。

为什么单独做一步
----------------
L1 的数据包是回放归档字节码产出的（`l1/restore_dashboard_packages.py`），
那套导出器的清单是“三批”（`l1` / `l1v2` / `l1v3`），而看板现在要按
**C1 恒幅+FBG+DFOS / C2 恒幅+DFOS / C3 变幅VA+FBG / C4 谱载+FBG** 分开选。
分组真源是 `datasets.py` 的 `L1_CAMPAIGNS`（含 2026-10-09 补入的 L1-22 / L1-23）。

产物
----
    dashboard/data/index_c1.js ... index_c4.js   （window.SHM_DATASETS["c1"] ... ["c4"]）
    数据集名：ReMAP C1 / ReMAP C2 / ReMAP C3 / ReMAP C4
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from datasets import CAMPAIGN_ALIAS, L1_CAMPAIGNS  # noqa: E402

DASH = os.path.join(PROJ, 'dashboard', 'data')
FIELDS = ('gid', 'nfr', 'dur', 'n', 'meta', 'foCols', 'spec', 'warn')
KIND = {'C1': '', 'C2': '', 'C3': '变幅 VA', 'C4': '谱载'}


def read_pkg(path):
    s = open(path, encoding='utf-8').read()
    i = s.index('window.SHM_DATA[') + len('window.SHM_DATA[')
    j = s.index(']=', i) + 2
    return json.loads(s[j:].rstrip().rstrip(';'))


def main():
    made = []
    for alias in ('C1', 'C2', 'C3', 'C4'):
        camp = CAMPAIGN_ALIAS[alias]
        members = L1_CAMPAIGNS[camp]['members']
        groups, missing = [], []
        for g in members:
            fp = os.path.join(DASH, g + '.js')
            if not os.path.exists(fp):
                missing.append(g)
                continue
            d = read_pkg(fp)
            entry = {k: d.get(k) for k in FIELDS}
            entry['spec'] = {'kind': KIND[alias], 'load': '', 'note': ''}
            groups.append(entry)
        ds = {'id': alias.lower(), 'name': 'ReMAP ' + alias, 'unit': 'cycle',
              'path': 'data/', 'groups': groups}
        out = os.path.join(DASH, f'index_{alias.lower()}.js')
        with open(out, 'w', encoding='utf-8') as fh:
            fh.write('window.SHM_DATASETS=window.SHM_DATASETS||{};')
            fh.write(f'window.SHM_DATASETS[{json.dumps(alias.lower())}]=' +
                     json.dumps(ds, ensure_ascii=False) + ';')
        made.append((alias, [g['gid'] for g in groups], missing))

    for alias, gids, missing in made:
        miss = f'  缺包: {",".join(missing)}' if missing else ''
        print(f'  ReMAP {alias}: {len(gids):2d} 组  {", ".join(gids)}{miss}')
    print(f'共写出 {len(made)} 个清单 → {DASH}')


if __name__ == '__main__':
    main()
