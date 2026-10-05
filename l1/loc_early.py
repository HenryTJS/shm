# -*- coding: utf-8 -*-
"""把「定位质心偏离图纸冲击点」拆成「源铺开」与「残余定位误差」（待办 1.1 / 1.2）。

背景
----
`details.md` §14：体检通过的组，**全寿命质心**与图纸冲击点仍差 25 至 87 mm。
两种解释必须分开：
  (a) **聚合假象** —— AE 源随损伤扩展而铺开，全寿命质心本就不该等于冲击点；
  (b) **残余误差** —— 定位本身有恒定偏置（探头偏晚、声速、几何）。
做法：按**事件时间**分位切窗，看三件事：
  1. **离散度**（r68 / r90）是否随寿命增长 —— 这是 (a) 的直接证据；
  2. **早期窗**质心是否明显比全寿命质心更靠近冲击点 —— (a) 的第二个证据；
  3. **小样本零分布** —— 早期窗事件少、质心本来就更噪，必须与「同样本量随机子集」
     的零分布对比，否则会把采样噪声当成信号。

⚠️ 口径限制（必须先说清，否则结论会被误读）
  · **C1/C2 批没有 AE 帧表、也没有循环轴**：`ae_frames.py` 只处理 `.DTA`（即 C3/C4），
    所以 L1-03/04/05/09 与 L1-49 至 L1-60 **只有 `_l1_loc_*.npz`，没有 `_l1cyc_*`**。
    ⇒ 这里只能用 **AE 相对时间**分位，不能用循环分位。
    对「早 vs 晚」这个判断够用：映射只要单调，「最早的一段」必定比「最后一段」早。
  · **真值只用图纸 mm 值**（`x_mm` / `y_mm`）+ 180° 旋转 `(165−X, 243−Y)`。
    **不要用** `l1_impact_truth.csv` 的 `prose_x_mm` / `prose_y_mm` —— 那两列由**文字描述**
    推得（旧名 `x_skin_mm` / `y_skin_mm`，列名已改以防误用），`details.md` §12 已判定文字描述不可用：
    同批 L1-49 与 L1-54 的图纸值完全相同，那两列却给出 50 与 115，自相矛盾。
    本工具读的是新加的 `flip_x_mm` / `flip_y_mm`（= 图纸值旋转），与上面手算一致。

输出
----
  results/l1_loc_early.csv          逐组逐窗的统计
  results/_logs/l1_loc_early.txt    人类可读报告

用法
----
    python l1/loc_early.py
    python l1/loc_early.py --groups L1-49,L1-54
"""

import os
import sys
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ae_locate as al                                          # noqa: E402
import pandas as pd                                             # noqa: E402

RES = al.RES
LOGDIR = os.path.join(RES, '_logs')
os.makedirs(LOGDIR, exist_ok=True)
OUT_CSV = os.path.join(RES, 'l1_loc_early.csv')
OUT_TXT = os.path.join(LOGDIR, 'l1_loc_early.txt')

LX, LY = 165.0, 243.0           # 试件尺寸（skin 长 x 宽），与 pdf_specimen_meta 一致

# 逐窗（相对于**事件数**的分位，不是时间分位 —— 事件在时间上极度不均匀）
WINS = [(0.00, 0.01, '首 1 %'), (0.01, 0.05, '1 至 5 %'),
        (0.05, 0.10, '5 至 10 %'), (0.10, 0.25, '10 至 25 %'),
        (0.25, 0.50, '25 至 50 %'), (0.50, 0.75, '50 至 75 %'),
        (0.75, 1.00, '75 至 100 %'), (0.00, 1.00, '全寿命')]

# §14 的探头体检结论：只有这 5 组四探头都正常
AUDIT_OK = {'L1-49', 'L1-54', 'L1-04', 'L1-05', 'L1-09'}

N_NULL = 200                    # 零分布抽样次数


def truth_of(gid, tdf):
    """真值：优先用表里的 flip_x_mm / flip_y_mm（图纸值 + 180° 旋转），
    没有则就地由 x_mm / y_mm 旋转得到。

    ⚠️ 绝不用 prose_x_mm / prose_y_mm（由文字描述推得，不可用）。
    """
    r = tdf[tdf['gid'] == gid]
    if r.empty:
        return None
    s = r.iloc[0]
    if 'flip_x_mm' in r.columns:
        fx, fy = s.get('flip_x_mm'), s.get('flip_y_mm')
        if not (pd.isna(fx) or pd.isna(fy)) and fx != '' and fy != '':
            return (float(fx), float(fy))
    x, y = s.get('x_mm'), s.get('y_mm')
    if pd.isna(x) or pd.isna(y) or x == '' or y == '':
        return None
    return (LX - float(x), LY - float(y))


def win_stat(x, y, sel, truth, rng, n_null=N_NULL):
    """一个事件窗的统计：质心、离散度、到真值距离，以及小样本零分布。

    `sel` 是**索引数组**（事件数分位切窗，不是布尔掩码）。
    """
    n = int(sel.size)
    if n < 20:
        return None
    cx, cy = float(x[sel].mean()), float(y[sel].mean())
    d = np.hypot(x[sel] - cx, y[sel] - cy)
    out = {'n': n, 'cx': cx, 'cy': cy,
           'r68': float(np.median(d)), 'r90': float(np.percentile(d, 90)),
           'sx': float(x[sel].std()), 'sy': float(y[sel].std())}
    if truth is None:
        out.update(d_truth=float('nan'), null_pct=float('nan'), null_med=float('nan'))
        return out
    tx, ty = truth
    out['d_truth'] = float(np.hypot(cx - tx, cy - ty))
    # 零分布：从**全组**里随机抽同样本量 n 的子集，看质心距真值的分布。
    # 若观测值落在零分布低分位，说明「这一窗的质心更近」不是采样噪声。
    N = x.size
    dd = np.empty(n_null)
    for i in range(n_null):
        idx = rng.choice(N, n, replace=False)
        dd[i] = np.hypot(x[idx].mean() - tx, y[idx].mean() - ty)
    out['null_pct'] = float((dd < out['d_truth']).mean() * 100.0)
    out['null_med'] = float(np.median(dd))
    return out


def analyse(gid, tdf, rng):
    fp = os.path.join(RES, '_l1_loc_%s.npz' % gid)
    if not os.path.exists(fp):
        return None
    z = np.load(fp)
    xa, ya, rms, ta = z['x'], z['y'], z['rms_us'], z['t_s']
    good = rms <= al.RMS_GOOD
    x, y, t = xa[good], ya[good], ta[good]
    if x.size < 200:
        return None
    truth = truth_of(gid, tdf)
    t0 = float(ta.min())
    span_h = (float(ta.max()) - t0) / 3600.0

    # 按**事件数**分位（时间分布极不均匀，按时间切会让各窗 n 相差上百倍）
    order = np.argsort(t, kind='stable')
    n = x.size
    rec = {'gid': gid, 'n_loc': int(n), 'n_all': int(xa.size),
           'keep': float(n) / max(xa.size, 1), 'span_h': span_h,
           'audit_ok': gid in AUDIT_OK, 'truth': truth,
           'rms_med': float(np.median(rms)), 'wins': [], 't_conc': None}
    for lo, hi, lab in WINS:
        a, b = int(round(lo * n)), int(round(hi * n))
        if b - a < 20:
            continue
        sel = order[a:b]
        s = win_stat(x, y, sel, truth, rng)
        if s is None:
            continue
        s['lab'] = lab
        s['lo'], s['hi'] = lo, hi
        s['t_from_h'] = (float(t[sel].min()) - t0) / 3600.0
        s['t_to_h'] = (float(t[sel].max()) - t0) / 3600.0
        rec['wins'].append(s)
    # 事件在时间上的集中度：最大的一个时间等分装了多少事件
    te = np.linspace(t0, t0 + span_h * 3600.0, 11)
    cnt = np.histogram(t, bins=te)[0]
    rec['t_conc'] = float(cnt.max()) / max(n, 1)
    return rec


def fmt(rec):
    L = ['=' * 92,
         '%s  定位事件 %d / 全部 %d（保留 %.0f%%）  AE 跨度 %.1f h  探头体检 %s'
         % (rec['gid'], rec['n_loc'], rec['n_all'], 100 * rec['keep'],
            rec['span_h'], '通过' if rec['audit_ok'] else '未通过/未测'),
         '  图纸冲击点（skin 系，已 180° 旋转）= %s'
         % ('(%.0f, %.0f)' % rec['truth'] if rec['truth'] else '缺'),
         '  事件在时间上的集中度：最满的一个时间等分占 %.0f%%（越高越不能按时间切窗）'
         % (100 * rec['t_conc']),
         '  事件数窗      时间范围(h)        n     质心(x, y)        r68   r90   到真值 零中位 分位']
    for s in rec['wins']:
        L.append('  %-12s %5.1f 至 %-7.1f %6d  (%6.1f, %6.1f) %6.1f %5.1f %7.1f %6s %6s'
                 % (s['lab'], s['t_from_h'], s['t_to_h'], s['n'], s['cx'], s['cy'],
                    s['r68'], s['r90'], s['d_truth'],
                    ('%.1f' % s['null_med']) if np.isfinite(s.get('null_med', np.nan)) else '—',
                    ('%.0f%%' % s['null_pct']) if np.isfinite(s['null_pct']) else '—'))
    body = [s for s in rec['wins'] if s['lab'] != '全寿命']
    if body:
        L.append('  离散度 r68 随寿命：%s（首 %.1f -> 末 %.1f，单调增 %s）'
                 % (' / '.join('%.1f' % s['r68'] for s in body),
                    body[0]['r68'], body[-1]['r68'],
                    '是' if all(body[i]['r68'] <= body[i + 1]['r68']
                                for i in range(len(body) - 1)) else '否'))
        cen = np.array([[s['cx'], s['cy']] for s in body])
        mig = np.hypot(np.diff(cen[:, 0]), np.diff(cen[:, 1]))
        L.append('  相邻窗质心迁移（mm）：%s；总迁移 %.1f mm'
                 % (' / '.join('%.1f' % v for v in mig), float(mig.sum())))
    return '\n'.join(L)


def main(argv):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    gids = []
    for i, a in enumerate(argv):
        if a == '--groups':
            gids = argv[i + 1].split(',')
    if not gids:
        gids = sorted(os.path.basename(p)[len('_l1_loc_'):-4]
                      for p in glob.glob(os.path.join(RES, '_l1_loc_*.npz')))
    tdf = pd.read_csv(os.path.join(RES, 'l1_impact_truth.csv'))
    rng = np.random.default_rng(0)

    recs, blocks, rows = [], [], []
    for g in gids:
        r = analyse(g, tdf, rng)
        if r is None:
            blocks.append('%s  跳过（无定位 npz 或可信事件过少）' % g)
            continue
        recs.append(r)
        blocks.append(fmt(r))
        for s in r['wins']:
            rows.append({'gid': g, 'win': s['lab'], 'lo': s['lo'], 'hi': s['hi'],
                         'n': s['n'],
                         't_from_h': round(s['t_from_h'], 1),
                         't_to_h': round(s['t_to_h'], 1),
                         'cx': round(s['cx'], 1), 'cy': round(s['cy'], 1),
                         'r68': round(s['r68'], 1), 'r90': round(s['r90'], 1),
                         'd_truth': round(s['d_truth'], 1) if np.isfinite(s['d_truth']) else '',
                         'null_pct': round(s['null_pct'], 1) if np.isfinite(s['null_pct']) else '',
                         'audit_ok': r['audit_ok']})

    head = ('C1/C2 批无帧表也无循环轴，故按 **AE 相对时间**分位切窗；'
            '真值 = 图纸 mm 值 + 180° 旋转 (165−X, 243−Y)。\n'
            '「零分布分位」= 从全组随机抽同样本量子集时，质心距真值小于观测值的概率；\n'
            '该值小（如 < 10%）才说明「早期更近」不是采样噪声。\n')
    with open(OUT_TXT, 'w', encoding='utf-8') as fh:
        fh.write(head + '\n'.join(blocks) + '\n')
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print('已写 %s\n     %s' % (OUT_CSV, OUT_TXT))
    print('\n'.join(blocks))


if __name__ == '__main__':
    main(sys.argv[1:])
