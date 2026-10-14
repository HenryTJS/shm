"""可行性探针：导波尾波的退相关图样是否与「已知的冲击位置」几何一致？

判据（先验、可证伪，不看数据就知道该看到什么）
    冲击在板中心 (120, 82.5) mm。PZT 坐标 (mm)：
      1(25,25) 2(25,100) 3(215,140) 4(215,66) 5(120,25) 6(120,140) 7(90,82.5) 8(180,82.5)
    - 路径 7-8：(90,82.5)→(180,82.5) 直线穿过中心
    - 路径 5-6：(120,25)→(120,140)  直线穿过中心
    ⇒ 这两条路径的敏感核椭圆应最贴近损伤，退相关应最强。
    反之，远端路径（如 1-2，都在 x=25 一侧）退相关应最弱。

做法
    尾波窗取直达波之后的晚段（点 500 至 1800 = 250 至 900 µs），带通 150 至 350 kHz。
    逐路径退相关 D = 1 − corr(尾波_REF, 尾波_该状态)。
    椭圆敏感核权重  W = exp( −(d_k + d_m − L_km)^2 / (2σ^2) )，
      d_* = 传感器到冲击点的距离，L_km = 直达距离，σ 做扫描。

两个对照（用来判断「REF 是否在冲击之前」这个前提）
    - REF→DAM#1：若冲击在 REF 之前，此步只有疲劳，图样应接近均匀
    - DAM#1→DAM#2：纯疲劳的一步，作为基线
  若 REF→DAM#1 明显比 DAM#1→DAM#2 大、且更贴合几何 ⇒ 冲击确实落在这一步里。

只读，不写任何产物。
"""
import importlib.util
import os
import sys

import numpy as np
import scipy.io as sio
from scipy.signal import hilbert

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


PC = _load('pc', os.path.join(HERE, 'pzt_crosslab.py'))
PI = PC.PI

FREQ = '250kHz_5cycles'
GEO = np.array([[25, 25], [25, 100], [215, 140], [215, 66],
                [120, 25], [120, 140], [90, 82.5], [180, 82.5]], float)
IMPACT = np.array([120.0, 82.5])        # 板中心（impact_center）
CODA = (500, 1800)                       # 兼容旧引用
CODA_WINS = [(320, 700), (320, 1100), (500, 1800)]   # 候选尾波窗（点）
SIGMA = 25.0                             # 椭圆核宽度（mm），下面做扫描


def snr_of_win(x, ba, win):
    """窗口内信噪比（dB）：窗口 RMS / 深尾段（1900 至 2000 点）RMS。

    若该值接近 0 dB，则窗口里本就没有信号 —— 那种情况下「退相关恒等于 1」
    是平凡结论，不能当成物理发现。
    """
    b, a = ba
    from scipy.signal import filtfilt
    y = filtfilt(b, a, x - x.mean())
    sig = float(np.sqrt(np.mean(y[win[0]:win[1]] ** 2)))
    noi = float(np.sqrt(np.mean(y[1900:2000] ** 2)))
    return 20 * np.log10(sig / noi) if noi > 0 else np.nan


def coda(x, ba, win):
    b, a = ba
    from scipy.signal import filtfilt
    return filtfilt(b, a, x - x.mean())[win[0]:win[1]]


BA = PI.bandpass(250.0)


def paths_of_state(gdir, pztN, state, win, rep=1):
    """一个状态下 56 条路径的尾波信号。"""
    out = {}
    for k in range(1, 9):
        p = os.path.join(gdir, pztN, state, FREQ, 'Actionneur%d' % k,
                         'measured_data_rep_%d.mat' % rep)
        if not os.path.exists(p):
            return None
        tr = sio.loadmat(p)['Time_Response']
        for j, m in enumerate(PI.rx_of(k)):
            out[(k, m)] = coda(np.asarray(tr[:, 2 + j], float), BA, win)
    return out


def decorr(a, b):
    """1 − 归一化相关（尾波退相关）。"""
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1.0 - float(np.dot(a, b) / (na * nb)) if na * nb > 0 else np.nan


def debias_by_actuator(keys, d):
    """减去每条路径所属作动器内 7 条路径的中位。

    动机（先验）：实测发现最强的路径按作动器聚堆（某组全被作动器 4 占据，
    另一组全是作动器 2），说明退相关被「哪个作动器发的」主导 —— 那是耦合/安装
    状态的漂移，与损伤几何无关。若几何信号真的存在，去偏移后应当浮现。
    预期：若存在几何信号，去偏移后的 ρ 应当**转正**；若仍不转正，则几何信号不存在。
    """
    out = np.array(d, float).copy()
    for k in range(1, 9):
        idx = [i for i, kk in enumerate(keys) if kk[0] == k]
        if len(idx) >= 3:
            out[idx] -= np.nanmedian(out[idx])
    return out


def kernel(sigma=SIGMA):
    """每条路径的椭圆敏感核权重（相对量，无需归一化）。"""
    w = {}
    for k in range(1, 9):
        for m in PI.rx_of(k):
            dk = np.hypot(*(GEO[k - 1] - IMPACT))
            dm = np.hypot(*(GEO[m - 1] - IMPACT))
            L = np.hypot(*(GEO[k - 1] - GEO[m - 1]))
            w[(k, m)] = float(np.exp(-((dk + dm - L) ** 2) / (2 * sigma ** 2)))
    return w


def main():
    print('椭圆的先验预测：哪些路径应当退相关最强（σ = %.0f mm）' % SIGMA)
    w = kernel()
    top = sorted(w.items(), key=lambda kv: -kv[1])[:12]
    print('  ' + '  '.join('%d-%d(%.2f)' % (k[0], k[1], v) for k, v in top))
    print('\n  对照：最弱的 6 条  ' + '  '.join(
        '%d-%d(%.3f)' % (k[0], k[1], v) for k, v in sorted(w.items(), key=lambda kv: kv[1])[:6]))
    print('  注：路径 7-8 与 5-6 是否进前列，是本探针的第一判据')

    for g in ['L1-24', 'L1-13', 'L1-06', 'L1-25', 'L1-44']:
        st = PC.dam_order(g)
        if len(st) < 3:
            print('\n[%s] DAM 不足，跳过' % g)
            continue
        gdir = os.path.join(HERE, g)
        refst = None
        for e in os.scandir(os.path.join(gdir, st[0][1])):
            if e.is_dir() and e.name.startswith('State_') and e.name not in [x[2] for x in st]:
                refst = e.name
        print('\n================ %s（REF=%s）================' % (g, refst))

        for win in CODA_WINS:
            s_ref = paths_of_state(gdir, st[0][1], refst, win) if refst else None
            if s_ref is None:
                print('  取不到 REF，跳过')
                break
            s1 = paths_of_state(gdir, st[0][1], st[0][2], win)
            s2 = paths_of_state(gdir, st[0][1], st[1][2], win)
            if s1 is None or s2 is None:
                print('  取不到 DAM 状态，跳过')
                break

            # 窗口内信噪比（若接近 0 dB，则该窗内没有信号，退相关恒为 1 是平凡结论）
            p = os.path.join(gdir, st[0][1], refst, FREQ, 'Actionneur1',
                             'measured_data_rep_1.mat')
            x1 = np.asarray(sio.loadmat(p)['Time_Response'][:, 2], float)
            snr = snr_of_win(x1, BA, win)

            keys = sorted(s_ref)
            d1 = np.array([decorr(s1[k], s_ref[k]) for k in keys])
            d2 = np.array([decorr(s2[k], s1[k]) for k in keys])
            wv = np.array([w[k] for k in keys])
            ok = np.isfinite(d1) & np.isfinite(d2)
            d1n = debias_by_actuator(keys, d1)
            d2n = debias_by_actuator(keys, d2)
            print('  窗 %4d-%4d（SNR %5.1f dB）：REF→DAM#1 中位 %.3f  DAM#1→DAM#2 中位 %.3f'
                  % (win[0], win[1], snr, np.nanmedian(d1), np.nanmedian(d2)))
            print('      与椭圆核 ρ： 原始 %+.3f / %+.3f    **按作动器去偏移** %+.3f / %+.3f'
                  % (_sp(d1[ok], wv[ok])[0], _sp(d2[ok], wv[ok])[0],
                     _sp(d1n[ok], wv[ok])[0], _sp(d2n[ok], wv[ok])[0]))
            if d1.max() < 1.5 and np.nanmedian(d1) < 0.6:
                order = np.argsort(-d1)
                print('       实测最强 6 条：' + '  '.join(
                    '%d-%d(%.2f|W=%.2f)' % (keys[i][0], keys[i][1], d1[i], wv[i])
                    for i in order[:6]))
                for kk in [(7, 8), (5, 6)]:
                    if kk in keys:
                        i = keys.index(kk)
                        print('       * 路径 %d-%d：%.3f（W=%.2f）排名 %d/%d'
                              % (kk[0], kk[1], d1[i], wv[i],
                                 int(np.where(order == i)[0][0]) + 1, len(keys)))


def _sp(a, b):
    from scipy import stats
    r = stats.spearmanr(a, b)
    return r.correlation, r.pvalue


if __name__ == '__main__':
    main()
