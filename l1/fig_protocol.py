"""SNHI 协议流程图（论文用图）。

左侧：协议的四步流程
右侧：上块 = 现实约束（为什么要它）；下块 = 验证结果（它是否管用）

风格与 l1/figures 下其他图一致（Microsoft YaHei、unicode_minus=False）。
只写 l1/figures/fig_protocol.png / .pdf，不读任何数据文件。
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 7.5

C_IN = '#4c78a8'
C_STEP = '#5b8c5a'
C_OUT = '#c0392b'
C_BG = '#f4f6f8'

# (标题, 正文, 边框色)
LEFT = [
    ('输入', '多机构 SHM 数据：波形 / 事件流，状态目录，n_f，\n'
             '以及数据集自带的采集设计文档\n'
             '障碍：时间戳不可靠 —— 错年份、原点不同、跨批无法对齐', C_IN),
    ('步骤 1　寿命轴重建', 'Δ ← 采集设计文档记载的测量间隔（5000 或 5200 cycles）\n'
                          'cycle(k) = k × Δ　→　life = cycle / n_f\n'
                          '自校验：与独立的绝对时钟轴互校（秩相关 ≥ 0.993）', C_STEP),
    ('步骤 2　无标度指标构造', 'C(t) ← 按寿命分箱后的累积量\n'
                              'Ĉ(t) = C(t) / C(T)　（自归一化）\n'
                              'HI_SNHI = 使 Ĉ(t) = x 的 t，取 x = 50% / 85%', C_STEP),
    ('步骤 3　可迁移性评价', 'R = 批间极差（各批中位数）/ 组内极差中位数\n'
                            '同一数据上并行跑绝对阈值法作为对照\n'
                            'R 越小，该指标越可跨机构迁移', C_STEP),
    ('输出', 'SNHI 值 · 可迁移性比 R · 跨批与跨模态的一致性报告', C_OUT),
]

RIGHT = [
    ('为什么需要它：绝对阈值不可迁移',
     '· 单阈值达阈时刻跨组离散：中位 46.2 pp，最大 65.8 pp\n'
     '· 光纤失效值跨组跨度：6.57 至 7.34 倍（CV 0.37 至 0.41）\n'
     '· 因果归一后失效值 CV 仍为 0.967 至 1.622\n'
     '· 失效时声发射计数在同一批内跨 50.3 倍（918 至 46221）\n'
     '因此统一绝对阈值在算术上不可能', '#a02020'),
    ('它管用吗：留一批对照（15 个试件）',
     '绝对阈值法：6/15 试件从不报警，\n'
     '　　　　　　组内极差中位 60.7 pp\n'
     'SNHI：　　0/15 漏报，\n'
     '　　　　　　组内极差中位 18.5 pp（小 3.3 倍）\n'
     '批间极差 2.4 pp', '#1a6b3c'),
]


def box(ax, x, y, w, h, title, body, ec, fc='white', ts=8.0, bs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.006,rounding_size=0.012',
                                linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + 0.012, y + h - 0.020, title, fontsize=ts, color=ec,
            fontweight='bold', va='top', ha='left', zorder=3)
    ax.text(x + 0.012, y + h - 0.052, body, fontsize=bs, va='top', ha='left',
            zorder=3, linespacing=1.45)


def arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle='-|>',
                                 mutation_scale=9, linewidth=1.0,
                                 color='0.35', zorder=1))


def main():
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.002, 0.002), 0.996, 0.996,
                                boxstyle='square,pad=0', linewidth=0, facecolor=C_BG))

    LX, LW = 0.028, 0.545            # 左栏
    RX, RW = 0.600, 0.372            # 右栏
    gap = 0.028
    hs = [0.158, 0.158, 0.158, 0.158, 0.100]      # 各框高度（自上而下）
    total = sum(hs) + gap * (len(hs) - 1)
    top = 0.966

    # 自上而下依次放置，否则 输入/输出 会颠倒
    ys = []
    yy = top
    for (t, b, c), h in zip(LEFT, hs):
        y0 = yy - h
        box(ax, LX, y0, LW, h, t, b, c)
        ys.append((y0, h))
        yy = y0 - gap
    for i in range(len(ys) - 1):
        y_from = ys[i][0]                                        # 上框底部
        y_to = ys[i + 1][0] + ys[i + 1][1] + 0.002               # 下框顶部
        arrow(ax, LX + LW / 2, y_from, y_to)

    # 右栏：两块从同一顶端往下排
    rh = (top - (top - total) - gap) / 2
    yy = top
    for (t, b, c) in RIGHT:
        box(ax, RX, yy - rh, RW, rh, t, b, c, fc='white', ts=7.4, bs=6.2)
        yy -= rh + gap

    ax.text(0.5, 0.014, 'SNHI 协议：形状归一化健康指标（Shape-Normalized Health Indicator）',
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='0.2')

    fig.tight_layout(pad=0.25)
    fp = os.path.join(FIG, 'fig_protocol.png')
    fig.savefig(fp, dpi=300)
    fig.savefig(fp.replace('.png', '.pdf'))
    plt.close(fig)
    print('已写', fp, '与同名 .pdf')


if __name__ == '__main__':
    main()
