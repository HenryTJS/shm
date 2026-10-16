#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文插图的统一版式（英文 · 期刊风格）—— fig_protocol / fig_two_systems / fig_four_modal 共用。

风格约定
--------
- 白底、衬线字（Times）、细线；不用圆角矩形、阴影、渐变、粗色块、大段图内文字
- 字号：面板标题 7 至 7.5，轴标签 6.2 至 7，刻度 6，注释 5.8 至 6.3
- 配色低饱和：标「机构」用 `BATCH`（Lab A 蓝系 / Lab B 红系），标「模态」用 `MODALITY`
- 数字与结论交给正文和图注，图内只留单位与必要的少量标注

用法
----
    import paper_style as PS
    PS.apply()
    ...
    PS.thin(ax)                       # 去掉上/右边框
    PS.save(fig, os.path.join(FIG, 'fig_xxx.png'))
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 按机构分色：Lab A 蓝系、Lab B 红系 —— 全局一致，读者一眼分辨机构
C_A, C_A2 = '#2f4b7c', '#7c9dc4'
C_B, C_B2 = '#a4453b', '#c98b7f'
BATCH = {'C1': C_A, 'C2': C_A2, 'C3': C_B, 'C4': C_B2}
LAB = {'C1': 'A', 'C2': 'A', 'C3': 'B', 'C4': 'B'}

# 按模态分色：四色低饱和，色相拉得够开
MODALITY = {'AE': '#2f4b7c', 'DFOS': '#3f7f6f',
            'PZT': '#a4453b', 'FBG': '#b08a3e'}


def apply():
    """装上全局版式。每个出图脚本开头调一次。"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 7,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.2,
        'ytick.major.size': 2.2,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6.2,
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def thin(ax, keep=('left', 'bottom')):
    """只留需要的边框（默认留左/下，去掉上/右）。"""
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(s in keep)


def save(fig, path, dpi=300):
    """同时写 .png 与 .pdf。"""
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.replace('.png', '.pdf'))
    plt.close(fig)
    print('已写 %s 与同名 .pdf' % path)
