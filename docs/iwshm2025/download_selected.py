# -*- coding: utf-8 -*-
"""按人工筛读清单下载 IWSHM 2025 论文（全部开放获取）。

清单分四组，每组对应你稿子里的一个待补强点：

  A 主题同源  —— 归一化、可比性、population-based、域适应、指标选择、数据集
  B 结构同源  —— 复合材料加筋板、导波、FBG、温度与载荷影响量
  C 模态同源  —— 声发射定位、传感器故障与可靠性
  D 统计方法  —— 不确定度、样本量、MAPOD、信息价值

用法
    python download_selected.py            # 下载
    python download_selected.py --list     # 只打印清单

URL 规则：https://www.ndt.net/article/iwshm2025/papers/IWSHM_<n>.pdf
"""
import argparse
import csv
import os
import sys
import time

import requests

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
PDFDIR = os.path.join(HERE, 'pdfs')
LOG = os.path.join(HERE, 'download_log.csv')
BASE = 'https://www.ndt.net/article/iwshm2025/papers/IWSHM_%s.pdf'
HDR = {'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/126.0 Safari/537.36'),
       'Accept': 'application/pdf,*/*'}

# (编号, 组, 理由)
PICKS = [
    # ---- A 主题同源 ----
    ('975', 'A', '数据归一化做无监督损伤诊断，与你的归一化主张最直接同题'),
    ('934', 'A', 'NDE/SHM 里的域偏移与自感知，是「跨机构」在 ML 语境下的同义问题'),
    ('1876', 'A', 'population-based SHM 应用于渡口码头，你这篇该挂的框架'),
    ('2189', 'A', '多源域适应，换结构不换标注的做法'),
    ('2513', 'A', '两座闸门之间的域适应泛化，跨结构可比性的实例'),
    ('1817', 'A', 'PINN 加迁移学习做损伤识别，迁移类写法的参考'),
    ('610',  'A', '自建数据集论文（在建），看人家怎么描述数据与可用性'),
    ('1232', 'A', '提出「估计精度比较指数」，与 SNHI 的指标定义属同一文体'),
    ('1403', 'A', '不确定性下的损伤指标选择，直接对你 §3 的指标论证'),
    ('2015', 'A', '损伤检测所需样本量的定量分析，对你 n=29/17 的辩护有用'),
    ('2358', 'A', '长期连续监测的异常检测与基线问题，等价于你的寿命轴'),
    ('1768', 'A', '温度补偿，影响量一类的标准处理'),
    ('1861', 'A', '用局部距离相关考察环境变化，另一种影响量归一化'),
    ('1220', 'A', '季节环境效应，影响量对结构行为的作用'),
    # ---- B 结构同源 ----
    ('1901', 'B', '复合材料加筋板加导波加多任务学习，与 ReMAP 结构最像'),
    ('1917', 'B', '加筋复合材料板的高分辨超声成像，同结构不同方法'),
    ('58',   'B', '后屈曲壁板的形状重构，分布式光纤加 DIC，与你的应变线同源'),
    ('265',  'B', '载荷与温度双影响量下的导波损伤检测，影响量分离的写法'),
    ('366',  'B', '无人机复材结构冲击分层的实时评估，光纤方案'),
    ('2425', 'B', '低温对 CFRP 导波信号的影响，影响量的定量化'),
    ('1395', 'B', '温度对 PZT 响应的影响，传感器侧的漂移'),
    ('117',  'B', '全尺寸复材机翼多参数监测系统，工程口径'),
    ('1338', 'B', '复材叶片多损伤早期检出与定位'),
    ('1279', 'B', '发动机复材风扇叶片的导波检测'),
    # ---- C 模态同源 ----
    ('402',  'C', 'SHM 系统传感器故障检测方法的横向比较，与你 §3.2 同题'),
    ('2278', 'C', '传感器故障的检测、定位与补偿鲁棒性'),
    ('2329', 'C', '同上，退化系统中的鲁棒性，可与 2278 对照读'),
    ('560',  'C', '声发射传感器自诊断，失效判据的来源'),
    ('306',  'C', '匹配滤波做传感器故障分类'),
    ('728',  'C', '声发射定位加矩张量反演，定位精度的另一条路'),
    ('545',  'C', '声发射加机器学习定位裂纹尖端'),
    ('1346', 'C', '疲劳裂纹自身发射的超声信号，被动监测的物理基础'),
    ('2072', 'C', '导波跟踪疲劳裂纹演化，时序跟踪写法'),
    ('465',  'C', '单面紧固件隐蔽疲劳裂纹检出'),
    ('1111', 'C', '面向疲劳损伤评估的 AI-SHM 方法'),
    # ---- D 统计方法 ----
    ('447',  'D', '自由振动模态识别的不确定度规律，不确定度写法参考'),
    ('1484', 'D', '全场超声测厚的不确定度量化'),
    ('1997', 'D', 'MAPOD，检出概率的模型辅助做法'),
    ('2561', 'D', '多个损伤模式下的信息价值分析，决策落点的写法'),
    ('1484b', 'D', '占位，忽略'),
]


def download(picks, dry=False):
    os.makedirs(PDFDIR, exist_ok=True)
    log = []
    for i, (num, grp, why) in enumerate(picks, 1):
        if not num.isdigit():
            continue
        url = BASE % num
        path = os.path.join(PDFDIR, 'IWSHM_%s.pdf' % num)
        if os.path.exists(path) and os.path.getsize(path) > 20000:
            note = 'exists'
        elif dry:
            note = 'dry'
        else:
            note = ''
            try:
                with requests.get(url, headers=HDR, timeout=30, stream=True) as r:
                    if r.status_code != 200:
                        note = 'HTTP %d' % r.status_code
                    else:
                        buf = bytearray()
                        for c in r.iter_content(65536):
                            buf += c
                        if bytes(buf[:5]).startswith(b'%PDF'):
                            open(path, 'wb').write(buf)
                            note = 'ok'
                        else:
                            note = 'not-pdf'
            except requests.RequestException as e:
                note = 'err:%s' % type(e).__name__
            time.sleep(0.4)
        log.append({'num': num, 'group': grp, 'why': why, 'note': note,
                    'kb': round(os.path.getsize(path) / 1024) if os.path.exists(path) else 0})
        print('  %2d/%-2d [%s] IWSHM_%-5s %-8s %s' % (i, len(picks), grp, num, note, why[:40]))

    if not dry:
        with open(LOG, 'w', newline='', encoding='utf-8-sig') as fh:
            wr = csv.DictWriter(fh, fieldnames=['num', 'group', 'why', 'note', 'kb'])
            wr.writeheader()
            wr.writerows(log)
    ok = sum(1 for x in log if x['note'] in ('ok', 'exists'))
    print('\n成功 %d / %d' % (ok, len([x for x in log if x['note'] != 'dry'])))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    picks = [p for p in PICKS if p[0].isdigit()]
    print('清单 %d 篇（A 主题 %d / B 结构 %d / C 模态 %d / D 统计 %d）'
          % (len(picks), sum(p[1] == 'A' for p in picks), sum(p[1] == 'B' for p in picks),
             sum(p[1] == 'C' for p in picks), sum(p[1] == 'D' for p in picks)))
    if a.list:
        for num, grp, why in picks:
            print('  [%s] %-5s %s' % (grp, num, why))
        return 0
    download(picks, dry=a.dry)
    return 0


if __name__ == '__main__':
    sys.exit(main())
