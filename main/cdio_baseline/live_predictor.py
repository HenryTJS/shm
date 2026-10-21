#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
飞行器结构件 实时剩余寿命百分比预测系统（最终封装版）
================================================================================
方法（与离线验证严格一致，详见最终版报告）：
  主时钟   CElog = cumsum( log1p( ae_SpectralEnergy ) )   —— 跨工况稳健的累积损伤时钟
  组成指纹 CP    = cumsum( ae_Peak )                      —— 提升不同试件间的轨迹辨识
  估计器C  容量-组成指纹 KNN 相似匹配（softmin, K=3, 50点归一化轨迹）
  估计器D  库 CElog 总容量对数正态贝叶斯在线反解（给 90% 概率区间，末期尾部外推）
  融合     在线同步对比约束动态门控（按两估计器分歧/存活库比例/邻域一致性/最近邻距离
           实时决定 KNN 权重，Ridge 门控，参数已在库内层留一标定并固化于模型工件）
  默认同时输出 动态权重 与 固定0.7 两套结果。

两种用法：
  1) Python 流式 API（随到随算，真正在线）：
        from live_predictor import LiveLifePredictor
        model = LiveLifePredictor("model_bundle_final.pkl")
        for window in stream:                 # 每个时间窗一行（dict，含 ae_SpectralEnergy[/ae_Peak]）
            model.update(window)
            out = model.predict()            # 随时取当前寿命百分比
            print(out["life_pct"], out["status"])

  2) 命令行（离线整文件 / 标准输入流式）：
        python live_predictor.py --input 实时数据.csv --every 200 --out 预测结果.csv
        tail -f 数据流.csv | python live_predictor.py --stream --every 200

输入规范（务必与训练库同口径）：
  - CSV，每行一个时间窗，窗长固定 0.1594 s，且 ae_SpectralEnergy / ae_Peak 的定义、
    频带与标定与建模数据一致（连续波形需先用相同流程提取窗级统计量）。
  - 必需列：ae_SpectralEnergy（非负能量）。
  - 强烈建议：ae_Peak（缺省时自动退化为仅 CElog 单时钟，跨件辨识力略降）。
  - 可选列：ae_time_s（仅用于输出时间戳）、strain、fiber_*（当前主线未参与，预留）。
  - 无需健康基线、无需总寿命、无需标签；负值/空值自动按 0 处理。
  - 列名兼容：SpectralEnergy/能量/ae_energy、Peak/峰值 等常见写法会自动映射。
================================================================================
"""
import os, sys, json, argparse, pickle, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------------- 中文字体（画图用） -----------------------------
def _setup_cn_font():
    for p in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
              r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\STZHONGS.TTF",
              r"C:\Windows\Fonts\simsun.ttc"]:
        if os.path.exists(p):
            try: font_manager.fontManager.addfont(p)
            except Exception: pass
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "STZhongsong", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

# ----------------------------- 列名兼容映射 -----------------------------
ALIAS = {
    "ae_SpectralEnergy": ["aespectralenergy","spectralenergy","spectral_energy","ae_energy",
                          "energy","能量","ae能量","声发射能量"],
    "ae_Peak": ["aepeak","peak","ae_peak","amplitude_peak","峰值","ae峰值","声发射峰值"],
    "ae_time_s": ["aetime_s","time_s","time","时间","窗时间"],
}

def _norm(name): return str(name).strip().lower().replace(" ","").replace("-","_")

def map_columns(df):
    """把任意常见列名映射到标准列，返回 (energy_col or None, peak_col or None, time_col or None)"""
    lookup={_norm(c):c for c in df.columns}
    def find(std):
        if _norm(std) in lookup: return lookup[_norm(std)]
        for a in ALIAS[std]:
            if _norm(a) in lookup: return lookup[_norm(a)]
        return None
    return find("ae_SpectralEnergy"), find("ae_Peak"), find("ae_time_s")

class LiveLifePredictor:
    """随数据流入、增量输出寿命百分比的实时预测器（加载自包含模型工件）。"""

    def __init__(self, bundle_path="model_bundle_final.pkl", gate="ridge", exclude_ids=None):
        """bundle_path:模型工件；gate: ridge动态权重 / fixed固定0.7；
        exclude_ids: 预测在库试件做留一验证时，需从库中剔除的试件编号（实际部署预测新试件时留空）。"""
        with open(bundle_path,"rb") as f: self.b=pickle.load(f)
        self.gate=gate
        excl=set(exclude_ids or [])
        self.lib=[r for r in self.b["library"] if r["id"] not in excl]; self.nlib=len(self.lib)
        self.M=self.b["M"]; self.K=self.b["knn"]
        self.mu=self.b["bayes"]["mu"]; self.sig=self.b["bayes"]["sig"]
        g=self.b["gate"]; self.gfeats=g["feats"]; self.gmu=g["mu"]; self.gsd=g["sd"]
        self.gcoef=g["coef"]; self.gint=g["intercept"]; self.wlo=g["w_lo"]; self.whi=g["w_hi"]; self.w0=g["w_default"]
        self.ygrid=np.linspace(0,80,200001)
        self.alarm=self.b["alarm"]
        self.reset()

    def reset(self):
        """清空流式状态（开始一个新试件前调用）。"""
        self.ce=[]; self.cp=[]; self.t=[]; self.has_peak=True; self.n=0; self._ce=0.0; self._cp=0.0
        return self

    # -------------------------- 数据摄入（增量） --------------------------
    def update(self, data):
        """喂入一个新时间窗（dict/pandas.Series）或一批时间窗（pandas.DataFrame）。返回 self，可链式。"""
        if isinstance(data,pd.DataFrame):
            ec,pc,tc=map_columns(data)
            if ec is None:
                raise ValueError("输入缺少能量列 ae_SpectralEnergy（或其别名：SpectralEnergy/能量/energy）")
            E=pd.to_numeric(data[ec],errors="coerce").fillna(0).clip(lower=0).values
            if pc is not None:
                P=pd.to_numeric(data[pc],errors="coerce").fillna(0).clip(lower=0).values
            else:
                P=np.zeros_like(E); self.has_peak=False
            if tc is not None: self.t.extend(pd.to_numeric(data[tc],errors="coerce").tolist())
            self._append(E,P); return self
        if isinstance(data,dict) or isinstance(data,pd.Series):
            one=pd.DataFrame([dict(data)])
            ec,pc,tc=map_columns(one)
            if ec is None:
                raise ValueError("输入缺少能量列 ae_SpectralEnergy（或其别名：SpectralEnergy/能量/energy）")
            E=pd.to_numeric(one[ec],errors="coerce").fillna(0).clip(lower=0).values
            if pc is not None:
                P=pd.to_numeric(one[pc],errors="coerce").fillna(0).clip(lower=0).values
            else:
                P=np.zeros_like(E); self.has_peak=False
            if tc is not None:
                tv=pd.to_numeric(one[tc],errors="coerce").iloc[0]
                if pd.notna(tv): self.t.append(float(tv))
            self._append(E,P); return self
        raise TypeError("update 仅接受 dict/Series（单窗）或 DataFrame（批量）")

    def _append(self,E,P):
        self._ce += float(np.log1p(E).sum()) if E.size else 0.0
        self._cp += float(P.sum())
        # 维护逐窗累积轨迹（供组成指纹插值）；批量时用 cumsum 对齐
        ce=np.cumsum(np.log1p(E))+(self.ce[-1] if self.ce else 0.0)
        cp=np.cumsum(P)+(self.cp[-1] if self.cp else 0.0)
        self.ce.extend(ce.tolist()); self.cp.extend(cp.tolist()); self.n+=E.size

    # ----------------------------- 两个估计器 -----------------------------
    def _knn(self):
        x=self.ce[-1]
        if x<=1e-12: return dict(pred=0.0,alive=1.0,dmin=0.0,cons=0.0)
        M=self.M; qx=np.linspace(0,x,M)
        ce=np.asarray(self.ce); cp=np.asarray(self.cp)
        comps=["CElog","CP"] if self.has_peak else ["CElog"]
        Pn={}
        for c in comps:
            Pn[c]=(qx/x) if c=="CElog" else (np.interp(qx,ce,cp)/(x+1e-12))
        rec=[]
        for r in self.lib:
            X=r["CElog"]; Xt=X[-1]
            if Xt<=x*1.001: continue
            ur=float(np.interp(x,X,r["u"])); qr=np.linspace(0,x,M); dd=0.0; nd=0
            for c in comps:
                src=(qr/x) if c=="CElog" else (np.interp(qr,X,r["CP"])/(x+1e-12))
                cat=np.concatenate([Pn[c],src])
                if np.abs(cat).max()<1e-9: continue
                sd=np.std(cat)+1e-9; dd+=np.mean(np.abs((Pn[c]-src)/sd)); nd+=1
            rec.append((dd/nd if nd else 1e6,ur))
        if len(rec)<2: return dict(pred=np.nan,alive=len(rec)/self.nlib,dmin=9.0,cons=9.0)
        rec.sort(key=lambda z:z[0]); top=rec[:self.K]
        ds=np.array([z[0] for z in top]); us=np.array([z[1] for z in top]); sig=np.median([z[0] for z in rec])+1e-9
        w=np.exp(-ds**2/(2*sig**2)); w=w/w.sum()
        return dict(pred=float(np.clip((w*us).sum()*100,0,100)),
                    alive=len(rec)/self.nlib, dmin=float(ds[0]),
                    cons=float(np.std(ds)/(np.mean(ds)+1e-9)))

    def _bayes(self):
        x=self.ce[-1]
        if x<=0: return 0.0,(0.0,0.0)
        y=self.ygrid; y=y[y>=np.log(x)-1e-9]
        from scipy.stats import norm
        w=norm.pdf(y,self.mu,self.sig)*np.exp(-y); w=w/w.sum()
        u=np.clip(x/np.exp(y),0,1)
        mean=float(np.sum(w*u)*100)
        order=np.argsort(u); cs=np.cumsum(w[order]); uo=u[order]
        lo=float(np.interp(0.05,cs,uo)*100); hi=float(np.interp(0.95,cs,uo)*100)
        return mean,(lo,hi)

    # ------------------------------- 预测 --------------------------------
    def predict(self):
        """输出当前时刻寿命百分比与诊断信息；预热不足返回 None。"""
        if self.n < self.alarm["min_warmup"]:
            return {"status":["WARMUP 预热中（窗数不足）"],"n_windows":self.n,"life_pct":None}
        k=self._knn(); (b,(lo,hi))=self._bayes()
        if np.isnan(k["pred"]):
            return {"status":["NO_NEIGHBOR 历史库中无容量更大的试件，已超出库覆盖范围"],
                    "n_windows":self.n,"life_pct":None,"bayes_pct":b,"bayes_p05":lo,"bayes_p95":hi,
                    "alive_frac":k["alive"]}
        disagree=abs(k["pred"]-b)/100.0
        feats=np.array([disagree,k["alive"],k["cons"],k["dmin"]])
        if self.gate=="ridge":
            z=(feats-self.gmu)/self.gsd
            wk=float(np.clip(z@self.gcoef+self.gint,self.wlo,self.whi))
        else:
            wk=self.w0
        dyn=float(wk*k["pred"]+(1-wk)*b); fix=float(self.w0*k["pred"]+(1-self.w0)*b)
        status=[]
        if k["alive"]<self.alarm["alive_low"]: status.append("CAPACITY_LOW 存活历史件不足：接近/超出库容量范围，末期不确定性高")
        if disagree>self.alarm["disagree_high"]: status.append("DISAGREE 两估计器分歧大：可能遇到新工况或库失配，建议人工复核/补库")
        if k["dmin"]>self.alarm["dmin_high"]: status.append("FAR_MATCH 无接近的相似历史件，结果外推成分高")
        if not status: status.append("OK 库覆盖充分、两估计器一致")
        return dict(life_pct=round(dyn,2), life_pct_fixed=round(fix,2),
                    knn_pct=round(k["pred"],2), bayes_pct=round(b,2),
                    bayes_p05=round(lo,2), bayes_p95=round(hi,2),
                    w_knn=round(wk,3), disagree=round(disagree,3),
                    alive_frac=round(k["alive"],3), nn_dist=round(k["dmin"],3),
                    n_windows=self.n,
                    time_s=(round(self.t[-1],3) if self.t else None),
                    status=status)

    # --------------------------- 整文件/分段便捷 ---------------------------
    def run_dataframe(self, df, every=200):
        """模拟实时：按每 every 个窗增量喂入并预测，返回预测序列 DataFrame。"""
        ec,pc,tc=map_columns(df)
        if ec is None: raise ValueError("输入缺少能量列 ae_SpectralEnergy")
        E=pd.to_numeric(df[ec],errors="coerce").fillna(0).clip(lower=0).values
        P=pd.to_numeric(df[pc],errors="coerce").fillna(0).clip(lower=0).values if pc else np.zeros_like(E)
        T=pd.to_numeric(df[tc],errors="coerce").values if tc else np.arange(len(df))*self.b["window_s"]
        self.reset(); rows=[]
        for i in range(0,len(E)):
            self._append(np.array([E[i]]),np.array([P[i]])); 
            if T is not None and not np.isnan(T[i]): self.t.append(float(T[i]))
            if (i+1)%every==0 or i==len(E)-1:
                o=self.predict()
                if o and o.get("life_pct") is not None:
                    o["window_index"]=i+1; rows.append(o)
        return pd.DataFrame(rows)

# ------------------------- 结果可视化与独立存档 -------------------------
_RESULT_COLS = ["window_index","time_s","life_pct","life_pct_fixed","knn_pct","bayes_pct",
                "bayes_p05","bayes_p95","w_knn","disagree","alive_frac","nn_dist","status"]
_STATUS_COLOR = {"OK":"#2e8b57","CAPACITY_LOW":"#e69500","DISAGREE":"#d9534f",
                 "FAR_MATCH":"#c0392b","NO_NEIGHBOR":"#8b0000","WARMUP":"#888888"}
_STATUS_ADVICE = {
    "OK": "状态正常：历史库覆盖充分、两种算法判断一致，预测值可直接采用。",
    "CAPACITY_LOW": "已接近历史库的损伤极限（通常对应寿命末期 >85%）：点值偏保守，请重点看区间上界并提前安排检修。",
    "DISAGREE": "两种算法分歧较大，可能遇到没见过的新工况/新载荷：结果谨慎采用，建议人工复核，并把该件拉断后补入历史库。",
    "FAR_MATCH": "没有与当前试件接近的历史件，结果主要靠外推：谨慎采用，需补充同类工况的历史件。",
    "NO_NEIGHBOR": "损伤量已超过所有历史件，基本临近失效：点值不再给出，请立即按临近断裂处置。",
    "WARMUP": "数据窗数不足，仍在预热：继续喂入数据即可。",
}

def _status_str(v):
    """把状态（可能是 Python list 或其字符串形式）转成干净中文，多条用；连接。"""
    if isinstance(v, (list, tuple)):
        return "；".join(str(x) for x in v)
    s = str(v)
    if s.startswith("["):
        try:
            import ast
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return "；".join(str(x) for x in obj)
        except Exception:
            pass
    return s

def _x_axis(res, window_s=0.1594):
    wi = res["window_index"].astype(float).values
    if "time_s" in res.columns:
        t = pd.to_numeric(res["time_s"], errors="coerce").values
        if np.isfinite(t).any():
            t = np.where(np.isfinite(t), t, wi*window_s)
            return t, "试验时间 (秒，已采集全程)"
    return wi*window_s, "试验时间 (秒，按每窗0.1594s折算)"

def plot_life_curve(res, png_path, source=""):
    """画一张'覆盖全程、一看就懂'的图：上=寿命%随时间(纵轴0-100覆盖寿命全程)+90%区间；
    下=0-100%寿命进度条(开始→断裂)，标当前位置与区间。"""
    _setup_cn_font()
    x, xlabel = _x_axis(res)
    y = res["life_pct"].astype(float).values
    yf = res["life_pct_fixed"].astype(float).values if "life_pct_fixed" in res else y
    lo = pd.to_numeric(res["bayes_p05"], errors="coerce").ffill().astype(float).values
    hi = pd.to_numeric(res["bayes_p95"], errors="coerce").ffill().astype(float).values
    last, llo, lhi = y[-1], lo[-1], hi[-1]
    status = _status_str(res["status"].iloc[-1]); st = status.split()[0]
    color = _STATUS_COLOR.get(st, "#4472C4")

    fig = plt.figure(figsize=(11.5, 7.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.34)
    ax = fig.add_subplot(gs[0])
    ax.fill_between(x, lo, hi, color="#7fbf7f", alpha=0.28, label="90% 概率区间（真实寿命90%落在此带内）")
    ax.plot(x, yf, color="#9a9a9a", lw=1.2, ls="--", label="固定权重对照")
    ax.plot(x, y, color="#1f4e79", lw=2.4, label="预测已消耗寿命（主结果）")
    ax.scatter([x[-1]], [last], color="#d92121", s=75, zorder=6)
    ax.annotate(f"当前 {last:.1f}%", (x[-1], last), xytext=(-78, 20),
                textcoords="offset points", fontsize=12.5, fontweight="bold", color="#d92121",
                arrowprops=dict(arrowstyle="->", color="#d92121", lw=1.4))
    ax.set_ylim(0, 100); ax.set_xlim(left=0)
    ax.grid(alpha=.3); ax.set_ylabel("已消耗寿命百分比 (%)", fontsize=11.5)
    ax.set_xlabel(xlabel, fontsize=11.5)
    ax.set_title("实时寿命预测曲线（纵轴0%%=刚开始，100%%=拉断）  数据：%s" % os.path.basename(str(source)),
                 fontsize=12.5, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=.92)

    ax2 = fig.add_subplot(gs[1]); ax2.set_xlim(0, 100); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.add_patch(plt.Rectangle((0, 0.40), 100, 0.22, color="#ececec"))                        # 全程底槽
    ax2.add_patch(plt.Rectangle((llo, 0.43), max(lhi-llo, 0.4), 0.16, color="#f0ad4e", alpha=.70))  # 90%区间（细条，置底）
    ax2.add_patch(plt.Rectangle((0, 0.40), max(last, 0.4), 0.22, color=color))                 # 已走过（置顶、醒目）
    ax2.plot([last, last], [0.30, 0.76], color="black", lw=2.4)                                # 当前位置
    for gv in range(0, 101, 20):
        ax2.text(gv, 0.24, f"{gv}%", ha="center", fontsize=9.5, color="#555")
    ax2.text(0, 0.80, "开始", ha="center", fontsize=10)
    ax2.text(100, 0.80, "拉断 100%", ha="center", fontsize=10)
    ax2.text(50, 1.02, f"当前已消耗寿命 {last:.1f}%　|　90%区间 {llo:.1f}% ~ {lhi:.1f}%　|　状态：{status}",
             ha="center", fontsize=11, fontweight="bold", color=color)
    ax2.text(50, 0.10, "绿色=已走过的寿命　黑竖线=当前预测位置　黑竖线右侧的橙色=寿命可能已到达的上界（90%）",
             ha="center", fontsize=8.8, color="#777")
    fig.savefig(png_path, dpi=150, bbox_inches="tight"); plt.close(fig)

def _summary_txt(res, source, every, gate, window_s=0.1594):
    last = res.iloc[-1]
    x, _ = _x_axis(res, window_s); span_s = float(x[-1])
    span = f"{span_s/60:.1f} 分钟（{span_s:.0f} 秒）" if span_s >= 120 else f"{span_s:.1f} 秒"
    status = _status_str(last["status"]); st = status.split()[0]
    nwin = int(last["window_index"])
    lines = [
        "="*56, "        飞行器结构件 · 实时寿命预测 结果摘要", "="*56, "",
        f"数据文件：{os.path.basename(str(source))}",
        f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"融合方式：{'动态权重 ridge' if gate=='ridge' else '固定权重 fixed'}；每 {every} 个时间窗输出一个点",
        f"已处理时间窗：{nwin} 个（约 {span}，每窗 {window_s} 秒）",
        f"本次共输出预测点：{len(res)} 个", "",
        "-"*56, "  当前（最新时刻）预测", "-"*56,
        f"  ★ 已消耗寿命：{float(last['life_pct']):.1f} %",
        f"    固定权重对照：{float(last['life_pct_fixed']):.1f} %",
        f"    KNN 相似匹配：{float(last['knn_pct']):.1f} %   贝叶斯容量反解：{float(last['bayes_pct']):.1f} %",
        f"    90% 概率区间：{float(last['bayes_p05']):.1f} %  ~  {float(last['bayes_p95']):.1f} %",
        f"    状态：{status}", "",
        "-"*56, "  怎么看图（预测曲线.png）", "-"*56,
        "  · 上图纵轴 0%=刚开始、100%=拉断；深蓝粗线=预测寿命随试验时间的变化，",
        "    绿色阴影带=90%概率区间（越往后越窄），红色圆点=当前位置。",
        "  · 下图是 0→100% 的寿命进度条：绿色=已走过，橙色=当前的90%不确定范围，",
        "    黑色竖线=当前预测位置。", "",
        "-"*56, "  判读建议", "-"*56,
        "  " + _STATUS_ADVICE.get(st, "请结合曲线与区间综合判断。"),
        "", "  完整逐点数值见同目录 [预测结果.csv]（可用 Excel 打开）。", "="*56, "",
    ]
    return "\n".join(lines)

def save_run(res, source, every, gate, root=None, window_s=0.1594):
    """把本次预测【单独】存到 预测结果/<数据名>_<时间戳>/ ，含 CSV + 全程曲线图 + 摘要。"""
    if res is None or len(res) == 0:
        return None
    res = res[[c for c in _RESULT_COLS if c in res.columns]].copy()
    if "status" in res.columns:
        res["status"] = res["status"].map(_status_str)
    here = os.path.dirname(os.path.abspath(__file__))
    root = root or os.path.join(here, "预测结果")
    base = os.path.splitext(os.path.basename(str(source)))[0] if source not in (None, "<stream>") else "实时流数据"
    folder = os.path.join(root, f"{base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(folder, exist_ok=True)
    csvp = os.path.join(folder, "预测结果.csv")
    pngp = os.path.join(folder, "预测曲线.png")
    txtp = os.path.join(folder, "结果摘要.txt")
    res.to_csv(csvp, index=False, encoding="utf-8-sig")
    plot_life_curve(res, pngp, source)
    with open(txtp, "w", encoding="utf-8") as f:
        f.write(_summary_txt(res, source, every, gate, window_s))
    return folder

# --------------------------------- CLI ---------------------------------
def main():
    ap=argparse.ArgumentParser(description="飞行器结构件实时寿命百分比预测（每次结果单独存档并自动出全程曲线图）")
    ap.add_argument("--input",help="输入窗级 CSV 文件（每行一个时间窗）")
    ap.add_argument("--model",default=os.path.join(os.path.dirname(os.path.abspath(__file__)),"model_bundle_final.pkl"))
    ap.add_argument("--every",type=int,default=200,help="每 N 个窗输出一次预测（默认200）")
    ap.add_argument("--out",help="可选：自定义结果 CSV 路径；不填则自动在“预测结果”文件夹下按时间戳单独存档（含曲线图+摘要）")
    ap.add_argument("--gate",default="ridge",choices=["ridge","fixed"],help="ridge=动态权重(默认) fixed=固定0.7")
    ap.add_argument("--stream",action="store_true",help="从标准输入逐行读取（实时流）")
    args=ap.parse_args()
    mdl=LiveLifePredictor(args.model,gate=args.gate)
    window_s=mdl.b.get("window_s",0.1594)

    if args.stream:
        import csv as _csv
        rdr=_csv.DictReader(sys.stdin); rows=[]; idx=0
        for row in rdr:
            mdl.update(row); idx+=1
            if idx%args.every==0:
                o=mdl.predict()
                if o:
                    if o.get("life_pct") is not None: o["window_index"]=idx; rows.append(o)
                    print(json.dumps(o,ensure_ascii=False),flush=True)
        o=mdl.predict()
        if o and o.get("life_pct") is not None:
            o["window_index"]=idx; rows.append(o); print(json.dumps(o,ensure_ascii=False),flush=True)
        if rows:
            folder=save_run(pd.DataFrame(rows),"<stream>",args.every,args.gate,window_s=window_s)
            print(f"\n[存档] 本次实时流结果（CSV+全程曲线图+摘要）已单独保存到：\n  {folder}",file=sys.stderr)
        return
    if not args.input:
        ap.error("请用 --input 指定 CSV，或用 --stream 从标准输入读取")
    df=pd.read_csv(args.input)
    print(f"读取 {len(df)} 个时间窗；能量/峰值列自动映射；每 {args.every} 窗输出一次...",file=sys.stderr)
    res=mdl.run_dataframe(df,every=args.every)
    if len(res)==0:
        print("数据窗数不足（少于预热所需的30窗），未能输出预测，请增加数据。",file=sys.stderr); return
    res=res[[c for c in _RESULT_COLS if c in res.columns]]
    print(res.tail(12).to_string(index=False))
    if args.out:
        res.to_csv(args.out,index=False,encoding="utf-8-sig")
        png=os.path.splitext(args.out)[0]+"_曲线.png"
        plot_life_curve(res,png,args.input)
        print(f"\n[存档] 结果CSV：{args.out}\n[存档] 全程曲线图：{png}",file=sys.stderr)
    else:
        folder=save_run(res,args.input,args.every,args.gate,window_s=window_s)
        print(f"\n[完成] 本次结果已【单独】保存（含 预测结果.csv / 预测曲线.png / 结果摘要.txt）到：\n  {folder}",file=sys.stderr)

if __name__=="__main__":
    main()
