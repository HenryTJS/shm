# -*- coding: utf-8 -*-
"""看板数据包的二进制编码（base64 + Int16/Int32/Float32）。

为什么
------
`dashboard/data/*.js` 里数值数组原本写成 JSON 十进制文本（平均约 6 字节/值），
占到单个包的 80% 以上。改成小端定宽二进制再 base64（1.33 倍膨胀）后：

    Int16 → 2.67 B/值（原约 6 B）   Int32/Float32 → 5.33 B/值

前端在 `dashboard.js` 的 `hydrateAll()` 里解码，字段名不变，旧包（纯数组）仍兼容。

编码格式（单个数组）
-------------------
    {"i16": "<base64>"}   小端 int16 序列
    {"i32": "<base64>"}   小端 int32 序列
    {"f32": "<base64>"}   小端 float32 序列（仅用于浮点数组，如时间/坐标）

约定：**不编码** `meta` / `warn` / `labels` / `chans` / `spec`（小且需要人读），
其余任意层级的"纯数字列表"都会被编码（含 `fo.*`、`dfos.*`、`indics[].series`）。
"""
from __future__ import annotations

import base64

import numpy as np

SKIP_KEYS = {"meta", "warn", "labels", "chans", "spec"}
I16_MIN, I16_MAX = -32768, 32767


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def enc_array(seq):
    """数值列表 → {"i16"/"i32"/"f32": base64}；不能编码时原样返回。"""
    a = np.asarray(seq)
    if a.size == 0 or a.dtype.kind not in "iuf":
        return seq
    if a.dtype.kind in "iu":
        lo, hi = int(a.min()), int(a.max())
        if I16_MIN <= lo and hi <= I16_MAX:
            return {"i16": base64.b64encode(a.astype("<i2").tobytes()).decode("ascii")}
        return {"i32": base64.b64encode(a.astype("<i4").tobytes()).decode("ascii")}
    return {"f32": base64.b64encode(a.astype("<f4").tobytes()).decode("ascii")}


def _walk(v):
    if isinstance(v, list):
        if v and all(_is_num(x) for x in v):
            return enc_array(v)
        return [_walk(x) for x in v]
    if isinstance(v, dict):
        return {k: _walk(x) for k, x in v.items()}
    return v


def encode_package(obj):
    """就地编码数据包（返回同一个对象）。"""
    for k in list(obj.keys()):
        if k in SKIP_KEYS:
            continue
        obj[k] = _walk(obj[k])
    return obj


def decode_package(obj):
    """反向解码（自检/对比用，前端不需要）。"""
    def _dec(v):
        if isinstance(v, dict):
            if set(v.keys()) == {"i16"}:
                return np.frombuffer(base64.b64decode(v["i16"]), dtype="<i2").astype(int).tolist()
            if set(v.keys()) == {"i32"}:
                return np.frombuffer(base64.b64decode(v["i32"]), dtype="<i4").astype(int).tolist()
            if set(v.keys()) == {"f32"}:
                return np.frombuffer(base64.b64decode(v["f32"]), dtype="<f4").astype(float).tolist()
            return {k: _dec(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_dec(x) for x in v]
        return v

    for k in list(obj.keys()):
        if k in SKIP_KEYS:
            continue
        obj[k] = _dec(obj[k])
    return obj
