# -*- coding: utf-8 -*-
"""
后端：Flask 简单计算服务
安装依赖：
    pip install flask

运行：
    python main.py
访问：
    http://127.0.0.1:5000
"""
from __future__ import annotations
import math
from typing import List, Union, Tuple
from flask import Flask, request, jsonify, send_from_directory
import os
app = Flask(__name__, static_folder='.', static_url_path='')


@app.route('/')
def index():
    # 直接从当前目录提供前端文件
    return send_from_directory('.', 'index.html')


@app.post('/api/calculate')
def api_calculate():
    """
    接收 JSON: { "expression": "12+3.5×4" }
    返回:
      成功: { "result": "26" }
      失败: { "error": "错误提示" }
    """
    data = request.get_json(silent=True) or {}
    expr = str(data.get('expression', '')).strip()

    if not expr:
        return jsonify(error="表达式不能为空")

    try:
        result = evaluate_expression(expr)
    except CalcError as e:
        return jsonify(error=str(e))
    except Exception:
        # 兜底异常
        return jsonify(error="无效的表达式")

    # 结果格式化为更友好的字符串
    return jsonify(result=format_number(result))


class CalcError(Exception):
    """计算相关的可预期错误"""
    pass


Number = float
Token = Union[str, Number]


def evaluate_expression(expr: str) -> float:
    """
    解析并计算仅含 + - × ÷ * / 和小数的表达式。
    处理优先级，支持负数（例如：-3+5、5*-2）。
    """
    # 1) 预处理：去空格、替换特殊运算符
    s = expr.replace(' ', '').replace('×', '*').replace('÷', '/')

    # 2) 基本字符检查
    allowed = set('0123456789.+-*/')
    if not s or any(ch not in allowed for ch in s):
        raise CalcError("表达式包含不支持的字符")

    # 3) 词法分析（含一元负号）
    tokens = tokenize(s)
    if not tokens:
        raise CalcError("无效的表达式")

    # 4) 转后缀表达式（逆波兰）
    rpn = to_rpn(tokens)

    # 5) 计算 RPN
    return eval_rpn(rpn)


def tokenize(s: str) -> List[Token]:
    tokens: List[Token] = []
    i, n = 0, len(s)

    while i < n:
        ch = s[i]

        if ch in '+*/':
            # 二元运算符：前一个必须是数字
            if not tokens or isinstance(tokens[-1], str):
                raise CalcError("表达式不能以运算符开头或出现连续运算符")
            tokens.append(ch)
            i += 1
            continue

        if ch == '-':
            # 可能是一元负号
            if not tokens or isinstance(tokens[-1], str):
                # 作为负号，后面必须能解析出数字
                num, j = parse_number(s, i + 1, negative=True)
                tokens.append(num)
                i = j
            else:
                # 作为二元减号
                tokens.append('-')
                i += 1
            continue

        if ch.isdigit() or ch == '.':
            num, j = parse_number(s, i, negative=False)
            tokens.append(num)
            i = j
            continue

        # 不应到达此处（已做字符过滤）
        raise CalcError("无效的表达式")

    # 结束时不应以运算符结尾
    if tokens and isinstance(tokens[-1], str):
        raise CalcError("表达式不能以运算符结尾")

    return tokens


def parse_number(s: str, start: int, negative: bool) -> Tuple[float, int]:
    """
    从 s[start:] 解析一个浮点数，返回 (值, 结束位置索引)
    要求：数字格式类似 [digits][.digits] 或 [.digits] 或 [digits.]
    至少包含一个数字。negative=True 时在数值前加负号。
    """
    i, n = start, len(s)
    dot = 0
    digits = 0

    # 允许以 '.' 开头
    while i < n and (s[i].isdigit() or s[i] == '.'):
        if s[i] == '.':
            dot += 1
            if dot > 1:
                raise CalcError("数字中包含多个小数点")
        else:
            digits += 1
        i += 1

    # 允许形如 "12."（digits>0, dot==1），此时视为整数形式的小数
    # 允许形如 ".5"（digits>0, dot==1）
    if digits == 0:
        raise CalcError("小数点位置不正确")

    num_str = s[start:i]
    try:
        val = float(num_str)
    except ValueError:
        raise CalcError("数字格式无效")

    if negative:
        val = -val
    return val, i


def to_rpn(tokens: List[Token]) -> List[Token]:
    """Shunting-yard 算法：仅 + - * /，左结合"""
    out: List[Token] = []
    ops: List[str] = []
    prec = {'+': 1, '-': 1, '*': 2, '/': 2}

    for t in tokens:
        if isinstance(t, (int, float)):
            out.append(float(t))
        else:
            # 运算符
            while ops and prec[ops[-1]] >= prec[t]:
                out.append(ops.pop())
            ops.append(t)

    while ops:
        out.append(ops.pop())

    return out


def eval_rpn(rpn: List[Token]) -> float:
    st: List[float] = []
    for t in rpn:
        if isinstance(t, (int, float)):
            st.append(float(t))
            continue
        # 运算符
        if len(st) < 2:
            raise CalcError("无效的表达式")
        b = st.pop()
        a = st.pop()
        if t == '+':
            st.append(a + b)
        elif t == '-':
            st.append(a - b)
        elif t == '*':
            st.append(a * b)
        elif t == '/':
            if abs(b) < 1e-12:
                raise CalcError("除数不能为 0")
            st.append(a / b)
        else:
            raise CalcError("未知运算符")
    if len(st) != 1:
        raise CalcError("无效的表达式")
    return st[0]


def format_number(x: float) -> str:
    """结果格式化：整数不带小数，浮点保留合理精度并去除尾零"""
    if not math.isfinite(x):
        raise CalcError("结果超出范围")
    # 近似整数
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    # 保留 12 位小数，去掉多余 0
    s = f"{x:.12f}".rstrip('0').rstrip('.')
    # 避免 "-0"
    if s in ("-0", "-0.0", "0.0"):
        s = "0"
    return s


if __name__ == '__main__':
    # 移除硬编码的 host 和 port，改用环境变量
    port = int(os.environ.get('PORT', 5000))  # Render 会自动分配 PORT
    app.run(host='0.0.0.0', port=port, debug=False)  # 关闭 debug 模式
