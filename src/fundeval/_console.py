"""命令行输出的编码兜底。

中文 Windows 的控制台被管道或重定向时，标准输出与标准错误的编码通常是 GBK（cp936），
报告与提示中的“²”“−”“½”等字符在 GBK 中没有，直接 print 会抛出 UnicodeEncodeError。
这里先按 CONSOLE_FALLBACK 换成 ASCII 写法（M² → M^2，− → -），仍无法编码的字符换成“?”，
保证命令不因输出编码中断。能编码时原样输出；写入文件（--out）一律 UTF-8，不经过这里。
"""

from __future__ import annotations

import sys

#: 目标编码无法表示时的替换写法
CONSOLE_FALLBACK = str.maketrans({
    "²": "^2",
    "½": "1/2",
    "−": "-",
    "⌈": "ceil(",
    "⌉": ")",
    "⌊": "floor(",
    "⌋": ")",
})


def console_safe(text: str, stream=None) -> str:
    """返回 ``stream`` 的编码能够写出的文本：能编码时原样返回，否则先换成 ASCII 写法，再把其余字符换成“?”。"""
    encoding = getattr(stream if stream is not None else sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        pass
    except LookupError:  # 未知编码名：按原文写出，由流自己处理
        return text
    return text.translate(CONSOLE_FALLBACK).encode(encoding, errors="replace").decode(encoding)


def console_write(text: str, stream=None) -> None:
    """按 console_safe 写出文本（不自动换行）。"""
    stream = stream if stream is not None else sys.stdout
    stream.write(console_safe(text, stream))


def console_print(*parts, sep: str = " ", end: str = "\n", file=None) -> None:
    """与 print 相同的用法，但输出前按 console_safe 处理编码。"""
    stream = file if file is not None else sys.stdout
    console_write(sep.join(str(p) for p in parts) + end, stream)
