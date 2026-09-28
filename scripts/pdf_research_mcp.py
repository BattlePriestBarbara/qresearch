#!/usr/bin/env python
"""
pdf_research_mcp.py
===================

面向「数据与统计 / 神经网络」科研场景的 PDF 解析 MCP Server（Model Context Protocol）。

暴露 3 个 Tool：
    1. extract_pdf_text    —— PyMuPDF 纯文本提取，支持页码区间（start_page / end_page）。
    2. extract_pdf_tables  —— pdfplumber 表格提取，输出行列对齐的 Markdown / CSV，
                              对合并单元格与空值做确定性填充，便于直接喂给 pandas / statsmodels。
    3. extract_pdf_figures —— PyMuPDF 提取内嵌位图（如神经网络架构图、实验结果图），
                              保存到本地并返回绝对路径。

工程约束（与 Qlib 仓库既有规范一致：line-length 120 / black 风格 / 详细中文注释）：
    * SDK：官方 mcp Python SDK 2.x（本环境 2.2.0）。2.x 采用「构造器注册 handler」装配
      （Server(name, on_list_tools=..., on_call_tool=...)），1.x 的 @server.call_tool()
      装饰器已移除。
    * 通信：asyncio + mcp.server.stdio 的 stdio 通道；stdout 只允许承载协议消息，
      所有日志走 stderr，避免污染 JSON-RPC 报文。
    * 返回：全部 Tool 一律返回含 TextContent 的 CallToolResult；任何异常都会被捕获并转换成
      可读的中文错误文本（isError=true），绝不让未捕获异常穿透导致 Server 退出。
    * 阻塞隔离：PyMuPDF / pdfplumber 都是同步阻塞库，统一用 asyncio.to_thread 丢到
      默认线程池执行，防止长 PDF 阻塞 MCP 事件循环（否则客户端会误判超时）。

运行方式（由 MCP 客户端拉起，一般不需要手动运行；下例路径按本机仓库位置填写）：
    <mcp-venv-python>  <repo>/scripts/pdf_research_mcp.py
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

# PyMuPDF：1.24+ 的官方导入名是 pymupdf，fitz 只是被弃用的历史别名（导入时会打印弃用告警）。
# 这里优先用 pymupdf，并对旧安装回退到 fitz，保证脚本在新旧环境都能跑。
try:
    import pymupdf
except ImportError:  # pragma: no cover - 仅用于兼容 PyMuPDF < 1.24
    import fitz as pymupdf

import pdfplumber
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, TextContent, Tool

# =====================================================================================
# 全局常量
# =====================================================================================
SERVER_NAME = "pdf-research-mcp"
SERVER_VERSION = "1.0.0"

#: 未显式指定 output_dir 时使用的子目录名（相对当前工作目录）
DEFAULT_IMAGE_DIRNAME = "pdf_images"

#: 小图判定阈值：任一边小于该像素值时，标注为“可能是公式/图标/页眉装饰”，但仍会保存
SMALL_FIGURE_EDGE_PX = 100

#: 图片默认输出目录的环境变量名，可在 MCP 配置的 env 中指定，使默认路径可预测
ENV_IMAGES_DIR = "PDF_MCP_IMAGES_DIR"

#: 表格提取：优先用“线框”策略（论文统计表通常有完整边框，列对齐最可靠）
TABLE_SETTINGS_LINES: dict[str, Any] = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "intersection_tolerance": 5,
}

#: 线框策略找不到表格时的兜底：“文本对齐”策略（无边框三线表常见）
TABLE_SETTINGS_TEXT: dict[str, Any] = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "text_x_tolerance": 3,
    "text_y_tolerance": 3,
}

#: 空值填充标记。用 "NaN" 而非空串：pandas.read_csv 会直接把 "NaN" 识别为浮点 NaN，
#: 统计建模时不会被误当成 0 或类别标签。
NAN_TOKEN = "NaN"

#: 各版本号仅在结果头部展示，便于复现实验环境
try:  # pragma: no cover - 纯信息性代码，不参与逻辑分支
    PYMUPDF_VERSION = str(getattr(pymupdf, "__version__", None) or getattr(pymupdf, "VersionBind", "unknown"))
except Exception:  # 版本号获取失败绝不能影响主流程
    PYMUPDF_VERSION = "unknown"
PDFPLUMBER_VERSION = str(getattr(pdfplumber, "__version__", "unknown"))


class PdfToolError(Exception):
    """工具层可预期的错误（文件不存在、加密、页码越界等），会被转换成友好中文文本返回。

    额外携带 hint，用于在错误结果里给出「下一步怎么做」的建议。
    注意：Python 的 Exception 默认不接受关键字参数，必须显式定义 __init__，
    否则 `PdfToolError("...", hint="...")` 会抛出 TypeError 而不是预期的业务错误。
    """

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


# =====================================================================================
# 基础工具函数
# =====================================================================================
def _log(message: str) -> None:
    """把诊断信息写到 stderr；绝不写 stdout（stdout 专供 MCP 协议报文）。"""
    print(f"[{SERVER_NAME}] {message}", file=sys.stderr, flush=True)


def _ok(text: str) -> CallToolResult:
    """成功结果：包装成 CallToolResult，内容块类型固定为 TextContent。"""
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=False)


def _fail(title: str, detail: str = "", hint: str = "") -> CallToolResult:
    """失败结果：仍然是合法的 CallToolResult（isError=true + 清晰中文说明），而不是抛异常。

    这样客户端能拿到完整错误文本并据此自我修正，同时 Server 进程不会因为一次失败调用退出。
    """
    lines = [f"❌ {title}"]
    if detail:
        lines.append(f"原因：{detail}")
    if hint:
        lines.append(f"建议：{hint}")
    return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))], is_error=True)


def _resolve_pdf(file_path: Any) -> Path:
    """校验并返回 PDF 的绝对路径。

    失败情形都抛出 PdfToolError（携带中文可读信息）：
        - 空参数 / 路径不存在 / 指向目录 / 空文件（下载未完成）；非 .pdf 后缀仅警告。
    相对路径按 Server 进程的当前工作目录解析。
    """
    if file_path is None or str(file_path).strip() == "":
        raise PdfToolError("参数 file_path 为空，请传入 PDF 文件的绝对路径（相对路径按 Server 当前工作目录解析）。")

    raw = str(file_path).strip().strip('"').strip("'")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        # 相对路径兜底：按 Server 进程 cwd 解析，便于用户传 "papers/xxx.pdf"
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        raise PdfToolError(f"文件不存在：{path}（请检查盘符、路径拼写，以及网络盘/云盘是否已挂载）")
    if path.is_dir():
        raise PdfToolError(f"路径指向的是文件夹而不是 PDF 文件：{path}")
    try:
        size = path.stat().st_size
    except OSError as exc:  # 权限不足或文件被其他进程独占
        raise PdfToolError(f"无法读取文件属性（可能权限不足或文件被独占）：{path}（{exc}）") from exc
    if size == 0:
        raise PdfToolError(f"文件大小为 0 字节，通常是下载未完成或已损坏：{path}")
    if path.suffix.lower() != ".pdf":
        # 只提醒不阻断：部分文献库导出的文件扩展名不规范，PyMuPDF 仍可能正常解析
        _log(f"警告：{path.name} 的扩展名不是 .pdf，仍将按 PDF 尝试解析。")
    return path


def _open_pdf(path: Path) -> pymupdf.Document:
    """打开 PDF 并统一处理损坏 / 加密 / 权限异常。调用方负责 close()。"""
    try:
        doc = pymupdf.open(path)
    except pymupdf.FileNotFoundError as exc:
        raise PdfToolError(f"PyMuPDF 找不到文件：{path}") from exc
    except pymupdf.FileDataError as exc:
        raise PdfToolError(f"文件不是有效的 PDF 或已损坏：{path}（底层报错：{exc}）") from exc
    except PermissionError as exc:
        raise PdfToolError(f"没有读取权限，文件可能被占用或受系统保护：{path}（{exc}）") from exc
    except Exception as exc:  # 兜底：任何打开失败都要转成可读文本
        raise PdfToolError(f"打开 PDF 失败：{type(exc).__name__}: {exc}") from exc

    # 加密保护处理：needs_pass=True 表示需要口令；先试空口令（部分 PDF 只设了权限口令）
    if getattr(doc, "needs_pass", False):
        try:
            opened = doc.authenticate("")
        except Exception:  # 某些实现下空口令会直接抛错，视作失败
            opened = 0
        if not opened:
            doc.close()
            raise PdfToolError(
                f"该 PDF 受密码保护，无法解析：{path}",
                hint="请先解除加密（例如 qpdf --decrypt in.pdf out.pdf 或用阅读器另存为），再传入解密后的文件。",
            )
    return doc


def _pdf_meta(doc: pymupdf.Document) -> dict[str, Any]:
    """抽取用于结果头部的元数据（标题/作者/页数/是否加密）。"""
    meta = doc.metadata or {}
    return {
        "title": str(meta.get("title") or "").strip(),
        "author": str(meta.get("author") or "").strip(),
        "page_count": int(doc.page_count),
        "encrypted": bool(getattr(doc, "needs_pass", False)),
    }


def _as_optional_int(value: Any, name: str) -> int | None:
    """把选填的页码参数转成 int 或 None；类型不合法时报出可读错误。"""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PdfToolError(f"参数 {name} 需要整数页码（如 3），实际收到：{value!r}") from exc


def _resolve_page_range(start_page: int | None, end_page: int | None, total_pages: int) -> tuple[int, int, str]:
    """把 1 起算、可缺省的页码规范化成合法的闭区间 [start, end]。

    返回 (start, end, 备注)。页码越界属于用户输入错误，直接报错；
    end_page 超过总页数时按“读到末页”收敛并在备注里说明，避免因一次越界而整体失败。
    """
    start = 1 if start_page is None else start_page
    end = total_pages if end_page is None else end_page
    note = ""

    if start < 1:
        raise PdfToolError(f"start_page 必须 >= 1（页码从 1 开始编号），实际收到 {start}。")
    if end < 1:
        raise PdfToolError(f"end_page 必须 >= 1（页码从 1 开始编号），实际收到 {end}。")
    if start > total_pages:
        raise PdfToolError(
            f"start_page={start} 超出文档范围：该 PDF 共 {total_pages} 页。",
            hint=f"请把 start_page 调整到 1 ~ {total_pages} 之间。",
        )
    if end > total_pages:
        note = f"（end_page={end} 超出总页数，已自动收敛到末页 {total_pages}）"
        end = total_pages
    if end < start:
        raise PdfToolError(f"end_page({end}) 必须大于等于 start_page({start})。")
    return start, end, note


def _tidy_text(text: str) -> str:
    """整理 PDF 抽出的文本：统一换行、去掉行尾空白、压缩连续空行。

    PDF 的硬换行会把一句话拆成多行，并且经常残留大量空白行；
    压缩空行有助于后续按段落切分（例如抽取方法章节做文本统计）。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text)


def _human_size(num_bytes: int) -> str:
    """字节数转人类可读字符串（图片体积展示用）。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _safe_stem(name: str) -> str:
    """把文件名主干清洗成 Windows 合法字符，用于生成图片文件名。"""
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", name).strip("_")
    return cleaned or "document"


# =====================================================================================
# Tool 1：extract_pdf_text —— PyMuPDF 纯文本提取
# =====================================================================================
def _extract_text_sync(path: Path, start_page: int | None, end_page: int | None) -> str:
    """同步实现：按页抽取纯文本，并返回带页码标记与统计摘要的报告文本。

    为什么用 PyMuPDF 而不是 pdfplumber：PyMuPDF 的文本抽取速度通常快一个数量级，
    且对多栏排版、连字符换行的还原更稳定，适合「先整体通读论文」的场景。
    """
    doc = _open_pdf(path)
    try:
        meta = _pdf_meta(doc)
        total = meta["page_count"]
        if total == 0:
            return f"⚠️ 该 PDF 不含任何页面（page_count=0），无法提取文本：{path}"

        start, end, range_note = _resolve_page_range(start_page, end_page, total)

        blocks: list[str] = []
        empty_pages: list[int] = []
        total_chars = 0
        # PyMuPDF 的页码是 0 起算，用户参数是 1 起算，这里统一做偏移转换
        for page_index in range(start - 1, end):
            page = doc.load_page(page_index)
            raw = page.get_text("text") or ""  # "text" 模式 = 按阅读顺序输出的纯文本
            text = _tidy_text(raw)
            total_chars += len(text)
            if not text.strip():
                empty_pages.append(page_index + 1)
                body = "（本页未提取到可复制文本：多为扫描件/纯图片页面，需要 OCR 才能取到文字）"
            else:
                body = text
            blocks.append(f"===== 第 {page_index + 1} 页 / 共 {total} 页 =====")
            blocks.append(body)

        header = [
            "【PDF 文本提取结果 · extract_pdf_text】",
            f"文件：{path}",
            f"文件大小：{_human_size(path.stat().st_size)}",
            f"总页数：{total} 页；本次提取：第 {start} ~ {end} 页{range_note}",
            f"正文总字符数：{total_chars}",
            f"引擎：PyMuPDF {PYMUPDF_VERSION}",
        ]
        if meta["title"]:
            header.append(f"标题（元数据）：{meta['title']}")
        if meta["author"]:
            header.append(f"作者（元数据）：{meta['author']}")
        if empty_pages:
            # 科研文献里扫描页很常见，明确提示可避免把“空页”误判成解析失败
            listed = "、".join(str(p) for p in empty_pages[:20])
            more = "" if len(empty_pages) <= 20 else f" 等共 {len(empty_pages)} 页"
            header.append(f"⚠️ 未提取到文字的第 {listed} 页{more}：疑似扫描件，如需文字请先做 OCR。")

        return "\n".join(header) + "\n\n--- 正文开始 ---\n" + "\n\n".join(blocks) + "\n--- 正文结束 ---"
    finally:
        doc.close()


# =====================================================================================
# Tool 2：extract_pdf_tables —— pdfplumber 表格提取（Markdown / CSV）
# =====================================================================================
def _normalize_matrix(table: Any) -> list[list[str]]:
    """把 pdfplumber 的 Table 对象规范化为「严格矩形」的字符串矩阵。

    这是统计表格能否被正确消费的关键，做三件事：

    1. 行补齐（列对齐）：跨列/跨行合并会让不同行的单元格个数不一致。统一补齐到最大列数，
       保证矩阵是矩形，从而 Markdown 表格的列与列严格对齐、CSV 的字段数恒定。
    2. 合并单元格填充：pdfplumber 用 ``None`` 表示该位置没有独立单元格
       （``table.rows[i].cells[j] is None``）。此时优先沿用左侧单元格的值（水平合并），
       否则沿用上一行同列的值（垂直合并），使每一行都是完整、可建模的记录。
    3. 真实空值：存在独立单元格但内容为空 —— 填 ``NaN``。这样 pandas 读取后直接是
       浮点 NaN，不会在统计推断里被误当成 0 或某个类别标签。

    单元格内部的换行/多空格会被压成单个空格：否则会破坏 Markdown 的行结构。
    """
    try:
        raw_rows = table.extract() or []
    except Exception as exc:  # 单个表格解析失败不应中断整篇文档
        _log(f"表格 extract() 失败：{type(exc).__name__}: {exc}")
        return []
    if not raw_rows:
        return []

    ncols = max(len(r) for r in raw_rows)

    # 单元格 bbox 存在性：False = 该位置被合并单元格占用（没有独立单元格边界）
    try:
        bbox_flags = [[cell is not None for cell in row.cells] for row in table.rows]
    except Exception as exc:  # 拿不到 bbox 时退化为纯空值填充
        _log(f"读取单元格 bbox 失败，将退化为空值填充：{type(exc).__name__}: {exc}")
        bbox_flags = []

    matrix: list[list[str]] = []
    for row_index, raw in enumerate(raw_rows):
        padded = list(raw) + [None] * (ncols - len(raw))  # 行补齐，保证矩形
        out: list[str] = []
        for col_index in range(ncols):
            value = padded[col_index]
            cell_text = "" if value is None else re.sub(r"\s+", " ", str(value)).strip()
            merged = (
                bool(bbox_flags)
                and row_index < len(bbox_flags)
                and col_index < len(bbox_flags[row_index])
                and not bbox_flags[row_index][col_index]
            )
            if cell_text == "":
                if merged and col_index > 0 and out[col_index - 1] not in ("", NAN_TOKEN):
                    cell_text = out[col_index - 1]  # 水平合并：沿用左侧值
                elif (
                    merged
                    and matrix
                    and col_index < len(matrix[row_index - 1])
                    and matrix[row_index - 1][col_index] not in ("", NAN_TOKEN)
                ):
                    cell_text = matrix[row_index - 1][col_index]  # 垂直合并：沿用上方值
                else:
                    cell_text = NAN_TOKEN  # 真实空值
            out.append(cell_text)
        matrix.append(out)
    return matrix


def _matrix_to_markdown(matrix: list[list[str]]) -> str:
    """矩阵转 Markdown 表格（第一行作为表头；列数固定，保证左右对齐可读）。"""
    ncols = max(len(r) for r in matrix)
    rows = [list(r) + [NAN_TOKEN] * (ncols - len(r)) for r in matrix]
    header, body = rows[0], rows[1:]

    def escape(cell_text: str) -> str:
        # '|' 是 Markdown 的列分隔符，必须转义，否则表格结构会被破坏
        return cell_text.replace("|", "\\|")

    lines = [
        "| " + " | ".join(escape(c) for c in header) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    lines.extend("| " + " | ".join(escape(c) for c in row) + " |" for row in body)
    return "\n".join(lines)


def _matrix_to_csv(matrix: list[list[str]]) -> str:
    """矩阵转 CSV 字符串。

    统一使用 ``\\n`` 换行、标准双引号转义，可以直接
    ``pandas.read_csv(io.StringIO(text))`` 转成 DataFrame 做统计推断。
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    ncols = max(len(r) for r in matrix)
    for row in matrix:
        writer.writerow(list(row) + [NAN_TOKEN] * (ncols - len(row)))
    return buffer.getvalue().rstrip("\n")


def _is_meaningful_table(matrix: list[list[str]]) -> bool:
    """过滤退化的「伪表格」。

    典型伪表格来自文本对齐策略：它会把一栏上下相邻的正文行误判成「N 行 × 1 列」的表格。
    统计表格至少要有 2 行 × 2 列，且有效单元格（非 NaN）不少于 3 个；否则一律丢弃，
    避免把正文段落当表格喂给下游分析。
    """
    rows = len(matrix)
    cols = max((len(r) for r in matrix), default=0)
    if rows < 2 or cols < 2:
        return False
    filled = sum(1 for row in matrix for cell in row if cell and cell != NAN_TOKEN)
    return filled >= 3


def _extract_tables_sync(path: Path, output_format: str) -> str:
    """同步实现：逐页检出表格并转成 Markdown / CSV。

    策略选择（论文表格的两大流派）：
        * “线框”（lines）策略：依赖表格边框线定位行列 —— 有完整边框的统计表首选，列切分最准。
        * “文本对齐”（text）策略：依赖字符间距聚类 —— 用于无边框的“三线表”。
    先试线框，只有在当页完全没有检出表格时才退化到文本对齐，避免同一张表被重复输出。
    """
    fmt = (output_format or "markdown").strip().lower()
    if fmt not in {"markdown", "csv", "both"}:
        raise PdfToolError(f"output_format 只支持 markdown / csv / both，实际收到：{output_format!r}")

    # 先用 PyMuPDF 做一次轻量校验：pdfplumber 对「加密 / 损坏 / 无权限」的底层报错很晦涩
    # （例如 PdfminerException: No /Root object!），统一交给 _open_pdf 转成可读的中文提示，
    # 保证三个 Tool 的错误口径一致。
    probe = _open_pdf(path)
    try:
        page_total = int(probe.page_count)
    finally:
        probe.close()

    collected: list[tuple[int, int, str, list[list[str]]]] = []
    rejected = 0  # 被过滤掉的退化伪表格数量（会在结果里说明，便于用户判断）
    try:
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages) != page_total:
                _log(f"提示：PyMuPDF 报 {page_total} 页、pdfplumber 报 {len(pdf.pages)} 页，以实际解析页数为准。")
                page_total = len(pdf.pages)
            for page_index, page in enumerate(pdf.pages, start=1):
                tables: list[Any] = []
                strategy = ""
                # 依次尝试两种策略，命中即止（线框优先，结果更可靠）
                for settings, tag in ((TABLE_SETTINGS_LINES, "线框"), (TABLE_SETTINGS_TEXT, "文本对齐")):
                    try:
                        tables = page.find_tables(table_settings=settings) or []
                    except Exception as exc:  # 单页失败不影响其它页
                        _log(f"第 {page_index} 页表格检测失败（{tag} 策略）：{type(exc).__name__}: {exc}")
                        tables = []
                    if tables:
                        strategy = tag
                        break
                for table_index, table in enumerate(tables, start=1):
                    matrix = _normalize_matrix(table)
                    if matrix and _is_meaningful_table(matrix):
                        collected.append((page_index, table_index, strategy, matrix))
                    elif matrix:
                        rejected += 1
                        _log(f"第 {page_index} 页第 {table_index} 个候选被判为伪表格，已丢弃。")
    except PdfToolError:
        raise
    except Exception as exc:
        raise PdfToolError(
            f"pdfplumber 解析失败：{type(exc).__name__}: {exc}",
            hint="若为加密文件请先解密；若文件损坏请重新下载；也可能是页面对象过于复杂，可先用 extract_pdf_text 确认文件可读。",
        ) from exc

    header = [
        "【PDF 表格提取结果 · extract_pdf_tables】",
        f"文件：{path}",
        f"总页数：{page_total} 页",
        f"检出表格：{len(collected)} 个",
        f"引擎：pdfplumber {PDFPLUMBER_VERSION}",
        f"填充约定：合并单元格 -> 沿用左/上单元格的值；真实空值 -> {NAN_TOKEN}（pandas 读取即为 NaN）",
    ]
    if rejected:
        header.append(f"（另有 {rejected} 个候选被判定为退化伪表格并已过滤，例如正文被误判成的 N 行 1 列表格）")
    if not collected:
        header.append("")
        header.append("⚠️ 未检出任何表格。常见原因与建议：")
        header.append("  1) 该 PDF 是扫描件（整幅图片），需先做 OCR；")
        header.append("  2) 表格没有边框线且字符间距不规则 —— 可先用 extract_pdf_text 看该页文本行；")
        header.append("  3) 该文档确实不含表格（可换几页确认，例如结果/实验章节）。")
        return "\n".join(header)

    parts: list[str] = ["\n".join(header)]
    for order, (page_index, table_index, strategy, matrix) in enumerate(collected, start=1):
        rows = len(matrix)
        cols = max(len(r) for r in matrix)
        title = (
            f"--- 表格 {order}/{len(collected)}：第 {page_index} 页 第 {table_index} 个表格"
            f"（策略：{strategy}），尺寸 {rows} 行 × {cols} 列 ---"
        )
        block = [title]
        if fmt in {"markdown", "both"}:
            block.append("[Markdown]")
            block.append(_matrix_to_markdown(matrix))
        if fmt in {"csv", "both"}:
            block.append("[CSV]")
            block.append(_matrix_to_csv(matrix))
        parts.append("\n".join(block))

    parts.append(
        "提示：CSV 段可直接用 `pd.read_csv(io.StringIO(csv_text))` 载入为 DataFrame；"
        "表头若被识别成数据行，可用 `header=None` 再自行指定列名。"
    )
    return "\n\n".join(parts)


# =====================================================================================
# Tool 3：extract_pdf_figures —— PyMuPDF 提取内嵌图片
# =====================================================================================
def _resolve_output_dir(output_dir: Any) -> Path:
    """确定图片输出目录（优先级：显式参数 > 环境变量 PDF_MCP_IMAGES_DIR > cwd/pdf_images）。"""
    if output_dir is not None and str(output_dir).strip():
        candidate = Path(str(output_dir).strip().strip('"').strip("'")).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()  # 相对路径按 Server 进程 cwd 解析
        return candidate
    env_dir = os.environ.get(ENV_IMAGES_DIR, "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path.cwd() / DEFAULT_IMAGE_DIRNAME).resolve()


def _extract_figures_sync(path: Path, output_dir: Any) -> str:
    """同步实现：抽取内嵌位图并存盘，返回绝对路径清单。

    实现要点：
        1. 通过 ``page.get_images(full=True)`` 取到页面上引用的图片 xref，
           再用 ``doc.extract_image(xref)`` 取出原始字节（不做重编码，保留画质）。
        2. 用 MD5 去重：论文里同一期刊 logo/装饰图会在每页重复出现，
           去重后只保存一次，并记录它出现过的页码，避免刷屏式产出几十个同名副本。
        3. 文件名带页码 + 页内序号 + 内容哈希前 8 位，便于回溯到原文位置，也不会互相覆盖。

    注意：``get_images`` 只能拿到「位图对象」。用矢量指令绘制的架构图（例如直接画矩形+文字的
    Transformer 框图）不在其中，需要把整页渲染为位图才能看到 —— 这是引擎能力的边界，已在
    返回文本中显式提示。
    """
    out_dir = _resolve_output_dir(output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PdfToolError(f"无法创建输出目录：{out_dir}（{exc}）", hint="请换一个有写权限的目录。") from exc
    if not os.access(out_dir, os.W_OK):
        raise PdfToolError(f"输出目录不可写：{out_dir}", hint="请检查目录权限或改用其它目录。")

    doc = _open_pdf(path)
    stem = _safe_stem(path.stem)
    saved: list[tuple[int, Path, int, int, int, str]] = []  # (页码, 路径, 宽, 高, 字节数, 内容哈希)
    seen: dict[str, tuple[Path, list[int]]] = {}  # 内容哈希 -> (首次保存路径, 出现过的页码)
    try:
        total_pages = doc.page_count
        for page_index in range(total_pages):
            page = doc.load_page(page_index)
            try:
                images = page.get_images(full=True) or []
            except Exception as exc:  # 单页异常不影响其它页
                _log(f"第 {page_index + 1} 页读取图片列表失败：{type(exc).__name__}: {exc}")
                continue
            for order, image in enumerate(images, start=1):
                xref = image[0]
                width, height = int(image[2]), int(image[3])
                try:
                    base = doc.extract_image(xref)
                except Exception as exc:  # 个别损坏对象直接跳过
                    _log(f"第 {page_index + 1} 页 xref={xref} 图片提取失败：{type(exc).__name__}: {exc}")
                    continue
                data = base.get("image") or b""
                if not data:
                    continue
                digest = hashlib.md5(data).hexdigest()  # md5 仅用于去重，不涉及安全用途
                if digest in seen:
                    seen[digest][1].append(page_index + 1)
                    continue  # 与之前保存的图片字节完全一致 -> 去重
                ext = str(base.get("ext") or "png").lower()
                filename = f"{stem}_p{page_index + 1:04d}_img{order:02d}_{digest[:8]}.{ext}"
                target = out_dir / filename
                try:
                    target.write_bytes(data)
                except OSError as exc:
                    raise PdfToolError(f"写入图片失败：{target}（{exc}）") from exc
                saved.append((page_index + 1, target, width, height, len(data), digest))
                seen[digest] = (target, [page_index + 1])
    finally:
        doc.close()

    header = [
        "【PDF 图片提取结果 · extract_pdf_figures】",
        f"源文件：{path}",
        f"总页数：{total_pages} 页",
        f"输出目录：{out_dir}",
        f"保存图片：{len(saved)} 张（已按内容 MD5 去重，重复出现的不再另存）",
        f"引擎：PyMuPDF {PYMUPDF_VERSION}",
    ]
    if not saved:
        header.append("")
        header.append("⚠️ 未提取到任何内嵌位图。可能原因：")
        header.append("  1) 架构图/示意图是矢量绘制（矩形+文字），不带位图对象；")
        header.append("  2) 图片以“嵌入整页扫描”形式存在（需先 OCR），或该文档确实没有插图。")
        header.append("  提示：若需要“所见即所得”的图，可把整页另存为 PNG 再人工裁剪。")
        return "\n".join(header)

    lines = [*header, "", "绝对路径清单："]
    for idx, (page_no, target, width, height, size, digest) in enumerate(saved, start=1):
        flags: list[str] = []
        if min(width, height) < SMALL_FIGURE_EDGE_PX:
            flags.append("小图(可能是公式/图标)")
        hits = seen[digest][1]  # 该内容出现过的全部页码（去重前的页）
        if len(hits) > 1:
            flags.append(f"重复出现于第 {'、'.join(str(p) for p in hits)} 页")
        suffix = f"  ← {'；'.join(flags)}" if flags else ""
        lines.append(
            f"{idx:>2}. {target}  [{page_no} 页 | {width}×{height}px | {base_ext(target)} | "
            f"{_human_size(size)}]{suffix}"
        )
    return "\n".join(lines)


# =====================================================================================
# Tool 定义（JSON Schema）—— 描述要让 LLM 明白「什么时候用、参数怎么给」
# =====================================================================================
def base_ext(target: Path) -> str:
    """取图片文件的扩展名（大写展示用）。"""
    return target.suffix.lstrip(".").upper() or "BIN"


def _build_tools() -> list[Tool]:
    """构造 3 个 Tool 的声明。参数名与类型必须与实际处理逻辑严格一致。"""
    return [
        Tool(
            name="extract_pdf_text",
            description=(
                "提取 PDF 的纯文本内容（基于 PyMuPDF，速度快、版面还原好）。"
                "适合通读论文正文、方法章节、统计描述与结论。"
                "返回按页分隔的文本，并附带页数、字符数、元数据；"
                "若某页是扫描件（无文字层）会明确提示需要 OCR。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "PDF 文件的绝对路径，例如 papers/attention.pdf",
                    },
                    "start_page": {
                        "type": "integer",
                        "description": "起始页码（1 起算，含该页）。省略表示从第 1 页开始。",
                    },
                    "end_page": {"type": "integer", "description": "结束页码（1 起算，含该页）。省略表示读到末页。"},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="extract_pdf_tables",
            description=(
                "提取 PDF 中的表格并输出结构化文本（基于 pdfplumber）。"
                "自动尝试「线框」与「文本对齐」两种策略，保证行列严格对齐；"
                "合并单元格沿用左/上单元格的值，真实空值填充为 NaN（pandas 读取即为 NaN）。"
                "默认输出 Markdown；output_format='csv' 或 'both' 可额外得到可直接喂给 pandas 的 CSV。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "PDF 文件的绝对路径。"},
                    "output_format": {
                        "type": "string",
                        "enum": ["markdown", "csv", "both"],
                        "description": "输出格式：markdown（默认）/ csv / both。",
                    },
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        ),
        Tool(
            name="extract_pdf_figures",
            description=(
                "提取 PDF 中的内嵌位图（如神经网络架构图、实验结果图），保存到本地并返回绝对路径列表。"
                "同名内容会自动去重，并标注小图（可能是公式/图标）与重复出现的页码。"
                "注意：矢量绘制的框图不在提取范围内。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "PDF 文件的绝对路径。"},
                    "output_dir": {
                        "type": "string",
                        "description": "图片保存目录（绝对或相对路径）；默认使用当前目录下的 pdf_images。",
                    },
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        ),
    ]


# =====================================================================================
# MCP Server 装配：tools/list、tools/call 两个 handler + stdio 主循环
#
# 注意 SDK 版本：本脚本针对官方 mcp SDK 2.x（本环境为 2.2.0）。
# 2.x 移除了 1.x 的 @server.list_tools() / @server.call_tool() 装饰器，改为
# 「构造器注册回调」：Server(name, on_list_tools=..., on_call_tool=...)。
# =====================================================================================
async def _handle_list_tools(ctx: Any, params: Any) -> ListToolsResult:
    """响应 tools/list：把 3 个 Tool 的声明交给客户端（Cline 据此生成调用入口）。"""
    return ListToolsResult(tools=_build_tools())


async def _handle_call_tool(ctx: Any, params: CallToolRequestParams) -> CallToolResult:
    """响应 tools/call：参数容错解析 -> 线程池执行 -> 任何异常都转成文本返回。

    两条硬约束：
        1. 无论成功还是失败，返回的都是含 TextContent 的 CallToolResult，绝不向外抛异常
           （未捕获异常会被客户端视为 Server 崩溃并断开连接）。
        2. 解析是 CPU/IO 密集的同步操作，统一走 asyncio.to_thread，
           避免阻塞 MCP 事件循环导致客户端误判超时。
    """
    name = str(getattr(params, "name", "") or "")
    args = getattr(params, "arguments", None) or {}
    _log(f"收到调用：{name}，参数={args}")
    try:
        if name == "extract_pdf_text":
            path = _resolve_pdf(args.get("file_path"))
            start = _as_optional_int(args.get("start_page"), "start_page")
            end = _as_optional_int(args.get("end_page"), "end_page")
            return _ok(await asyncio.to_thread(_extract_text_sync, path, start, end))

        if name == "extract_pdf_tables":
            path = _resolve_pdf(args.get("file_path"))
            fmt = args.get("output_format") or "markdown"
            return _ok(await asyncio.to_thread(_extract_tables_sync, path, fmt))

        if name == "extract_pdf_figures":
            path = _resolve_pdf(args.get("file_path"))
            return _ok(await asyncio.to_thread(_extract_figures_sync, path, args.get("output_dir")))

        return _fail(
            f"未知的工具名称：{name}",
            hint="可用工具：extract_pdf_text / extract_pdf_tables / extract_pdf_figures。",
        )
    except PdfToolError as exc:
        _log(f"{name} 参数/数据错误：{exc}")
        return _fail("PDF 解析失败", str(exc), getattr(exc, "hint", "") or "")
    except asyncio.CancelledError:
        raise  # 客户端主动取消：交回事件循环，不能吞掉
    except Exception as exc:  # 兜底：绝不让异常穿透导致 Server 崩溃
        _log(traceback.format_exc())
        return _fail(
            "发生未预期的内部错误",
            f"{type(exc).__name__}: {exc}",
            "详细堆栈已写入 Server 的 stderr；请把该信息反馈给开发者，Server 仍在正常服务。",
        )


# Server 实例：mcp 2.x 用「构造器注册 handler」装配；title/instructions 会展示在客户端的
# 工具列表与上下文里，帮助模型判断「何时该用这套 PDF 工具」。
server = Server(
    name=SERVER_NAME,
    version=SERVER_VERSION,
    title="PDF Research MCP Server",
    instructions=(
        "面向数据与统计、神经网络等科研场景的 PDF 解析服务："
        "extract_pdf_text 读正文；extract_pdf_tables 取统计表（Markdown/CSV，行列对齐、空值填 NaN）；"
        "extract_pdf_figures 导出架构图/结果图并返回绝对路径。"
    ),
    on_list_tools=_handle_list_tools,
    on_call_tool=_handle_call_tool,
)


async def main() -> None:
    """以 stdio 通道运行 Server：stdin 收 JSON-RPC，stdout 回 JSON-RPC，stderr 记日志。"""
    _log(f"启动 {SERVER_NAME} v{SERVER_VERSION}（PyMuPDF {PYMUPDF_VERSION} / pdfplumber {PDFPLUMBER_VERSION}）")
    _log(f"当前工作目录={Path.cwd()}；图片默认输出目录={_resolve_output_dir(None)}")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _log("收到中断信号，Server 正常退出。")
    except Exception as err:  # 启动期失败要打印堆栈，便于排查 MCP 配置问题
        _log("Server 启动失败：\n" + traceback.format_exc())
        raise SystemExit(1) from err
