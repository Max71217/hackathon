"""
智能路由 PDF 解析器
第1阶段：使用 PyMuPDF 读取 PDF，智能判断文本型/扫描型页面
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF

from schema import Chunk, Chapter, Textbook


# =============================================================================
# 类型枚举
# =============================================================================

class PageType(Enum):
    """页面类型枚举"""
    TEXT = "text"          # 文本型：可直接提取文字
    SCAN = "scan"         # 扫描型/图表：需转图像供视觉处理


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class PageResult:
    """单页解析结果"""
    page_num: int                     # 页码（从1开始）
    page_type: PageType               # 页面类型
    text: Optional[str] = None        # 文本内容（文本型页面）
    image_bytes: Optional[bytes] = None  # 图像字节流（扫描型页面）
    char_count: int = 0              # 提取字符数


@dataclass
class ParseResult:
    """整本教材解析结果"""
    book_name: str
    total_pages: int
    pages: list[PageResult]
    textbook: Optional[Textbook] = None  # 结构化教材对象


# =============================================================================
# 核心解析器
# =============================================================================

TEXT_THRESHOLD = 400  # 文本型/扫描型分界线：字符数 > 400 判定为文本型


class SmartPdfParser:
    """
    智能路由 PDF 解析器
    逐页读取，自动判定页面类型并分流处理
    """

    def __init__(self, pdf_path: str, book_name: str = "未知教材"):
        """
        初始化解析器

        Args:
            pdf_path: PDF 文件路径
            book_name: 教材名称
        """
        self.pdf_path = pdf_path
        self.book_name = book_name
        self._doc: Optional[fitz.Document] = None

    def open(self) -> "SmartPdfParser":
        """打开 PDF 文件"""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF 文件不存在: {self.pdf_path}")
        self._doc = fitz.open(self.pdf_path)
        return self

    def close(self) -> None:
        """关闭 PDF 文件"""
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def __enter__(self) -> "SmartPdfParser":
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def doc(self) -> fitz.Document:
        """获取文档对象（懒加载）"""
        if self._doc is None:
            self.open()
        assert self._doc is not None
        return self._doc

    # -------------------------------------------------------------------------
    # 解析入口
    # -------------------------------------------------------------------------

    def parse(self) -> ParseResult:
        """
        解析整本教材

        Returns:
            ParseResult: 包含所有页面解析结果
        """
        pages: list[PageResult] = []
        for page_num in range(self.doc.page_count):
            page_result = self._parse_page(page_num)
            pages.append(page_result)

        return ParseResult(
            book_name=self.book_name,
            total_pages=self.doc.page_count,
            pages=pages,
        )

    def parse_range(self, start_page: int, end_page: int) -> ParseResult:
        """
        解析指定页码范围

        Args:
            start_page: 起始页码（从1开始）
            end_page: 结束页码（从1开始， inclusive）

        Returns:
            ParseResult: 包含指定范围的页面解析结果
        """
        pages: list[PageResult] = []
        for page_num in range(start_page - 1, end_page):
            page_result = self._parse_page(page_num)
            pages.append(page_result)

        return ParseResult(
            book_name=self.book_name,
            total_pages=end_page - start_page + 1,
            pages=pages,
        )

    # -------------------------------------------------------------------------
    # 私有方法
    # -------------------------------------------------------------------------

    def _parse_page(self, page_index: int) -> PageResult:
        """
        解析单个页面，自动判定类型

        Args:
            page_index: 页码索引（从0开始）

        Returns:
            PageResult: 单页解析结果
        """
        page = self.doc[page_index]
        raw_text = page.get_text()
        char_count = len(raw_text.strip())

        # 智能路由判定
        if char_count > TEXT_THRESHOLD:
            # 文本型页面：直接返回文本
            return PageResult(
                page_num=page_index + 1,
                page_type=PageType.TEXT,
                text=raw_text,
                image_bytes=None,
                char_count=char_count,
            )
        else:
            # 扫描型/图表页面：转图像字节流
            pixmap = page.get_pixmap(dpi=300)
            image_bytes = pixmap.tobytes("png")
            return PageResult(
                page_num=page_index + 1,
                page_type=PageType.SCAN,
                text=None,
                image_bytes=image_bytes,
                char_count=char_count,
            )

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------

    def get_text_pages(self) -> list[PageResult]:
        """获取所有文本型页面"""
        result = self.parse()
        return [p for p in result.pages if p.page_type == PageType.TEXT]

    def get_scan_pages(self) -> list[PageResult]:
        """获取所有扫描型页面"""
        result = self.parse()
        return [p for p in result.pages if p.page_type == PageType.SCAN]

    def build_chunks(self, min_chunk_size: int = 500, max_chunk_size: int = 800) -> list[Chunk]:
        """
        将文本型页面内容切分为 RAG 切片

        Args:
            min_chunk_size: 最小切片字符数
            max_chunk_size: 最大切片字符数

        Returns:
            list[Chunk]: RAG 切片列表
        """
        chunks: list[Chunk] = []
        text_pages = self.get_text_pages()

        for page in text_pages:
            if page.text is None:
                continue

            # 简单按段落切分（实际可使用更复杂策略）
            paragraphs = [p.strip() for p in page.text.split("\n") if p.strip()]
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
                    chunks.append(Chunk(
                        text=current_chunk,
                        book_name=self.book_name,
                        chapter="未知章节",  # 后续可按标题识别
                        page=page.page_num,
                    ))
                    current_chunk = para
                else:
                    current_chunk += "\n" + para if current_chunk else para

            # 处理最后一块
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(Chunk(
                    text=current_chunk,
                    book_name=self.book_name,
                    chapter="未知章节",
                    page=page.page_num,
                ))

        return chunks
