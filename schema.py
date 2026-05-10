"""
医学知识整合智能体 - 核心数据模型
第0阶段：契约驱动开发
仅定义数据结构，不涉及 UI、PDF 解析或 LLM 调用逻辑
"""

from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# 1. 文档解析契约 (Document Schema)
# =============================================================================

class Chunk(BaseModel):
    """
    RAG 切片单元
    用于检索的最小文本片段，建议长度 500-800 字符
    """
    text: str = Field(..., description="切片后的正文内容")
    book_name: str = Field(..., description="所属教材书名")
    chapter: str = Field(..., description="所属章节标题")
    page: int = Field(..., description="起始页码")


class Chapter(BaseModel):
    """
    章节单元
    对应教材中的单个章节
    """
    title: str = Field(..., description="章节标题")
    start_page: int = Field(..., description="起始页码")
    content: str = Field(..., description="章节正文内容（纯文本）")
    character_count: int = Field(..., description="字符数统计")


class Textbook(BaseModel):
    """
    教材文档
    对应一本完整的医学教材
    """
    title: str = Field(..., description="书名")
    total_pages: int = Field(..., description="总页数")
    total_characters: int = Field(..., description="总字符数")
    chapters: list[Chapter] = Field(default_factory=list, description="章节列表")


# =============================================================================
# 2. 知识图谱契约 (Graph Schema)
# =============================================================================

class ConceptNode(BaseModel):
    """
    知识节点
    代表一个医学专有名词或概念
    """
    id: str = Field(..., description="唯一标识符")
    name: str = Field(..., description="医学专有名词/术语")
    definition: str = Field(..., description="严格的医学定义")
    category: str = Field(..., description="概念分类（如：解剖、生理学、病理学等）")
    chapter: str = Field(..., description="所属章节")
    page: int = Field(..., description="页码")
    textbook_source: str = Field(..., description="来源教材书名")


class RelationEdge(BaseModel):
    """
    关系边
    连接两个知识节点，表示它们之间的逻辑关系
    """
    source_node_id: str = Field(..., description="源节点 ID")
    target_node_id: str = Field(..., description="目标节点 ID")
    relation_type: str = Field(
        ...,
        description=(
            "关系类型，必须为以下字面值之一：\n"
            "  - prerequisite: 前置依赖（A 是 B 的前置知识）\n"
            "  - parallel: 并列（A 与 B 为平行概念）\n"
            "  - contains: 包含（A 包含 B）\n"
            "  - applies_to: 应用（A 应用于 B）"
        )
    )


class KnowledgeGraph(BaseModel):
    """
    知识图谱
    包含节点列表和关系边列表
    """
    nodes: list[ConceptNode] = Field(default_factory=list, description="知识节点列表")
    edges: list[RelationEdge] = Field(default_factory=list, description="关系边列表")


# =============================================================================
# 3. 跨书整合契约 (Merge Schema)
# =============================================================================

class MergeDecision(BaseModel):
    """
    整合决策
    记录跨教材知识整合时的决策过程

    注意：当执行 'merge' 动作生成 result_node 时，
    该新节点将继承所有原 affected_node_ids 的 RelationEdge，
    以保证教学连贯性不断裂。
    """
    decision_id: str = Field(..., description="决策唯一标识符")
    action: str = Field(
        ...,
        description=(
            "整合动作，必须为以下字面值之一：\n"
            "  - merge: 合并多个节点为一个新节点\n"
            "  - keep: 保留原节点不做修改\n"
            "  - remove: 移除指定节点"
        )
    )
    affected_node_ids: list[str] = Field(
        ...,
        description="受影响的原始节点 ID 列表"
    )
    result_node: Optional[ConceptNode] = Field(
        None,
        description=(
            "合并后的新 ConceptNode\n"
            "注意：该节点 definition 字数必须不超过受影响节点定义总字数的 30%"
        )
    )
    reason: str = Field(..., description="整合理由说明")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="决策置信度，范围 [0.0, 1.0]"
    )
