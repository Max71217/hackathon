"""
医学知识整合智能体 - Streamlit 单页应用
Track 1 前端界面
第2阶段：前后端打通，真实知识图谱提取
"""

import os
import tempfile
import shutil
from datetime import datetime

import streamlit as st
from streamlit_echarts import st_echarts

from schema import ConceptNode, RelationEdge, KnowledgeGraph
from pdf_parser import SmartPdfParser
from ai_engine import StepfunAgent

st.set_page_config(page_title="医学知识整合智能体", layout="wide")

# ============================================================================
# 初始化 session_state
# ============================================================================
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

if "knowledge_graph" not in st.session_state:
    st.session_state["knowledge_graph"] = None

if "graph_generated" not in st.session_state:
    st.session_state["graph_generated"] = False

if "graphs_dict" not in st.session_state:
    st.session_state["graphs_dict"] = {}

# ============================================================================
# 配置
# ============================================================================
TEMP_DIR = tempfile.mkdtemp(prefix="medical_agent_")

# 类别颜色映射
CATEGORY_COLORS = {
    "生理学": "#4CAF50",
    "病理学": "#F44336",
    "解剖学": "#2196F3",
    "药理学": "#9C27B0",
    "诊断学": "#FF9800",
    "其他": "#888888",
}

# 默认 Mock 数据（未上传时展示）
MOCK_NODES = [
    ConceptNode(id="n01", name="心肌收缩", definition="心肌细胞通过钙离子介导的交叉桥循环产生收缩力", category="生理学", chapter="心脏生理", page=102, textbook_source="生理学"),
    ConceptNode(id="n02", name="心力衰竭", definition="心脏泵血功能下降，无法满足机体代谢需求", category="病理学", chapter="心血管疾病", page=205, textbook_source="病理学"),
    ConceptNode(id="n03", name="钙离子通道", definition="细胞膜上允许Ca2+选择性通过的蛋白质复合物", category="生理学", chapter="细胞电生理", page=58, textbook_source="生理学"),
    ConceptNode(id="n04", name="洋地黄中毒", definition="洋地黄类药物过量导致心律失常和恶心", category="病理学", chapter="药物中毒", page=310, textbook_source="病理学"),
    ConceptNode(id="n05", name="肾素-血管紧张素系统", definition="调节血压和体液平衡的激素级联系统", category="生理学", chapter="肾功能", page=178, textbook_source="生理学"),
    ConceptNode(id="n06", name="高血压", definition="体循环动脉血压持续升高的病理状态", category="病理学", chapter="心血管病理", page=192, textbook_source="病理学"),
    ConceptNode(id="n07", name="肺通气", definition="气体在肺泡与外界之间的交换过程", category="生理学", chapter="呼吸生理", page=134, textbook_source="生理学"),
    ConceptNode(id="n08", name="COPD", definition="以持续气流受限为特征的慢性阻塞性肺疾病", category="病理学", chapter="呼吸系统疾病", page=228, textbook_source="病理学"),
    ConceptNode(id="n09", name="血气屏障", definition="肺泡上皮与毛细血管之间的薄层结构", category="生理学", chapter="呼吸生理", page=140, textbook_source="生理学"),
    ConceptNode(id="n10", name="低氧血症", definition="血液氧含量低于正常的病理状态", category="病理学", chapter="呼吸衰竭", page=260, textbook_source="病理学"),
    ConceptNode(id="n11", name="交感神经", definition="自主神经系统分支，驱动战斗或逃跑反应", category="生理学", chapter="神经生理", page=88, textbook_source="生理学"),
    ConceptNode(id="n12", name="应激性溃疡", definition="严重应激状态下胃黏膜的急性损伤", category="病理学", chapter="消化系统疾病", page=302, textbook_source="病理学"),
]

MOCK_EDGES = [
    RelationEdge(source_node_id="n03", target_node_id="n01", relation_type="prerequisite"),
    RelationEdge(source_node_id="n01", target_node_id="n02", relation_type="applies_to"),
    RelationEdge(source_node_id="n04", target_node_id="n02", relation_type="contains"),
    RelationEdge(source_node_id="n05", target_node_id="n06", relation_type="applies_to"),
    RelationEdge(source_node_id="n07", target_node_id="n08", relation_type="prerequisite"),
    RelationEdge(source_node_id="n09", target_node_id="n07", relation_type="contains"),
    RelationEdge(source_node_id="n08", target_node_id="n10", relation_type="applies_to"),
    RelationEdge(source_node_id="n11", target_node_id="n02", relation_type="applies_to"),
    RelationEdge(source_node_id="n11", target_node_id="n12", relation_type="applies_to"),
]

MOCK_GRAPH = KnowledgeGraph(nodes=MOCK_NODES, edges=MOCK_EDGES)

# ============================================================================
# 辅助函数
# ============================================================================

def build_echarts_options(graph: KnowledgeGraph) -> dict:
    """将 KnowledgeGraph 转换为 ECharts 力导向图配置"""
    # 收集所有出现的 category
    categories = list(set(n.textbook_source for n in graph.nodes))
    color_map = {cat: CATEGORY_COLORS.get(cat, "#888888") for cat in categories}

    return {
        "backgroundColor": "#0d1117",
        "tooltip": {
            "formatter": "{b}<br/><span style='color:#888'>来源:</span> {a}<br/><span style='color:#888'>定义:</span> {c}",
        },
        "series": [
            {
                "type": "graph",
                "layout": "force",
                "symbolSize": 55,
                "roam": True,
                "draggable": True,
                "label": {"show": True, "color": "#fff", "fontSize": 12, "fontWeight": "bold"},
                "force": {
                    "repulsion": 350,
                    "edgeLength": 130,
                    "gravity": 0.12,
                    "layoutAnimation": True,
                },
                "lineStyle": {"color": "#555", "curveness": 0.3, "width": 2},
                "emphasis": {
                    "focus": "adjacency",
                    "lineStyle": {"color": "#00ffcc", "width": 4},
                },
                "data": [
                    {
                        "id": n.id,
                        "name": n.name,
                        "category": n.textbook_source,
                        "value": n.definition[:80] + ("..." if len(n.definition) > 80 else ""),
                        "itemStyle": {"color": color_map.get(n.textbook_source, "#888")},
                    }
                    for n in graph.nodes
                ],
                "categories": [
                    {"name": cat, "itemStyle": {"color": color_map.get(cat, "#888")}}
                    for cat in categories
                ],
                "edges": [
                    {
                        "source": e.source_node_id,
                        "target": e.target_node_id,
                        "name": e.relation_type,
                    }
                    for e in graph.edges
                ],
            }
        ],
        "legend": {
            "data": categories,
            "textStyle": {"color": "#ccc"},
            "top": 15,
        },
    }


def process_pdf_to_graph(pdf_path: str, book_name: str, agent: StepfunAgent) -> KnowledgeGraph:
    """
    解析 PDF 并提取知识图谱
    逐页处理文本型页面，聚合同步到 AI 引擎
    """
    all_nodes: list[ConceptNode] = []
    all_edges: list[RelationEdge] = []

    with SmartPdfParser(pdf_path, book_name) as parser:
        text_pages = parser.get_text_pages()

        for page in text_pages:
            if page.text and len(page.text.strip()) > 50:
                try:
                    graph = agent.extract_knowledge_graph(
                        text_content=page.text,
                        image_bytes=None,
                        book_name=book_name,
                        chapter="待提取章节",
                        page=page.page_num,
                    )
                    # 合并节点（追加 ID 前缀避免冲突）
                    for node in graph.nodes:
                        node.id = f"{book_name}_{page.page_num}_{node.id}"
                    all_nodes.extend(graph.nodes)

                    # 合并边
                    for edge in graph.edges:
                        edge.source_node_id = f"{book_name}_{page.page_num}_{edge.source_node_id}"
                        edge.target_node_id = f"{book_name}_{page.page_num}_{edge.target_node_id}"
                    all_edges.extend(graph.edges)

                except Exception as e:
                    print(f"⚠️ 页面 {page.page_num} 文本提取失败，已自动跳过。错误原因: {e}")
                    st.toast(f"⚠️ 页面 {page.page_num} 提取失败，已自动跳过")
                    continue

        # 扫描型页面（图像）单独处理
        scan_pages = parser.get_scan_pages()
        for page in scan_pages:
            if page.image_bytes:
                try:
                    graph = agent.extract_knowledge_graph(
                        text_content="",
                        image_bytes=page.image_bytes,
                        book_name=book_name,
                        chapter="图表内容",
                        page=page.page_num,
                    )
                    for node in graph.nodes:
                        node.id = f"{book_name}_scan_{page.page_num}_{node.id}"
                    all_nodes.extend(graph.nodes)

                    for edge in graph.edges:
                        edge.source_node_id = f"{book_name}_scan_{page.page_num}_{edge.source_node_id}"
                        edge.target_node_id = f"{book_name}_scan_{page.page_num}_{edge.target_node_id}"
                    all_edges.extend(graph.edges)
                except Exception as e:
                    print(f"⚠️ 扫描页 {page.page_num} 图像提取失败，已自动跳过。错误原因: {e}")
                    st.toast(f"⚠️ 扫描页 {page.page_num} 提取失败，已自动跳过")
                    continue

    # 去重（同名节点保留一个）
    unique_nodes: dict[str, ConceptNode] = {}
    for node in all_nodes:
        if node.name not in unique_nodes:
            unique_nodes[node.name] = node

    # 边去重
    seen_edges = set()
    dedup_edges: list[RelationEdge] = []
    for edge in all_edges:
        key = (edge.source_node_id, edge.target_node_id, edge.relation_type)
        if key not in seen_edges:
            seen_edges.add(key)
            dedup_edges.append(edge)

    return KnowledgeGraph(nodes=list(unique_nodes.values()), edges=dedup_edges)


# ============================================================================
# 页面布局
# ============================================================================
st.title("🩺 医学知识整合智能体")

left_col, main_col, right_col = st.columns([1, 3, 1])

# ============================================================================
# 左侧边栏：教材上传区
# ============================================================================
with left_col:
    st.markdown("### 📚 教材上传区")
    uploaded = st.file_uploader(
        "支持 PDF/TXT 多文件上传",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    process_btn = st.button("🔍 开始分析与提取图谱", type="primary", use_container_width=True)

    if process_btn and uploaded:
        api_key = os.environ.get("STEPFUN_API_KEY")
        if not api_key:
            st.error("⚠️ 请先设置环境变量 STEPFUN_API_KEY\n\n命令：export STEPFUN_API_KEY=你的密钥")
        else:
            with st.status("🚀 初始化 AI 引擎...", expanded=True) as status:
                try:
                    agent = StepfunAgent()
                    status.update(label="✅ AI 引擎就绪", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"AI 引擎初始化失败: {e}")
                    agent = None

            if agent:
                # 逐个处理文件
                for f in uploaded:
                    # 【核心修复】跳过已经提取成功的文件，避免重复消耗和卡顿
                    if f.name in st.session_state["graphs_dict"]:
                        continue

                    # 保存到临时目录
                    tmp_path = os.path.join(TEMP_DIR, f.name)
                    with open(tmp_path, "wb") as tmp:
                        tmp.write(f.read())

                    status_info = st.status(f"📄 正在解析 PDF：{f.name}", expanded=True)
                    try:
                        # 解析 PDF
                        with SmartPdfParser(tmp_path, f.name) as parser:
                            result = parser.parse()
                            text_page_count = sum(1 for p in result.pages if p.page_type.value == "text")
                            scan_page_count = sum(1 for p in result.pages if p.page_type.value == "scan")

                        status_info.update(
                            label=f"✅ {f.name} 解析完成",
                            state="complete",
                            expanded=False,
                        )
                    except Exception as e:
                        st.error(f"PDF 解析失败: {e}")
                        continue

                    # AI 提取知识图谱
                    with st.spinner(f"🤖 AI 正在从 {f.name} 提取知识点，请稍候..."):
                        try:
                            graph = process_pdf_to_graph(tmp_path, f.name, agent)
                            st.session_state["knowledge_graph"] = graph
                            st.session_state["graphs_dict"][f.name] = graph
                            st.session_state["graph_generated"] = True

                            # 更新文件状态
                            for item in st.session_state["uploaded_files"]:
                                if item["name"] == f.name:
                                    item["status"] = f"✅ 已提取 {len(graph.nodes)} 个节点"
                                    break

                            st.success(
                                f"✅ {f.name} 处理完成！"
                                f"\n提取到 **{len(graph.nodes)} 个节点**，**{len(graph.edges)} 条边**"
                            )
                        except Exception as e:
                            import traceback
                            error_detail = traceback.format_exc()
                            print(f"⚠️ 图谱提取遇到问题，已自动跳过。错误原因: {e}\n堆栈: {error_detail}")
                            st.toast(f"⚠️ 图谱提取遇到问题，已自动跳过，请查看终端日志")

                    # 清理临时文件
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    # 记录已上传文件
    if uploaded:
        for f in uploaded:
            if f.name not in [x["name"] for x in st.session_state["uploaded_files"]]:
                st.session_state["uploaded_files"].append({
                    "name": f.name,
                    "size": f.size,
                    "status": "已上传",
                })

    st.markdown("---")
    st.markdown("### 📋 已上传教材列表")
    if st.session_state["uploaded_files"]:
        for item in st.session_state["uploaded_files"]:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"📄 `{item['name']}`")
                with col2:
                    st.caption(item["status"])
    else:
        st.info("暂无上传教材")

# ============================================================================
# 主工作区：知识图谱力导向图
# ============================================================================
with main_col:
    if st.session_state.get("graphs_dict"):
        graph_names = list(st.session_state["graphs_dict"].keys())
        selected_name = st.selectbox("👁️ 切换当前查看的图谱", graph_names, index=len(graph_names)-1)
        graph = st.session_state["graphs_dict"][selected_name]

        st.markdown(f"### 🕸️ 知识图谱（来源: {selected_name}）")
        echarts_opts = build_echarts_options(graph)
        st_echarts(echarts_opts, height="620px")

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("节点数", len(graph.nodes))
        with col_s2:
            st.metric("边数", len(graph.edges))
        with col_s3:
            textbooks = set(n.textbook_source for n in graph.nodes)
            st.metric("涉及教材", len(textbooks))
    else:
        st.markdown("### 🕸️ 知识图谱（示例数据）")
        st.info("⬆️ 请在左侧上传 PDF 教材，点击『开始分析』触发真实图谱提取")
        echarts_opts = build_echarts_options(MOCK_GRAPH)
        st_echarts(echarts_opts, height="620px")

        st.markdown(
            """
            <div style="display:flex; gap:20px; margin-top:8px;">
                <span style="color:#4CAF50;">● 生理学</span>
                <span style="color:#F44336;">● 病理学</span>
                <span style="color:#888;">― 前置/包含/应用</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================================
# 右侧面板：功能区
# ============================================================================
with right_col:
    tab1, tab2, tab3 = st.tabs(["⚙️ 整合操作", "💬 RAG 问答", "🔧 聊天微调"])

    with tab1:
        st.markdown("##### 一键整合")
        if st.button("🚀 执行跨书整合", use_container_width=True):
            if "graphs_dict" not in st.session_state or len(st.session_state["graphs_dict"]) < 2:
                st.warning("⚠️ 请先在左侧上传并提取至少两本不同教材的图谱！")
            else:
                book_names = list(st.session_state["graphs_dict"].keys())
                g1, g2 = st.session_state["graphs_dict"][book_names[0]], st.session_state["graphs_dict"][book_names[1]]
                with st.spinner(f"🧠 deepseek-v4-pro 正在深层对齐与融合: {book_names[0]} & {book_names[1]}..."):
                    try:
                        agent = StepfunAgent()
                        merged_graph = agent.merge_graphs(g1, g2)
                        st.session_state["knowledge_graph"] = merged_graph
                        st.session_state["graphs_dict"]["跨书整合版"] = merged_graph
                        st.success(f"🎉 融合成功！新图谱包含 {len(merged_graph.nodes)} 个节点。")
                        st.rerun()
                    except Exception as e:
                        st.error(f"融合失败: {e}")

    with tab2:
        st.markdown("##### RAG 问答")
        if "rag_messages" not in st.session_state:
            st.session_state["rag_messages"] = []

        for msg in st.session_state["rag_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query = st.chat_input("输入医学问题...")
        if query:
            st.session_state["rag_messages"].append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            graph = st.session_state.get("knowledge_graph")

            with st.chat_message("assistant"):
                if not graph or not graph.nodes:
                    st.warning("⚠️ 当前没有可用的知识图谱，请先在左侧提取教材或执行整合。")
                    st.session_state["rag_messages"].append({"role": "assistant", "content": "当前没有可用的知识图谱，请先提取数据。"})
                else:
                    message_placeholder = st.empty()
                    message_placeholder.markdown(f"🧠 正在检索包含 **{len(graph.nodes)}** 个节点的图谱记忆...")

                    try:
                        sys_prompt = st.session_state.get("sys_prompt", "你是一个专业的医学知识助手。")
                        temperature = st.session_state.get("temp_slider", 0.3)

                        agent = StepfunAgent()
                        answer = agent.rag_answer(query, graph, sys_prompt, temperature)

                        message_placeholder.markdown(answer)
                        st.session_state["rag_messages"].append({"role": "assistant", "content": answer})
                    except Exception as e:
                        message_placeholder.error(f"问答失败: {e}")

    with tab3:
        st.markdown("##### 聊天微调")
        st.text_area(
            "系统提示词 (System Prompt)",
            value="你是一个专业的医学知识助手，擅长从教材中提取结构化知识。",
            height=80,
            key="sys_prompt",
        )
        st.slider("Temperature", 0.0, 1.0, 0.3, key="temp_slider")
        st.button("💾 保存微调配置", use_container_width=True, key="save_tune")

st.divider()
st.caption("医学知识整合智能体 v0.2 — Track 1 前端 · 第2阶段前后端打通")

# ============================================================================
# 清理临时目录
# ============================================================================
def cleanup_temp():
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    except Exception:
        pass

import atexit
atexit.register(cleanup_temp)