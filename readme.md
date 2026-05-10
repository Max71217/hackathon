# 医学知识整合智能体 (Medical Knowledge Integration Agent)

本项目是一个基于多模态大模型（LLM）的医学知识提取、融合与问答平台。针对多本医学教材，系统能自动进行智能分流解析（文本/视觉），提取结构化知识图谱，并使用高阶大模型进行跨教材图谱的端到端对齐，最终提供基于图谱增强的精准医学 RAG 问答服务。

## 🌟 核心功能
- **智能路由解析**：基于字符密度判定，自动将纯文本与解剖插图分流至不同大模型处理。
- **高精度图谱抽取**：利用智谱 GLM-4.6V 视觉模型与 GLM-4-Flash 文本模型，提取标准医学节点与关系。
- **跨教材图谱融合**：使用 DeepSeek-V4-Pro 的强逻辑推理与原生 JSON 输出能力，对不同教材的重复概念进行去重、对齐与融合。
- **Graph-Augmented RAG**：将知识图谱的拓扑连线转化为上下文，提供深度逻辑医学问答。

---

## 🛠️ 环境依赖
- **操作系统**：Linux / macOS / Windows
- **基础环境**：Python 3.10+ 
- **核心组件库**：
  - `streamlit` (>=1.30.0)：交互式 Web 前端框架
  - `streamlit-echarts`：知识图谱动态力导向可视化
  - `zhipuai`：智谱 AI SDK（负责文本提取与视觉识别）
  - `PyMuPDF` (fitz)：极速 PDF 逐页解析与图像转换
  - `pydantic` (>=2.0)：结构化数据类型约束与契约管理
  - `requests`：DeepSeek V4 Pro 官方 API 调用
  - `python-dotenv`：本地开发环境变量管理

---

## 🚀 安装步骤

1. **克隆项目到本地**:
```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
创建并激活虚拟环境 (强烈建议):
code
Bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# Windows 用户请使用: venv\Scripts\activate
安装依赖项:
code
Bash
pip install -r requirements.txt
⚙️ 配置说明
本项目依赖外部大模型 API 进行知识计算。分为本地开发配置与云端部署配置：
1. 本地开发配置 (Local Development)
在项目根目录下创建一个 .env 隐藏文件，并填入以下内容：
code
Env
# 智谱 AI API Key (用于视觉提取 GLM-4.6V 和文本处理 GLM-4-Flash)
ZHIPU_API_KEY=

# DeepSeek API Key (用于跨书图谱融合对齐 DeepSeek-V4-Pro)
DEEPSEEK_API_KEY=

# 阶跃星辰 API Key (可选保留)
STEPFUN_API_KEY=
(注：为保证数据安全，.env 文件已被 .gitignore 排除，不会被提交至代码库。)
2. 魔搭云端部署配置 (ModelScope Deployment)
项目根目录已提供 ms_deploy.json 配置文件。在 ModelScope 创空间选择“编程式创建应用”时，系统将自动读取配置使用 Streamlit SDK 构建。
安全注意：请在创空间构建完成后，于魔搭的 「设置」->「环境变量」 面板中手动注入真实的 API Key，切勿硬编码在 JSON 文件中上传。
🏃 运行命令
确认环境及配置无误后，在项目根目录执行以下命令启动系统：
code
Bash
streamlit run app.py
启动成功后，浏览器会自动打开 http://localhost:8501。
进入系统后，您可以在左侧边栏上传 PDF 教材片段，提取图谱后在右侧控制台进行“跨书整合”与“RAG 问答”。