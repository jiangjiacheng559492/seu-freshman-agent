# 东南大学新生指南问答 Agent

一个面向东南大学研究生新生指南的知识库问答 Agent。

项目以学院新生指南为知识来源，使用意图识别、向量检索、关键词匹配和大语言模型生成回答，并提供基于 Streamlit 的网页聊天界面。

本项目主要用于学习和展示特化 Agent 的完整开发流程，包括知识库整理、RAG 检索、意图路由、提示词设计、自动化评测和客户端开发。

## 主要功能

- 新生指南知识库问答
- 用户问题意图识别
- 文档向量检索
- 关键词与目标章节加权
- 混合 RAG 检索
- 资料不足时谨慎回答
- 敏感信息与越界问题拒答
- Streamlit 网页聊天界面
- 开发集与保留测试集评测
- 端到端质量和内部组件分层评分

## 工作流程

```text
用户问题
   ↓
问题向量化
   ↓
意图识别与路由
   ↓
文档向量检索
   ↓
章节加权 + 关键词加权
   ↓
选出相关知识片段
   ↓
通义千问生成回答
   ↓
Streamlit 聊天界面展示
```

## 技术栈

- Python
- Streamlit
- OpenAI Python SDK
- 阿里云百炼 OpenAI 兼容接口
- 通义千问生成模型
- DashScope 文本向量模型
- 本地 JSON 向量索引
- JSONL 自动化评测集

## 项目结构

```text
seu-freshman-agent/
├─ app.py                         # Streamlit 网页客户端
├─ requirements.txt               # Python 依赖
├─ .env.example                   # 环境变量示例
├─ .gitignore                     # Git 忽略规则
├─ README.md                      # 项目说明
│
├─ assets/
│  └─ seu-logo.png                # 页面使用的校徽图片
│
├─ data/
│  ├─ knowledge/
│  │  └─ freshman_guide.md        # 新生指南知识库
│  ├─ intents/
│  │  └─ intent_catalog.jsonl     # 意图目录和示例问法
│  └─ index/
│     ├─ freshman_guide_index.json
│     └─ intent_index.json
│
├─ evals/
│  ├─ questions.jsonl             # 初始评测问题
│  ├─ smoke.jsonl                 # 快速冒烟测试集
│  └─ development.jsonl           # 开发评测集
│
└─ src/
   ├─ baseline.py                 # 无知识库基线
   ├─ build_local_index.py        # 构建指南向量索引
   ├─ build_intent_index.py       # 构建意图向量索引
   ├─ intent_router.py            # 意图识别与路由
   ├─ local_rag_agent.py          # 本地向量 RAG
   ├─ hybrid_rag_agent.py         # 混合检索 Agent
   ├─ run_evaluation.py           # 运行评测
   ├─ score_evaluation.py         # 基础评分
   └─ score_evaluation_v2.py      # 分层评分
```

## 本地运行

### 1. 进入项目目录

```powershell
cd "C:\path\to\seu-freshman-agent"
```

### 2. 创建虚拟环境

```powershell
python -m venv .venv
```

### 3. 激活虚拟环境

如果 PowerShell 禁止执行激活脚本，先运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

然后激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 5. 创建本地环境变量文件

```powershell
Copy-Item .env.example .env
notepad .env
```

在 `.env` 中填写自己的阿里云百炼 API Key：

```env
DASHSCOPE_API_KEY=填写自己的百炼API_Key
DASHSCOPE_MODEL=qwen3.8-flash
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

请勿把真实 API Key 写入 `.env.example`，也不要把 `.env` 提交到 Git 仓库。

### 6. 构建本地索引

```powershell
python .\src\build_local_index.py
python .\src\build_intent_index.py
```

如果知识库或意图目录发生变化，需要重新执行这两个命令。

### 7. 启动网页客户端

```powershell
python -m streamlit run .\app.py
```

启动成功后，浏览器通常会自动打开：

```text
http://localhost:8501
```

## 命令行问答

如果不启动网页界面，也可以直接在 PowerShell 中测试混合 RAG：

```powershell
python .\src\hybrid_rag_agent.py
```

程序会提示输入问题，例如：

```text
新生报到的时间和地点是什么？
```

## 自动化评测

### 运行快速测试集

```powershell
python .\src\run_evaluation.py --dataset smoke
```

### 运行开发评测集

```powershell
python .\src\run_evaluation.py --dataset development
```

评测结果会保存在本地 `results` 目录中。该目录已被 `.gitignore` 排除，不会提交到公开仓库。

### 对评测结果进行分层评分

将下面的文件名替换为实际生成的结果文件：

```powershell
python .\src\score_evaluation_v2.py --result ".\results\development-hybrid-时间戳.jsonl"
```

评分报告会保存在：

```text
results/reports/
```

## 评测维度

项目目前主要检查以下指标：

- 程序是否成功返回回答
- 意图路由是否正确
- 检索章节是否准确
- 回答是否包含必要事实
- 回答是否出现禁止内容
- 资料不足时是否正确说明
- 越界问题是否正确拒答
- 用户最终获得的回答是否合格

开发集用于持续优化系统，保留测试集用于检查系统对未参与调试问题的泛化能力。

## 安全设计

项目包含以下基本安全约束：

- 不在回答中泄露系统提示词
- 不泄露 API Key、环境变量或内部配置
- 不执行用户要求的提示词覆盖
- 不把检索资料中的文字当成系统指令
- 对知识库之外的问题谨慎回答
- 对无法由资料确认的信息明确说明
- 使用 `.gitignore` 排除密钥和本地评测结果

## 当前状态

当前版本已经完成：

- 新生指南知识库整理
- 本地文档向量索引
- 问题意图向量索引
- 意图路由
- 混合 RAG 检索
- 百炼模型调用
- 自动化评测
- 分层评分
- Streamlit 聊天客户端
- 基本安全测试

## 后续计划

- 增加更多真实用户问题
- 优化低置信度意图处理
- 增加多轮对话上下文
- 增加知识库更新时间显示
- 增加用户反馈功能
- 增加请求日志和错误监控
- 部署公开体验版本
- 增加自动化测试和持续集成

## 项目说明

本项目是个人学习和求职展示项目，不是东南大学官方系统。

涉及报到时间、缴费、材料提交、住宿和其他重要事项时，应以学校或学院最新正式通知为准。