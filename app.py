"""东南大学新生指南智能问答客户端。"""

from __future__ import annotations

import base64
import html
import logging
import sys
import time
from pathlib import Path
from textwrap import dedent

import streamlit as st
from openai import OpenAI


# ==================================================
# 项目路径
# ==================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
LOGO_PATH = ROOT / "assets" / "seu-logo.png"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from hybrid_rag_agent import hybrid_retrieve
from intent_router import (
    load_intent_index,
    rank_intents,
    select_intents,
)
from local_rag_agent import (
    embed_question,
    generate_answer,
    load_index,
    load_settings,
)


LOGGER = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 500
MAX_QUESTIONS_PER_SESSION = 20
MIN_REQUEST_INTERVAL = 1.5


# ==================================================
# 页面配置
# ==================================================

st.set_page_config(
    page_title="东南大学新生指南助手",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(markup: str) -> None:
    """渲染经过整理的 HTML。"""
    st.markdown(
        dedent(markup).strip(),
        unsafe_allow_html=True,
    )


def image_to_data_uri(
    image_path: Path,
) -> str:
    """将本地图片转换为 Data URI。"""
    if not image_path.exists():
        return ""

    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }

    mime_type = mime_types.get(
        image_path.suffix.lower(),
        "image/png",
    )

    encoded = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    return (
        f"data:{mime_type};"
        f"base64,{encoded}"
    )


LOGO_DATA_URI = image_to_data_uri(
    LOGO_PATH
)

ASSISTANT_AVATAR = (
    str(LOGO_PATH)
    if LOGO_PATH.exists()
    else "🎓"
)


# ==================================================
# 页面样式
# ==================================================

render_html(
    """
    <style>
    :root {
        --seu-green: #0f5945;
        --seu-green-dark: #0a4033;
        --seu-green-soft: #eef6f2;
        --seu-gold: #c9a45c;
        --background: #ffffff;
        --sidebar: #f9f9f8;
        --border: #e7e7e4;
        --border-dark: #d2d3cf;
        --text: #202123;
        --muted: #70716d;
    }

    html,
    body,
    [class*="css"] {
        font-family:
            Inter,
            "Microsoft YaHei",
            "PingFang SC",
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    html,
    body {
        background: var(--background);
    }

    #MainMenu,
    footer,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none !important;
    }

    [data-testid="stHeader"] {
        height: 0;
        background: transparent;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--background);
    }

    [data-testid="stAppViewBlockContainer"],
    .block-container {
        max-width: 820px;
        padding-top: 0.8rem;
        padding-bottom: 8rem;
    }

    /* 左侧栏 */

    [data-testid="stSidebar"] {
        width: 272px !important;
        min-width: 272px !important;
        background: var(--sidebar);
        border-right: 1px solid #eeeeeb;
    }

    [data-testid="stSidebarContent"] {
        padding-top: 0.75rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin: 0.2rem 0 1.1rem;
        padding: 0 0.15rem;
    }

    .sidebar-logo-image {
        display: block;
        width: 40px;
        height: 40px;
        flex: 0 0 40px;
        object-fit: contain;
    }

    .sidebar-logo-fallback {
        display: flex;
        width: 38px;
        height: 38px;
        flex: 0 0 38px;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: var(--seu-green);
        color: white;
        font-size: 0.66rem;
        font-weight: 750;
        letter-spacing: 0.04rem;
    }

    .sidebar-brand-name {
        color: var(--text);
        font-size: 0.88rem;
        font-weight: 650;
        line-height: 1.3;
    }

    .sidebar-brand-subtitle {
        margin-top: 0.1rem;
        color: var(--muted);
        font-size: 0.66rem;
    }

    .sidebar-heading {
        margin: 1.35rem 0 0.5rem;
        padding-left: 0.35rem;
        color: #8a8b87;
        font-size: 0.66rem;
        font-weight: 600;
    }

    .history-item {
        overflow: hidden;
        margin-bottom: 0.12rem;
        padding: 0.56rem 0.65rem;
        border-radius: 8px;
        color: #454642;
        font-size: 0.76rem;
        line-height: 1.35;
        text-overflow: ellipsis;
        white-space: nowrap;
        transition: background 0.15s ease;
    }

    .history-item:hover {
        background: #eeeeeb;
    }

    .empty-history {
        padding: 0.5rem 0.65rem;
        color: #999a96;
        font-size: 0.73rem;
    }

    .status-card {
        margin-top: 0.3rem;
        padding: 0.72rem;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.75);
    }

    .status-line {
        display: flex;
        align-items: center;
        gap: 0.48rem;
        color: #4c514e;
        font-size: 0.73rem;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #24a36a;
        box-shadow:
            0 0 0 3px
            rgba(36, 163, 106, 0.11);
    }

    .status-description {
        margin-top: 0.5rem;
        color: #8b8c88;
        font-size: 0.65rem;
        line-height: 1.55;
    }

    [data-testid="stSidebar"]
    div.stButton > button {
        min-height: 2.55rem;
        justify-content: flex-start;
        border: 1px solid var(--border-dark);
        border-radius: 9px;
        background: white;
        color: #343532;
        font-size: 0.76rem;
        font-weight: 500;
        box-shadow: none;
    }

    [data-testid="stSidebar"]
    div.stButton > button:hover {
        border-color: #c5c5c0;
        background: #f1f1ee;
        color: #20211f;
    }

    /* 顶部栏 */

    .chat-header {
        position: sticky;
        z-index: 20;
        top: 0;
        display: flex;
        min-height: 54px;
        align-items: center;
        margin-bottom: 0.8rem;
        background:
            rgba(255, 255, 255, 0.94);
        border-bottom:
            1px solid var(--border);
        backdrop-filter: blur(12px);
    }

    .chat-header-title {
        color: #30312f;
        font-size: 0.86rem;
        font-weight: 600;
    }

    .chat-header-arrow {
        margin-left: 0.28rem;
        color: #858681;
        font-size: 0.68rem;
    }

    /* 欢迎页面 */

    .welcome-logo-wrapper {
        display: flex;
        min-height: 90px;
        align-items: center;
        justify-content: center;
        margin-top: 7vh;
    }

    .welcome-logo-image {
        display: block;
        width: 84px;
        height: 84px;
        object-fit: contain;
        filter:
            drop-shadow(
                0 8px 18px
                rgba(15, 89, 69, 0.12)
            );
    }

    .welcome-logo-fallback {
        display: flex;
        width: 64px;
        height: 64px;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: var(--seu-green);
        color: white;
        font-size: 0.75rem;
        font-weight: 750;
    }

    .welcome {
        max-width: 650px;
        margin: 1.15rem auto 2rem;
        text-align: center;
    }

    .welcome-title {
        margin: 0;
        color: var(--text);
        font-size: 1.62rem;
        font-weight: 650;
        letter-spacing: -0.035rem;
    }

    .welcome-description {
        max-width: 560px;
        margin: 0.7rem auto 0;
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.8;
    }

    .welcome-meta {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        margin-top: 0.95rem;
        color: var(--seu-green);
        font-size: 0.68rem;
        font-weight: 600;
    }

    .welcome-meta-dot {
        width: 4px;
        height: 4px;
        border-radius: 50%;
        background: var(--seu-gold);
    }

    /* 推荐问题按钮 */

    div.stButton > button {
        min-height: 3.1rem;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: white;
        color: #424340;
        font-size: 0.76rem;
        font-weight: 400;
        line-height: 1.4;
        box-shadow: none;
        transition:
            border-color 0.15s ease,
            background 0.15s ease,
            color 0.15s ease,
            transform 0.15s ease;
    }

    div.stButton > button:hover {
        border-color: #c9cec9;
        background: #fafbf9;
        color: var(--seu-green);
        transform: translateY(-1px);
    }

    /* 对话消息 */

    [data-testid="stChatMessage"] {
        max-width: 100%;
        margin: 0;
        padding: 1.3rem 0.25rem;
        border: none;
        background: transparent;
    }

    [data-testid="stChatMessage"] p {
        color: #292a28;
        font-size: 0.9rem;
        line-height: 1.85;
    }

    [data-testid="stChatMessageAvatarUser"] {
        background: #eeeeeb;
        color: #4d4e4a;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        overflow: hidden;
        border: 1px solid #e2e2de;
        background: white;
    }

    [data-testid="stChatMessageAvatarAssistant"]
    img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    [data-testid="stChatMessageContent"] {
        padding-top: 0.05rem;
    }

    [data-testid="stChatMessageContent"] ul,
    [data-testid="stChatMessageContent"] ol {
        padding-left: 1.35rem;
    }

    [data-testid="stChatMessageContent"] li {
        margin-bottom: 0.35rem;
    }

    [data-testid="stChatMessageContent"] code {
        border-radius: 5px;
        background: #f2f3f1;
        color: #34443e;
    }

    /* 输入框 */

    [data-testid="stChatInput"] {
        border: 1px solid #d2d3cf;
        border-radius: 25px;
        background: white;
        box-shadow:
            0 7px 24px
            rgba(0, 0, 0, 0.07),
            0 1px 3px
            rgba(0, 0, 0, 0.04);
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #aeb8b2;
        box-shadow:
            0 7px 28px
            rgba(15, 89, 69, 0.09),
            0 1px 3px
            rgba(0, 0, 0, 0.04);
    }

    [data-testid="stChatInput"] textarea {
        min-height: 28px;
        color: #2b2c29;
        font-size: 0.86rem;
    }

    [data-testid="stChatInput"] button {
        color: var(--seu-green);
    }

    @media (max-width: 760px) {
        [data-testid="stAppViewBlockContainer"],
        .block-container {
            padding-right: 1rem;
            padding-left: 1rem;
        }

        .welcome-logo-wrapper {
            margin-top: 4vh;
        }

        .welcome-title {
            font-size: 1.4rem;
        }

        .welcome-description {
            font-size: 0.8rem;
        }
    }
    </style>
    """
)


# ==================================================
# Agent
# ==================================================

@st.cache_resource
def load_agent() -> dict:
    """加载模型客户端与本地索引。"""
    api_key, model, base_url = (
        load_settings()
    )

    document_index = load_index()
    intent_index = load_intent_index()

    document_metadata = (
        document_index["metadata"]
    )

    intent_metadata = (
        intent_index["metadata"]
    )

    if (
        document_metadata["embedding_model"]
        != intent_metadata["embedding_model"]
        or document_metadata["dimensions"]
        != intent_metadata["dimensions"]
    ):
        raise RuntimeError(
            "知识索引与意图索引的"
            "向量配置不一致。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    return {
        "client": client,
        "model": model,
        "document_index": document_index,
        "intent_index": intent_index,
    }


def ask_agent(
    question: str,
) -> str:
    """执行意图识别、检索和回答生成。"""
    resources = load_agent()

    client = resources["client"]
    model = resources["model"]

    document_index = (
        resources["document_index"]
    )

    intent_index = (
        resources["intent_index"]
    )

    metadata = document_index["metadata"]

    query_embedding = embed_question(
        client=client,
        embedding_model=(
            metadata["embedding_model"]
        ),
        dimensions=metadata["dimensions"],
        question=question,
    )

    ranked_intents = rank_intents(
        question=question,
        query_embedding=query_embedding,
        entries=intent_index["entries"],
    )

    selected_intents = select_intents(
        ranked_intents
    )

    retrieved_chunks = hybrid_retrieve(
        query_embedding=query_embedding,
        chunks=document_index["chunks"],
        selected_intents=selected_intents,
    )

    return generate_answer(
        client=client,
        model=model,
        question=question,
        retrieved_chunks=retrieved_chunks,
    )


def validate_question(
    question: str,
) -> str | None:
    """检查用户输入。"""
    if not question.strip():
        return "请输入一个具体问题。"

    if len(question) > MAX_QUESTION_LENGTH:
        return (
            f"问题有点长，请控制在 "
            f"{MAX_QUESTION_LENGTH} 个字符以内。"
        )

    question_count = sum(
        1
        for message in st.session_state.messages
        if message["role"] == "user"
    )

    if (
        question_count
        >= MAX_QUESTIONS_PER_SESSION
    ):
        return (
            "这次对话已经达到 20 个问题啦。"
            "你可以点击左侧“新建对话”继续提问。"
        )

    elapsed = (
        time.time()
        - st.session_state.last_request_time
    )

    if elapsed < MIN_REQUEST_INTERVAL:
        return "提问得有点快，请稍等一下再试。"

    return None


# ==================================================
# 初始化
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0.0


try:
    resources = load_agent()

except Exception:
    LOGGER.exception(
        "Agent initialization failed"
    )

    st.error(
        "问答服务暂时没有准备好，"
        "请稍后刷新页面再试。"
    )

    st.stop()


# ==================================================
# 侧边栏
# ==================================================

with st.sidebar:
    if LOGO_DATA_URI:
        sidebar_logo = (
            f'<img class="sidebar-logo-image" '
            f'src="{LOGO_DATA_URI}" '
            f'alt="东南大学校徽">'
        )
    else:
        sidebar_logo = (
            '<div class="sidebar-logo-fallback">'
            'SEU'
            '</div>'
        )

    render_html(
        f"""
        <div class="sidebar-brand">
        {sidebar_logo}
        <div>
        <div class="sidebar-brand-name">新生指南助手</div>
        <div class="sidebar-brand-subtitle">Southeast University</div>
        </div>
        </div>
        """
    )

    if st.button(
        "＋　新建对话",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.last_request_time = 0.0
        st.rerun()

    render_html(
        """
        <div class="sidebar-heading">最近对话</div>
        """
    )

    user_questions = [
        message["content"]
        for message in st.session_state.messages
        if message["role"] == "user"
    ]

    if user_questions:
        for item in reversed(
            user_questions[-7:]
        ):
            safe_item = html.escape(item)

            if len(safe_item) > 25:
                safe_item = (
                    safe_item[:25] + "…"
                )

            render_html(
                f"""
                <div class="history-item">{safe_item}</div>
                """
            )
    else:
        render_html(
            """
            <div class="empty-history">暂无历史对话</div>
            """
        )

    render_html(
        """
        <div class="sidebar-heading">服务状态</div>
        """
    )

    chunk_count = len(
        resources["document_index"]["chunks"]
    )

    render_html(
        f"""
        <div class="status-card">
        <div class="status-line">
        <span class="status-dot"></span>
        知识库已连接
        </div>
        <div class="status-description">
        已加载 {chunk_count} 个指南知识片段<br>
        当前对话最多可提问 {MAX_QUESTIONS_PER_SESSION} 次
        </div>
        </div>
        """
    )


# ==================================================
# 顶部
# ==================================================

render_html(
    """
    <div class="chat-header">
    <div class="chat-header-title">
    新生指南助手
    <span class="chat-header-arrow">⌄</span>
    </div>
    </div>
    """
)


quick_question = None


# ==================================================
# 欢迎页面
# ==================================================

if not st.session_state.messages:
    if LOGO_DATA_URI:
        welcome_logo = (
            f'<img class="welcome-logo-image" '
            f'src="{LOGO_DATA_URI}" '
            f'alt="东南大学校徽">'
        )
    else:
        welcome_logo = (
            '<div class="welcome-logo-fallback">'
            'SEU'
            '</div>'
        )

    render_html(
        f"""
        <div class="welcome-logo-wrapper">{welcome_logo}</div>
        <div class="welcome">
        <h1 class="welcome-title">有什么可以帮你的？</h1>
        <div class="welcome-description">我会查阅学院新生指南，帮你了解报到、交通、住宿、缴费、材料准备和校园生活。</div>
        <div class="welcome-meta">
        <span>2026级研究生新生指南</span>
        <span class="welcome-meta-dot"></span>
        <span>智能问答</span>
        </div>
        </div>
        """
    )

    quick_questions = [
        "新生报到的时间和地点是什么？",
        "从南京禄口机场到江北校区怎么走？",
        "报到当天需要携带哪些材料？",
        "宿舍什么时候可以查询和入住？",
    ]

    left_column, right_column = (
        st.columns(2)
    )

    for index, item in enumerate(
        quick_questions
    ):
        target_column = (
            left_column
            if index % 2 == 0
            else right_column
        )

        with target_column:
            if st.button(
                item,
                key=(
                    f"quick_question_{index}"
                ),
                use_container_width=True,
            ):
                quick_question = item


# ==================================================
# 历史对话
# ==================================================

for message in st.session_state.messages:
    avatar = (
        "👤"
        if message["role"] == "user"
        else ASSISTANT_AVATAR
    )

    with st.chat_message(
        message["role"],
        avatar=avatar,
    ):
        st.markdown(
            message["content"]
        )


# ==================================================
# 用户输入
# ==================================================

typed_question = st.chat_input(
    "给新生指南助手发送消息",
    max_chars=MAX_QUESTION_LENGTH,
)

question = (
    typed_question
    or quick_question
)


# ==================================================
# 生成回答
# ==================================================

if question:
    question = question.strip()

    validation_error = validate_question(
        question
    )

    if validation_error:
        st.warning(
            validation_error
        )

    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user",
            avatar="👤",
        ):
            st.markdown(
                question
            )

        with st.chat_message(
            "assistant",
            avatar=ASSISTANT_AVATAR,
        ):
            try:
                st.session_state.last_request_time = (
                    time.time()
                )

                with st.spinner(
                    "正在帮你查阅新生指南……"
                ):
                    answer = ask_agent(
                        question
                    )

                st.markdown(
                    answer
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception:
                LOGGER.exception(
                    "Question answering failed"
                )

                st.error(
                    "这次查询没有成功，"
                    "稍后再试一次吧。"
                )