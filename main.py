# -*- coding: utf-8 -*-
"""
===============================================================================
旅游景点智能问答系统 - Streamlit 主界面
===============================================================================
作者: 王思琪团队
描述: 本模块是旅游景点智能问答系统的主入口，整合了三种问答方法：
      1. 基于规则的问答（Rule-based QA）
      2. 基于文本分类的问答（Text Classification QA）
      3. 基于大模型RAG的问答（LLM + Retrieval QA）
      
系统流程:
      用户输入问题 -> 选择问答方法 -> 意图识别/实体识别 -> 知识库检索 -> 生成回答
===============================================================================
"""

import streamlit as st
import pandas as pd
import os
import sys
import time

# 添加代码目录到路径
当前文件目录 = os.path.dirname(os.path.abspath(__file__))
代码目录 = os.path.join(当前文件目录, "code")
if 代码目录 not in sys.path:
    sys.path.insert(0, 代码目录)

# 导入三种问答方法
from rule_based_qa import rule_based_qa
from text_classification_qa import text_classification_qa, 意图标签描述
from llm_rag_qa import llm_rag_qa, 更新RAG配置

# ============================================================================
# 页面配置与样式
# ============================================================================

# 设置页面配置
st.set_page_config(
    page_title="旅游景点智能问答系统",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式 - 现代简洁风格，紫色主题
自定义样式 = """
<style>
/* 全局字体和背景 */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans SC', sans-serif;
}

/* 主背景 - 简约白色 */
.stApp {
    background: #ffffff;
}

/* 主内容区域 */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* 侧边栏样式 - 浅紫色主题，提高可读性 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f3e8ff 0%, #e9d5ff 100%);
}

[data-testid="stSidebar"] .stMarkdown {
    color: #1f2937;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: #5b21b6 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #374151 !important;
}

/* 卡片样式 */
.card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
}

/* 卡片标题 */
.card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #5b21b6;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e9d5ff;
}

/* 卡片内容 */
.card-content {
    font-size: 0.95rem;
    color: #374151;
    line-height: 1.7;
}

/* 大标题样式 */
.main-title {
    text-align: center;
    padding: 2rem 0;
    background: #f8fafc;
    border-radius: 20px;
    margin-bottom: 2rem;
    border: 1px solid #e2e8f0;
}

.main-title h1 {
    color: #5b21b6;
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
}

.main-title p {
    color: #6b7280;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* 输入框样式 */
.stTextInput > div > div > input {
    border-radius: 12px;
    border: 2px solid #e9d5ff;
    padding: 0.8rem 1rem;
    font-size: 1rem;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #8b5cf6;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
}

/* 按钮样式 */
.stButton > button {
    background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
}

/* 选择框样式 */
.stSelectbox > div > div {
    border-radius: 12px;
}

/* 标签页样式 */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #f3f4f6;
    padding: 0.5rem;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1rem;
    background: transparent;
    color: #374151;
}

.stTabs [aria-selected="true"] {
    background: #ffffff;
    color: #5b21b6;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* 展开器样式 */
.streamlit-expanderHeader {
    background: rgba(139, 92, 246, 0.1);
    border-radius: 12px;
    font-weight: 600;
}

/* 进度条样式 */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #8b5cf6, #6d28d9);
}

/* 成功/信息/警告提示样式 */
.stSuccess, .stInfo, .stWarning {
    border-radius: 12px;
}

/* 分隔线 */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    margin: 1.5rem 0;
}

/* 方法介绍卡片 */
.method-card {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(109, 40, 217, 0.1) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* 结果高亮 */
.highlight {
    background: linear-gradient(120deg, #fef3c7 0%, #fde68a 100%);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
}

/* 检索文档卡片 */
.doc-card {
    background: #f8fafc;
    border-left: 4px solid #8b5cf6;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
}

/* 隐藏Streamlit默认元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

# 注入自定义样式
st.markdown(自定义样式, unsafe_allow_html=True)


# ============================================================================
# 辅助函数
# ============================================================================

@st.cache_data
def 加载知识库():
    """
    加载景点知识库数据
    使用缓存避免重复加载
    """
    数据路径 = os.path.join(当前文件目录, "data", "merged_attractions.csv")
    try:
        知识库 = pd.read_csv(数据路径)
        return 知识库, None
    except Exception as e:
        return None, str(e)


def 渲染卡片(标题: str, 内容: str, 图标: str = ""):
    """
    渲染一个美观的卡片组件
    
    参数:
        标题: 卡片标题
        内容: 卡片内容
        图标: 可选的图标（不使用emoji）
    """
    卡片HTML = f"""
    <div class="card">
        <div class="card-title">{图标} {标题}</div>
        <div class="card-content">{内容}</div>
    </div>
    """
    st.markdown(卡片HTML, unsafe_allow_html=True)


def 渲染检索文档(文档列表: list):
    """
    渲染检索到的文档列表
    """
    if not 文档列表:
        st.info("未检索到相关文档")
        return
    
    for i, 文档 in enumerate(文档列表, 1):
        相似度 = 文档.get('score', 0) * 100
        with st.expander(f"文档 {i}: {文档.get('name', '未知景点')} (相似度: {相似度:.1f}%)"):
            st.markdown(f"""
            <div class="doc-card">
                <strong>景点名称:</strong> {文档.get('name', '未知')}<br>
                <strong>所在城市:</strong> {文档.get('city', '未知')}<br>
                <strong>相似度得分:</strong> {相似度:.2f}%
            </div>
            """, unsafe_allow_html=True)
            # 显示部分文档内容
            内容 = 文档.get('content', '')
            if len(内容) > 500:
                内容 = 内容[:500] + "..."
            st.text(内容)


# ============================================================================
# 主界面布局
# ============================================================================

def main():
    """主函数 - 构建Streamlit界面"""
    
    # 加载知识库
    知识库, 错误信息 = 加载知识库()
    
    if 错误信息:
        st.error(f"加载知识库失败: {错误信息}")
        st.stop()
    
    # ========== 侧边栏 ==========
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #5b21b6; margin: 0;">问答方法选择</h2>
            <p style="color: #6b7280; font-size: 0.9rem;">选择您想使用的问答模式</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 方法选择
        问答方法 = st.radio(
            "选择问答方法",
            ["基于规则问答 (Rule-based)", 
             "文本分类问答 (Classification)", 
             "大模型RAG问答 (LLM-RAG)"],
            index=0,
            help="选择不同的问答方法体验不同的技术实现"
        )
        
        st.markdown("---")
        
        # 方法介绍
        st.markdown("""
        <div style="color: #5b21b6;">
            <h4>方法介绍</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if "规则" in 问答方法:
            st.markdown("""
            <div class="method-card" style="background: rgba(139,92,246,0.15); color: #374151; border: 1px solid #c4b5fd;">
                <strong style="color: #5b21b6;">基于规则的问答</strong><br>
                通过关键词匹配识别意图，使用词典匹配识别实体，根据预定义模板生成回答。
                <br><br>
                <em style="color: #6b7280;">优点：响应快、可控性强</em>
            </div>
            """, unsafe_allow_html=True)
        elif "分类" in 问答方法:
            st.markdown("""
            <div class="method-card" style="background: rgba(139,92,246,0.15); color: #374151; border: 1px solid #c4b5fd;">
                <strong style="color: #5b21b6;">基于文本分类的问答</strong><br>
                使用TF-IDF特征提取+LinearSVC分类器识别用户意图，再结合知识库生成回答。
                <br><br>
                <em style="color: #6b7280;">优点：泛化能力强、准确率高</em>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="method-card" style="background: rgba(139,92,246,0.15); color: #374151; border: 1px solid #c4b5fd;">
                <strong style="color: #5b21b6;">基于LLM RAG的问答</strong><br>
                检索相关文档，构建Prompt，利用大语言模型生成自然流畅的回答。
                <br><br>
                <em style="color: #6b7280;">优点：回答自然、理解深入</em>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # RAG配置（仅在选择RAG时显示）
        if "RAG" in 问答方法:
            st.markdown("""
            <div style="color: #5b21b6;">
                <h4>RAG 配置</h4>
            </div>
            """, unsafe_allow_html=True)

            # 大模型RAG问答必须使用大模型，默认启用
            使用大模型 = True

            模型类型 = st.selectbox(
                "选择模型类型",
                ["siliconflow", "ollama", "openai", "qianwen"],
                index=0,
                help="推荐使用硅基流动(SiliconFlow)，国内访问稳定"
            )

            # 根据模型类型设置默认模型名和API地址
            if 模型类型 == "siliconflow":
                默认模型 = "deepseek-ai/DeepSeek-V3.2-Exp"
                默认API地址 = "https://api.siliconflow.cn/v1/chat/completions"
                模型帮助 = "硅基流动支持: deepseek-ai/DeepSeek-V3.2-Exp 等"
            elif 模型类型 == "ollama":
                默认模型 = "qwen2.5:7b"
                默认API地址 = "http://localhost:11434/api/generate"
                模型帮助 = "Ollama本地模型: qwen2.5:7b, llama3, granite4:tiny-h 等"
            elif 模型类型 == "openai":
                默认模型 = "gpt-3.5-turbo"
                默认API地址 = "https://api.openai.com/v1/chat/completions"
                模型帮助 = "OpenAI模型"
            else:  # qianwen
                默认模型 = "qwen-turbo"
                默认API地址 = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
                模型帮助 = "通义千问模型"

            模型名称 = st.text_input(
                "模型名称",
                value=默认模型,
                help=模型帮助
            )

            if 模型类型 != "ollama":
                API密钥 = st.text_input(
                    "API密钥",
                    type="password",
                    help="硅基流动API密钥请从 https://cloud.siliconflow.cn 获取"
                )
            else:
                API密钥 = ""

            # 更新配置（包含API地址）
            更新RAG配置(
                模型类型=模型类型,
                模型名称=模型名称,
                API地址=默认API地址,
                API密钥=API密钥
            )

            检索数量 = st.slider("检索文档数量", 1, 5, 3)
        else:
            使用大模型 = False
            检索数量 = 3
        
        st.markdown("---")
        
        # 数据统计
        st.markdown("""
        <div style="color: #5b21b6;">
            <h4>数据统计</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("景点总数", len(知识库))
        st.metric("覆盖城市", 知识库['city'].nunique())
        st.metric("覆盖省份", 知识库['province'].nunique())
    
    # ========== 主内容区 ==========
    
    # 大标题
    st.markdown("""
    <div class="main-title">
        <h1>旅游景点智能问答系统</h1>
        <p>Tourism Attraction Intelligent Q&A System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 输入区域
    # 示例问题点击回调 - 直接修改输入框的key对应的session_state
    def 设置示例问题(问题文本):
        st.session_state["user_question_input"] = 问题文本
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        用户问题 = st.text_input(
            "请输入您的问题",
            placeholder="例如：上海迪士尼的门票多少钱？/ 外滩什么时候开放？/ 给我介绍一下东方明珠",
            key="user_question_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        提交按钮 = st.button("提问", use_container_width=True)
    
    # 示例问题
    with st.expander("点击查看示例问题"):
        示例列表 = [
            "上海迪士尼的门票多少钱？",
            "外滩的开放时间是什么时候？",
            "给我介绍一下东方明珠",
            "豫园在哪里？怎么去？",
            "上海有什么好玩的景点推荐？",
            "朱家角古镇玩多久合适？",
            "上海博物馆的联系电话是多少？"
        ]
        
        cols = st.columns(3)
        for i, 示例 in enumerate(示例列表):
            with cols[i % 3]:
                st.button(示例, key=f"example_{i}", on_click=设置示例问题, args=(示例,))
    
    st.markdown("---")
    
    # 处理问答
    if 提交按钮 and 用户问题:
        # 显示加载动画
        with st.spinner("正在分析您的问题..."):
            time.sleep(0.5)  # 短暂延迟以显示动画
            
            # 根据选择的方法执行问答
            if "规则" in 问答方法:
                结果 = rule_based_qa(用户问题, 知识库)
                方法名称 = "基于规则的问答"
            elif "分类" in 问答方法:
                结果 = text_classification_qa(用户问题, 知识库)
                方法名称 = "基于文本分类的问答"
            else:
                结果 = llm_rag_qa(用户问题, 知识库, use_llm=使用大模型)
                方法名称 = "基于LLM RAG的问答"
        
        # 显示结果
        st.success(f"使用方法: {方法名称}")
        
        # 使用标签页展示不同内容
        tab1, tab2, tab3, tab4 = st.tabs(["最终回答", "意图识别", "实体识别", "详细信息"])
        
        with tab1:
            # 最终回答卡片
            回答内容 = 结果.get('answer', '未能生成回答')
            渲染卡片("系统回答", 回答内容.replace('\n', '<br>'))
        
        with tab2:
            # 意图识别结果
            意图 = 结果.get('intent', 'UNKNOWN')
            置信度 = 结果.get('intent_confidence', 0)
            意图描述 = 结果.get('intent_description', 意图标签描述.get(意图, '未知意图'))
            
            col1, col2 = st.columns(2)
            with col1:
                渲染卡片("识别到的意图", f"""
                    <strong>意图标签:</strong> {意图}<br>
                    <strong>意图描述:</strong> {意图描述}<br>
                    <strong>置信度:</strong> {置信度:.2%}
                """)
            
            with col2:
                # 置信度可视化
                st.markdown("""
                <div class="card">
                    <div class="card-title">置信度指示</div>
                    <div class="card-content">
                """, unsafe_allow_html=True)
                st.progress(min(置信度, 1.0))
                st.markdown("</div></div>", unsafe_allow_html=True)
        
        with tab3:
            # 实体识别结果
            实体 = 结果.get('entities', {})
            匹配景点 = 结果.get('matched_attraction', None)
            
            渲染卡片("识别到的实体", f"""
                <strong>景点实体:</strong> {', '.join(实体.get('景点', [])) or '未识别到'}<br>
                <strong>城市实体:</strong> {', '.join(实体.get('城市', [])) or '未识别到'}<br>
                <strong>省份实体:</strong> {', '.join(实体.get('省份', [])) or '未识别到'}<br>
                <strong>匹配景点:</strong> {匹配景点 or '未匹配'}
            """)
        
        with tab4:
            # 详细信息（根据方法不同显示不同内容）
            if "RAG" in 问答方法:
                st.markdown("### 检索到的文档")
                检索文档 = 结果.get('retrieved_docs', [])
                渲染检索文档(检索文档)
                
                # 显示Prompt（可折叠）
                if 结果.get('prompt'):
                    with st.expander("查看发送给大模型的Prompt"):
                        st.code(结果.get('prompt', ''), language='text')
                
                st.info(f"是否使用大模型: {'是' if 结果.get('use_llm') else '否（使用模板回答）'}")
            else:
                # 显示槽位值
                槽位值 = 结果.get('slots', {})
                if 槽位值:
                    st.markdown("### 槽位填充结果")
                    槽位文本 = "<br>".join([f"<strong>{k}:</strong> {v}" for k, v in 槽位值.items()])
                    渲染卡片("槽位信息", 槽位文本)
    
    elif 提交按钮 and not 用户问题:
        st.warning("请输入您想咨询的问题")
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.7); padding: 1rem;">
        <p>旅游景点智能问答系统 v1.0 | 王思琪团队开发</p>
        <p style="font-size: 0.8rem;">自然语言处理大作业 - 2025</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    main()
