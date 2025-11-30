# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个旅游景点智能问答系统，实现了三种不同的问答方法：
1. **基于规则的问答** (Rule-based QA) - 关键词匹配 + 词典实体识别
2. **基于文本分类的问答** (Text Classification QA) - TF-IDF + LinearSVC 意图分类
3. **基于大模型 RAG 的问答** (LLM RAG QA) - 检索增强生成

## 常用命令

### 运行系统
```bash
# 启动 Streamlit Web 应用（主入口）
streamlit run main.py

# 默认会在 http://localhost:8501 启动
```

### 测试单个模块
```bash
# 测试基于规则的问答
python code/rule_based_qa.py

# 测试基于文本分类的问答（会自动训练模型）
python code/text_classification_qa.py

# 测试基于 LLM RAG 的问答
python code/llm_rag_qa.py
```

### 数据预处理
```bash
# 运行数据预处理脚本（如果需要重新处理原始数据）
python data_preprocessing.py

# 修复银川景点名称（特定数据修复脚本）
python fix_yinchuan_names.py
```

### 调试检索功能
```bash
# 调试 RAG 检索功能
python debug_retrieval.py
```

## 核心架构

### 三层问答架构

系统采用**三种独立的问答方法**，每种方法都是完整的端到端实现：

```
用户问题 → [方法选择] → 问答处理 → 生成回答
                ↓
        ┌───────┴────────┐
        │                │
   规则匹配        文本分类        RAG检索
   (关键词)        (机器学习)      (向量+LLM)
```

#### 1. 基于规则的问答 (`code/rule_based_qa.py`)

**核心流程：**
```
用户问题 → 意图识别(关键词匹配) → 实体识别(词典匹配) → 槽位填充 → 模板生成回答
```

**关键组件：**
- `意图关键词映射表`: 12种意图的关键词列表
- `识别意图()`: 基于关键词匹配统计识别意图
- `识别实体()`: 从知识库构建词典，匹配景点/城市/省份
- `填充槽位()`: 根据意图提取对应字段
- `生成回答()`: 使用预定义模板生成回答

**主函数：** `rule_based_qa(question, kb) -> Dict`

#### 2. 基于文本分类的问答 (`code/text_classification_qa.py`)

**核心流程：**
```
用户问题 → 中文分词 → TF-IDF向量化 → LinearSVC分类 → 意图预测 → 知识库查询 → 生成回答
```

**关键组件：**
- `训练数据集`: 12种意图，每种15+条训练样本
- `意图分类器类`: 封装 TF-IDF + LinearSVC 的训练和预测
- `中文分词()`: 使用 jieba 分词
- `predict_intent()`: 全局预测接口
- 模型持久化: 训练后保存到 `models/intent_classifier.pkl`

**主函数：** `text_classification_qa(question, kb) -> Dict`

**模型训练：**
- 首次运行时自动训练并保存模型
- 训练集/测试集 = 80%/20%
- 使用 stratified split 保持类别平衡

#### 3. 基于 LLM RAG 的问答 (`code/llm_rag_qa.py`)

**核心流程：**
```
用户问题 → 向量化 → Top-K检索(混合策略) → 构建Prompt → LLM生成 → 最终回答
                                ↓
                        TF-IDF相似度 + 实体加权
```

**关键组件：**
- `向量检索器类`: 基于 TF-IDF 的向量检索（支持实体感知加权）
- `大模型客户端类`: 支持多种 LLM API（Ollama、OpenAI、硅基流动等）
- `构建景点文档()`: 将 CSV 数据转换为标准文档格式
- `构建Prompt()`: 组合检索结果和用户问题
- `基于模板生成回答()`: 备用方案（当 LLM 不可用时）

**主函数：** `llm_rag_qa(question, kb, use_llm=True) -> Dict`

**混合检索策略：**
- 基础分数：TF-IDF 余弦相似度
- 实体加权：识别城市(+0.5)、省份(+0.3)、景点名(+0.8)
- 综合分数 = 0.3 × TF-IDF + 0.7 × 实体加权（有实体时）

**支持的 LLM：**
- `siliconflow`: 硅基流动（推荐，国内访问稳定）
- `ollama`: 本地部署
- `openai`: OpenAI API
- `qianwen`: 通义千问

### 数据流

```
data/merged_attractions.csv (知识库)
         ↓
    [加载到内存]
         ↓
    pd.DataFrame
         ↓
    ┌────┴────┐
    │         │
规则方法   分类方法   RAG方法
    │         │         │
词典构建  实体提取  文档构建+向量化
    │         │         │
    └────┬────┘         │
         ↓              ↓
    知识库查询      向量检索
         ↓              ↓
    模板回答      LLM生成回答
```

### 主界面 (`main.py`)

**Streamlit 应用结构：**
- 侧边栏：方法选择、RAG 配置、数据统计
- 主内容区：问题输入、示例问题、结果展示
- 结果展示：4个标签页（最终回答、意图识别、实体识别、详细信息）

**关键函数：**
- `加载知识库()`: 使用 `@st.cache_data` 缓存知识库
- `渲染卡片()`: 统一的卡片组件渲染
- `main()`: 主界面逻辑

## 数据结构

### 知识库字段 (`data/merged_attractions.csv`)

| 字段 | 说明 | 用途 |
|------|------|------|
| `dataid` | 数据ID | 唯一标识 |
| `name` | 景点名称 | 实体识别、检索 |
| `city` | 所在城市 | 实体识别、城市推荐 |
| `province` | 所在省份 | 地理信息 |
| `address` | 详细地址 | 位置查询 |
| `phone` | 联系电话 | 联系方式查询 |
| `website` | 官方网站 | 网站查询 |
| `rating` | 用户评分 | 排序、推荐 |
| `ticket_price` | 门票价格 | 价格查询 |
| `open_time` | 开放时间 | 时间查询 |
| `suggest_time` | 建议游玩时间 | 行程规划 |
| `tags` | 景点标签 | 分类、推荐 |
| `description` | 景点描述 | 简介查询 |
| `tips` | 游玩贴士 | 攻略查询 |

### 问答返回格式

所有三种方法都返回统一的字典格式：

```python
{
    "intent": str,              # 识别到的意图标签
    "intent_confidence": float, # 意图置信度 (0-1)
    "entities": {               # 识别到的实体
        "景点": List[str],
        "城市": List[str],
        "省份": List[str]
    },
    "answer": str,              # 最终回答
    "matched_attraction": str,  # 匹配到的景点名称
    "slots": Dict,              # 槽位值（规则/分类方法）
    "retrieved_docs": List,     # 检索文档（RAG方法）
    "prompt": str,              # LLM Prompt（RAG方法）
    "use_llm": bool            # 是否使用了LLM（RAG方法）
}
```

## 意图标签体系

系统支持 12 种意图类别：

| 意图标签 | 中文描述 | 示例问题 |
|---------|---------|---------|
| `ASK_TICKET` | 询问门票价格 | "门票多少钱？" |
| `ASK_OPEN_TIME` | 询问开放时间 | "几点开门？" |
| `ASK_INTRO` | 询问景点简介 | "介绍一下这个景点" |
| `ASK_ADDRESS` | 询问地址位置 | "在哪里？" |
| `ASK_PHONE` | 询问联系电话 | "电话是多少？" |
| `ASK_WEBSITE` | 询问官方网站 | "官网是什么？" |
| `ASK_RATING` | 询问评分评价 | "评分多少？" |
| `ASK_SUGGEST_TIME` | 询问建议游玩时间 | "玩多久合适？" |
| `ASK_TAGS` | 询问景点标签 | "是什么类型的景点？" |
| `ASK_TIPS` | 询问游玩贴士 | "有什么注意事项？" |
| `ASK_LOCATION` | 询问所在城市/省份 | "在哪个城市？" |
| `ASK_RECOMMEND` | 景点推荐 | "有什么好玩的？" |

## 重要实现细节

### 1. 实体识别策略

**规则方法和分类方法：**
- 从知识库动态构建实体词典
- 使用字符串包含匹配（`景点名 in 问题`）
- 支持模糊匹配（部分匹配）

**RAG 方法：**
- 在检索阶段进行实体感知加权
- 城市匹配权重最高（解决"银川有什么好玩的"类问题）
- 综合 TF-IDF 相似度和实体匹配分数

### 2. 模型持久化

**文本分类模型：**
- 首次运行时自动训练
- 保存到 `models/intent_classifier.pkl`
- 后续运行直接加载，无需重新训练

**RAG 向量索引：**
- 每次启动时重新构建（基于 TF-IDF）
- 未使用持久化向量数据库（简化部署）

### 3. 配置管理

**RAG 配置更新：**
```python
from code.llm_rag_qa import 更新RAG配置

更新RAG配置(
    模型类型="siliconflow",
    模型名称="Qwen/Qwen2.5-7B-Instruct",
    API地址="https://api.siliconflow.cn/v1/chat/completions",
    API密钥="your-api-key"
)
```

### 4. 错误处理

**LLM 调用失败时：**
- 自动回退到模板生成方法
- 基于关键词匹配提取文档字段
- 确保系统始终能返回有效回答

**知识库查询失败时：**
- 规则方法：返回引导性提示
- 分类方法：提供示例问题
- RAG 方法：返回"未找到相关信息"

## 开发注意事项

### 添加新意图

1. 在 `rule_based_qa.py` 中添加关键词映射
2. 在 `text_classification_qa.py` 中添加训练样本（至少10条）
3. 在 `意图槽位映射表` 中定义需要的字段
4. 在 `回答模板库` 中添加回答模板
5. 删除 `models/intent_classifier.pkl` 重新训练

### 修改检索策略

RAG 检索的核心逻辑在 `向量检索器.检索()` 方法中：
- 调整实体加权分数（第274-302行）
- 修改综合分数计算公式（第306-311行）
- 调整相似度阈值（`RAG配置.相似度阈值`）

### 更换 LLM 提供商

在 `大模型客户端` 类中添加新的调用方法：
1. 实现 `调用XXX()` 方法
2. 在 `生成回答()` 中添加分支
3. 更新 `RAG配置` 的默认值

### 自定义 UI 样式

主界面样式在 `main.py` 的 `自定义样式` 变量中（第47-258行）：
- 使用 CSS 定义卡片、按钮、输入框等样式
- 紫色主题色系：`#667eea`, `#764ba2`, `#8b5cf6`

## 依赖说明

核心依赖（`requirements.txt`）：
- `streamlit`: Web 界面框架
- `pandas`: 数据处理
- `scikit-learn`: 机器学习（TF-IDF + SVC）
- `jieba`: 中文分词
- `requests`: HTTP 请求（LLM API 调用）
- `ollama`: 本地大模型客户端（可选）

## 性能特点

| 方法 | 响应时间 | 准确率 | 适用场景 |
|------|---------|--------|---------|
| 规则方法 | <100ms | 高（限定域） | 固定模式问题 |
| 分类方法 | 100-500ms | 中高 | 常见意图识别 |
| RAG 方法 | 1-10s | 高 | 复杂问题、自然回答 |

## 团队信息

- **组长**: 王思琪
- **组员**: 禹红倩、马成龙、张楠、罗应萍
- **项目**: 自然语言处理大作业 - 2025
