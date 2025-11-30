# 旅游景点智能问答系统

## 项目概述

本项目是一个基于自然语言处理技术的旅游景点智能问答系统，实现了三种不同的问答方法：
1. **基于规则的问答（Rule-based QA）**
2. **基于文本分类的问答（Text Classification QA）**
3. **基于大模型RAG的问答（LLM + Retrieval-Augmented Generation）**

## 项目结构

```
wsq/
├── main.py                    # Streamlit主程序入口
├── requirements.txt           # 项目依赖
├── README.md                  # 项目说明文档
├── code/                      # 源代码目录
│   ├── rule_based_qa.py       # 基于规则的问答实现
│   ├── text_classification_qa.py  # 基于文本分类的问答实现
│   └── llm_rag_qa.py          # 基于LLM RAG的问答实现
├── data/                      # 数据目录
│   └── merged_attractions.csv # 景点数据集
├── models/                    # 模型存储目录
│   └── intent_classifier.pkl  # 训练好的意图分类模型
└── docs/                      # 文档目录
    ├── 方法对比分析.md         # 三种方法对比文档
    └── 小组分工说明.md         # 小组成员分工文档
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（可选）
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行系统

```bash
# 进入项目目录
cd wsq

# 启动Streamlit应用
streamlit run main.py
```

### 3. 访问系统

打开浏览器访问 `http://localhost:8501`

## 数据集说明

数据集包含以下字段：
| 字段名 | 说明 |
|--------|------|
| dataid | 数据ID |
| name | 景点名称 |
| city | 所在城市 |
| province | 所在省份 |
| address | 详细地址 |
| phone | 联系电话 |
| website | 官方网站 |
| rating | 用户评分 |
| ticket_price | 门票价格 |
| open_time | 开放时间 |
| suggest_time | 建议游玩时间 |
| tags | 景点标签 |
| description | 景点描述 |
| tips | 游玩贴士 |

## 功能特性

### 基于规则的问答
- 关键词意图识别
- 词典实体识别
- 槽位填充
- 模板化回答

### 基于文本分类的问答
- TF-IDF特征提取
- LinearSVC分类器
- 12种意图类别
- 自动模型训练

### 基于LLM RAG的问答
- 文档向量化索引
- 相似度检索
- Prompt工程
- 支持多种大模型（Ollama、OpenAI等）

## 系统截图

（运行系统后可截图添加）

## 团队成员

- **组长**: 王思琪
- **组员**: 禹红倩、马成龙、张楠、罗应萍

## 技术栈

- Python 3.8+
- Streamlit（Web框架）
- Pandas（数据处理）
- Scikit-learn（机器学习）
- Jieba（中文分词）
- Requests（HTTP请求）

## 许可证

本项目仅供学习交流使用。

---

*自然语言处理大作业 - 2025*
