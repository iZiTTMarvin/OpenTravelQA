# -*- coding: utf-8 -*-
"""
===============================================================================
基于大模型 RAG 的旅游景点问答系统 (LLM + Retrieval-Augmented Generation)
===============================================================================
作者: 王思琪团队
描述: 本模块实现了基于大语言模型和检索增强生成（RAG）的问答方法。
      通过将景点数据转换为文档，构建向量索引，检索相关文档，
      然后利用大模型根据检索到的上下文生成准确的回答。

核心组件:
    1. 文档构建 - 将CSV数据转换为标准景点文档
    2. 向量化与索引 - 使用Embedding模型构建向量库
    3. 相似度检索 - 基于问题向量检索Top-K相关文档
    4. Prompt构建 - 组合检索结果和用户问题
    5. 大模型生成 - 调用LLM生成最终回答
===============================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 导入规则模块中的意图识别和实体识别函数
from rule_based_qa import 识别意图, 识别实体, 构建实体词典

# ============================================================================
# 第一部分：配置项
# ============================================================================

@dataclass
class RAG配置:
    """RAG系统的配置参数"""
    # 检索相关配置
    检索数量: int = 3                    # Top-K 检索的文档数量
    相似度阈值: float = 0.1              # 最小相似度阈值
    
    # 大模型相关配置
    # 支持多种大模型API：OpenAI、通义千问、硅基流动、本地Ollama等
    模型类型: str = "siliconflow"        # 可选: "openai", "qianwen", "ollama", "siliconflow", "zhipu"
    模型名称: str = "deepseek-ai/DeepSeek-V3.2-Exp"  # 硅基流动默认模型
    API地址: str = "https://api.siliconflow.cn/v1/chat/completions"  # 硅基流动API地址（主要用这个）
    API密钥: str = ""                    # API密钥（硅基流动需要）
    
    # 生成相关配置
    最大生成长度: int = 1024             # 最大生成token数
    温度: float = 0.3                    # 生成温度，越低越确定
    
    # 向量化配置
    向量维度: int = 5000                 # TF-IDF特征维度


# 全局配置实例
默认配置 = RAG配置()


# ============================================================================
# 第二部分：景点文档模板与构建
# ============================================================================

景点文档模板 = """
【景点名称】{name}
【所在城市】{city}
【所在省份】{province}
【详细地址】{address}
【联系电话】{phone}
【官方网站】{website}
【用户评分】{rating}
【门票价格】{ticket_price}
【开放时间】{open_time}
【建议游玩时间】{suggest_time}
【景点标签】{tags}
【景点简介】{description}
【游玩贴士】{tips}
"""


def 构建景点文档(景点数据: pd.Series) -> str:
    """
    将单条景点数据转换为标准文档格式
    
    参数:
        景点数据: 包含景点信息的pandas Series
        
    返回:
        格式化的景点文档字符串
        
    说明:
        此函数将DataFrame中的一行数据转换为便于检索和阅读的文档格式。
        所有字段都会被包含在文档中，空值会被标记为"暂无信息"。
    """
    # 定义需要包含的字段
    字段列表 = [
        "name", "city", "province", "address", "phone", "website",
        "rating", "ticket_price", "open_time", "suggest_time", 
        "tags", "description", "tips"
    ]
    
    # 准备模板数据
    模板数据 = {}
    for 字段 in 字段列表:
        if 字段 in 景点数据.index:
            值 = 景点数据[字段]
            # 处理空值和NaN
            if pd.isna(值) or 值 == "" or 值 is None:
                模板数据[字段] = "暂无信息"
            else:
                模板数据[字段] = str(值)
        else:
            模板数据[字段] = "暂无信息"
    
    # 生成文档
    文档 = 景点文档模板.format(**模板数据)
    return 文档.strip()


def 批量构建文档(知识库: pd.DataFrame) -> List[Dict]:
    """
    将整个知识库转换为文档列表
    
    参数:
        知识库: 景点数据DataFrame
        
    返回:
        文档列表，每个元素包含 {"id": 索引, "name": 景点名, "content": 文档内容}
    """
    文档列表 = []
    
    for 索引, 行数据 in 知识库.iterrows():
        文档内容 = 构建景点文档(行数据)
        文档列表.append({
            "id": 索引,
            "name": 行数据.get("name", f"景点{索引}"),
            "city": 行数据.get("city", "未知城市"),
            "content": 文档内容
        })
    
    return 文档列表


# ============================================================================
# 第三部分：向量化与索引构建
# ============================================================================

class 向量检索器:
    """
    基于TF-IDF的向量检索器
    
    说明:
        这是一个简化版的向量检索实现，使用TF-IDF进行文本向量化。
        在实际生产环境中，建议使用专业的向量数据库（如FAISS、Milvus）
        和更强大的Embedding模型（如BGE、text2vec）。
        
    属性:
        向量化器: TF-IDF向量化器
        文档向量: 所有文档的向量表示
        文档列表: 原始文档数据
        是否已索引: 是否已建立索引
    """
    
    def __init__(self, 配置: RAG配置 = None):
        """
        初始化检索器
        
        参数:
            配置: RAG配置对象，如果为None则使用默认配置
        """
        self.配置 = 配置 or 默认配置
        
        # 初始化TF-IDF向量化器
        self.向量化器 = TfidfVectorizer(
            max_features=self.配置.向量维度,
            ngram_range=(1, 2),
            analyzer='char_wb',  # 使用字符级别的n-gram，适合中文
            min_df=1
        )
        
        self.文档向量 = None
        self.文档列表 = []
        self.是否已索引 = False
    
    def 构建索引(self, 知识库: pd.DataFrame):
        """
        从知识库构建向量索引
        
        参数:
            知识库: 景点数据DataFrame
            
        说明:
            1. 将知识库转换为文档列表
            2. 使用TF-IDF对文档进行向量化
            3. 保存向量和文档数据
        """
        print("开始构建向量索引...")
        
        # 构建文档
        self.文档列表 = 批量构建文档(知识库)
        print(f"已构建 {len(self.文档列表)} 个文档")
        
        # 提取文档内容用于向量化
        文档内容列表 = [doc["content"] for doc in self.文档列表]
        
        # 向量化
        self.文档向量 = self.向量化器.fit_transform(文档内容列表)
        self.是否已索引 = True
        
        print(f"向量索引构建完成，向量维度: {self.文档向量.shape}")
    
    def 检索(self, 查询文本: str, top_k: int = None, 知识库: pd.DataFrame = None) -> List[Dict]:
        """
        根据查询文本检索相关文档（混合检索：TF-IDF相似度 + 实体匹配加权）
        
        参数:
            查询文本: 用户的问题或查询
            top_k: 返回的文档数量，默认使用配置值
            知识库: 原始知识库DataFrame，用于实体识别
            
        返回:
            检索结果列表，每个元素包含文档信息和相似度分数
            
        检索策略:
            1. 首先识别查询中的城市、省份、景点实体
            2. 计算TF-IDF余弦相似度作为基础分数
            3. 对匹配实体的文档进行加权提升
            4. 综合排序返回Top-K结果
        """
        if not self.是否已索引:
            raise Exception("索引尚未构建！请先调用 构建索引() 方法。")
        
        top_k = top_k or self.配置.检索数量
        
        # 将查询文本向量化
        查询向量 = self.向量化器.transform([查询文本])
        
        # 计算余弦相似度作为基础分数
        基础相似度 = cosine_similarity(查询向量, self.文档向量).flatten()
        
        # ========== 实体感知加权 ==========
        # 从查询中提取城市、省份、景点名称
        识别到的城市 = []
        识别到的省份 = []
        识别到的景点 = []
        
        # 提取知识库中的实体词典
        if 知识库 is not None:
            所有城市 = 知识库["city"].dropna().unique().tolist()
            所有省份 = 知识库["province"].dropna().unique().tolist()
            所有景点 = 知识库["name"].dropna().unique().tolist()
        else:
            # 从文档列表中提取
            所有城市 = list(set(doc["city"] for doc in self.文档列表))
            所有省份 = []
            所有景点 = [doc["name"] for doc in self.文档列表]
        
        # 识别查询中的实体
        for 城市 in 所有城市:
            if 城市 and 城市 in 查询文本:
                识别到的城市.append(城市)
        
        for 省份 in 所有省份:
            if 省份 and 省份 in 查询文本:
                识别到的省份.append(省份)
                
        for 景点 in 所有景点:
            if 景点 and 景点 in 查询文本:
                识别到的景点.append(景点)
        
        # 计算实体匹配加权分数
        实体加权分数 = np.zeros(len(self.文档列表))
        
        for i, 文档 in enumerate(self.文档列表):
            加权值 = 0
            文档城市 = 文档.get("city", "")
            文档名称 = 文档.get("name", "")
            文档内容 = 文档.get("content", "")
            
            # 城市匹配：高权重（解决"银川有什么好玩的"类问题）
            if 文档城市 in 识别到的城市:
                加权值 += 0.5  # 城市匹配加0.5分
            
            # 省份匹配
            for 省份 in 识别到的省份:
                if 省份 in 文档内容:
                    加权值 += 0.3
                    break
            
            # 景点名称匹配：最高权重
            if 文档名称 in 识别到的景点:
                加权值 += 0.8  # 精确匹配景点名加0.8分
            else:
                for 景点 in 识别到的景点:
                    if 景点 in 文档名称 or 文档名称 in 景点:
                        加权值 += 0.6  # 部分匹配加0.6分
                        break
            
            实体加权分数[i] = 加权值
        
        # 综合分数 = 基础相似度 + 实体加权分数
        # 使用混合策略：如果有实体匹配，实体权重更重要
        if np.max(实体加权分数) > 0:
            # 有实体匹配时，综合分数 = 0.3*TF-IDF + 0.7*实体加权
            综合分数 = 0.3 * 基础相似度 + 0.7 * 实体加权分数
        else:
            # 无实体匹配时，仅使用TF-IDF
            综合分数 = 基础相似度
        
        # 获取Top-K索引
        top_k_indices = np.argsort(综合分数)[::-1][:top_k]
        
        # 构建检索结果
        检索结果 = []
        for 索引 in top_k_indices:
            分数 = 综合分数[索引]
            if 分数 >= self.配置.相似度阈值:
                检索结果.append({
                    "id": self.文档列表[索引]["id"],
                    "name": self.文档列表[索引]["name"],
                    "city": self.文档列表[索引]["city"],
                    "content": self.文档列表[索引]["content"],
                    "score": float(分数),
                    "tfidf_score": float(基础相似度[索引]),  # 保留原始TF-IDF分数供调试
                    "entity_boost": float(实体加权分数[索引])  # 实体加权分数
                })
        
        return 检索结果
    
    def 按景点名检索(self, 景点名: str) -> Optional[Dict]:
        """
        根据景点名称精确检索文档
        
        参数:
            景点名: 景点名称
            
        返回:
            匹配的文档，如果没找到则返回None
        """
        for 文档 in self.文档列表:
            if 文档["name"] == 景点名 or 景点名 in 文档["name"]:
                return 文档
        return None


# ============================================================================
# 第四部分：Prompt模板设计
# ============================================================================

# 系统提示词：指导大模型只基于提供的文档回答
系统提示词 = """你是一个专业的旅游景点智能问答助手。你的任务是根据提供的景点信息文档，准确、友好地回答用户的问题。

重要规则：
1. 只能根据提供的文档内容回答问题，不要编造或推测任何信息
2. 如果文档中没有相关信息，请诚实地告诉用户"根据现有资料，暂无相关信息"
3. 回答要简洁明了，突出关键信息
4. 使用友好、专业的语气
5. 如果用户问的是多个景点，请分别回答"""

# 用户Prompt模板
用户Prompt模板 = """请根据以下景点信息文档，回答用户的问题。

【参考文档】
{检索文档}

【用户问题】
{用户问题}

请基于以上文档信息回答用户的问题："""


def 构建Prompt(用户问题: str, 检索文档列表: List[Dict]) -> Tuple[str, str]:
    """
    构建发送给大模型的Prompt
    
    参数:
        用户问题: 用户提出的问题
        检索文档列表: 检索到的相关文档列表
        
    返回:
        (系统提示词, 用户Prompt) 的元组
    """
    # 将检索到的文档拼接成文本
    文档文本列表 = []
    for i, 文档 in enumerate(检索文档列表, 1):
        文档文本 = f"---文档{i}: {文档['name']}---\n{文档['content']}"
        文档文本列表.append(文档文本)
    
    检索文档文本 = "\n\n".join(文档文本列表)
    
    # 构建用户Prompt
    用户Prompt = 用户Prompt模板.format(
        检索文档=检索文档文本,
        用户问题=用户问题
    )
    
    return 系统提示词, 用户Prompt


# ============================================================================
# 第五部分：大模型调用接口
# ============================================================================

class 大模型客户端:
    """
    大语言模型调用客户端
    
    支持多种大模型API：
    - Ollama（本地部署）
    - OpenAI API
    - 通义千问 API
    - 智谱AI API
    """
    
    def __init__(self, 配置: RAG配置 = None):
        """
        初始化大模型客户端
        
        参数:
            配置: RAG配置对象
        """
        self.配置 = 配置 or 默认配置
    
    def 调用Ollama(self, 系统提示: str, 用户提示: str) -> str:
        """
        调用本地Ollama模型
        
        参数:
            系统提示: 系统提示词
            用户提示: 用户Prompt
            
        返回:
            模型生成的回答
        """
        try:
            # 优先使用 ollama Python 库（更稳定）
            import ollama
            
            完整提示 = f"{系统提示}\n\n{用户提示}"
            
            响应 = ollama.generate(
                model=self.配置.模型名称,
                prompt=完整提示,
                options={
                    "temperature": self.配置.温度,
                    "num_predict": self.配置.最大生成长度
                }
            )
            
            return 响应.get("response", "模型未返回有效回答")
            
        except ImportError:
            # 如果没有安装 ollama 库，回退到 REST API
            return self._调用Ollama_REST(系统提示, 用户提示)
        except Exception as e:
            return f"调用Ollama出错: {str(e)}"
    
    def _调用Ollama_REST(self, 系统提示: str, 用户提示: str) -> str:
        """
        通过REST API调用Ollama（备用方法）
        """
        try:
            完整提示 = f"{系统提示}\n\n{用户提示}"
            
            请求数据 = {
                "model": self.配置.模型名称,
                "prompt": 完整提示,
                "stream": False,
                "options": {
                    "temperature": self.配置.温度,
                    "num_predict": self.配置.最大生成长度
                }
            }
            
            响应 = requests.post(
                self.配置.API地址,
                json=请求数据,
                timeout=120
            )
            
            if 响应.status_code == 200:
                结果 = 响应.json()
                return 结果.get("response", "模型未返回有效回答")
            else:
                return f"调用Ollama失败: HTTP {响应.status_code}，请确保Ollama服务正常运行"
                
        except requests.exceptions.ConnectionError:
            return "无法连接到Ollama服务，请确保Ollama已启动并运行在 http://localhost:11434"
        except requests.exceptions.Timeout:
            return "Ollama请求超时，模型可能正在加载中，请稍后重试"
        except Exception as e:
            return f"调用Ollama出错: {str(e)}"
    
    def 调用OpenAI(self, 系统提示: str, 用户提示: str) -> str:
        """
        调用OpenAI API
        
        参数:
            系统提示: 系统提示词
            用户提示: 用户Prompt
            
        返回:
            模型生成的回答
        """
        try:
            import openai
            
            openai.api_key = self.配置.API密钥
            
            响应 = openai.ChatCompletion.create(
                model=self.配置.模型名称,
                messages=[
                    {"role": "system", "content": 系统提示},
                    {"role": "user", "content": 用户提示}
                ],
                max_tokens=self.配置.最大生成长度,
                temperature=self.配置.温度
            )
            
            return 响应.choices[0].message.content
            
        except ImportError:
            return "请先安装openai库: pip install openai"
        except Exception as e:
            return f"调用OpenAI出错: {str(e)}"
    
    def 调用通义千问(self, 系统提示: str, 用户提示: str) -> str:
        """
        调用阿里通义千问API
        
        参数:
            系统提示: 系统提示词
            用户提示: 用户Prompt
            
        返回:
            模型生成的回答
        """
        try:
            请求头 = {
                "Authorization": f"Bearer {self.配置.API密钥}",
                "Content-Type": "application/json"
            }
            
            请求数据 = {
                "model": self.配置.模型名称,
                "input": {
                    "messages": [
                        {"role": "system", "content": 系统提示},
                        {"role": "user", "content": 用户提示}
                    ]
                },
                "parameters": {
                    "max_tokens": self.配置.最大生成长度,
                    "temperature": self.配置.温度
                }
            }
            
            响应 = requests.post(
                self.配置.API地址,
                headers=请求头,
                json=请求数据,
                timeout=60
            )
            
            if 响应.status_code == 200:
                结果 = 响应.json()
                return 结果.get("output", {}).get("text", "模型未返回有效回答")
            else:
                return f"调用通义千问失败: HTTP {响应.status_code}"
                
        except Exception as e:
            return f"调用通义千问出错: {str(e)}"
    
    def 调用硅基流动(self, 系统提示: str, 用户提示: str) -> str:
        """
        调用硅基流动 SiliconFlow API
        
        参数:
            系统提示: 系统提示词
            用户提示: 用户Prompt
            
        返回:
            模型生成的回答
            
        说明:
            硅基流动(SiliconFlow)是国内领先的AI云服务平台，提供多种大模型API。
            支持的模型包括：
            - Qwen/Qwen2.5-7B-Instruct (默认)
            - Qwen/Qwen2.5-14B-Instruct
            - deepseek-ai/DeepSeek-V2-Chat
            - 更多模型请参考: https://siliconflow.cn/models
            
            API文档: https://docs.siliconflow.cn/
        """
        try:
            请求头 = {
                "Authorization": f"Bearer {self.配置.API密钥}",
                "Content-Type": "application/json"
            }
            
            请求数据 = {
                "model": self.配置.模型名称,
                "messages": [
                    {"role": "system", "content": 系统提示},
                    {"role": "user", "content": 用户提示}
                ],
                "max_tokens": self.配置.最大生成长度,
                "temperature": self.配置.温度,
                "stream": False
            }
            
            响应 = requests.post(
                self.配置.API地址,
                headers=请求头,
                json=请求数据,
                timeout=60
            )
            
            if 响应.status_code == 200:
                结果 = 响应.json()
                # 硅基流动返回格式与OpenAI兼容
                if "choices" in 结果 and len(结果["choices"]) > 0:
                    return 结果["choices"][0].get("message", {}).get("content", "模型未返回有效回答")
                return "模型未返回有效回答"
            else:
                错误信息 = 响应.json().get("error", {}).get("message", f"HTTP {响应.status_code}")
                return f"调用硅基流动失败: {错误信息}"
                
        except requests.exceptions.Timeout:
            return "请求硅基流动API超时，请稍后重试"
        except requests.exceptions.ConnectionError:
            return "无法连接到硅基流动API，请检查网络连接"
        except Exception as e:
            return f"调用硅基流动出错: {str(e)}"
    
    def 生成回答(self, 系统提示: str, 用户提示: str) -> str:
        """
        根据配置的模型类型调用相应的API生成回答
        
        参数:
            系统提示: 系统提示词
            用户提示: 用户Prompt
            
        返回:
            模型生成的回答
        """
        模型类型 = self.配置.模型类型.lower()
        
        if 模型类型 == "ollama":
            return self.调用Ollama(系统提示, 用户提示)
        elif 模型类型 == "openai":
            return self.调用OpenAI(系统提示, 用户提示)
        elif 模型类型 in ["qianwen", "tongyi", "通义千问"]:
            return self.调用通义千问(系统提示, 用户提示)
        elif 模型类型 in ["siliconflow", "硅基流动", "silicon"]:
            return self.调用硅基流动(系统提示, 用户提示)
        else:
            # 默认使用硅基流动
            return self.调用硅基流动(系统提示, 用户提示)


# ============================================================================
# 第六部分：备用回答生成（无需大模型）
# ============================================================================

def 基于模板生成回答(用户问题: str, 检索文档: Dict) -> str:
    """
    当大模型不可用时，使用模板生成回答
    
    参数:
        用户问题: 用户的问题
        检索文档: 检索到的最相关文档
        
    返回:
        基于模板生成的回答
    """
    # 简单的关键词匹配来确定回答内容
    文档内容 = 检索文档["content"]
    景点名 = 检索文档["name"]
    
    # 定义关键词和对应的字段提取规则
    关键词规则 = {
        "门票": "门票价格",
        "票价": "门票价格",
        "多少钱": "门票价格",
        "开放时间": "开放时间",
        "几点": "开放时间",
        "营业": "开放时间",
        "地址": "详细地址",
        "在哪": "详细地址",
        "位置": "详细地址",
        "电话": "联系电话",
        "联系": "联系电话",
        "网站": "官方网站",
        "官网": "官方网站",
        "评分": "用户评分",
        "评价": "用户评分",
        "玩多久": "建议游玩时间",
        "游玩时间": "建议游玩时间",
        "标签": "景点标签",
        "类型": "景点标签",
        "介绍": "景点简介",
        "简介": "景点简介",
        "贴士": "游玩贴士",
        "注意": "游玩贴士",
        "建议": "游玩贴士"
    }
    
    # 查找匹配的字段
    匹配字段 = None
    for 关键词, 字段 in 关键词规则.items():
        if 关键词 in 用户问题:
            匹配字段 = 字段
            break
    
    # 从文档中提取对应信息
    if 匹配字段:
        # 在文档中查找对应字段的值
        行列表 = 文档内容.split("\n")
        for 行 in 行列表:
            if 匹配字段 in 行:
                值 = 行.split("】")[-1].strip()
                if 值 and 值 != "暂无信息":
                    return f"【{景点名}】的{匹配字段}：{值}"
                else:
                    return f"抱歉，关于【{景点名}】的{匹配字段}，目前暂无相关信息。"
    
    # 如果没有匹配到特定字段，返回景点简介
    for 行 in 文档内容.split("\n"):
        if "景点简介" in 行:
            简介 = 行.split("】")[-1].strip()
            if 简介 and 简介 != "暂无信息":
                return f"关于【{景点名}】：\n{简介[:300]}..." if len(简介) > 300 else f"关于【{景点名}】：\n{简介}"
    
    return f"已找到【{景点名}】的相关信息，但未能准确理解您的问题。您可以尝试询问该景点的门票、开放时间、地址等具体信息。"


# ============================================================================
# 第七部分：RAG问答主类
# ============================================================================

class RAG问答系统:
    """
    基于检索增强生成的问答系统主类
    
    属性:
        配置: RAG配置对象
        检索器: 向量检索器实例
        大模型客户端: 大模型调用客户端
        知识库: 原始知识库DataFrame
    """
    
    def __init__(self, 配置: RAG配置 = None):
        """
        初始化RAG问答系统
        
        参数:
            配置: RAG配置对象，如果为None则使用默认配置
        """
        self.配置 = 配置 or 默认配置
        self.检索器 = 向量检索器(self.配置)
        self.大模型客户端 = 大模型客户端(self.配置)
        self.知识库 = None
        self.是否已初始化 = False
    
    def 初始化(self, 知识库: pd.DataFrame):
        """
        使用知识库初始化系统
        
        参数:
            知识库: 景点数据DataFrame
        """
        self.知识库 = 知识库
        self.检索器.构建索引(知识库)
        self.是否已初始化 = True
        print("RAG问答系统初始化完成！")
    
    def 问答(self, 问题: str, 使用大模型: bool = True) -> Dict:
        """
        回答用户问题
        
        参数:
            问题: 用户的问题
            使用大模型: 是否使用大模型生成回答，如果为False则使用模板
            
        返回:
            包含以下字段的字典:
            - question: 用户问题
            - retrieved_docs: 检索到的文档列表
            - prompt: 发送给大模型的Prompt
            - answer: 最终回答
            - use_llm: 是否使用了大模型
        """
        if not self.是否已初始化:
            return {
                "question": 问题,
                "retrieved_docs": [],
                "prompt": "",
                "answer": "系统尚未初始化，请先加载知识库。",
                "use_llm": False
            }
        
        # 第一步：检索相关文档（传入知识库以支持实体感知检索）
        检索结果 = self.检索器.检索(问题, 知识库=self.知识库)
        
        if not 检索结果:
            return {
                "question": 问题,
                "retrieved_docs": [],
                "prompt": "",
                "answer": "抱歉，未找到与您问题相关的景点信息。请尝试更换关键词或询问其他景点。",
                "use_llm": False
            }
        
        # 第二步：构建Prompt
        系统提示, 用户Prompt = 构建Prompt(问题, 检索结果)
        
        # 第三步：生成回答
        if 使用大模型:
            回答 = self.大模型客户端.生成回答(系统提示, 用户Prompt)
            # 检查是否调用失败
            if "出错" in 回答 or "失败" in 回答 or "无法连接" in 回答:
                # 使用备用模板生成
                回答 = 基于模板生成回答(问题, 检索结果[0])
                使用大模型 = False
        else:
            回答 = 基于模板生成回答(问题, 检索结果[0])
        
        return {
            "question": 问题,
            "retrieved_docs": 检索结果,
            "prompt": 用户Prompt,
            "answer": 回答,
            "use_llm": 使用大模型
        }


# ============================================================================
# 第八部分：全局接口函数
# ============================================================================

# 全局RAG系统实例
_全局RAG系统 = None


def 获取RAG系统(知识库: pd.DataFrame = None) -> RAG问答系统:
    """
    获取或初始化全局RAG系统实例
    
    参数:
        知识库: 景点数据DataFrame（首次调用时必须提供）
        
    返回:
        初始化好的RAG问答系统实例
    """
    global _全局RAG系统
    
    if _全局RAG系统 is None:
        _全局RAG系统 = RAG问答系统()
    
    if 知识库 is not None and not _全局RAG系统.是否已初始化:
        _全局RAG系统.初始化(知识库)
    
    return _全局RAG系统


def llm_rag_qa(question: str, kb: pd.DataFrame, use_llm: bool = True) -> Dict:
    """
    基于LLM RAG的问答主函数（全局接口）
    
    参数:
        question: 用户输入的问题
        kb: 知识库DataFrame
        use_llm: 是否使用大模型，默认为True
        
    返回:
        包含以下字段的字典:
        - question: 用户问题
        - intent: 识别到的意图
        - intent_confidence: 意图识别的置信度
        - entities: 识别到的实体
        - matched_attraction: 匹配到的景点名称
        - retrieved_docs: 检索到的文档列表（包含name、score等信息）
        - prompt: 发送给大模型的Prompt（用于展示）
        - answer: 最终回答
        - use_llm: 是否使用了大模型
        
    使用示例:
        >>> import pandas as pd
        >>> kb = pd.read_csv("data/merged_attractions.csv")
        >>> result = llm_rag_qa("上海迪士尼有什么好玩的？", kb)
        >>> print(result["answer"])
    """
    # 获取或初始化RAG系统
    rag系统 = 获取RAG系统(kb)
    
    # 执行问答获取基础结果
    基础结果 = rag系统.问答(question, use_llm)
    
    # 添加意图识别
    意图, 置信度 = 识别意图(question)
    基础结果["intent"] = 意图
    基础结果["intent_confidence"] = 置信度
    
    # 添加实体识别
    实体词典 = 构建实体词典(kb)
    实体 = 识别实体(question, 实体词典)
    基础结果["entities"] = 实体
    
    # 添加匹配景点信息（从检索结果中获取）
    检索文档 = 基础结果.get("retrieved_docs", [])
    if 检索文档:
        基础结果["matched_attraction"] = 检索文档[0].get("name", None)
    else:
        # 如果没有检索到文档，尝试从实体中获取
        if 实体.get("景点"):
            基础结果["matched_attraction"] = 实体["景点"][0]
        else:
            基础结果["matched_attraction"] = None
    
    return 基础结果


# ============================================================================
# 第九部分：配置更新函数
# ============================================================================

def 更新RAG配置(
    模型类型: str = None,
    模型名称: str = None,
    API地址: str = None,
    API密钥: str = None,
    检索数量: int = None,
    温度: float = None
) -> RAG配置:
    """
    更新RAG系统配置
    
    参数:
        模型类型: "ollama", "openai", "qianwen" 等
        模型名称: 具体的模型名称
        API地址: API服务地址
        API密钥: API密钥
        检索数量: Top-K检索数量
        温度: 生成温度
        
    返回:
        更新后的配置对象
    """
    global 默认配置, _全局RAG系统
    
    if 模型类型 is not None:
        默认配置.模型类型 = 模型类型
    if 模型名称 is not None:
        默认配置.模型名称 = 模型名称
    if API地址 is not None:
        默认配置.API地址 = API地址
    if API密钥 is not None:
        默认配置.API密钥 = API密钥
    if 检索数量 is not None:
        默认配置.检索数量 = 检索数量
    if 温度 is not None:
        默认配置.温度 = 温度
    
    # 重新初始化RAG系统以应用新配置
    if _全局RAG系统 is not None:
        知识库 = _全局RAG系统.知识库
        _全局RAG系统 = RAG问答系统(默认配置)
        if 知识库 is not None:
            _全局RAG系统.初始化(知识库)
    
    return 默认配置


# ============================================================================
# 第十部分：测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("基于LLM RAG的旅游景点问答系统 - 测试")
    print("=" * 60)
    
    # 加载知识库
    当前目录 = os.path.dirname(os.path.abspath(__file__))
    数据路径 = os.path.join(当前目录, "..", "data", "merged_attractions.csv")
    
    try:
        知识库 = pd.read_csv(数据路径)
        print(f"成功加载知识库，共 {len(知识库)} 条景点数据")
    except Exception as e:
        print(f"加载知识库失败: {e}")
        exit(1)
    
    # 测试问题
    测试问题 = [
        "上海迪士尼有什么好玩的？",
        "外滩的门票多少钱？",
        "东方明珠几点开门？",
        "给我介绍一下豫园",
        "上海有哪些值得去的景点？"
    ]
    
    print("\n" + "-" * 60)
    print("开始测试（使用模板回答，不调用大模型）...")
    print("-" * 60)
    
    for 问题 in 测试问题:
        print(f"\n【用户问题】{问题}")
        # 使用模板回答，不调用大模型
        结果 = llm_rag_qa(问题, 知识库, use_llm=False)
        
        print(f"【检索结果】找到 {len(结果['retrieved_docs'])} 个相关文档:")
        for doc in 结果['retrieved_docs'][:2]:
            print(f"  - {doc['name']} (相似度: {doc['score']:.3f})")
        
        print(f"【系统回答】\n{结果['answer']}")
        print("-" * 40)
    
    # 如果想测试大模型，取消以下注释
    # print("\n" + "-" * 60)
    # print("测试大模型调用（需要Ollama运行）...")
    # print("-" * 60)
    # 
    # 问题 = "上海迪士尼有什么好玩的项目？"
    # print(f"\n【用户问题】{问题}")
    # 结果 = llm_rag_qa(问题, 知识库, use_llm=True)
    # print(f"【是否使用大模型】{结果['use_llm']}")
    # print(f"【系统回答】\n{结果['answer']}")
