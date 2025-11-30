# -*- coding: utf-8 -*-
"""
===============================================================================
基于文本分类的旅游景点问答系统 (Text Classification QA)
===============================================================================
作者: 王思琪团队
描述: 本模块实现了基于机器学习文本分类的问答方法，使用 TF-IDF 特征提取
      结合 LinearSVC 分类器来识别用户意图，然后结合知识库生成回答。

核心组件:
    1. 意图标签体系 - 定义旅游问答的意图分类
    2. 训练数据集 - 每个意图标签的示例问题（不少于10条）
    3. TF-IDF + LinearSVC 分类器 - 构建意图分类模型
    4. 模型保存与加载 - 持久化模型
    5. 预测接口 - predict_intent() 函数
    6. 问答逻辑 - 结合分类结果和知识库生成回答
===============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
import jieba
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# 第一部分：意图标签体系定义

# 定义旅游问答系统的意图标签，每个标签代表用户的一种询问意图

意图标签列表 = [
    "ASK_TICKET",       # 询问门票价格
    "ASK_OPEN_TIME",    # 询问开放时间
    "ASK_INTRO",        # 询问景点简介
    "ASK_ADDRESS",      # 询问地址位置
    "ASK_PHONE",        # 询问联系电话
    "ASK_WEBSITE",      # 询问官方网站
    "ASK_RATING",       # 询问评分评价
    "ASK_SUGGEST_TIME", # 询问建议游玩时间
    "ASK_TAGS",         # 询问景点标签/类型
    "ASK_TIPS",         # 询问游玩贴士
    "ASK_LOCATION",     # 询问所在城市/省份
    "ASK_RECOMMEND"     # 景点推荐
]

意图标签描述 = {
    "ASK_TICKET": "询问门票价格",
    "ASK_OPEN_TIME": "询问开放时间",
    "ASK_INTRO": "询问景点简介",
    "ASK_ADDRESS": "询问地址位置",
    "ASK_PHONE": "询问联系电话",
    "ASK_WEBSITE": "询问官方网站",
    "ASK_RATING": "询问评分评价",
    "ASK_SUGGEST_TIME": "询问建议游玩时间",
    "ASK_TAGS": "询问景点标签/类型",
    "ASK_TIPS": "询问游玩贴士",
    "ASK_LOCATION": "询问所在城市/省份",
    "ASK_RECOMMEND": "景点推荐"
}



# 第二部分：训练数据集

# 每个意图标签至少包含10条示例训练数据
# 这些数据用于训练意图分类器

训练数据集 = {
    # 询问门票价格的示例问题
    "ASK_TICKET": [
        "这个景点门票多少钱",
        "请问门票价格是多少",
        "入场需要花多少钱",
        "票价贵不贵",
        "门票收费标准是什么",
        "去玩要多少钱门票",
        "景区门票费用多少",
        "要买门票吗多少钱一张",
        "成人票和儿童票分别多少钱",
        "门票有优惠吗",
        "学生票多少钱",
        "老年人门票免费吗",
        "一张票要多少钱",
        "进去要收费吗",
        "入园门票价格"
    ],
    
    # 询问开放时间的示例问题
    "ASK_OPEN_TIME": [
        "景点几点开门",
        "开放时间是什么时候",
        "几点到几点营业",
        "什么时候关门",
        "营业时间是多少",
        "早上几点可以进去",
        "晚上开到几点",
        "周末开放吗",
        "节假日开门吗",
        "全天开放吗",
        "运营时间是怎样的",
        "现在开门了吗",
        "最晚几点可以入园",
        "开园时间是几点",
        "闭园时间是什么时候"
    ],
    
    # 询问景点简介的示例问题
    "ASK_INTRO": [
        "介绍一下这个景点",
        "这个地方有什么特色",
        "景点简介是什么",
        "这里有什么好玩的",
        "能给我讲讲这个景点吗",
        "这个景区怎么样",
        "有什么看点",
        "景点概况是什么",
        "这是一个什么样的地方",
        "好玩吗这里",
        "值得去吗",
        "主要有什么内容",
        "景点的特点是什么",
        "详细介绍一下",
        "说说这个景点的情况"
    ],
    
    # 询问地址位置的示例问题
    "ASK_ADDRESS": [
        "景点地址在哪",
        "怎么去这个地方",
        "具体位置在哪里",
        "在什么路",
        "详细地址是什么",
        "坐什么车能到",
        "离市中心远吗",
        "交通方便吗",
        "景点位于哪里",
        "地址门牌号是多少",
        "在哪个区",
        "怎么走能到",
        "导航地址是什么",
        "所在位置",
        "这个景点坐落在哪"
    ],
    
    # 询问联系电话的示例问题
    "ASK_PHONE": [
        "景点电话是多少",
        "联系方式是什么",
        "客服电话多少",
        "咨询热线是什么",
        "预约电话是多少",
        "怎么联系景区",
        "服务电话号码",
        "订票电话是什么",
        "有没有联系电话",
        "工作人员电话",
        "售票处电话多少",
        "景区的电话号码",
        "投诉电话是什么",
        "可以打电话咨询吗",
        "有客服热线吗"
    ],
    
    # 询问官方网站的示例问题
    "ASK_WEBSITE": [
        "官网是什么",
        "官方网站地址是多少",
        "网址是什么",
        "有官方网站吗",
        "在哪个网站可以了解更多",
        "网上购票网址",
        "官方主页是什么",
        "景区网站是什么",
        "网站链接是多少",
        "哪里可以看到官方信息",
        "预约网站是什么",
        "网上平台是什么",
        "官方网页地址",
        "有没有官网可以查询",
        "线上平台网址是什么"
    ],
    
    # 询问评分评价的示例问题
    "ASK_RATING": [
        "这个景点评分多少",
        "口碑怎么样",
        "游客评价好吗",
        "推荐指数是多少",
        "值不值得去",
        "评价高不高",
        "星级是多少",
        "大家觉得好玩吗",
        "景区评级是什么",
        "用户打分多少",
        "好评率高吗",
        "景点质量怎么样",
        "受欢迎程度如何",
        "游客满意度怎样",
        "评分高吗这个景点"
    ],
    
    # 询问建议游玩时间的示例问题
    "ASK_SUGGEST_TIME": [
        "玩多久合适",
        "需要玩多长时间",
        "建议游玩时间是多少",
        "一般游客玩多久",
        "游览需要几个小时",
        "半天够不够",
        "要安排多少时间",
        "能玩一整天吗",
        "大概需要多久",
        "参观时间要多长",
        "走完全程要多久",
        "推荐游玩时长",
        "预计游玩时间",
        "多久能逛完",
        "安排几个小时合适"
    ],
    
    # 询问景点标签/类型的示例问题
    "ASK_TAGS": [
        "这是什么类型的景点",
        "景点标签是什么",
        "属于什么主题",
        "是自然景观还是人文景观",
        "这个景点的分类是什么",
        "适合什么人群",
        "是亲子游景点吗",
        "属于什么风格",
        "景点类别是什么",
        "主题是什么",
        "有什么特色标签",
        "是历史古迹还是现代建筑",
        "景区性质是什么",
        "这里是公园还是博物馆",
        "景点定位是什么"
    ],
    
    # 询问游玩贴士的示例问题
    "ASK_TIPS": [
        "有什么注意事项",
        "游玩攻略是什么",
        "有什么建议",
        "需要注意什么",
        "有什么小贴士",
        "去之前要准备什么",
        "有什么经验分享",
        "玩的时候要注意啥",
        "有什么禁忌吗",
        "给点建议吧",
        "最佳游玩方式是什么",
        "有什么省钱技巧",
        "避坑指南有吗",
        "怎么玩比较好",
        "有没有游览须知"
    ],
    
    # 询问所在城市/省份的示例问题
    "ASK_LOCATION": [
        "这个景点在哪个城市",
        "属于哪个省份",
        "是哪里的景点",
        "所在城市是哪",
        "位于什么地方",
        "归属哪个地区",
        "是哪个城市的",
        "在什么省",
        "地理位置在哪",
        "这是哪个省的景点",
        "所在地是哪里",
        "隶属于哪个城市",
        "这个景区在什么地方",
        "区域位置是哪",
        "是国内哪个城市的"
    ],
    
    # 景点推荐的示例问题
    "ASK_RECOMMEND": [
        "有什么好玩的景点推荐",
        "推荐几个值得去的地方",
        "哪些景点比较好",
        "有什么必去的景点",
        "推荐一些热门景点",
        "最值得去的是哪个",
        "有什么好玩的地方",
        "去哪玩比较好",
        "有没有推荐的景区",
        "想找个好玩的地方",
        "周边有什么景点",
        "给我推荐几个景点",
        "哪里比较好玩",
        "有什么旅游景点",
        "帮我推荐一下景点"
    ]
}



# 第三部分：文本预处理函数

def 中文分词(文本: str) -> str:
    """
    对中文文本进行分词处理
    
    参数:
        文本: 输入的中文文本字符串
        
    返回:
        分词后用空格连接的字符串（适用于TF-IDF向量化）
        
    说明:
        使用jieba分词库进行中文分词，将分词结果用空格连接
        这样TfidfVectorizer可以正确处理中文文本
    """
    # 使用jieba进行分词
    分词结果 = jieba.lcut(文本)
    # 过滤掉空白字符和单字符（通常是标点符号）
    分词结果 = [词 for 词 in 分词结果 if len(词.strip()) > 0]
    # 用空格连接
    return " ".join(分词结果)


def 准备训练数据() -> Tuple[List[str], List[str]]:
    """
    将训练数据集转换为适合模型训练的格式
    
    返回:
        (问题列表, 标签列表) 的元组
        
    说明:
        遍历训练数据集字典，将所有问题和对应的标签提取出来
    """
    问题列表 = []
    标签列表 = []
    
    for 意图标签, 问题集 in 训练数据集.items():
        for 问题 in 问题集:
            # 对问题进行分词处理
            分词后问题 = 中文分词(问题)
            问题列表.append(分词后问题)
            标签列表.append(意图标签)
    
    return 问题列表, 标签列表



# 第四部分：分类器构建与训练

class 意图分类器:
    """
    基于TF-IDF + LinearSVC的意图分类器
    
    属性:
        模型管道: sklearn Pipeline，包含TF-IDF向量化器和SVC分类器
        是否已训练: 布尔值，指示模型是否已经训练
        
    方法:
        训练(): 使用训练数据训练分类器
        预测(): 预测单个问题的意图
        评估(): 在测试集上评估模型性能
        保存模型(): 将模型保存到文件
        加载模型(): 从文件加载模型
    """
    
    def __init__(self):
        """初始化分类器"""
        # 创建模型管道：TF-IDF向量化 -> LinearSVC分类
        self.模型管道 = Pipeline([
            # TF-IDF向量化器配置
            ('tfidf', TfidfVectorizer(
                max_features=5000,      # 最大特征数量
                ngram_range=(1, 2),     # 使用unigram和bigram
                min_df=1,               # 最小文档频率
                max_df=0.95,            # 最大文档频率
                sublinear_tf=True       # 使用对数TF
            )),
            # LinearSVC分类器配置
            ('svc', LinearSVC(
                C=1.0,                  # 正则化参数
                class_weight='balanced', # 平衡类别权重
                max_iter=10000,         # 最大迭代次数
                random_state=42         # 随机种子
            ))
        ])
        self.是否已训练 = False
    
    def 训练(self, 问题列表: List[str] = None, 标签列表: List[str] = None, 
            显示评估报告: bool = True) -> Dict:
        """
        训练意图分类器
        
        参数:
            问题列表: 训练问题列表（如果为None则使用默认训练数据）
            标签列表: 对应的标签列表
            显示评估报告: 是否显示训练后的评估报告
            
        返回:
            包含训练信息的字典
        """
        # 如果没有提供数据，使用默认训练数据
        if 问题列表 is None or 标签列表 is None:
            问题列表, 标签列表 = 准备训练数据()
        
        print(f"开始训练意图分类器...")
        print(f"训练样本数量: {len(问题列表)}")
        print(f"意图类别数量: {len(set(标签列表))}")
        
        # 划分训练集和测试集（80%训练，20%测试）
        X_train, X_test, y_train, y_test = train_test_split(
            问题列表, 标签列表, 
            test_size=0.2, 
            random_state=42,
            stratify=标签列表  # 保持类别比例
        )
        
        # 训练模型
        self.模型管道.fit(X_train, y_train)
        self.是否已训练 = True
        
        # 评估模型
        训练集准确率 = self.模型管道.score(X_train, y_train)
        测试集准确率 = self.模型管道.score(X_test, y_test)
        
        评估结果 = {
            "训练样本数": len(X_train),
            "测试样本数": len(X_test),
            "训练集准确率": 训练集准确率,
            "测试集准确率": 测试集准确率
        }
        
        print(f"训练完成！")
        print(f"训练集准确率: {训练集准确率:.4f}")
        print(f"测试集准确率: {测试集准确率:.4f}")
        
        # 显示详细评估报告
        if 显示评估报告:
            y_pred = self.模型管道.predict(X_test)
            print("\n分类报告:")
            print(classification_report(y_test, y_pred, target_names=意图标签列表, zero_division=0))
        
        return 评估结果
    
    def 预测(self, 问题: str) -> Tuple[str, float]:
        """
        预测单个问题的意图
        
        参数:
            问题: 用户输入的问题文本
            
        返回:
            (预测的意图标签, 置信度分数) 的元组
            
        说明:
            置信度基于决策函数的值计算
        """
        if not self.是否已训练:
            raise Exception("模型尚未训练！请先调用 训练() 方法。")
        
        # 对问题进行分词
        分词后问题 = 中文分词(问题)
        
        # 预测意图
        预测结果 = self.模型管道.predict([分词后问题])[0]
        
        # 计算置信度（基于决策函数）
        决策分数 = self.模型管道.decision_function([分词后问题])[0]
        # 使用softmax将决策分数转换为概率
        概率分布 = np.exp(决策分数) / np.sum(np.exp(决策分数))
        置信度 = float(np.max(概率分布))
        
        return 预测结果, 置信度
    
    def 批量预测(self, 问题列表: List[str]) -> List[Tuple[str, float]]:
        """
        批量预测多个问题的意图
        
        参数:
            问题列表: 问题文本列表
            
        返回:
            预测结果列表，每个元素为 (意图标签, 置信度) 元组
        """
        return [self.预测(问题) for 问题 in 问题列表]
    
    def 保存模型(self, 模型路径: str):
        """
        将训练好的模型保存到文件
        
        参数:
            模型路径: 保存模型的文件路径
        """
        if not self.是否已训练:
            raise Exception("模型尚未训练！无法保存。")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(模型路径) if os.path.dirname(模型路径) else '.', exist_ok=True)
        
        # 使用pickle保存模型
        with open(模型路径, 'wb') as f:
            pickle.dump(self.模型管道, f)
        
        print(f"模型已保存到: {模型路径}")
    
    def 加载模型(self, 模型路径: str):
        """
        从文件加载预训练的模型
        
        参数:
            模型路径: 模型文件路径
        """
        if not os.path.exists(模型路径):
            raise FileNotFoundError(f"模型文件不存在: {模型路径}")
        
        with open(模型路径, 'rb') as f:
            self.模型管道 = pickle.load(f)
        
        self.是否已训练 = True
        print(f"模型已从 {模型路径} 加载")



# 第五部分：全局预测接口函数

# 全局分类器实例
_全局分类器 = None


def 获取分类器() -> 意图分类器:
    """
    获取或初始化全局分类器实例
    
    返回:
        训练好的意图分类器实例
    """
    global _全局分类器
    
    if _全局分类器 is None:
        _全局分类器 = 意图分类器()
        
        # 尝试加载已保存的模型
        当前目录 = os.path.dirname(os.path.abspath(__file__))
        模型路径 = os.path.join(当前目录, "..", "models", "intent_classifier.pkl")
        
        if os.path.exists(模型路径):
            _全局分类器.加载模型(模型路径)
        else:
            # 如果没有保存的模型，重新训练
            _全局分类器.训练(显示评估报告=False)
            # 保存模型
            _全局分类器.保存模型(模型路径)
    
    return _全局分类器


def predict_intent(question: str) -> Tuple[str, float]:
    """
    预测用户问题的意图（全局接口函数）
    
    参数:
        question: 用户输入的问题文本
        
    返回:
        (预测的意图标签, 置信度分数) 的元组
        
    使用示例:
        >>> intent, confidence = predict_intent("上海迪士尼门票多少钱？")
        >>> print(f"意图: {intent}, 置信度: {confidence:.2f}")
    """
    分类器 = 获取分类器()
    return 分类器.预测(question)



# 第六部分：知识库查询与回答生成

# 意图到数据字段的映射
意图字段映射 = {
    "ASK_TICKET": ["name", "ticket_price"],
    "ASK_OPEN_TIME": ["name", "open_time"],
    "ASK_INTRO": ["name", "description"],
    "ASK_ADDRESS": ["name", "address", "city", "province"],
    "ASK_PHONE": ["name", "phone"],
    "ASK_WEBSITE": ["name", "website"],
    "ASK_RATING": ["name", "rating"],
    "ASK_SUGGEST_TIME": ["name", "suggest_time"],
    "ASK_TAGS": ["name", "tags"],
    "ASK_TIPS": ["name", "tips"],
    "ASK_LOCATION": ["name", "city", "province"],
    "ASK_RECOMMEND": ["name", "rating", "tags", "description"]
}

# 回答模板
回答模板 = {
    "ASK_TICKET": "【{name}】的门票价格为：{ticket_price}",
    "ASK_OPEN_TIME": "【{name}】的开放时间为：{open_time}",
    "ASK_INTRO": "【{name}】景点简介：\n{description}",
    "ASK_ADDRESS": "【{name}】的地址信息：\n省份：{province}\n城市：{city}\n详细地址：{address}",
    "ASK_PHONE": "【{name}】的联系电话为：{phone}",
    "ASK_WEBSITE": "【{name}】的官方网站为：{website}",
    "ASK_RATING": "【{name}】的用户评分为：{rating} 分",
    "ASK_SUGGEST_TIME": "【{name}】的建议游玩时间为：{suggest_time}",
    "ASK_TAGS": "【{name}】的景点标签：{tags}",
    "ASK_TIPS": "【{name}】的游玩贴士：\n{tips}",
    "ASK_LOCATION": "【{name}】位于{province}{city}",
    "ASK_RECOMMEND": "为您推荐【{name}】（评分：{rating}）\n标签：{tags}\n简介：{description}"
}


def 从知识库查询景点(景点名称: str, 知识库: pd.DataFrame) -> Optional[pd.Series]:
    """
    从知识库中查询景点信息
    
    参数:
        景点名称: 要查询的景点名称
        知识库: 景点数据DataFrame
        
    返回:
        如果找到则返回景点数据，否则返回None
    """
    # 精确匹配
    匹配结果 = 知识库[知识库["name"] == 景点名称]
    if not 匹配结果.empty:
        return 匹配结果.iloc[0]
    
    # 模糊匹配
    for _, 行 in 知识库.iterrows():
        if 景点名称 in 行["name"] or 行["name"] in 景点名称:
            return 行
    
    return None


def 提取实体(问题: str, 知识库: pd.DataFrame) -> Dict[str, List[str]]:
    """
    从问题中提取景点和城市实体
    
    参数:
        问题: 用户问题
        知识库: 景点数据DataFrame
        
    返回:
        识别到的实体字典
    """
    实体 = {"景点": [], "城市": []}
    
    # 提取景点名称
    景点列表 = 知识库["name"].dropna().unique().tolist()
    for 景点名 in 景点列表:
        if 景点名 in 问题:
            实体["景点"].append(景点名)
    
    # 提取城市名称
    城市列表 = 知识库["city"].dropna().unique().tolist()
    for 城市名 in 城市列表:
        if 城市名 in 问题:
            实体["城市"].append(城市名)
    
    return 实体


def 生成回答(意图: str, 景点数据: pd.Series) -> str:
    """
    根据意图和景点数据生成回答
    
    参数:
        意图: 预测的意图标签
        景点数据: 匹配到的景点数据
        
    返回:
        生成的回答文本
    """
    模板 = 回答模板.get(意图, "关于【{name}】：{description}")
    需要的字段 = 意图字段映射.get(意图, ["name", "description"])
    
    # 准备模板数据
    模板数据 = {}
    for 字段 in 需要的字段:
        if 字段 in 景点数据.index:
            值 = 景点数据[字段]
            模板数据[字段] = str(值) if pd.notna(值) and 值 != "" else "暂无信息"
        else:
            模板数据[字段] = "暂无信息"
    
    try:
        return 模板.format(**模板数据)
    except KeyError:
        return f"关于【{模板数据.get('name', '该景点')}】：暂时无法获取您询问的具体信息。"



# 第七部分：主问答函数

def text_classification_qa(question: str, kb: pd.DataFrame) -> Dict:
    """
    基于文本分类的问答主函数
    
    参数:
        question: 用户输入的问题
        kb: 知识库DataFrame
        
    返回:
        包含以下字段的字典:
        - intent: 预测的意图
        - intent_confidence: 意图置信度
        - intent_description: 意图的中文描述
        - entities: 识别到的实体
        - answer: 生成的回答
        - matched_attraction: 匹配到的景点
        
    使用示例:
        >>> import pandas as pd
        >>> kb = pd.read_csv("data/merged_attractions.csv")
        >>> result = text_classification_qa("外滩的门票多少钱？", kb)
        >>> print(result["answer"])
    """
    结果 = {
        "intent": "UNKNOWN",
        "intent_confidence": 0.0,
        "intent_description": "未知意图",
        "entities": {"景点": [], "城市": []},
        "answer": "",
        "matched_attraction": None
    }
    
    # 第一步：使用分类器预测意图
    意图, 置信度 = predict_intent(question)
    结果["intent"] = 意图
    结果["intent_confidence"] = 置信度
    结果["intent_description"] = 意图标签描述.get(意图, "未知意图")
    
    # 第二步：提取实体
    实体 = 提取实体(question, kb)
    结果["entities"] = 实体
    
    # 第三步：根据实体和意图生成回答
    if 实体["景点"]:
        # 找到了景点实体
        景点名 = 实体["景点"][0]
        景点数据 = 从知识库查询景点(景点名, kb)
        
        if 景点数据 is not None:
            结果["matched_attraction"] = 景点数据["name"]
            结果["answer"] = 生成回答(意图, 景点数据)
        else:
            结果["answer"] = f"抱歉，未找到关于【{景点名}】的详细信息。"
    
    elif 实体["城市"]:
        # 只找到了城市实体，推荐该城市的景点
        城市名 = 实体["城市"][0]
        城市景点 = kb[kb["city"] == 城市名].sort_values(by="rating", ascending=False)
        
        if not 城市景点.empty:
            推荐列表 = []
            for i, (_, 景点) in enumerate(城市景点.head(3).iterrows(), 1):
                推荐列表.append(f"{i}. 【{景点['name']}】- 评分：{景点['rating']}")
            
            结果["answer"] = f"为您推荐{城市名}的热门景点：\n" + "\n".join(推荐列表)
            结果["matched_attraction"] = "城市推荐"
        else:
            结果["answer"] = f"抱歉，暂未收录{城市名}的景点信息。"
    
    else:
        # 没有找到任何实体
        结果["answer"] = (
            "抱歉，我没有识别到您询问的具体景点。\n\n"
            "您可以这样问：\n"
            "- 上海迪士尼的门票多少钱？\n"
            "- 外滩什么时候开放？\n"
            "- 介绍一下东方明珠\n"
            "- 上海有什么好玩的景点？"
        )
    
    return 结果


# 第八部分：模型训练脚本

def 训练并保存模型():
    """
    训练意图分类模型并保存到文件
    
    这个函数用于初始化或更新分类模型
    """
    print("=" * 60)
    print("意图分类模型训练")
    print("=" * 60)
    
    # 创建分类器
    分类器 = 意图分类器()
    
    # 训练模型
    分类器.训练(显示评估报告=True)
    
    # 保存模型
    当前目录 = os.path.dirname(os.path.abspath(__file__))
    模型路径 = os.path.join(当前目录, "..", "models", "intent_classifier.pkl")
    分类器.保存模型(模型路径)
    
    return 分类器



# 第九部分：测试代码

if __name__ == "__main__":
    print("=" * 60)
    print("基于文本分类的旅游景点问答系统 - 测试")
    print("=" * 60)
    
    # 训练模型
    分类器 = 训练并保存模型()
    
    # 加载知识库
    当前目录 = os.path.dirname(os.path.abspath(__file__))
    数据路径 = os.path.join(当前目录, "..", "data", "merged_attractions.csv")
    
    try:
        知识库 = pd.read_csv(数据路径)
        print(f"\n成功加载知识库，共 {len(知识库)} 条景点数据")
    except Exception as e:
        print(f"加载知识库失败: {e}")
        exit(1)
    
    # 测试问题
    测试问题 = [
        "上海迪士尼门票多少钱",
        "外滩几点开门",
        "介绍一下东方明珠",
        "豫园在哪里",
        "上海有什么好玩的景点",
        "静安寺的联系电话是多少",
        "田子坊玩多久合适"
    ]
    
    print("\n" + "-" * 60)
    print("开始测试...")
    print("-" * 60)
    
    for 问题 in 测试问题:
        print(f"\n【用户问题】{问题}")
        结果 = text_classification_qa(问题, 知识库)
        print(f"【预测意图】{结果['intent']} ({结果['intent_description']})")
        print(f"【置信度】{结果['intent_confidence']:.2f}")
        print(f"【识别实体】{结果['entities']}")
        print(f"【匹配景点】{结果['matched_attraction']}")
        print(f"【系统回答】\n{结果['answer']}")
        print("-" * 40)
