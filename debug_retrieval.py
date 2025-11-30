# -*- coding: utf-8 -*-
"""调试检索算法 - 修复后测试"""
import sys
sys.path.insert(0, 'd:/visual_ProgrammingSoftware/Qoder_projects/wsq/code')
import pandas as pd

# 重新加载模块
import importlib
import llm_rag_qa
importlib.reload(llm_rag_qa)
from llm_rag_qa import llm_rag_qa

kb = pd.read_csv('d:/visual_ProgrammingSoftware/Qoder_projects/wsq/data/merged_attractions.csv')
result = llm_rag_qa('银川有什么好玩的呢？推荐一个最好玩的给我', kb)

print('修复后的检索结果:')
print('=' * 60)
for doc in result.get('retrieved_docs', []):
    name = doc.get("name", "未知")
    city = doc.get("city", "未知")
    score = doc.get("score", 0)
    tfidf = doc.get("tfidf_score", 0)
    entity = doc.get("entity_boost", 0)
    print(f'  {name} ({city})')
    print(f'    综合分数: {score:.4f} = TF-IDF:{tfidf:.4f} + 实体加权:{entity:.4f}')
    print()

# 看看银川有哪些景点
银川景点 = kb[kb['city'] == '银川']['name'].tolist()
print(f'银川的景点: {银川景点}')
