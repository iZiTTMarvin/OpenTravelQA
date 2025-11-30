# -*- coding: utf-8 -*-
"""
修复银川景点名称问题
银川数据的name字段误用了英文名，需要更正为中文名
"""

import pandas as pd

# 银川景点名称映射表（英文名 -> 中文名）
银川名称映射 = {
    "Xixia Imperial Tombs": "西夏王陵",
    "Zhengbeibao Western Film Studio": "镇北堡西部影视城",
    "Helan Mountain Rock Art": "贺兰山岩画",
    "Ningxia Museum": "宁夏博物馆",
    "Yinchuan Mingcui Lake National Wetland Park": "鸣翠湖国家湿地公园",
    "Haibao Pagoda": "海宝塔",
    "Suyukou National Forest Park": "贺兰山苏峪口国家森林公园"
}

def 修复银川名称():
    """修复银川景点的中文名称"""
    # 读取合并数据
    数据路径 = "d:/visual_ProgrammingSoftware/Qoder_projects/wsq/data/merged_attractions.csv"
    df = pd.read_csv(数据路径)
    
    # 统计修复数量
    修复数量 = 0
    
    # 遍历银川景点并修复名称
    for 索引, 行 in df.iterrows():
        if 行['city'] == '银川':
            原名称 = 行['name']
            if 原名称 in 银川名称映射:
                新名称 = 银川名称映射[原名称]
                df.at[索引, 'name'] = 新名称
                print(f"修复: {原名称} -> {新名称}")
                修复数量 += 1
    
    # 保存修复后的数据
    df.to_csv(数据路径, index=False, encoding='utf-8-sig')
    
    print(f"\n共修复 {修复数量} 条银川景点名称")
    print(f"数据已保存到: {数据路径}")
    
    # 验证修复结果
    print("\n验证修复结果:")
    银川数据 = df[df['city'] == '银川'][['dataid', 'name', 'city']]
    print(银川数据.to_string())

if __name__ == '__main__':
    修复银川名称()
