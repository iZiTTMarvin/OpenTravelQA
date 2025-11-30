# -*- coding: utf-8 -*-
"""
数据预处理脚本
将多个城市的旅游景点数据整理合并为标准化格式
"""

import pandas as pd
import re
import os
from pathlib import Path

# 城市与省份映射
CITY_PROVINCE_MAP = {
    '上海': '上海',
    '银川': '宁夏',
    '海口': '海南',
    '固原': '宁夏',
    '呼和浩特': '内蒙古',
    '景德镇': '江西'
}

def clean_text(text):
    """清理文本中的多余空白和换行"""
    if pd.isna(text) or text is None:
        return ''
    text = str(text)
    # 移除多余空白和换行
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_address(address_text):
    """从地址字段提取实际地址"""
    if pd.isna(address_text) or not address_text:
        return ''
    text = str(address_text)
    # 查找"地址:"后面的内容
    match = re.search(r'地址[：:]\s*([^\n电话官网]+)', text)
    if match:
        return clean_text(match.group(1))
    return clean_text(text)

def extract_phone(address_text):
    """从地址字段提取电话号码"""
    if pd.isna(address_text) or not address_text:
        return ''
    text = str(address_text)
    # 查找"电话:"后面的内容
    match = re.search(r'电话[：:]\s*([\d\-,，\s]+)', text)
    if match:
        phone = match.group(1)
        # 清理电话号码
        phone = re.sub(r'[，\s]+', ',', phone).strip(',')
        return phone
    return ''

def extract_website(address_text):
    """从地址字段提取网站"""
    if pd.isna(address_text) or not address_text:
        return ''
    text = str(address_text)
    # 查找"官网:"后面的URL
    match = re.search(r'官网[：:]\s*(https?://[^\s\n]+)', text)
    if match:
        return match.group(1).strip()
    return ''

def clean_name(name):
    """清理景点名称，去除英文名"""
    if pd.isna(name) or not name:
        return ''
    name = str(name)
    # 分离中文和英文
    # 尝试匹配中文名称（在英文名称之前）
    match = re.match(r'^([\u4e00-\u9fa5·（）\(\)\d]+)', name)
    if match:
        return clean_text(match.group(1))
    return clean_text(name)

def clean_rating(rating):
    """清理评分"""
    if pd.isna(rating):
        return None
    try:
        rating = float(rating)
        if 0 <= rating <= 5:
            return rating
        return None
    except:
        return None

def clean_ticket_price(ticket_text):
    """清理门票价格"""
    if pd.isna(ticket_text) or not ticket_text:
        return ''
    text = str(ticket_text)
    # 如果是字典格式，提取价格信息
    if text.startswith('{') or '¥' in text:
        prices = re.findall(r'¥(\d+)', text)
        if prices:
            min_price = min(int(p) for p in prices)
            return f'¥{min_price}起'
    # 处理"免费"等情况
    if '免费' in text:
        return '免费'
    # 处理普通价格文本
    text = clean_text(text)
    if text and text != 'nan':
        return text
    return ''

def clean_open_time(time_text):
    """清理开放时间"""
    if pd.isna(time_text) or not time_text:
        return ''
    text = str(time_text)
    text = clean_text(text)
    # 简化过长的开放时间描述
    if len(text) > 100:
        # 尝试提取最主要的时间信息
        match = re.search(r'(\d{1,2}[：:]\d{2}\s*[-至到]\s*\d{1,2}[：:]\d{2})', text)
        if match:
            return match.group(1)
        if '全天' in text:
            return '全天开放'
    return text if text != 'nan' else ''

def clean_suggest_time(time_text):
    """清理建议游玩时间"""
    if pd.isna(time_text) or not time_text:
        return ''
    text = str(time_text)
    # 提取时间信息
    text = re.sub(r'建议游览时间[：:]?\s*', '', text)
    return clean_text(text)

def extract_tags(row):
    """从介绍和建议季节等字段提取标签"""
    tags = []
    
    intro = str(row.get('介绍', '')) if pd.notna(row.get('介绍')) else ''
    name = str(row.get('名字', '')) if pd.notna(row.get('名字')) else ''
    
    # 根据关键词添加标签
    tag_keywords = {
        '博物馆': '博物馆|历史',
        '古镇': '古镇|历史',
        '海滩': '海滩|休闲',
        '公园': '公园|自然',
        '寺': '寺庙|宗教',
        '塔': '古迹|历史',
        '山': '山岳|自然',
        '湖': '湖泊|自然',
        '森林': '森林|自然',
        '动物': '动物园|亲子',
        '水族': '水族馆|亲子',
        '影视': '影视|娱乐',
        '老街': '老街|历史',
        '陶瓷': '陶瓷|文化',
        '遗址': '遗址|历史',
        '夜景': '夜景',
        '建筑': '建筑',
        '迪士尼': '主题乐园|亲子',
        '游乐': '游乐场|娱乐',
        '漂流': '漂流|户外',
        '温泉': '温泉|休闲',
        '电影': '电影|娱乐',
        '草原': '草原|自然',
        '峡谷': '峡谷|自然',
        '瀑布': '瀑布|自然',
        '石窟': '石窟|历史',
        '长城': '长城|历史',
    }
    
    combined_text = name + intro
    for keyword, tag in tag_keywords.items():
        if keyword in combined_text:
            tags.extend(tag.split('|'))
    
    # 去重并限制标签数量
    tags = list(dict.fromkeys(tags))[:5]
    return '|'.join(tags) if tags else ''

def truncate_description(text, max_length=500):
    """截断过长的描述"""
    if pd.isna(text) or not text:
        return ''
    text = clean_text(str(text))
    if len(text) > max_length:
        return text[:max_length] + '...'
    return text

def process_csv_file(filepath, city_name):
    """处理单个CSV文件"""
    print(f"正在处理: {filepath}")
    
    try:
        # 尝试不同的编码
        for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"  无法读取文件: {filepath}")
            return pd.DataFrame()
        
        print(f"  读取到 {len(df)} 条记录")
        
        # 处理每一行数据
        processed_data = []
        for idx, row in df.iterrows():
            # 跳过没有名称的行
            name = clean_name(row.get('名字', ''))
            if not name:
                continue
            
            address_field = row.get('地址', '')
            
            processed_row = {
                'name': name,
                'city': city_name,
                'province': CITY_PROVINCE_MAP.get(city_name, ''),
                'address': extract_address(address_field),
                'phone': extract_phone(address_field),
                'website': extract_website(address_field),
                'rating': clean_rating(row.get('评分')),
                'ticket_price': clean_ticket_price(row.get('门票', '')),
                'open_time': clean_open_time(row.get('开放时间', '')),
                'suggest_time': clean_suggest_time(row.get('建议游玩时间', '')),
                'tags': extract_tags(row),
                'description': truncate_description(row.get('介绍', '')),
                'tips': clean_text(str(row.get('小贴士', ''))) if pd.notna(row.get('小贴士')) else ''
            }
            processed_data.append(processed_row)
        
        return pd.DataFrame(processed_data)
    
    except Exception as e:
        print(f"  处理出错: {e}")
        return pd.DataFrame()

def main():
    """主函数"""
    # 数据目录
    data_dir = Path(r'd:\visual_ProgrammingSoftware\Qoder_projects\wsq\data')
    
    # 所有城市CSV文件
    csv_files = {
        '上海.csv': '上海',
        '银川.csv': '银川',
        '海口.csv': '海口',
        '固原.csv': '固原',
        '呼和浩特.csv': '呼和浩特',
        '景德镇.csv': '景德镇'
    }
    
    # 处理所有文件
    all_data = []
    for filename, city in csv_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            df = process_csv_file(filepath, city)
            if not df.empty:
                all_data.append(df)
        else:
            print(f"文件不存在: {filepath}")
    
    if not all_data:
        print("没有处理到任何数据!")
        return
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 添加 dataid
    combined_df.insert(0, 'dataid', range(1, len(combined_df) + 1))
    
    # 去除完全重复的记录
    combined_df = combined_df.drop_duplicates(subset=['name', 'city'], keep='first')
    
    # 重新生成 dataid
    combined_df['dataid'] = range(1, len(combined_df) + 1)
    
    # 确保列顺序
    columns_order = [
        'dataid', 'name', 'city', 'province', 'address', 'phone', 
        'website', 'rating', 'ticket_price', 'open_time', 
        'suggest_time', 'tags', 'description', 'tips'
    ]
    combined_df = combined_df[columns_order]
    
    # 保存结果
    output_path = data_dir / 'merged_attractions.csv'
    combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n处理完成!")
    print(f"总共处理了 {len(combined_df)} 条景点数据")
    print(f"输出文件: {output_path}")
    
    # 显示统计信息
    print(f"\n各城市景点数量:")
    print(combined_df['city'].value_counts())
    
    # 显示前几条数据预览
    print(f"\n数据预览 (前5条):")
    print(combined_df.head().to_string())
    
    return combined_df

if __name__ == '__main__':
    df = main()
