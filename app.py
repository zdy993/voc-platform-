# -*- coding: utf-8 -*-
"""
VOC 智能分析平台 - 完整版（含自主学习维度）
包含：数据概览、战略洞察、维度分析、购买动机、情绪分析、用户画像、使用场景、机会发现、一键导出
新增：自主学习维度 - AI自动发现和记忆新维度，并应用到分析中
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import re
import requests
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="VOC 智能洞察平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# API 配置
# =========================
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def call_llm(api_key: str, prompt: str, max_tokens: int = 2000) -> str:
    """调用DeepSeek API，带重试机制"""
    if not api_key:
        return ""
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }
    
    for attempt in range(3):
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=25)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                time.sleep(2)
                continue
        except:
            if attempt == 2:
                return ""
            time.sleep(1)
    return ""

# =========================
# 基础维度库（种子维度）
# =========================
BASE_DIMENSIONS = {
    "磁吸能力": ["magsafe", "magnetic", "磁力", "磁吸", "吸附力", "吸力", "磁铁", "吸得稳", "吸得住", "磁吸力"],
    "手感": ["手感", "触感", "握感", "feel", "texture", "舒服", "舒适"],
    "防滑性": ["滑", "防滑", "打滑", "容易掉", "slippery", "grip"],
    "保护性": ["保护", "防摔", "防撞", "protection", "drop", "防震", "安全", "防刮"],
    "耐用性": ["变黄", "发黄", "黄变", "耐用", "durability", "褪色", "老化", "yellowing"],
    "外观设计": ["设计", "颜色", "颜值", "外观", "design", "color", "漂亮", "好看", "美观", "开孔"],
    "清洁度": ["沾指纹", "指纹", "油污", "脏", "fingerprint", "stain", "灰尘", "易脏"],
    "安装体验": ["拆卸", "安装", "fit", "贴合", "尺寸", "松动", "紧", "难拆", "拆卸费力"],
    "性价比": ["价格", "值", "性价比", "price", "value", "worth", "便宜", "贵"],
    "相机控制": ["相机按键", "相机控制", "camera button", "按键", "按钮", "控制键"],
    "无线充电": ["无线充电", "wireless charging", "magsafe充电", "充电"],
    "物流服务": ["物流", "快递", "包装", "送货", "shipping", "delivery"],
    "售后服务": ["售后", "客服", "退货", "换货", "service", "support"]
}

# =========================
# 维度学习器类
# =========================
class DimensionLearner:
    """自主维度学习器 - 自动发现和记忆新维度"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_dimensions = BASE_DIMENSIONS.copy()
        self.learned_dimensions = {}  # 已学习的新维度 {维度名: {mentions: 次数, keywords: [关键词]}}
        self._load_from_session()
    
    def _load_from_session(self):
        """从session加载已学习的维度"""
        if "learned_dimensions" in st.session_state:
            self.learned_dimensions = st.session_state.learned_dimensions
    
    def _save_to_session(self):
        """保存学习的维度到session"""
        st.session_state.learned_dimensions = self.learned_dimensions
    
    def match_dimensions(self, text: str) -> List[str]:
        """匹配已知维度（基础维度 + 已学习维度）"""
        text_lower = text.lower()
        matched = []
        
        # 匹配基础维度
        for dim, keywords in self.base_dimensions.items():
            for kw in keywords:
                if kw in text_lower:
                    matched.append(dim)
                    break
        
        # 匹配已学习的新维度
        for dim, info in self.learned_dimensions.items():
            if dim in text:
                matched.append(dim)
            else:
                for kw in info.get("keywords", []):
                    if kw in text_lower:
                        matched.append(dim)
                        break
        
        return list(set(matched))[:4]  # 最多4个维度
    
    def discover_new_dimensions(self, text: str, rating: int, sentiment: str) -> List[str]:
        """使用AI发现新维度"""
        if not self.api_key or len(text) < 30:
            return []
        
        # 获取已有维度列表
        existing_dims = list(self.base_dimensions.keys()) + list(self.learned_dimensions.keys())
        
        prompt = f"""分析用户评论，提取用户提到的产品属性维度。

评论：{text[:200]}
评分：{rating}/5
情感：{sentiment}

已有维度：{', '.join(existing_dims[:15])}

重要：如果用户提到了**不在上述列表中**的新维度，请提取出来。

输出JSON格式（只输出JSON）：
{{
    "new_dimensions": [
        {{"name": "新维度名称", "keywords": ["关键词1", "关键词2"]}}
    ]
}}

如果没有新维度，输出：{{"new_dimensions": []}}"""
        
        try:
            result = call_llm(self.api_key, prompt, max_tokens=300)
            if not result:
                return []
            
            clean = re.sub(r'```json\s*|```\s*', '', result.strip())
            data = json.loads(clean)
            
            new_dims = []
            for new_dim in data.get("new_dimensions", []):
                dim_name = new_dim.get("name", "")
                if dim_name and dim_name not in self.base_dimensions and dim_name not in self.learned_dimensions:
                    if dim_name not in self.learned_dimensions:
                        self.learned_dimensions[dim_name] = {
                            "mentions": 1,
                            "keywords": new_dim.get("keywords", []),
                            "first_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    else:
                        self.learned_dimensions[dim_name]["mentions"] += 1
                        for kw in new_dim.get("keywords", []):
                            if kw not in self.learned_dimensions[dim_name]["keywords"]:
                                self.learned_dimensions[dim_name]["keywords"].append(kw)
                    new_dims.append(dim_name)
            
            self._save_to_session()
            return new_dims
        except:
            return []
    
    def get_all_dimensions(self) -> Dict:
        """获取所有维度（基础+学习）"""
        all_dims = self.base_dimensions.copy()
        for dim, info in self.learned_dimensions.items():
            if info["mentions"] >= 2:  # 只显示出现2次以上的
                all_dims[dim] = info.get("keywords", [dim])
        return all_dims
    
    def get_emerging_dimensions(self, min_mentions: int = 2) -> Dict:
        """获取新兴维度（出现次数>=min_mentions）"""
        return {k: v for k, v in self.learned_dimensions.items() if v["mentions"] >= min_mentions}
    
    def merge_to_base(self, dimension: str) -> bool:
        """将新维度合并到基础库"""
        if dimension in self.learned_dimensions:
            self.base_dimensions[dimension] = self.learned_dimensions[dimension].get("keywords", [dimension])
            del self.learned_dimensions[dimension]
            self._save_to_session()
            return True
        return False

# =========================
# 维度提取函数（使用学习器）
# =========================
def extract_dimensions_with_learner(review_text: str, star_rating: int, api_key: str, learner: DimensionLearner) -> Tuple[str, List[str], List[str]]:
    """使用学习器提取维度和情感，同时发现新维度"""
    if not api_key:
        sentiment = "正面" if star_rating >= 4 else "负面" if star_rating <= 2 else "中性"
        dimensions = learner.match_dimensions(review_text)
        return sentiment, dimensions, []
    
    # 使用AI分析
    prompt = f"""分析评论，输出JSON：
评论：{review_text[:250]}
星级：{star_rating}/5

输出格式：{{"sentiment":"正面/负面/中性","dimensions":["维度1","维度2"]}}
只输出JSON："""
    
    try:
        result = call_llm(api_key, prompt, max_tokens=150)
        if not result:
            sentiment = "正面" if star_rating >= 4 else "负面" if star_rating <= 2 else "中性"
            return sentiment, learner.match_dimensions(review_text), []
        
        clean = re.sub(r'```json\s*|```\s*', '', result.strip())
        data = json.loads(clean)
        sentiment = data.get("sentiment", "中性")
        ai_dimensions = data.get("dimensions", [])
        
        # 归一化维度
        normalized = []
        for d in ai_dimensions:
            # 先尝试匹配已有维度
            matched = False
            for std_dim in learner.base_dimensions:
                if d.lower() in std_dim.lower() or std_dim.lower() in d.lower():
                    normalized.append(std_dim)
                    matched = True
                    break
            if not matched:
                normalized.append(d)
        
        # 同时用学习器匹配
        learner_dims = learner.match_dimensions(review_text)
        all_dims = list(set(normalized + learner_dims))[:3]
        
        # 发现新维度
        new_dims = learner.discover_new_dimensions(review_text, star_rating, sentiment)
        
        return sentiment, all_dims, new_dims
    except:
        sentiment = "正面" if star_rating >= 4 else "负面" if star_rating <= 2 else "中性"
        return sentiment, learner.match_dimensions(review_text), []

def extract_motivation(text: str, api_key: str) -> str:
    """提取购买动机"""
    if not api_key:
        return "日常使用"
    prompt = f"分析购买动机（只输出一个词）：{text[:150]}\n选项：车载使用、商务办公、防摔保护、旅行使用、送礼、日常使用、游戏使用\n输出："
    try:
        r = call_llm(api_key, prompt, max_tokens=20)
        if not r:
            return "日常使用"
        for opt in ["车载使用", "商务办公", "防摔保护", "旅行使用", "送礼", "日常使用", "游戏使用"]:
            if opt in r:
                return opt
        return "日常使用"
    except:
        return "日常使用"

def extract_emotion(text: str, rating: int, api_key: str) -> str:
    """提取情绪"""
    if not api_key:
        return "满意" if rating >= 4 else "失望" if rating <= 2 else "平静"
    prompt = f"分析情绪（只输出一个词）：{text[:150]}\n选项：惊喜、满意、平静、失望、焦虑、愤怒、后悔\n输出："
    try:
        r = call_llm(api_key, prompt, max_tokens=20)
        if not r:
            return "满意" if rating >= 4 else "失望"
        for opt in ["惊喜", "满意", "平静", "失望", "焦虑", "愤怒", "后悔"]:
            if opt in r:
                return opt
        return "满意" if rating >= 4 else "失望"
    except:
        return "满意" if rating >= 4 else "失望"

def extract_persona(text: str, api_key: str) -> str:
    """提取用户画像"""
    if not api_key:
        return "普通用户"
    prompt = f"判断用户身份（只输出一个词）：{text[:150]}\n选项：商务人士、学生、旅行用户、家庭用户、科技爱好者、游戏用户、普通用户\n输出："
    try:
        r = call_llm(api_key, prompt, max_tokens=20)
        if not r:
            return "普通用户"
        for opt in ["商务人士", "学生", "旅行用户", "家庭用户", "科技爱好者", "游戏用户", "普通用户"]:
            if opt in r:
                return opt
        return "普通用户"
    except:
        return "普通用户"

def extract_scenario(text: str, api_key: str) -> str:
    """提取使用场景"""
    if not api_key:
        return "日常"
    prompt = f"判断使用场景（只输出一个词）：{text[:150]}\n选项：车载、办公室、旅行、健身房、家庭、户外、通勤\n输出："
    try:
        r = call_llm(api_key, prompt, max_tokens=20)
        if not r:
            return "日常"
        for opt in ["车载", "办公室", "旅行", "健身房", "家庭", "户外", "通勤"]:
            if opt in r:
                return opt
        return "日常"
    except:
        return "日常"

# =========================
# 完整版战略洞察报告
# =========================
def generate_strategic_insights(
    positive_dims: dict, 
    negative_dims: dict, 
    emotion_dist: dict,
    persona_dist: dict,
    motivation_dist: dict,
    scenario_dist: dict,
    emerging_dims: dict,
    sample_reviews: List[str],
    api_key: str
) -> str:
    """生成完整的战略洞察报告"""
    pos_total = sum(positive_dims.values()) if positive_dims else 1
    neg_total = sum(negative_dims.values()) if negative_dims else 1
    
    top_pos = list(positive_dims.items())[:5] if positive_dims else []
    top_neg = list(negative_dims.items())[:5] if negative_dims else []
    top_emerging = list(emerging_dims.items())[:3] if emerging_dims else []
    
    pos_str = "\n".join([f"  - {dim}: {count}次 ({count/pos_total*100:.1f}%)" for dim, count in top_pos])
    neg_str = "\n".join([f"  - {dim}: {count}次 ({count/neg_total*100:.1f}%)" for dim, count in top_neg])
    emerging_str = "\n".join([f"  - {dim}: 出现{info['mentions']}次" for dim, info in top_emerging])
    
    sample_str = "\n".join([f"- {text[:100]}..." for text in sample_reviews[:8]])
    
    prompt = f"""你是资深产品战略分析师。基于以下数据，生成战略洞察报告。

## 好评维度 TOP5
{pos_str}

## 差评维度 TOP5
{neg_str}

## 新发现的新兴维度
{emerging_str if emerging_str else "暂无新发现"}

## 代表性评论
{sample_str}

请生成以下格式的报告（简洁专业，500字以内）：

# 📊 用户评论深度洞察报告

## 一、核心发现摘要
（3-5个核心发现）

## 二、好评与痛点分析
### 2.1 核心优势
### 2.2 主要痛点

## 三、新维度洞察
（分析新发现的维度及其意义）

## 四、产品优化建议
### 4.1 短期改进
### 4.2 长期战略

## 五、行动优先级
1. 紧急
2. 重要
3. 持续

请用专业语言输出："""
    
    try:
        report = call_llm(api_key, prompt, max_tokens=2000)
        if report and len(report) > 200:
            return report
    except:
        pass
    
    # 降级报告
    return generate_fallback_report(positive_dims, negative_dims, emotion_dist, persona_dist, motivation_dist, emerging_dims)

def generate_fallback_report(positive_dims, negative_dims, emotion_dist, persona_dist, motivation_dist, emerging_dims):
    """降级报告"""
    pos_total = sum(positive_dims.values()) if positive_dims else 1
    neg_total = sum(negative_dims.values()) if negative_dims else 1
    top_pos = list(positive_dims.items())[:3] if positive_dims else []
    top_neg = list(negative_dims.items())[:3] if negative_dims else []
    
    report = f"""
# 📊 用户评论深度洞察报告

## 一、核心发现摘要
- 用户最认可：{top_pos[0][0] if top_pos else '核心优势'}，占比{top_pos[0][1]/pos_total*100:.1f}%
- 主要痛点：{top_neg[0][0] if top_neg else '待改进'}，占比{top_neg[0][1]/neg_total*100:.1f}%
- 正向情绪：{emotion_dist.get('惊喜',0)+emotion_dist.get('满意',0):.1f}%

## 二、好评与痛点分析
| 类型 | 维度 | 占比 |
|------|------|------|
"""
    for dim, count in top_pos[:3]:
        report += f"| 好评 | {dim} | {count/pos_total*100:.1f}% |\n"
    for dim, count in top_neg[:3]:
        report += f"| 差评 | {dim} | {count/neg_total*100:.1f}% |\n"

    if emerging_dims:
        report += f"""
## 三、新发现维度
"""
        for dim, info in list(emerging_dims.items())[:3]:
            report += f"- **{dim}**：出现{info['mentions']}次，关键词：{', '.join(info.get('keywords', [])[:3])}\n"

    report += f"""
## 四、产品优化建议
1. **立即改进**：{top_neg[0][0] if top_neg else '主要痛点'}
2. **持续强化**：{top_pos[0][0] if top_pos else '核心优势'}

---
*报告时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    return report

# =========================
# 详细报告生成
# =========================
def generate_detailed_dimension_report(positive_dims: dict, negative_dims: dict) -> str:
    pos_total = sum(positive_dims.values()) if positive_dims else 1
    neg_total = sum(negative_dims.values()) if negative_dims else 1
    
    report = f"""# 📊 维度分析详细报告

## 好评维度 TOP10
| 排名 | 维度 | 提及次数 | 占比 |
|------|------|---------|------|
"""
    for i, (dim, count) in enumerate(list(positive_dims.items())[:10], 1):
        report += f"| {i} | {dim} | {count} | {count/pos_total*100:.1f}% |\n"
    
    report += f"""
## 差评维度 TOP10
| 排名 | 维度 | 提及次数 | 占比 |
|------|------|---------|------|
"""
    for i, (dim, count) in enumerate(list(negative_dims.items())[:10], 1):
        report += f"| {i} | {dim} | {count} | {count/neg_total*100:.1f}% |\n"
    
    return report

def generate_detailed_persona_report(persona_dist: dict, total: int) -> str:
    report = f"""# 👤 用户画像详细报告

| 用户类型 | 数量 | 占比 |
|---------|------|------|
"""
    for persona, pct in list(persona_dist.items())[:10]:
        count = int(pct * total / 100)
        report += f"| {persona} | {count} | {pct:.1f}% |\n"
    return report

def generate_detailed_emotion_report(emotion_dist: dict) -> str:
    report = f"""# 😊 情绪分析报告

| 情绪 | 占比 |
|------|------|
"""
    for emotion, pct in emotion_dist.items():
        report += f"| {emotion} | {pct:.1f}% |\n"
    
    report += f"""
## 总结
- 正向情绪：{emotion_dist.get('惊喜', 0) + emotion_dist.get('满意', 0):.1f}%
- 负向情绪：{emotion_dist.get('失望', 0) + emotion_dist.get('愤怒', 0):.1f}%
"""
    return report

def generate_detailed_motivation_report(motivation_dist: dict) -> str:
    report = f"""# 💭 购买动机报告

| 动机 | 占比 |
|------|------|
"""
    for moti, pct in list(motivation_dist.items())[:10]:
        report += f"| {moti} | {pct:.1f}% |\n"
    return report

def generate_detailed_opportunity_report(opportunities: list) -> str:
    report = f"""# 🎯 机会发现报告

| 维度 | 机会分数 | 差评率 |
|------|---------|--------|
"""
    for opp in opportunities[:10]:
        report += f"| {opp['dimension']} | {opp['score']} | {opp['complaint_rate']}% |\n"
    return report

def generate_detailed_scenario_report(scenario_dist: dict, total: int) -> str:
    report = f"""# 📍 使用场景报告

| 场景 | 数量 | 占比 |
|------|------|------|
"""
    for scene, pct in list(scenario_dist.items())[:10]:
        count = int(pct * total / 100)
        report += f"| {scene} | {count} | {pct:.1f}% |\n"
    return report

def generate_emerging_dimensions_report(emerging_dims: dict) -> str:
    """生成新维度学习报告"""
    if not emerging_dims:
        return "暂无新发现的维度，继续分析更多评论后会自动学习。"
    
    report = f"""# 🧠 自主学习发现的新维度

本次分析共发现 **{len(emerging_dims)}** 个新兴维度：

| 维度名称 | 出现次数 | 关键词 | 首次发现 |
|---------|---------|--------|---------|
"""
    for dim, info in sorted(emerging_dims.items(), key=lambda x: x[1]["mentions"], reverse=True):
        keywords = ", ".join(info.get("keywords", [dim])[:3])
        report += f"| {dim} | {info['mentions']} | {keywords} | {info.get('first_seen', '未知')} |\n"
    
    report += f"""
## 💡 说明
- 这些维度是AI自动从评论中发现的新概念
- 出现次数越多，说明用户越关注
- 可点击下方按钮将高频维度加入基础库
"""
    return report

# =========================
# 图表函数
# =========================
def make_bar_chart(data: dict, title: str, color: str):
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="暂无数据", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title, height=400)
        return fig
    items = list(data.items())[:10]
    fig = go.Figure(data=[go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation='h', marker_color=color)])
    fig.update_layout(title=title, height=400, xaxis_title="提及次数")
    return fig

def make_pie_chart(data: dict, title: str):
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="暂无数据", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title, height=400)
        return fig
    fig = go.Figure(data=[go.Pie(labels=list(data.keys()), values=list(data.values()), hole=0.4)])
    fig.update_layout(title=title, height=400)
    return fig

def make_emotion_chart(emotion_dist: dict):
    if not emotion_dist:
        fig = go.Figure()
        fig.add_annotation(text="暂无数据", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="用户情绪分布", height=400)
        return fig
    order = ["惊喜", "满意", "平静", "失望", "焦虑", "愤怒", "后悔"]
    colors = {"惊喜": "#2ecc71", "满意": "#27ae60", "平静": "#95a5a6", 
              "失望": "#e67e22", "焦虑": "#e74c3c", "愤怒": "#c0392b", "后悔": "#e74c3c"}
    values = [emotion_dist.get(e, 0) for e in order if e in emotion_dist]
    labels = [e for e in order if e in emotion_dist]
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=[colors[l] for l in labels])])
    fig.update_layout(title="用户情绪分布", height=400, xaxis_title="情绪", yaxis_title="占比(%)")
    return fig

# =========================
# 数据处理
# =========================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_text"] = df["review_text"].fillna("").astype(str)
    if "star_rating" in df.columns:
        df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce").fillna(3).astype(int)
    else:
        df["star_rating"] = 3
    df["sentiment"] = ""
    df["dimensions"] = ""
    df["motivation"] = ""
    df["emotion"] = ""
    df["persona"] = ""
    df["scenario"] = ""
    df["analysis_status"] = "未分析"
    return df

def get_sample_data():
    return pd.DataFrame({
        "review_text": [
            "磁力很强，开车用很稳，商务出差必备，强烈推荐",
            "太滑了，用了一个月就发黄，后悔买这个牌子",
            "惊喜！磁吸力超强，搭配车载支架完美，质感也很好",
            "失望，边框发黄严重，才用两周就变色了",
            "办公用很好，质感不错，按键灵敏",
            "学生党，性价比高，防摔效果好，值得购买",
            "旅行时用，磁吸很稳，拍照方便，满意",
            "环保材质很加分，摸着很舒服，支持环保",
            "充电速度很快，支持快充，很方便",
            "重量很轻，拿着不累手，好评",
            "散热效果不错，玩游戏不烫手",
            "防水性能好，下雨天也不怕"
        ],
        "star_rating": [5, 2, 5, 1, 4, 4, 5, 5, 5, 4, 5, 5]
    })

def discover_opportunities(positive_dims: dict, negative_dims: dict, total: int) -> List[dict]:
    opportunities = []
    for dim, neg_count in negative_dims.items():
        pos_count = positive_dims.get(dim, 0)
        total_mentions = pos_count + neg_count
        if total_mentions > 0:
            score = (total_mentions / total) * (neg_count / total_mentions) * 100
            opportunities.append({
                "dimension": dim,
                "score": round(score, 2),
                "mentions": total_mentions,
                "complaint_rate": round(neg_count / total_mentions * 100, 1)
            })
    opportunities.sort(key=lambda x: x["score"], reverse=True)
    return opportunities[:10]

def export_all_data(df: pd.DataFrame, analysis_data: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='原始数据', index=False)
        
        if analysis_data.get("positive_dims"):
            pd.DataFrame([{"维度": k, "提及次数": v} for k, v in analysis_data["positive_dims"].items()]).to_excel(writer, sheet_name='好评维度', index=False)
        if analysis_data.get("negative_dims"):
            pd.DataFrame([{"维度": k, "提及次数": v} for k, v in analysis_data["negative_dims"].items()]).to_excel(writer, sheet_name='差评维度', index=False)
        if analysis_data.get("persona_dist"):
            pd.DataFrame([{"用户类型": k, "占比": f"{v:.1f}%"} for k, v in analysis_data["persona_dist"].items()]).to_excel(writer, sheet_name='用户画像', index=False)
        if analysis_data.get("motivation_dist"):
            pd.DataFrame([{"购买动机": k, "占比": f"{v:.1f}%"} for k, v in analysis_data["motivation_dist"].items()]).to_excel(writer, sheet_name='购买动机', index=False)
        if analysis_data.get("emotion_dist"):
            pd.DataFrame([{"情绪": k, "占比": f"{v:.1f}%"} for k, v in analysis_data["emotion_dist"].items()]).to_excel(writer, sheet_name='情绪分布', index=False)
        if analysis_data.get("scenario_dist"):
            pd.DataFrame([{"使用场景": k, "占比": f"{v:.1f}%"} for k, v in analysis_data["scenario_dist"].items()]).to_excel(writer, sheet_name='使用场景', index=False)
        if analysis_data.get("opportunities"):
            pd.DataFrame(analysis_data["opportunities"]).to_excel(writer, sheet_name='机会点', index=False)
        if analysis_data.get("emerging_dimensions"):
            pd.DataFrame([{"新维度": k, "出现次数": v["mentions"], "关键词": ", ".join(v.get("keywords", []))} 
                         for k, v in analysis_data["emerging_dimensions"].items()]).to_excel(writer, sheet_name='新发现维度', index=False)
    return output.getvalue()

# =========================
# 主分析函数（批量处理，高速）
# =========================
def run_analysis(df: pd.DataFrame, api_key: str, learner: DimensionLearner, progress_callback=None):
    df = df.copy()
    total = len(df)
    
    positive_dims = Counter()
    negative_dims = Counter()
    motivations = []
    emotions = []
    personas = []
    scenarios = []
    all_new_dims = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 批量处理，每批3条（平衡速度和API限制）
    batch_size = 3
    
    for idx in range(total):
        progress = (idx + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"🧠 分析中: {idx+1}/{total} | 已发现 {len(learner.learned_dimensions)} 个新维度")
        
        if progress_callback:
            progress_callback(idx + 1, total)
        
        try:
            row = df.iloc[idx]
            text = row["review_text"]
            rating = row["star_rating"]
            
            # 使用学习器分析
            sentiment, dimensions, new_dims = extract_dimensions_with_learner(text, rating, api_key, learner)
            
            all_new_dims.extend(new_dims)
            
            df.at[idx, "sentiment"] = sentiment
            df.at[idx, "dimensions"] = ", ".join(dimensions)
            
            for dim in dimensions:
                if sentiment == "正面":
                    positive_dims[dim] += 1
                elif sentiment == "负面":
                    negative_dims[dim] += 1
            
            # 提取其他属性
            df.at[idx, "motivation"] = extract_motivation(text, api_key)
            df.at[idx, "emotion"] = extract_emotion(text, rating, api_key)
            df.at[idx, "persona"] = extract_persona(text, api_key)
            df.at[idx, "scenario"] = extract_scenario(text, api_key)
            df.at[idx, "analysis_status"] = "已分析"
            
            motivations.append(df.at[idx, "motivation"])
            emotions.append(df.at[idx, "emotion"])
            personas.append(df.at[idx, "persona"])
            scenarios.append(df.at[idx, "scenario"])
            
            # 避免API限流
            time.sleep(0.1)
            
        except Exception as e:
            df.at[idx, "sentiment"] = "中性"
            df.at[idx, "dimensions"] = ""
            df.at[idx, "motivation"] = "日常使用"
            df.at[idx, "emotion"] = "平静"
            df.at[idx, "persona"] = "普通用户"
            df.at[idx, "scenario"] = "日常"
            df.at[idx, "analysis_status"] = "失败"
            
            motivations.append("日常使用")
            emotions.append("平静")
            personas.append("普通用户")
            scenarios.append("日常")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # 计算分布
    total_count = len(df)
    motivation_dist = {k: v/total_count*100 for k, v in Counter(motivations).items()}
    emotion_dist = {k: v/total_count*100 for k, v in Counter(emotions).items()}
    persona_dist = {k: v/total_count*100 for k, v in Counter(personas).items()}
    scenario_dist = {k: v/total_count*100 for k, v in Counter(scenarios).items()}
    
    opportunities = discover_opportunities(dict(positive_dims), dict(negative_dims), total_count)
    
    # 获取新兴维度
    emerging_dims = learner.get_emerging_dimensions(min_mentions=2)
    
    # 生成报告
    strategic_insights = generate_strategic_insights(
        dict(positive_dims), dict(negative_dims), emotion_dist,
        persona_dist, motivation_dist, scenario_dist, emerging_dims,
        df["review_text"].tolist()[:30], api_key
    )
    
    dimension_report = generate_detailed_dimension_report(dict(positive_dims), dict(negative_dims))
    persona_report = generate_detailed_persona_report(persona_dist, total_count)
    emotion_report = generate_detailed_emotion_report(emotion_dist)
    motivation_report = generate_detailed_motivation_report(motivation_dist)
    opportunity_report = generate_detailed_opportunity_report(opportunities)
    scenario_report = generate_detailed_scenario_report(scenario_dist, total_count)
    emerging_report = generate_emerging_dimensions_report(emerging_dims)
    
    analysis_data = {
        "total": total_count,
        "positive_dims": dict(positive_dims.most_common(20)),
        "negative_dims": dict(negative_dims.most_common(20)),
        "motivation_dist": motivation_dist,
        "emotion_dist": emotion_dist,
        "persona_dist": persona_dist,
        "scenario_dist": scenario_dist,
        "opportunities": opportunities,
        "strategic_insights": strategic_insights,
        "dimension_report": dimension_report,
        "persona_report": persona_report,
        "emotion_report": emotion_report,
        "motivation_report": motivation_report,
        "opportunity_report": opportunity_report,
        "scenario_report": scenario_report,
        "emerging_dimensions": emerging_dims,
        "emerging_report": emerging_report
    }
    
    return df, analysis_data

# =========================
# 侧边栏
# =========================
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ 配置")
        st.info("🧠 **自主学习模式**\n- AI自动发现新维度\n- 新维度应用到分析\n- 支持合并到基础库")
        
        api_key = st.text_input("DeepSeek API Key", type="password", placeholder="sk-...")
        
        st.markdown("---")
        uploaded_file = st.file_uploader("上传评论文件", type=["csv", "xlsx"])
        start_analysis = st.button("🚀 开始智能分析", use_container_width=True, type="primary")
        
        st.markdown("---")
        if st.button("📝 加载示例数据", use_container_width=True):
            return api_key, get_sample_data(), True
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.success(f"✅ 已加载 {len(df)} 条评论")
                return api_key, df, start_analysis
            except Exception as e:
                st.error(f"读取失败: {e}")
        return api_key, None, start_analysis

# =========================
# 主函数
# =========================
def main():
    st.title("🧠 VOC 智能洞察平台 - 自主学习版")
    st.caption("AI驱动 | 自动发现新维度 | 智能分析 | 批量处理")
    
    api_key, input_df, start_analysis = render_sidebar()
    
    if input_df is None:
        st.info("👈 左侧上传文件或点击「加载示例数据」")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🎯 核心功能
            - **📊 数据概览**：KPI指标、数据预览
            - **🎯 战略洞察**：AI深度分析报告
            - **📊 维度分析**：好评/差评维度
            - **👤 用户画像**：身份分布
            - **😊 情绪分析**：细粒度情绪
            - **💭 购买动机**：驱动因素
            - **📍 使用场景**：场景分布
            - **🎯 机会发现**：量化改进
            - **📥 一键导出**：Excel完整数据
            """)
        with col2:
            st.markdown("""
            ### 🧠 自主学习能力
            - **自动发现**：AI识别新维度（如"环保材质"）
            - **智能记忆**：记录出现次数和关键词
            - **报告应用**：新维度出现在洞察报告中
            - **可合并**：高频维度可加入基础库
            """)
        return
    
    if "review_text" not in input_df.columns:
        st.error("文件缺少 `review_text` 列")
        return
    
    df = preprocess_data(input_df)
    
    # 初始化学习器
    learner = DimensionLearner(api_key=api_key)
    
    if start_analysis:
        if not api_key:
            st.warning("⚠️ 请输入 DeepSeek API Key")
        else:
            with st.spinner(f"🧠 AI正在分析 {len(df)} 条评论，自动学习新维度..."):
                start_time = time.time()
                df, analysis_data = run_analysis(df, api_key, learner)
                elapsed = time.time() - start_time
                
                st.session_state["df"] = df
                st.session_state["analysis_data"] = analysis_data
                st.session_state["learner"] = learner
                st.success(f"✅ 分析完成！{len(df)} 条评论，用时 {elapsed:.1f} 秒")
                st.balloons()
    
    df = st.session_state.get("df", df)
    analysis_data = st.session_state.get("analysis_data", {})
    learner = st.session_state.get("learner", learner)
    
    if not analysis_data:
        return
    
    # 数据概览
    st.markdown("## 📊 数据概览")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("总评论数", analysis_data.get("total", 0))
    col2.metric("好评维度", len(analysis_data.get("positive_dims", {})))
    col3.metric("差评维度", len(analysis_data.get("negative_dims", {})))
    col4.metric("用户画像", len(analysis_data.get("persona_dist", {})))
    col5.metric("使用场景", len(analysis_data.get("scenario_dist", {})))
    col6.metric("新发现维度", len(analysis_data.get("emerging_dimensions", {})))
    
    # 核心指标
    st.markdown("---")
    st.markdown("### 📈 核心指标")
    pos_total = sum(analysis_data.get("positive_dims", {}).values())
    neg_total = sum(analysis_data.get("negative_dims", {}).values())
    col1, col2, col3 = st.columns(3)
    col1.metric("好评提及总数", pos_total)
    col2.metric("差评提及总数", neg_total)
    col3.metric("总提及次数", pos_total + neg_total)
    
    # 原始数据预览
    with st.expander("📋 原始数据预览", expanded=False):
        display_cols = ["review_text", "star_rating", "sentiment", "dimensions", "motivation", "emotion", "persona", "scenario"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True, height=300)
    
    # 9个Tab
    tabs = st.tabs([
        "🎯 战略洞察", "📊 维度分析", "🧠 新维度学习", 
        "💭 购买动机", "😊 情绪分析", "👤 用户画像", 
        "📍 使用场景", "🎯 机会发现", "📥 一键导出"
    ])
    
    # Tab 1: 战略洞察
    with tabs[0]:
        if analysis_data.get("strategic_insights"):
            st.markdown(analysis_data["strategic_insights"])
            st.download_button("📥 导出战略报告", analysis_data["strategic_insights"], 
                              f"strategic_insights_{datetime.now().strftime('%Y%m%d')}.md")
        else:
            st.info("点击「开始分析」生成报告")
    
    # Tab 2: 维度分析
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(make_bar_chart(analysis_data.get("positive_dims", {}), "好评 TOP 维度", "#2ecc71"), use_container_width=True)
        with col2:
            st.plotly_chart(make_bar_chart(analysis_data.get("negative_dims", {}), "差评 TOP 维度", "#e74c3c"), use_container_width=True)
        
        if analysis_data.get("dimension_report"):
            with st.expander("📄 查看详细维度报告"):
                st.markdown(analysis_data["dimension_report"])
    
    # Tab 3: 新维度学习
    with tabs[2]:
        st.markdown("## 🧠 自主学习发现的新维度")
        
        emerging = analysis_data.get("emerging_dimensions", {})
        if emerging:
            st.success(f"✨ AI自动发现了 {len(emerging)} 个新维度")
            
            for dim, info in sorted(emerging.items(), key=lambda x: x[1]["mentions"], reverse=True):
                with st.expander(f"🔍 {dim} (出现 {info['mentions']} 次)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**关键词**：{', '.join(info.get('keywords', []))}")
                        st.markdown(f"**首次发现**：{info.get('first_seen', '未知')}")
                    with col2:
                        if info["mentions"] >= 3:
                            st.info(f"💡 该维度已出现 {info['mentions']} 次")
                            if st.button(f"➕ 加入基础库", key=f"merge_{dim}"):
                                if learner.merge_to_base(dim):
                                    st.success(f"已添加 {dim} 到基础维度库！")
                                    st.rerun()
        else:
            st.info("🤖 暂无新发现维度，继续分析更多评论后会自动学习")
        
        if analysis_data.get("emerging_report"):
            with st.expander("📄 查看详细学习报告"):
                st.markdown(analysis_data["emerging_report"])
    
    # Tab 4: 购买动机
    with tabs[3]:
        st.plotly_chart(make_pie_chart(analysis_data.get("motivation_dist", {}), "购买动机分布"), use_container_width=True)
        if analysis_data.get("motivation_report"):
            with st.expander("📄 查看详细报告"):
                st.markdown(analysis_data["motivation_report"])
    
    # Tab 5: 情绪分析
    with tabs[4]:
        st.plotly_chart(make_emotion_chart(analysis_data.get("emotion_dist", {})), use_container_width=True)
        if analysis_data.get("emotion_report"):
            with st.expander("📄 查看详细报告"):
                st.markdown(analysis_data["emotion_report"])
    
    # Tab 6: 用户画像
    with tabs[5]:
        st.plotly_chart(make_pie_chart(analysis_data.get("persona_dist", {}), "用户画像分布"), use_container_width=True)
        if analysis_data.get("persona_report"):
            with st.expander("📄 查看详细报告"):
                st.markdown(analysis_data["persona_report"])
    
    # Tab 7: 使用场景
    with tabs[6]:
        st.plotly_chart(make_bar_chart(analysis_data.get("scenario_dist", {}), "使用场景分布", "#3498db"), use_container_width=True)
        if analysis_data.get("scenario_report"):
            with st.expander("📄 查看详细报告"):
                st.markdown(analysis_data["scenario_report"])
    
    # Tab 8: 机会发现
    with tabs[7]:
        opportunities = analysis_data.get("opportunities", [])
        if opportunities:
            for opp in opportunities[:5]:
                with st.expander(f"🎯 {opp['dimension']} - 机会分数 {opp['score']}"):
                    st.write(f"提及次数：{opp['mentions']}")
                    st.write(f"差评率：{opp['complaint_rate']}%")
                    if opp['complaint_rate'] > 50:
                        st.warning("⚠️ 紧急改进项")
                    elif opp['complaint_rate'] > 30:
                        st.warning("📌 建议改进")
            if analysis_data.get("opportunity_report"):
                with st.expander("📄 查看详细报告"):
                    st.markdown(analysis_data["opportunity_report"])
        else:
            st.info("暂无机会点数据")
    
    # Tab 9: 一键导出
    with tabs[8]:
        st.markdown("## 📥 一键导出")
        excel_data = export_all_data(df, analysis_data)
        st.download_button("📥 导出全部数据 (Excel)", excel_data,
                          file_name=f"voc_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                          use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📄 各模块报告")
        col1, col2 = st.columns(2)
        with col1:
            if analysis_data.get("strategic_insights"):
                st.download_button("战略洞察报告", analysis_data["strategic_insights"], "strategic_insights.md")
            if analysis_data.get("dimension_report"):
                st.download_button("维度分析报告", analysis_data["dimension_report"], "dimension_report.md")
            if analysis_data.get("emerging_report"):
                st.download_button("新维度学习报告", analysis_data["emerging_report"], "emerging_report.md")
        with col2:
            if analysis_data.get("persona_report"):
                st.download_button("用户画像报告", analysis_data["persona_report"], "persona_report.md")
            if analysis_data.get("opportunity_report"):
                st.download_button("机会发现报告", analysis_data["opportunity_report"], "opportunity_report.md")

if __name__ == "__main__":
    main()
