# -*- coding: utf-8 -*-
"""
VOC 智能分析平台 - 完整功能版（已修复大量评论卡死问题）
包含所有原有功能 + 断点续传 + 自动重试
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

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="VOC 智能洞察平台",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# API 配置
# =========================
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def call_llm(api_key: str, prompt: str, max_tokens: int = 2500) -> str:
    """调用DeepSeek API，带重试和超时控制"""
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
            else:
                if attempt == 2:
                    return ""
                time.sleep(1)
        except requests.exceptions.Timeout:
            if attempt == 2:
                return ""
            time.sleep(2)
        except:
            if attempt == 2:
                return ""
            time.sleep(1)
    return ""

# =========================
# 维度归一化映射（完整版）
# =========================
DIMENSION_MAPPING = {
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

def normalize_dimension(dimension: str) -> str:
    dim_lower = dimension.lower()
    for standard, variants in DIMENSION_MAPPING.items():
        for variant in variants:
            if variant in dim_lower or dim_lower in variant:
                return standard
    return dimension

# =========================
# 维度提取函数（完整版）
# =========================
def extract_dimensions(review_text: str, star_rating: int, api_key: str) -> Tuple[str, List[str]]:
    """提取维度和情感 - 使用AI"""
    if not api_key:
        return "正面" if star_rating >= 4 else "负面" if star_rating <= 2 else "中性", []
    
    prompt = f"""分析评论，输出JSON：
评论：{review_text[:300]}
星级：{star_rating}/5

输出格式：{{"sentiment":"正面/负面/中性","dimensions":["维度1","维度2"]}}
只输出JSON："""
    
    try:
        result = call_llm(api_key, prompt, max_tokens=150)
        if not result:
            return "中性", []
        clean = re.sub(r'```json\s*|```\s*', '', result.strip())
        data = json.loads(clean)
        sentiment = data.get("sentiment", "中性")
        dimensions = [normalize_dimension(d) for d in data.get("dimensions", [])[:3]]
        return sentiment, dimensions
    except:
        return "中性", []

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
# 完整版战略洞察报告（9大模块）
# =========================
def generate_strategic_insights(
    positive_dims: dict, 
    negative_dims: dict, 
    emotion_dist: dict,
    persona_dist: dict,
    motivation_dist: dict,
    scenario_dist: dict,
    sample_reviews: List[str],
    api_key: str
) -> str:
    """生成完整的9大模块战略洞察报告"""
    pos_total = sum(positive_dims.values()) if positive_dims else 1
    neg_total = sum(negative_dims.values()) if negative_dims else 1
    
    top_pos = list(positive_dims.items())[:5] if positive_dims else []
    top_neg = list(negative_dims.items())[:5] if negative_dims else []
    top_emotion = list(emotion_dist.items())[:3] if emotion_dist else []
    top_persona = list(persona_dist.items())[:3] if persona_dist else []
    top_motivation = list(motivation_dist.items())[:3] if motivation_dist else []
    top_scenario = list(scenario_dist.items())[:3] if scenario_dist else []
    
    pos_str = "\n".join([f"  - {dim}: {count}次 ({count/pos_total*100:.1f}%)" for dim, count in top_pos])
    neg_str = "\n".join([f"  - {dim}: {count}次 ({count/neg_total*100:.1f}%)" for dim, count in top_neg])
    emotion_str = "\n".join([f"  - {emotion}: {pct:.1f}%" for emotion, pct in top_emotion])
    persona_str = "\n".join([f"  - {p}: {pct:.1f}%" for p, pct in top_persona])
    motivation_str = "\n".join([f"  - {m}: {pct:.1f}%" for m, pct in top_motivation])
    scenario_str = "\n".join([f"  - {s}: {pct:.1f}%" for s, pct in top_scenario])
    
    sample_str = "\n".join([f"- {text[:120]}..." for text in sample_reviews[:10]])
    
    prompt = f"""你是资深产品战略分析师。基于以下数据，生成一份专业、详细的战略洞察报告。

## 用户好评维度 TOP5
{pos_str}

## 用户差评维度 TOP5
{neg_str}

## 用户情绪分布 TOP3
{emotion_str}

## 用户画像分布 TOP3
{persona_str}

## 购买动机分布 TOP3
{motivation_str}

## 使用场景分布 TOP3
{scenario_str}

## 代表性用户评论
{sample_str}

## 请生成以下格式的详细报告：

# 📊 用户评论深度洞察报告

## 一、核心发现摘要
（200字以内，总结最重要的3-5个发现）

## 二、用户核心关注点分析
### 2.1 最受关注的维度
### 2.2 关注度变化趋势判断
### 2.3 与行业基准对比（如有）

## 三、好评深度分析
### 3.1 核心好评维度及占比
### 3.2 用户满意的具体场景
### 3.3 好评背后的用户需求

## 四、痛点深度分析
### 4.1 核心痛点维度及占比
### 4.2 痛点的具体表现和场景
### 4.3 痛点背后的根本原因
### 4.4 用户矛盾心理分析

## 五、用户画像与行为洞察
### 5.1 核心用户群特征
### 5.2 不同画像的诉求差异
### 5.3 购买决策驱动因素

## 六、情绪洞察
### 6.1 情绪分布概况
### 6.2 触发正向情绪的关键因素
### 6.3 触发负向情绪的关键因素

## 七、产品优化建议
### 7.1 短期改进（1-2周可执行）
### 7.2 中期改进（1-2个月）
### 7.3 长期战略方向

## 八、差异化竞争策略
### 8.1 当前市场机会点
### 8.2 建议打造的独特卖点
### 8.3 竞品对比建议

## 九、行动优先级
（按紧急重要程度排序）

请用专业、清晰、有洞察力的语言输出。"""
    
    try:
        report = call_llm(api_key, prompt, max_tokens=3000)
        if report and len(report) > 500:
            return report
    except:
        pass
    
    return generate_fallback_report(positive_dims, negative_dims, emotion_dist, persona_dist, motivation_dist)

def generate_fallback_report(positive_dims, negative_dims, emotion_dist, persona_dist, motivation_dist):
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

## 二、好评维度 TOP5
| 维度 | 提及次数 | 占比 |
|------|---------|------|
"""
    for dim, count in top_pos[:5]:
        report += f"| {dim} | {count} | {count/pos_total*100:.1f}% |\n"
    
    report += f"""
## 三、差评维度 TOP5
| 维度 | 提及次数 | 占比 |
|------|---------|------|
"""
    for dim, count in top_neg[:5]:
        report += f"| {dim} | {count} | {count/neg_total*100:.1f}% |\n"
    
    report += f"""
## 四、产品优化建议
1. 立即改进：{top_neg[0][0] if top_neg else '主要痛点'}
2. 持续强化：{top_pos[0][0] if top_pos else '核心优势'}

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

## 好评维度 TOP15
| 排名 | 维度 | 提及次数 | 占比 |
|------|------|---------|------|
"""
    for i, (dim, count) in enumerate(list(positive_dims.items())[:15], 1):
        report += f"| {i} | {dim} | {count} | {count/pos_total*100:.1f}% |\n"
    
    report += f"""
## 差评维度 TOP15
| 排名 | 维度 | 提及次数 | 占比 |
|------|------|---------|------|
"""
    for i, (dim, count) in enumerate(list(negative_dims.items())[:15], 1):
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
- 负向情绪：{emotion_dist.get('失望', 0) + emotion_dist.get('愤怒', 0) + emotion_dist.get('后悔', 0):.1f}%
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
            "环保材质很加分，摸着很舒服",
            "充电速度很快，支持快充"
        ],
        "star_rating": [5, 2, 5, 1, 4, 4, 5, 5, 5]
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
    return output.getvalue()

# =========================
# 【核心】主分析函数 - 修复大量评论卡死 + 断点续传
# =========================
def run_analysis(df: pd.DataFrame, api_key: str, progress_callback=None):
    """完整版分析 - 支持大量评论 + 断点续传"""
    df = df.copy()
    total = len(df)
    
    # 检查是否有未完成的分析（断点续传）
    start_idx = 0
    if "analysis_progress" in st.session_state:
        start_idx = st.session_state.analysis_progress
        st.info(f"📌 从第 {start_idx + 1} 条继续...")
    
    positive_dims = Counter()
    negative_dims = Counter()
    motivations = []
    emotions = []
    personas = []
    scenarios = []
    
    # 恢复已有数据
    for i in range(start_idx):
        if df.at[i, "sentiment"]:
            sentiment = df.at[i, "sentiment"]
            dims = df.at[i, "dimensions"].split(", ") if df.at[i, "dimensions"] else []
            for dim in dims:
                if dim and sentiment == "正面":
                    positive_dims[dim] += 1
                elif dim and sentiment == "负面":
                    negative_dims[dim] += 1
            motivations.append(df.at[i, "motivation"])
            emotions.append(df.at[i, "emotion"])
            personas.append(df.at[i, "persona"])
            scenarios.append(df.at[i, "scenario"])
    
    progress_bar = st.progress(start_idx / total if total > 0 else 0)
    status_text = st.empty()
    failed_indices = []
    
    for idx in range(start_idx, total):
        progress = (idx + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"📊 分析进度: {idx + 1}/{total} ({progress*100:.1f}%)")
        
        if progress_callback:
            progress_callback(idx + 1, total)
        
        try:
            row = df.iloc[idx]
            text = row["review_text"]
            rating = row["star_rating"]
            
            # 提取维度和情感
            sentiment, dimensions = extract_dimensions(text, rating, api_key)
            
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
            
            # 每10条保存进度
            if (idx + 1) % 10 == 0:
                st.session_state.analysis_progress = idx + 1
                
        except Exception as e:
            failed_indices.append(idx)
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
        
        # 避免API限流
        time.sleep(0.08)
    
    # 重试失败的
    if failed_indices:
        status_text.text(f"🔄 重试 {len(failed_indices)} 条失败评论...")
        for idx in failed_indices[:]:
            try:
                row = df.iloc[idx]
                text = row["review_text"]
                rating = row["star_rating"]
                sentiment, dimensions = extract_dimensions(text, rating, api_key)
                df.at[idx, "sentiment"] = sentiment
                df.at[idx, "dimensions"] = ", ".join(dimensions)
                df.at[idx, "analysis_status"] = "已分析"
                failed_indices.remove(idx)
                time.sleep(0.1)
            except:
                pass
    
    status_text.empty()
    progress_bar.empty()
    
    if "analysis_progress" in st.session_state:
        del st.session_state.analysis_progress
    
    # 计算分布
    total_count = len(df)
    motivation_dist = {k: v/total_count*100 for k, v in Counter(motivations).items()}
    emotion_dist = {k: v/total_count*100 for k, v in Counter(emotions).items()}
    persona_dist = {k: v/total_count*100 for k, v in Counter(personas).items()}
    scenario_dist = {k: v/total_count*100 for k, v in Counter(scenarios).items()}
    
    opportunities = discover_opportunities(dict(positive_dims), dict(negative_dims), total_count)
    
    # 生成报告
    strategic_insights = generate_strategic_insights(
        dict(positive_dims), dict(negative_dims), emotion_dist,
        persona_dist, motivation_dist, scenario_dist,
        df["review_text"].tolist()[:30], api_key
    )
    
    dimension_report = generate_detailed_dimension_report(dict(positive_dims), dict(negative_dims))
    persona_report = generate_detailed_persona_report(persona_dist, total_count)
    emotion_report = generate_detailed_emotion_report(emotion_dist)
    motivation_report = generate_detailed_motivation_report(motivation_dist)
    opportunity_report = generate_detailed_opportunity_report(opportunities)
    scenario_report = generate_detailed_scenario_report(scenario_dist, total_count)
    
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
        "scenario_report": scenario_report
    }
    
    return df, analysis_data

# =========================
# 侧边栏
# =========================
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ 配置")
        st.info("💡 DeepSeek API\n注册：https://platform.deepseek.com")
        api_key = st.text_input("API Key", type="password", placeholder="sk-...")
        
        st.markdown("---")
        uploaded_file = st.file_uploader("上传评论文件", type=["csv", "xlsx"])
        
        if st.button("🔄 清除缓存", use_container_width=True):
            if "analysis_progress" in st.session_state:
                del st.session_state.analysis_progress
            st.success("已清除")
        
        start_analysis = st.button("🚀 开始分析", use_container_width=True, type="primary")
        
        st.markdown("---")
        if st.button("📝 加载示例数据", use_container_width=True):
            return api_key, get_sample_data(), True
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.success(f"✅ 已加载 {len(df)} 条")
                return api_key, df, start_analysis
            except Exception as e:
                st.error(f"读取失败: {e}")
        return api_key, None, start_analysis

# =========================
# 主函数
# =========================
def main():
    st.title("🎯 VOC 智能洞察平台")
    st.caption("完整功能版 | 已修复大量评论卡死问题 | 支持断点续传")
    
    api_key, input_df, start_analysis = render_sidebar()
    
    if input_df is None:
        st.info("👈 左侧上传文件或点击「加载示例数据」")
        st.markdown("""
        ### 📌 全部功能
        - **数据概览**：KPI卡片、核心指标
        - **战略洞察**：完整9大模块AI报告
        - **维度分析**：好评/差评维度
        - **购买动机**：驱动因素分析
        - **情绪分析**：细粒度情绪
        - **用户画像**：身份分布
        - **使用场景**：场景分布
        - **机会发现**：量化改进机会
        - **一键导出**：Excel完整数据
        
        ### ⚡ 稳定性保障
        - ✅ 支持500+评论
        - ✅ 断点续传
        - ✅ 自动重试
        - ✅ 进度保存
        """)
        return
    
    if "review_text" not in input_df.columns:
        st.error("文件缺少 `review_text` 列")
        return
    
    df = preprocess_data(input_df)
    
    if start_analysis:
        if not api_key:
            st.warning("⚠️ 请输入 API Key")
        else:
            start_time = time.time()
            df, analysis_data = run_analysis(df, api_key)
            elapsed = time.time() - start_time
            
            st.session_state["df"] = df
            st.session_state["analysis_data"] = analysis_data
            st.success(f"✅ 分析完成！{len(df)} 条，用时 {elapsed:.1f} 秒")
            st.balloons()
    
    df = st.session_state.get("df", df)
    analysis_data = st.session_state.get("analysis_data", {})
    
    if not analysis_data:
        return
    
    # 数据概览
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("总评论", analysis_data.get("total", 0))
    col2.metric("好评维度", len(analysis_data.get("positive_dims", {})))
    col3.metric("差评维度", len(analysis_data.get("negative_dims", {})))
    col4.metric("用户画像", len(analysis_data.get("persona_dist", {})))
    col5.metric("使用场景", len(analysis_data.get("scenario_dist", {})))
    col6.metric("机会点", len(analysis_data.get("opportunities", [])))
    
    # 原始数据
    with st.expander("📋 原始数据预览", expanded=False):
        display_cols = ["review_text", "star_rating", "sentiment", "dimensions", "analysis_status"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True, height=300)
    
    # 8个Tab
    tabs = st.tabs(["🎯 战略洞察", "📊 维度分析", "💭 购买动机", "😊 情绪分析", "👤 用户画像", "📍 使用场景", "🎯 机会发现", "📥 一键导出"])
    
    with tabs[0]:
        if analysis_data.get("strategic_insights"):
            st.markdown(analysis_data["strategic_insights"])
            st.download_button("📥 导出战略报告", analysis_data["strategic_insights"], 
                              f"strategic_{datetime.now().strftime('%Y%m%d')}.md")
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(make_bar_chart(analysis_data.get("positive_dims", {}), "好评 TOP", "#2ecc71"), use_container_width=True)
            if analysis_data.get("dimension_report"):
                with st.expander("📄 详细报告"):
                    st.markdown(analysis_data["dimension_report"])
        with col2:
            st.plotly_chart(make_bar_chart(analysis_data.get("negative_dims", {}), "差评 TOP", "#e74c3c"), use_container_width=True)
    
    with tabs[2]:
        st.plotly_chart(make_pie_chart(analysis_data.get("motivation_dist", {}), "购买动机"), use_container_width=True)
        if analysis_data.get("motivation_report"):
            with st.expander("📄 详细报告"):
                st.markdown(analysis_data["motivation_report"])
    
    with tabs[3]:
        st.plotly_chart(make_emotion_chart(analysis_data.get("emotion_dist", {})), use_container_width=True)
        if analysis_data.get("emotion_report"):
            with st.expander("📄 详细报告"):
                st.markdown(analysis_data["emotion_report"])
    
    with tabs[4]:
        st.plotly_chart(make_pie_chart(analysis_data.get("persona_dist", {}), "用户画像"), use_container_width=True)
        if analysis_data.get("persona_report"):
            with st.expander("📄 详细报告"):
                st.markdown(analysis_data["persona_report"])
    
    with tabs[5]:
        st.plotly_chart(make_bar_chart(analysis_data.get("scenario_dist", {}), "使用场景", "#3498db"), use_container_width=True)
        if analysis_data.get("scenario_report"):
            with st.expander("📄 详细报告"):
                st.markdown(analysis_data["scenario_report"])
    
    with tabs[6]:
        opportunities = analysis_data.get("opportunities", [])
        for opp in opportunities[:5]:
            with st.expander(f"🎯 {opp['dimension']} - 机会分数 {opp['score']}"):
                st.write(f"提及: {opp['mentions']} | 差评率: {opp['complaint_rate']}%")
                if opp['complaint_rate'] > 50:
                    st.warning("⚠️ 紧急改进")
        if analysis_data.get("opportunity_report"):
            with st.expander("📄 详细报告"):
                st.markdown(analysis_data["opportunity_report"])
    
    with tabs[7]:
        excel_data = export_all_data(df, analysis_data)
        st.download_button("📥 导出Excel", excel_data, 
                          f"voc_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                          use_container_width=True)

if __name__ == "__main__":
    main()
