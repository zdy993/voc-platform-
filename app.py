# -*- coding: utf-8 -*-
"""
VOC 智能分析平台 - 完整版（修复 + 维度自主学习）
包含：数据概览、战略洞察、维度分析、购买动机、情绪分析、用户画像、使用场景、机会发现、一键导出
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
from typing import Dict, List, Tuple, Optional
import io
import time
import copy

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

# 默认维度映射（初始化时会复制到 session_state）
DEFAULT_DIMENSION_MAPPING = {
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
# 辅助函数：安全调用 LLM
# =========================
def call_llm(api_key: str, prompt: str, max_tokens: int = 1500, retries: int = 2) -> str:
    """带重试和超时的LLM调用"""
    if not api_key:
        return ""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }
    for attempt in range(retries + 1):
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                if attempt < retries:
                    time.sleep(2)
                    continue
                return f"API错误: {response.status_code}"
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            return f"请求失败: {str(e)}"
    return ""

# =========================
# 维度归一化（使用 session_state 中的动态映射）
# =========================
def normalize_dimension(dimension: str) -> str:
    """根据当前映射表归一化维度"""
    mapping = st.session_state.get("dimension_mapping", DEFAULT_DIMENSION_MAPPING)
    dim_lower = dimension.lower()
    for standard, variants in mapping.items():
        for variant in variants:
            if variant in dim_lower or dim_lower in variant:
                return standard
    # 未匹配到，记录为新维度
    return dimension

# =========================
# 批量属性提取（一次API调用提取所有字段）
# =========================
def batch_extract_all(reviews: List[str], ratings: List[int], api_key: str) -> List[Dict]:
    """批量提取多条评论的所有属性，返回列表，每个元素包含 sentiment, dimensions, motivation, emotion, persona, scenario"""
    if not api_key or not reviews:
        return []
    
    # 构造批量请求
    items = []
    for i, (text, rating) in enumerate(zip(reviews, ratings)):
        items.append(f"评论{i+1}: {text[:200]} (星级:{rating}/5)")
    batch_text = "\n".join(items)
    
    prompt = f"""请分析以下{len(reviews)}条评论，为每条评论提取以下字段，并以JSON数组格式返回（数组元素顺序与评论顺序一致）。
每个元素需包含：sentiment（正面/负面/中性）, dimensions（维度列表，最多3个）, motivation（购买动机：车载使用/商务办公/防摔保护/旅行使用/送礼/日常使用/游戏使用）, emotion（情绪：惊喜/满意/平静/失望/焦虑/愤怒/后悔）, persona（用户身份：商务人士/学生/旅行用户/家庭用户/科技爱好者/游戏用户/普通用户）, scenario（使用场景：车载/办公室/旅行/健身房/家庭/户外/通勤）。

评论列表：
{batch_text}

只输出JSON数组，不要有其他文字。"""
    
    result = call_llm(api_key, prompt, max_tokens=2000)
    if not result or result.startswith("API错误") or result.startswith("请求失败"):
        # 降级：逐条处理
        return [extract_single_attributes(text, rating, api_key) for text, rating in zip(reviews, ratings)]
    
    try:
        # 清理可能的 markdown 标记
        clean = re.sub(r'```json\s*|```\s*', '', result.strip())
        data = json.loads(clean)
        if isinstance(data, list) and len(data) == len(reviews):
            # 确保每个元素都有所有字段
            for item in data:
                item.setdefault("sentiment", "中性")
                item.setdefault("dimensions", [])
                item.setdefault("motivation", "日常使用")
                item.setdefault("emotion", "平静")
                item.setdefault("persona", "普通用户")
                item.setdefault("scenario", "日常")
            return data
        else:
            # 格式不对，降级
            return [extract_single_attributes(text, rating, api_key) for text, rating in zip(reviews, ratings)]
    except Exception:
        # 降级
        return [extract_single_attributes(text, rating, api_key) for text, rating in zip(reviews, ratings)]

def extract_single_attributes(text: str, rating: int, api_key: str) -> Dict:
    """单条提取（降级方案）"""
    prompt = f"""分析评论：{text[:200]} (星级:{rating}/5)
请输出JSON格式：{{"sentiment":"正面/负面/中性","dimensions":["维度1","维度2"],"motivation":"购买动机","emotion":"情绪","persona":"用户身份","scenario":"使用场景"}}
只输出JSON："""
    result = call_llm(api_key, prompt, max_tokens=300)
    try:
        clean = re.sub(r'```json\s*|```\s*', '', result.strip())
        data = json.loads(clean)
        data.setdefault("sentiment", "中性")
        data.setdefault("dimensions", [])
        data.setdefault("motivation", "日常使用")
        data.setdefault("emotion", "平静")
        data.setdefault("persona", "普通用户")
        data.setdefault("scenario", "日常")
        return data
    except:
        return {
            "sentiment": "中性",
            "dimensions": [],
            "motivation": "日常使用",
            "emotion": "平静",
            "persona": "普通用户",
            "scenario": "日常"
        }

# =========================
# 维度自主学习：记录新维度
# =========================
def record_new_dimensions(dimensions: List[str]):
    """将未映射的维度记录到 session_state 的新维度池"""
    mapping = st.session_state.get("dimension_mapping", DEFAULT_DIMENSION_MAPPING)
    known = set(mapping.keys())
    new_dims = set()
    for d in dimensions:
        if d not in known:
            new_dims.add(d)
    if new_dims:
        if "new_dimensions" not in st.session_state:
            st.session_state.new_dimensions = set()
        st.session_state.new_dimensions.update(new_dims)

# =========================
# 战略洞察报告生成
# =========================
def generate_strategic_insights(
    positive_dims: dict, 
    negative_dims: dict, 
    emotion_dist: dict,
    persona_dist: dict,
    motivation_dist: dict,
    sample_reviews: List[str],
    api_key: str
) -> str:
    pos_total = sum(positive_dims.values()) if positive_dims else 1
    neg_total = sum(negative_dims.values()) if negative_dims else 1
    
    top_pos = list(positive_dims.items())[:5] if positive_dims else []
    top_neg = list(negative_dims.items())[:5] if negative_dims else []
    top_emotion = list(emotion_dist.items())[:3] if emotion_dist else []
    top_persona = list(persona_dist.items())[:3] if persona_dist else []
    top_motivation = list(motivation_dist.items())[:3] if motivation_dist else []
    
    pos_str = "\n".join([f"  - {dim}: {count}次 ({count/pos_total*100:.1f}%)" for dim, count in top_pos])
    neg_str = "\n".join([f"  - {dim}: {count}次 ({count/neg_total*100:.1f}%)" for dim, count in top_neg])
    emotion_str = "\n".join([f"  - {emotion}: {pct:.1f}%" for emotion, pct in top_emotion])
    persona_str = "\n".join([f"  - {p}: {pct:.1f}%" for p, pct in top_persona])
    motivation_str = "\n".join([f"  - {m}: {pct:.1f}%" for m, pct in top_motivation])
    
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
### 4.4 用户矛盾心理分析（如既要A又要B）

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
（按紧急重要程度排序，给出具体可执行项）

请用专业、清晰、有洞察力的语言输出。"""
    
    try:
        report = call_llm(api_key, prompt, max_tokens=2500)
        if report and not report.startswith("API错误") and not report.startswith("请求失败"):
            return report
    except:
        pass
    
    return generate_fallback_report(positive_dims, negative_dims, emotion_dist, persona_dist, motivation_dist)

def generate_fallback_report(positive_dims, negative_dims, emotion_dist, persona_dist, motivation_dist):
    pos_total = sum(positive_dims.values()) if positive_dims else 1
    neg_total = sum(negative_dims.values()) if negative_dims else 1
    top_pos = list(positive_dims.items())[:3] if positive_dims else []
    top_neg = list(negative_dims.items())[:3] if negative_dims else []
    
    report = f"""
# 📊 用户评论深度洞察报告

## 一、核心发现摘要
- 用户最认可的是{top_pos[0][0] if top_pos else '产品核心功能'}，提及占比{top_pos[0][1]/pos_total*100:.1f}%
- 用户最不满意的是{top_neg[0][0] if top_neg else '待改进项'}，提及占比{top_neg[0][1]/neg_total*100:.1f}%
- 核心用户群为{list(persona_dist.keys())[0] if persona_dist else '主流用户'}，占比{list(persona_dist.values())[0] if persona_dist else 0:.1f}%

## 二、用户核心关注点分析
用户最关心的维度：{', '.join([d for d, _ in top_pos[:3]])}
改进机会最大的维度：{', '.join([d for d, _ in top_neg[:3]])}

## 三、好评深度分析
| 维度 | 提及次数 | 占比 |
|------|---------|------|
"""
    for dim, count in top_pos[:5]:
        report += f"| {dim} | {count} | {count/pos_total*100:.1f}% |\n"
    
    report += f"""
## 四、痛点深度分析
| 维度 | 提及次数 | 占比 |
|------|---------|------|
"""
    for dim, count in top_neg[:5]:
        report += f"| {dim} | {count} | {count/neg_total*100:.1f}% |\n"
    
    report += f"""
## 五、产品优化建议
1. **强化优势**：继续优化{top_pos[0][0] if top_pos else '核心优势'}体验
2. **改进痛点**：优先解决{top_neg[0][0] if top_neg else '主要痛点'}问题
3. **差异化竞争**：在{top_pos[0][0] if top_pos else '核心'}维度打造独特卖点

---
*报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    return report

# =========================
# 详细报告生成（各模块）
# =========================
def generate_detailed_dimension_report(positive_dims: dict, negative_dims: dict) -> str:
    pos_total = sum(positive_dims.values()) if positive_dims else 1
    neg_total = sum(negative_dims.values()) if negative_dims else 1
    
    report = f"""# 📊 维度分析详细报告

## 一、好评维度分析

| 排名 | 维度 | 提及次数 | 占比 |
|------|------|---------|------|
"""
    for i, (dim, count) in enumerate(list(positive_dims.items())[:15], 1):
        report += f"| {i} | {dim} | {count} | {count/pos_total*100:.1f}% |\n"
    
    report += f"""
## 二、差评维度分析

| 排名 | 维度 | 提及次数 | 占比 |
|------|------|---------|------|
"""
    for i, (dim, count) in enumerate(list(negative_dims.items())[:15], 1):
        report += f"| {i} | {dim} | {count} | {count/neg_total*100:.1f}% |\n"
    
    return report

def generate_detailed_persona_report(persona_dist: dict, total: int) -> str:
    report = f"""# 👤 用户画像详细报告

## 用户画像分布

| 排名 | 用户类型 | 数量 | 占比 |
|------|---------|------|------|
"""
    for i, (persona, pct) in enumerate(list(persona_dist.items())[:10], 1):
        count = int(pct * total / 100)
        report += f"| {i} | {persona} | {count} | {pct:.1f}% |\n"
    
    report += f"""
## 画像解读

"""
    top_persona = list(persona_dist.keys())[0] if persona_dist else "未知"
    report += f"核心用户群是 **{top_persona}**，占整体的 {list(persona_dist.values())[0] if persona_dist else 0:.1f}%。\n"
    
    return report

def generate_detailed_emotion_report(emotion_dist: dict) -> str:
    report = f"""# 😊 情绪分析详细报告

## 情绪分布

| 排名 | 情绪类型 | 占比 |
|------|---------|------|
"""
    emotion_order = ["惊喜", "满意", "平静", "失望", "焦虑", "愤怒", "后悔"]
    sorted_emotions = [(e, emotion_dist.get(e, 0)) for e in emotion_order if e in emotion_dist]
    for i, (emotion, pct) in enumerate(sorted_emotions, 1):
        report += f"| {i} | {emotion} | {pct:.1f}% |\n"
    
    report += f"""
## 情绪解读

- **正向情绪**（惊喜+满意）：{emotion_dist.get('惊喜', 0) + emotion_dist.get('满意', 0):.1f}%
- **负向情绪**（失望+愤怒+后悔+焦虑）：{emotion_dist.get('失望', 0) + emotion_dist.get('愤怒', 0) + emotion_dist.get('后悔', 0) + emotion_dist.get('焦虑', 0):.1f}%
- **中性情绪**（平静）：{emotion_dist.get('平静', 0):.1f}%
"""
    return report

def generate_detailed_motivation_report(motivation_dist: dict) -> str:
    report = f"""# 💭 购买动机详细报告

## 购买动机分布

| 排名 | 购买动机 | 占比 |
|------|---------|------|
"""
    for i, (moti, pct) in enumerate(list(motivation_dist.items())[:10], 1):
        report += f"| {i} | {moti} | {pct:.1f}% |\n"
    
    return report

def generate_detailed_opportunity_report(opportunities: list) -> str:
    report = f"""# 🎯 机会发现详细报告

## 机会点排行榜

| 排名 | 维度 | 机会分数 | 提及次数 | 差评率 |
|------|------|---------|---------|--------|
"""
    for i, opp in enumerate(opportunities[:10], 1):
        report += f"| {i} | {opp['dimension']} | {opp['score']} | {opp['mentions']} | {opp['complaint_rate']}% |\n"
    
    report += f"""
## 行动建议

"""
    for opp in opportunities[:5]:
        report += f"### {opp['dimension']}\n"
        report += f"- 机会分数：{opp['score']}\n"
        report += f"- 差评率：{opp['complaint_rate']}%\n"
        if opp['complaint_rate'] > 50:
            report += f"- ⚠️ 紧急建议：这是用户最不满意的地方，建议优先改进\n\n"
        elif opp['complaint_rate'] > 30:
            report += f"- 📌 建议：用户对此有明显不满，建议尽快优化\n\n"
        else:
            report += f"- 💡 建议：虽有改进空间，但优先级可适当降低\n\n"
    
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
# 数据预处理
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
            "用几天就变黄了，而且很滑，垃圾产品，不会再买",
            "手感很好，防滑设计不错，不沾指纹",
            "沾指纹严重，看着很脏，影响心情",
            "保护性很好，摔了几次手机没事，边框结实",
            "拆卸太费力了，差点把手机刮花，设计有问题",
            "相机按键很灵敏，但是太灵敏了容易误触",
            "磁力弱，吸不住车载支架，开车时掉了",
            "包装精美，物流很快，整体满意"
        ],
        "star_rating": [5, 2, 5, 1, 4, 4, 5, 1, 5, 2, 5, 2, 3, 2, 4]
    })

# =========================
# 导出所有数据
# =========================
def export_all_data(df: pd.DataFrame, analysis_data: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='原始数据', index=False)
        if analysis_data.get("positive_dims"):
            pos_df = pd.DataFrame([
                {"维度": k, "提及次数": v, "类型": "好评"} 
                for k, v in analysis_data["positive_dims"].items()
            ])
            pos_df.to_excel(writer, sheet_name='好评维度', index=False)
        if analysis_data.get("negative_dims"):
            neg_df = pd.DataFrame([
                {"维度": k, "提及次数": v, "类型": "差评"} 
                for k, v in analysis_data["negative_dims"].items()
            ])
            neg_df.to_excel(writer, sheet_name='差评维度', index=False)
        if analysis_data.get("persona_dist"):
            persona_df = pd.DataFrame([
                {"用户类型": k, "占比": f"{v:.1f}%"} 
                for k, v in analysis_data["persona_dist"].items()
            ])
            persona_df.to_excel(writer, sheet_name='用户画像', index=False)
        if analysis_data.get("motivation_dist"):
            moti_df = pd.DataFrame([
                {"购买动机": k, "占比": f"{v:.1f}%"} 
                for k, v in analysis_data["motivation_dist"].items()
            ])
            moti_df.to_excel(writer, sheet_name='购买动机', index=False)
        if analysis_data.get("emotion_dist"):
            emotion_df = pd.DataFrame([
                {"情绪": k, "占比": f"{v:.1f}%"} 
                for k, v in analysis_data["emotion_dist"].items()
            ])
            emotion_df.to_excel(writer, sheet_name='情绪分布', index=False)
        if analysis_data.get("opportunities"):
            opp_df = pd.DataFrame(analysis_data["opportunities"])
            opp_df.to_excel(writer, sheet_name='机会点', index=False)
    return output.getvalue()

# =========================
# 机会发现
# =========================
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

# =========================
# 主分析函数（优化版：批量处理）
# =========================
def run_analysis(df: pd.DataFrame, api_key: str, progress_callback=None):
    df = df.copy()
    total = len(df)
    batch_size = 10  # 每批处理10条
    
    positive_dims = Counter()
    negative_dims = Counter()
    motivations = []
    emotions = []
    personas = []
    scenarios = []
    
    # 分批处理
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_df = df.iloc[start_idx:end_idx]
        texts = batch_df["review_text"].tolist()
        ratings = batch_df["star_rating"].tolist()
        
        # 批量提取
        batch_results = batch_extract_all(texts, ratings, api_key)
        
        # 填充结果
        for idx_in_batch, (row_idx, row) in enumerate(batch_df.iterrows()):
            if idx_in_batch < len(batch_results):
                res = batch_results[idx_in_batch]
            else:
                res = {"sentiment": "中性", "dimensions": [], "motivation": "日常使用", 
                       "emotion": "平静", "persona": "普通用户", "scenario": "日常"}
            
            sentiment = res.get("sentiment", "中性")
            dimensions = res.get("dimensions", [])
            # 归一化维度
            normalized_dims = [normalize_dimension(d) for d in dimensions]
            # 记录新维度（用于自主学习）
            record_new_dimensions(normalized_dims)
            
            df.at[row_idx, "sentiment"] = sentiment
            df.at[row_idx, "dimensions"] = ", ".join(normalized_dims[:3])
            
            for dim in normalized_dims:
                if sentiment == "正面":
                    positive_dims[dim] += 1
                elif sentiment == "负面":
                    negative_dims[dim] += 1
            
            df.at[row_idx, "motivation"] = res.get("motivation", "日常使用")
            df.at[row_idx, "emotion"] = res.get("emotion", "平静")
            df.at[row_idx, "persona"] = res.get("persona", "普通用户")
            df.at[row_idx, "scenario"] = res.get("scenario", "日常")
            df.at[row_idx, "analysis_status"] = "已分析"
            
            motivations.append(df.at[row_idx, "motivation"])
            emotions.append(df.at[row_idx, "emotion"])
            personas.append(df.at[row_idx, "persona"])
            scenarios.append(df.at[row_idx, "scenario"])
        
        # 更新进度
        if progress_callback:
            progress_callback(end_idx, total)
    
    total_count = len(df)
    motivation_dist = {k: v/total_count*100 for k, v in Counter(motivations).items()}
    emotion_dist = {k: v/total_count*100 for k, v in Counter(emotions).items()}
    persona_dist = {k: v/total_count*100 for k, v in Counter(personas).items()}
    scenario_dist = {k: v/total_count*100 for k, v in Counter(scenarios).items()}
    
    opportunities = discover_opportunities(dict(positive_dims), dict(negative_dims), total_count)
    
    # 生成各类报告
    strategic_insights = generate_strategic_insights(
        dict(positive_dims), dict(negative_dims), emotion_dist, 
        persona_dist, motivation_dist, df["review_text"].tolist(), api_key
    )
    
    dimension_report = generate_detailed_dimension_report(dict(positive_dims), dict(negative_dims))
    persona_report = generate_detailed_persona_report(persona_dist, total_count)
    emotion_report = generate_detailed_emotion_report(emotion_dist)
    motivation_report = generate_detailed_motivation_report(motivation_dist)
    opportunity_report = generate_detailed_opportunity_report(opportunities)
    
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
        "opportunity_report": opportunity_report
    }
    
    return df, analysis_data

# =========================
# 侧边栏
# =========================
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ 配置")
        st.info("💡 使用 DeepSeek API（新用户免费500万tokens）\n注册：https://platform.deepseek.com")
        api_key = st.text_input("API Key", type="password", placeholder="sk-...")
        
        st.markdown("---")
        uploaded_file = st.file_uploader("上传评论文件", type=["csv", "xlsx"])
        start_analysis = st.checkbox("🚀 开始分析", value=False)
        
        st.markdown("---")
        if st.button("📝 加载示例数据", use_container_width=True):
            return api_key, get_sample_data(), start_analysis
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                return api_key, df, start_analysis
            except Exception as e:
                st.error(f"读取失败: {e}")
        return api_key, None, start_analysis

# =========================
# 维度管理UI（修复合并bug + 智能合并增强）
# =========================
def render_dimension_management():
    st.markdown("### 🧠 维度自主学习与管理")
    st.caption("管理维度映射，将新维度纳入分析体系。")

    # 获取当前映射和新维度
    mapping = st.session_state.get("dimension_mapping", DEFAULT_DIMENSION_MAPPING)
    new_dims = st.session_state.get("new_dimensions", set())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**现有维度映射**")
        for standard, variants in mapping.items():
            st.write(f"- **{standard}**: {', '.join(variants)}")
    with col2:
        st.markdown("**待学习的新维度**")
        if new_dims:
            for dim in new_dims:
                st.write(f"- {dim}")
            # 提供操作按钮
            if st.button("✨ 智能合并新维度（使用LLM）"):
                api_key = st.session_state.get("api_key", "")
                if not api_key:
                    st.error("❌ 请先在左侧输入有效的 API Key 才能使用智能合并")
                elif not new_dims:
                    st.info("当前没有待学习的新维度")
                else:
                    with st.spinner("AI 正在分析新维度归属..."):
                        prompt = f"""现有维度体系：{list(mapping.keys())}
新出现的维度：{list(new_dims)}
请将新维度智能归类到现有维度中，如果无法归类则建议创建新维度。
输出JSON格式：{{"merge": {{"新维度1": "现有维度A", "新维度2": "现有维度B"}}, "create": ["新维度X", "新维度Y"]}}
只输出JSON。"""
                        result = call_llm(api_key, prompt, max_tokens=500)
                        if result.startswith("API错误") or result.startswith("请求失败"):
                            st.error(f"智能合并 API 调用失败：{result}")
                        else:
                            try:
                                clean = re.sub(r'```json\s*|```\s*', '', result.strip())
                                data = json.loads(clean)
                                merge_map = data.get("merge", {})
                                create_list = data.get("create", [])
                                
                                # 处理合并
                                for new_dim, target in merge_map.items():
                                    if target in mapping:
                                        if new_dim not in mapping[target]:
                                            mapping[target].append(new_dim)
                                    else:
                                        # 目标不存在则创建
                                        mapping[new_dim] = [new_dim]
                                    # 从新维度池中移除
                                    if new_dim in st.session_state.new_dimensions:
                                        st.session_state.new_dimensions.remove(new_dim)
                                
                                # 处理新建
                                for new_dim in create_list:
                                    if new_dim not in mapping:
                                        mapping[new_dim] = [new_dim]
                                    if new_dim in st.session_state.new_dimensions:
                                        st.session_state.new_dimensions.remove(new_dim)
                                
                                st.session_state.dimension_mapping = mapping
                                st.success("✅ 智能合并完成！")
                                st.rerun()
                            except json.JSONDecodeError as e:
                                st.error(f"❌ AI 返回格式错误，无法解析：{e}\n原始返回：{result[:200]}...")
                            except Exception as e:
                                st.error(f"❌ 智能合并处理异常：{e}")
            if st.button("❌ 清空所有新维度记录"):
                st.session_state.new_dimensions = set()
                st.rerun()
        else:
            st.info("暂无新维度，所有维度已映射。")

    # 手动添加维度
    with st.expander("✏️ 手动管理维度映射"):
        col1, col2 = st.columns(2)
        with col1:
            new_standard = st.text_input("新维度名称（标准）")
            new_variants = st.text_input("同义词（用逗号分隔）")
            if st.button("➕ 添加新维度"):
                if new_standard and new_variants:
                    variants_list = [v.strip() for v in new_variants.split(",") if v.strip()]
                    if new_standard not in mapping:
                        mapping[new_standard] = variants_list
                    else:
                        mapping[new_standard].extend(variants_list)
                    st.session_state.dimension_mapping = mapping
                    st.success(f"已添加维度 {new_standard}")
                    st.rerun()
        with col2:
            st.markdown("**合并维度**（将现有维度或新词合并到目标）")
            # 选择目标维度
            target = st.selectbox("选择目标维度", list(mapping.keys()))
            merge_input = st.text_input("要合并的维度名称（支持标准维度或新词）")
            if st.button("🔗 合并到目标"):
                if merge_input and target and merge_input != target:
                    merge_clean = merge_input.strip()
                    target_clean = target.strip()
                    # 1. 如果 merge_clean 是现有的标准维度（键），需要彻底转移并删除旧键
                    if merge_clean in mapping:
                        # 将旧维度的所有同义词转移到目标
                        if merge_clean not in mapping[target_clean]:
                            mapping[target_clean].extend(mapping[merge_clean])
                        # 删除被合并的旧维度键
                        del mapping[merge_clean]
                    else:
                        # 2. 否则当作新词/同义词，直接追加
                        if merge_clean not in mapping[target_clean]:
                            mapping[target_clean].append(merge_clean)
                    # 3. 从新维度池中清理（不区分大小写，避免残留）
                    new_dims_set = st.session_state.get("new_dimensions", set())
                    to_remove = [d for d in new_dims_set if d.lower() == merge_clean.lower() or d == target_clean]
                    for d in to_remove:
                        new_dims_set.discard(d)
                    st.session_state.new_dimensions = new_dims_set
                    # 4. 更新映射并刷新
                    st.session_state.dimension_mapping = mapping
                    st.success(f"✅ 已将「{merge_clean}」成功合并到「{target_clean}」")
                    st.rerun()
                elif merge_input == target:
                    st.warning("不能将维度合并到自身")
                else:
                    st.warning("请输入要合并的维度名称")
        if st.button("🔄 重置为默认映射"):
            st.session_state.dimension_mapping = copy.deepcopy(DEFAULT_DIMENSION_MAPPING)
            st.session_state.new_dimensions = set()
            st.success("已重置默认映射")
            st.rerun()

# =========================
# 主函数
# =========================
def main():
    # 初始化 session_state
    if "dimension_mapping" not in st.session_state:
        st.session_state.dimension_mapping = copy.deepcopy(DEFAULT_DIMENSION_MAPPING)
    if "new_dimensions" not in st.session_state:
        st.session_state.new_dimensions = set()
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    st.title("🎯 VOC 智能洞察平台")
    st.caption("AI驱动的消费者洞察 | 完整数据概览 + 战略分析报告 | 一键导出所有数据")

    api_key, input_df, start_analysis = render_sidebar()
    st.session_state.api_key = api_key  # 保存用于维度管理

    if input_df is None:
        st.info("👈 左侧上传文件或点击「加载示例数据」")
        st.markdown("""
        ### 📌 功能说明
        - **数据概览**：KPI卡片、完整数据表格、核心指标
        - **战略洞察**：AI生成的深度战略分析报告
        - **维度分析**：归一化后的好评/差评维度 + 维度自主学习管理
        - **用户画像**：用户身份分布和特征
        - **情绪分析**：细粒度情绪分布
        - **购买动机**：用户购买驱动因素
        - **机会发现**：量化改进机会
        - **一键导出**：导出所有分析数据为Excel
        """)
        return

    if "review_text" not in input_df.columns:
        st.error("文件缺少 `review_text` 列")
        st.write("当前列名:", input_df.columns.tolist())
        return

    df = preprocess_data(input_df)

    if start_analysis:
        if not api_key:
            st.warning("⚠️ 请输入 API Key")
        else:
            with st.spinner("分析中（批量处理，请耐心等待）..."):
                progress_bar = st.progress(0)
                def update(p, t):
                    progress_bar.progress(p / t)

                df, analysis_data = run_analysis(df, api_key, update)
                st.session_state["df"] = df
                st.session_state["analysis_data"] = analysis_data
                st.success("✅ 分析完成！")

    df = st.session_state.get("df", df)
    analysis_data = st.session_state.get("analysis_data", {})

    if not analysis_data:
        analysis_data = {
            "total": len(df),
            "positive_dims": {},
            "negative_dims": {},
            "motivation_dist": {},
            "emotion_dist": {},
            "persona_dist": {},
            "scenario_dist": {},
            "opportunities": [],
            "strategic_insights": "",
            "dimension_report": "",
            "persona_report": "",
            "emotion_report": "",
            "motivation_report": "",
            "opportunity_report": ""
        }

    # ========== 数据概览 ==========
    st.markdown("---")
    st.markdown("## 📊 数据概览")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("总评论数", analysis_data.get("total", 0))
    col2.metric("好评维度", len(analysis_data.get("positive_dims", {})))
    col3.metric("差评维度", len(analysis_data.get("negative_dims", {})))
    col4.metric("用户画像", len(analysis_data.get("persona_dist", {})))
    col5.metric("识别情绪", len(analysis_data.get("emotion_dist", {})))

    st.markdown("---")
    st.markdown("### 📈 核心指标")

    pos_total = sum(analysis_data.get("positive_dims", {}).values())
    neg_total = sum(analysis_data.get("negative_dims", {}).values())

    col1, col2, col3 = st.columns(3)
    col1.metric("好评提及总数", pos_total if pos_total > 0 else "待分析")
    col2.metric("差评提及总数", neg_total if neg_total > 0 else "待分析")
    col3.metric("总提及次数", pos_total + neg_total if (pos_total + neg_total) > 0 else "待分析")

    with st.expander("📋 原始数据预览", expanded=False):
        display_cols = ["review_text", "star_rating", "sentiment", "dimensions", "motivation", "emotion", "persona", "scenario", "analysis_status"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True, height=300)

    # ========== 8个Tab ==========
    tabs = st.tabs([
        "🎯 战略洞察", "📊 维度分析", "💭 购买动机",
        "😊 情绪分析", "👤 用户画像", "📍 使用场景",
        "🎯 机会发现", "📥 一键导出"
    ])

    # Tab 1: 战略洞察
    with tabs[0]:
        if analysis_data.get("strategic_insights"):
            st.markdown(analysis_data["strategic_insights"])
            st.download_button(
                "📥 导出战略洞察报告",
                analysis_data["strategic_insights"],
                file_name=f"strategic_insights_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
        else:
            st.info("点击「开始分析」生成战略洞察报告")

    # Tab 2: 维度分析 + 维度管理
    with tabs[1]:
        st.markdown("### 维度分析（已归一化）")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(make_bar_chart(analysis_data.get("positive_dims", {}), "好评 TOP 维度", "#2ecc71"), use_container_width=True)
            if analysis_data.get("dimension_report"):
                with st.expander("📄 查看详细维度报告"):
                    st.markdown(analysis_data["dimension_report"])
        with col2:
            st.plotly_chart(make_bar_chart(analysis_data.get("negative_dims", {}), "差评 TOP 维度", "#e74c3c"), use_container_width=True)

        # 维度管理（新增修复）
        with st.expander("🧠 维度自主学习与管理", expanded=False):
            render_dimension_management()

    # Tab 3: 购买动机
    with tabs[2]:
        st.plotly_chart(make_pie_chart(analysis_data.get("motivation_dist", {}), "购买动机分布"), use_container_width=True)
        if analysis_data.get("motivation_report"):
            with st.expander("📄 查看详细动机报告"):
                st.markdown(analysis_data["motivation_report"])

    # Tab 4: 情绪分析
    with tabs[3]:
        st.plotly_chart(make_emotion_chart(analysis_data.get("emotion_dist", {})), use_container_width=True)
        if analysis_data.get("emotion_report"):
            with st.expander("📄 查看详细情绪报告"):
                st.markdown(analysis_data["emotion_report"])

    # Tab 5: 用户画像
    with tabs[4]:
        st.plotly_chart(make_pie_chart(analysis_data.get("persona_dist", {}), "用户画像分布"), use_container_width=True)
        if analysis_data.get("persona_report"):
            with st.expander("📄 查看详细画像报告"):
                st.markdown(analysis_data["persona_report"])

    # Tab 6: 使用场景
    with tabs[5]:
        st.plotly_chart(make_bar_chart(analysis_data.get("scenario_dist", {}), "使用场景分布", "#3498db"), use_container_width=True)

    # Tab 7: 机会发现
    with tabs[6]:
        opportunities = analysis_data.get("opportunities", [])
        if opportunities:
            for opp in opportunities[:5]:
                with st.expander(f"🎯 {opp['dimension']} - 机会分数 {opp['score']}"):
                    st.write(f"**提及次数**：{opp['mentions']}")
                    st.write(f"**差评率**：{opp['complaint_rate']}%")
                    if opp['complaint_rate'] > 50:
                        st.warning("⚠️ 紧急改进项：用户对此非常不满")
                    elif opp['complaint_rate'] > 30:
                        st.warning("📌 建议改进项：用户对此有明显不满")
                    else:
                        st.info("💡 优化机会：改进空间较大")
            if analysis_data.get("opportunity_report"):
                with st.expander("📄 查看详细机会报告"):
                    st.markdown(analysis_data["opportunity_report"])
        else:
            st.info("暂无机会点数据")

    # Tab 8: 一键导出
    with tabs[7]:
        st.markdown("## 📥 一键导出所有数据")
        st.info("点击下方按钮，将导出所有分析数据为 Excel 文件，包含以下工作表：")
        st.markdown("""
        - 原始数据
        - 好评维度
        - 差评维度
        - 用户画像
        - 购买动机
        - 情绪分布
        - 机会点
        """)

        excel_data = export_all_data(df, analysis_data)
        st.download_button(
            "📥 导出全部数据 (Excel)",
            excel_data,
            file_name=f"voc_all_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("### 📄 各模块详细报告导出")

        col1, col2 = st.columns(2)
        with col1:
            if analysis_data.get("strategic_insights"):
                st.download_button("战略洞察报告", analysis_data["strategic_insights"], "strategic_insights.md")
            if analysis_data.get("dimension_report"):
                st.download_button("维度分析报告", analysis_data["dimension_report"], "dimension_report.md")
            if analysis_data.get("persona_report"):
                st.download_button("用户画像报告", analysis_data["persona_report"], "persona_report.md")
        with col2:
            if analysis_data.get("emotion_report"):
                st.download_button("情绪分析报告", analysis_data["emotion_report"], "emotion_report.md")
            if analysis_data.get("motivation_report"):
                st.download_button("购买动机报告", analysis_data["motivation_report"], "motivation_report.md")
            if analysis_data.get("opportunity_report"):
                st.download_button("机会发现报告", analysis_data["opportunity_report"], "opportunity_report.md")

if __name__ == "__main__":
    main()
