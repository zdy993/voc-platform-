# -*- coding: utf-8 -*-
"""
VOC 智能分析平台 - 自主学习版
自动发现新维度 | 持续学习优化 | 智能分析
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
from typing import List, Tuple, Dict
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import hashlib

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="VOC 智能洞察平台 - 自主学习版",
    page_icon="🧠",
    layout="wide"
)

# =========================
# API 配置
# =========================
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def call_llm(api_key: str, prompt: str, max_tokens: int = 2500) -> str:
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
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
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
# 智能维度学习器
# =========================
class DimensionLearner:
    """自主维度学习器 - 能够发现和记忆新维度"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_dimensions = BASE_DIMENSIONS.copy()
        self.emerging_dimensions = {}  # 新兴维度 {维度名: {mentions: 次数, keywords: [关键词], first_seen: 时间}}
        self.dimension_synonyms = {}  # 同义词映射
        self._load_learned_dimensions()  # 加载已学习的维度
        
    def _load_learned_dimensions(self):
        """加载之前学习到的维度（从session或文件）"""
        if "learned_dimensions" in st.session_state:
            self.emerging_dimensions = st.session_state.learned_dimensions
            
    def _save_learned_dimensions(self):
        """保存学习到的维度"""
        st.session_state.learned_dimensions = self.emerging_dimensions
        
    def match_dimensions(self, text: str) -> List[str]:
        """匹配已知维度（基于关键词）"""
        text_lower = text.lower()
        matched = []
        
        # 匹配基础维度
        for dim, keywords in self.base_dimensions.items():
            for kw in keywords:
                if kw in text_lower:
                    matched.append(dim)
                    break
        
        # 匹配学习到的新兴维度
        for dim, info in self.emerging_dimensions.items():
            if dim in text or any(kw in text_lower for kw in info.get('keywords', [])):
                matched.append(dim)
                
        return list(set(matched))[:3]  # 最多3个维度
    
    def discover_new_dimensions(self, text: str, rating: int, sentiment: str) -> List[str]:
        """使用AI发现新维度"""
        if not self.api_key:
            return []
        
        # 只对长文本进行探索（短文本信息量少）
        if len(text) < 30:
            return []
        
        prompt = f"""分析用户评论，提取用户提到的产品属性维度。

用户评论：{text[:200]}
用户评分：{rating}/5
情感倾向：{sentiment}

重要：如果用户提到了以下列表之外的新维度，一定要提取出来。
已有维度：{', '.join(list(self.base_dimensions.keys())[:10])}

输出JSON格式：
{{
    "dimensions": ["已有维度1", "已有维度2"],
    "new_dimensions": [
        {{"name": "新维度名称", "keywords": ["关键词1", "关键词2"], "description": "简要描述"}}
    ]
}}

注意：
- 新维度名称要简洁（2-4个字）
- keywords是触发这个词的原文关键词
- 如果用户提到了新概念，一定要发现

只输出JSON："""
        
        try:
            result = call_llm(self.api_key, prompt, max_tokens=500)
            if not result:
                return []
            
            # 清理JSON
            clean = re.sub(r'```json\s*|```\s*', '', result.strip())
            data = json.loads(clean)
            
            # 记录新发现的维度
            for new_dim in data.get("new_dimensions", []):
                dim_name = new_dim.get("name", "")
                if dim_name and dim_name not in self.base_dimensions:
                    if dim_name not in self.emerging_dimensions:
                        self.emerging_dimensions[dim_name] = {
                            "mentions": 1,
                            "keywords": new_dim.get("keywords", []),
                            "description": new_dim.get("description", ""),
                            "first_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "sample_reviews": [text[:100]]
                        }
                    else:
                        self.emerging_dimensions[dim_name]["mentions"] += 1
                        # 合并关键词
                        for kw in new_dim.get("keywords", []):
                            if kw not in self.emerging_dimensions[dim_name]["keywords"]:
                                self.emerging_dimensions[dim_name]["keywords"].append(kw)
                        # 保存样本
                        if len(self.emerging_dimensions[dim_name].get("sample_reviews", [])) < 5:
                            self.emerging_dimensions[dim_name].setdefault("sample_reviews", []).append(text[:100])
            
            self._save_learned_dimensions()
            return [d["name"] for d in data.get("new_dimensions", [])]
            
        except Exception as e:
            print(f"维度发现失败: {e}")
            return []
    
    def get_emerging_dimensions_report(self, min_mentions: int = 3) -> Dict:
        """获取新兴维度报告"""
        emerging = {}
        for dim, info in self.emerging_dimensions.items():
            if info["mentions"] >= min_mentions:
                emerging[dim] = info
        return emerging
    
    def suggest_merge_to_base(self, threshold: int = 5):
        """建议将高频新维度合并到基础维度库"""
        suggestions = []
        for dim, info in self.emerging_dimensions.items():
            if info["mentions"] >= threshold:
                suggestions.append({
                    "dimension": dim,
                    "mentions": info["mentions"],
                    "keywords": info.get("keywords", []),
                    "sample": info.get("sample_reviews", [""])[0],
                    "confidence": min(100, info["mentions"] * 10)
                })
        return sorted(suggestions, key=lambda x: x["mentions"], reverse=True)
    
    def merge_to_base(self, dimension: str):
        """手动将维度合并到基础库"""
        if dimension in self.emerging_dimensions:
            info = self.emerging_dimensions[dimension]
            self.base_dimensions[dimension] = info.get("keywords", [dimension])
            del self.emerging_dimensions[dimension]
            self._save_learned_dimensions()
            return True
        return False
    
    def get_all_dimensions(self) -> Dict:
        """获取所有维度（基础+学习）"""
        all_dims = self.base_dimensions.copy()
        for dim, info in self.emerging_dimensions.items():
            if info["mentions"] >= 2:  # 只显示出现2次以上的
                all_dims[dim] = info.get("keywords", [dim])
        return all_dims

# =========================
# AI分析函数
# =========================
def analyze_with_learner(text: str, rating: int, api_key: str, learner: DimensionLearner) -> Tuple[str, List[str], List[str]]:
    """使用学习器分析评论"""
    # 1. 先用关键词快速匹配
    matched_dims = learner.match_dimensions(text)
    
    # 2. 判断情感（基于评分和关键词）
    if rating >= 4:
        sentiment = "正面"
    elif rating <= 2:
        sentiment = "负面"
    else:
        sentiment = "中性"
    
    # 3. 用AI发现新维度（仅当有API Key且文本有意义时）
    new_dims = []
    if api_key and len(text) > 30:
        new_dims = learner.discover_new_dimensions(text, rating, sentiment)
    
    # 4. 合并结果
    all_dims = list(set(matched_dims + new_dims))
    
    return sentiment, all_dims, new_dims

# =========================
# 批量处理函数
# =========================
def batch_analyze(df: pd.DataFrame, api_key: str, learner: DimensionLearner, progress_callback=None):
    """批量分析评论"""
    df = df.copy()
    total = len(df)
    
    # 统计
    positive_dims = Counter()
    negative_dims = Counter()
    motivations = []
    emotions = []
    personas = []
    scenarios = []
    all_new_dimensions = Counter()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 并行处理参数
    batch_size = 3  # 每批3条（避免API过载）
    
    for idx in range(total):
        # 更新进度
        progress = (idx + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"正在分析第 {idx + 1}/{total} 条评论... (已发现 {len(learner.emerging_dimensions)} 个新维度)")
        
        if progress_callback:
            progress_callback(idx + 1, total)
        
        try:
            row = df.iloc[idx]
            text = row["review_text"]
            rating = row["star_rating"]
            
            # 使用学习器分析
            sentiment, dimensions, new_dims = analyze_with_learner(text, rating, api_key, learner)
            
            # 统计新维度
            for nd in new_dims:
                all_new_dimensions[nd] += 1
            
            # 统计正负面维度
            for dim in dimensions:
                if sentiment == "正面":
                    positive_dims[dim] += 1
                elif sentiment == "负面":
                    negative_dims[dim] += 1
            
            # 保存到DataFrame
            df.at[idx, "sentiment"] = sentiment
            df.at[idx, "dimensions"] = ", ".join(dimensions)
            
            # 简化的属性提取（使用关键词匹配，更快）
            df.at[idx, "motivation"] = fast_motivation(text)
            df.at[idx, "emotion"] = fast_emotion(rating)
            df.at[idx, "persona"] = fast_persona(text)
            df.at[idx, "scenario"] = fast_scenario(text)
            
            motivations.append(df.at[idx, "motivation"])
            emotions.append(df.at[idx, "emotion"])
            personas.append(df.at[idx, "persona"])
            scenarios.append(df.at[idx, "scenario"])
            
            # 每分析10条，休息一下避免过载
            if (idx + 1) % 10 == 0:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"第{idx}条分析失败: {e}")
            # 使用默认值
            df.at[idx, "sentiment"] = "中性"
            df.at[idx, "dimensions"] = ""
            df.at[idx, "motivation"] = "日常使用"
            df.at[idx, "emotion"] = "平静"
            df.at[idx, "persona"] = "普通用户"
            df.at[idx, "scenario"] = "日常"
            
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
    
    # 机会发现
    opportunities = discover_opportunities(dict(positive_dims), dict(negative_dims), total_count)
    
    # 生成报告
    strategic_insights = generate_strategic_insights(
        dict(positive_dims), dict(negative_dims), emotion_dist,
        persona_dist, motivation_dist, df["review_text"].tolist()[:30], api_key
    )
    
    # 新兴维度报告
    emerging_report = generate_emerging_dimensions_report(learner)
    
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
        "emerging_dimensions": learner.get_emerging_dimensions_report(),
        "emerging_report": emerging_report,
        "new_dimensions_summary": dict(all_new_dimensions.most_common(20))
    }
    
    return df, analysis_data

# =========================
# 快速匹配函数（无需API）
# =========================
def fast_motivation(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["车载", "开车"]):
        return "车载使用"
    if any(w in text_lower for w in ["办公", "商务"]):
        return "商务办公"
    if any(w in text_lower for w in ["摔", "保护"]):
        return "防摔保护"
    if any(w in text_lower for w in ["旅行"]):
        return "旅行使用"
    if any(w in text_lower for w in ["送", "送礼"]):
        return "送礼"
    if any(w in text_lower for w in ["游戏"]):
        return "游戏使用"
    return "日常使用"

def fast_emotion(rating: int) -> str:
    if rating >= 5:
        return "惊喜"
    elif rating == 4:
        return "满意"
    elif rating == 3:
        return "平静"
    elif rating == 2:
        return "失望"
    else:
        return "愤怒"

def fast_persona(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["商务", "出差"]):
        return "商务人士"
    if any(w in text_lower for w in ["学生", "宿舍"]):
        return "学生"
    if any(w in text_lower for w in ["旅行", "旅游"]):
        return "旅行用户"
    if any(w in text_lower for w in ["家庭", "孩子"]):
        return "家庭用户"
    if any(w in text_lower for w in ["科技", "数码"]):
        return "科技爱好者"
    if any(w in text_lower for w in ["游戏"]):
        return "游戏用户"
    return "普通用户"

def fast_scenario(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["车载", "开车"]):
        return "车载"
    if any(w in text_lower for w in ["办公", "工位"]):
        return "办公室"
    if any(w in text_lower for w in ["旅行"]):
        return "旅行"
    if any(w in text_lower for w in ["健身"]):
        return "健身房"
    return "日常"

# =========================
# 报告生成函数
# =========================
def generate_strategic_insights(positive_dims, negative_dims, emotion_dist, persona_dist, motivation_dist, sample_reviews, api_key):
    """生成战略洞察报告"""
    pos_total = sum(positive_dims.values()) if positive_dims else 1
    neg_total = sum(negative_dims.values()) if negative_dims else 1
    
    top_pos = list(positive_dims.items())[:5] if positive_dims else []
    top_neg = list(negative_dims.items())[:5] if negative_dims else []
    
    pos_str = "\n".join([f"  - {dim}: {count}次 ({count/pos_total*100:.1f}%)" for dim, count in top_pos])
    neg_str = "\n".join([f"  - {dim}: {count}次 ({count/neg_total*100:.1f}%)" for dim, count in top_neg])
    sample_str = "\n".join([f"- {text[:100]}..." for text in sample_reviews[:5]])
    
    prompt = f"""你是资深产品战略分析师。基于以下数据生成战略洞察报告：

## 好评TOP5
{pos_str}

## 差评TOP5
{neg_str}

## 代表评论
{sample_str}

请输出：
1. 核心发现（3-5点）
2. 产品优化建议（短期+长期）
3. 差异化策略

简洁专业输出："""
    
    try:
        report = call_llm(api_key, prompt, max_tokens=1500)
        if report and len(report) > 200:
            return report
    except:
        pass
    
    return f"""## 📊 战略洞察报告

### 核心发现
- 用户最满意：{top_pos[0][0] if top_pos else '待分析'}
- 主要痛点：{top_neg[0][0] if top_neg else '待分析'}

### 产品建议
1. 强化{top_pos[0][0] if top_pos else '优势'}体验
2. 改进{top_neg[0][0] if top_neg else '痛点'}问题"""

def generate_emerging_dimensions_report(learner: DimensionLearner) -> str:
    """生成新兴维度报告"""
    emerging = learner.get_emerging_dimensions_report(min_mentions=2)
    suggestions = learner.suggest_merge_to_base(threshold=3)
    
    if not emerging:
        return "暂无发现新维度，继续分析更多评论后会自动学习。"
    
    report = f"""## 🧠 自主学习发现的新维度

本次分析共发现 **{len(emerging)}** 个新兴维度：

| 维度名称 | 出现次数 | 关键词 | 首次发现 |
|---------|---------|--------|---------|
"""
    for dim, info in sorted(emerging.items(), key=lambda x: x[1]["mentions"], reverse=True)[:10]:
        keywords = ", ".join(info.get("keywords", [dim])[:3])
        report += f"| {dim} | {info['mentions']} | {keywords} | {info.get('first_seen', '未知')} |\n"
    
    if suggestions:
        report += f"""
### 💡 建议添加到基础维度库
以下维度出现频率较高，建议确认后永久加入维度库：

"""
        for s in suggestions[:5]:
            report += f"- **{s['dimension']}** (出现{s['mentions']}次) - 示例：{s['sample'][:50]}...\n"
    
    return report

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
    order = ["惊喜", "满意", "平静", "失望", "愤怒"]
    colors = {"惊喜": "#2ecc71", "满意": "#27ae60", "平静": "#95a5a6", "失望": "#e67e22", "愤怒": "#e74c3c"}
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
        ],
        "star_rating": [5, 2, 5, 1, 4, 4, 5, 5, 5, 4]
    })

def export_all_data(df: pd.DataFrame, analysis_data: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='原始数据', index=False)
        
        if analysis_data.get("positive_dims"):
            pd.DataFrame([{"维度": k, "提及次数": v} for k, v in analysis_data["positive_dims"].items()]).to_excel(writer, sheet_name='好评维度', index=False)
        
        if analysis_data.get("negative_dims"):
            pd.DataFrame([{"维度": k, "提及次数": v} for k, v in analysis_data["negative_dims"].items()]).to_excel(writer, sheet_name='差评维度', index=False)
        
        if analysis_data.get("emerging_dimensions"):
            emerging_df = pd.DataFrame([
                {"维度": k, "出现次数": v["mentions"], "关键词": ", ".join(v.get("keywords", [])), "示例": v.get("sample_reviews", [""])[0]}
                for k, v in analysis_data["emerging_dimensions"].items()
            ])
            emerging_df.to_excel(writer, sheet_name='新发现维度', index=False)
        
        if analysis_data.get("opportunities"):
            pd.DataFrame(analysis_data["opportunities"]).to_excel(writer, sheet_name='机会点', index=False)
    
    return output.getvalue()

# =========================
# 侧边栏
# =========================
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ 配置")
        st.info("🧠 **自主学习模式**\n- 自动发现新维度\n- 持续学习优化\n- 智能识别趋势")
        
        api_key = st.text_input("DeepSeek API Key", type="password", placeholder="sk-...", 
                                help="用于AI深度分析和新维度发现")
        
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
    st.caption("AI驱动的消费者洞察 | 自动发现新维度 | 持续学习优化")
    
    api_key, input_df, start_analysis = render_sidebar()
    
    if input_df is None:
        st.info("👈 左侧上传文件或点击「加载示例数据」")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🎯 核心功能
            - **🧠 自主学习**：自动发现和记忆新维度
            - **📊 维度分析**：智能识别用户关注点
            - **🎯 战略洞察**：AI生成深度报告
            - **📥 一键导出**：Excel完整数据
            """)
        with col2:
            st.markdown("""
            ### 🔬 学习能力
            - 发现新概念（如"环保材质"）
            - 自动记录出现频率
            - 智能建议加入维度库
            - 持续优化识别准确率
            """)
        return
    
    if "review_text" not in input_df.columns:
        st.error("文件缺少 `review_text` 列")
        return
    
    df = preprocess_data(input_df)
    
    # 初始化学习器
    learner = DimensionLearner(api_key=api_key)
    
    # 显示当前维度库状态
    with st.expander("📚 当前维度库状态", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**基础维度**：{len(learner.base_dimensions)} 个")
            st.markdown(f"**已学习维度**：{len(learner.emerging_dimensions)} 个")
        with col2:
            if learner.emerging_dimensions:
                st.markdown("**最近发现的新维度**：")
                for dim in list(learner.emerging_dimensions.keys())[-5:]:
                    st.markdown(f"- {dim}")
    
    if start_analysis:
        if not api_key:
            st.warning("⚠️ 请输入 DeepSeek API Key（用于新维度发现）")
        else:
            with st.spinner(f"🧠 AI正在分析 {len(df)} 条评论，并学习新维度..."):
                start_time = time.time()
                df, analysis_data = batch_analyze(df, api_key, learner)
                elapsed = time.time() - start_time
                
                st.session_state["df"] = df
                st.session_state["analysis_data"] = analysis_data
                st.session_state["learner"] = learner
                st.success(f"✅ 分析完成！共 {len(df)} 条评论，用时 {elapsed:.1f} 秒")
                st.balloons()
    
    df = st.session_state.get("df", df)
    analysis_data = st.session_state.get("analysis_data", {})
    
    if not analysis_data:
        return
    
    # 数据概览
    st.markdown("## 📊 数据概览")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("总评论数", analysis_data.get("total", 0))
    col2.metric("好评维度", len(analysis_data.get("positive_dims", {})))
    col3.metric("差评维度", len(analysis_data.get("negative_dims", {})))
    col4.metric("已学习维度", len(analysis_data.get("emerging_dimensions", {})))
    col5.metric("新发现概念", len(analysis_data.get("new_dimensions_summary", {})))
    
    # 8个Tab（新增新维度发现Tab）
    tabs = st.tabs(["🎯 战略洞察", "📊 维度分析", "🧠 新维度发现", "💭 购买动机", "😊 情绪分析", "👤 用户画像", "📍 使用场景", "🎯 机会发现", "📥 一键导出"])
    
    with tabs[0]:
        if analysis_data.get("strategic_insights"):
            st.markdown(analysis_data["strategic_insights"])
            st.caption("💡 战略洞察由AI深度分析生成")
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(make_bar_chart(analysis_data.get("positive_dims", {}), "好评 TOP 维度", "#2ecc71"), use_container_width=True)
        with col2:
            st.plotly_chart(make_bar_chart(analysis_data.get("negative_dims", {}), "差评 TOP 维度", "#e74c3c"), use_container_width=True)
        
        # 显示所有维度
        with st.expander("📋 完整维度列表"):
            all_dims = learner.get_all_dimensions()
            st.dataframe(pd.DataFrame([
                {"维度": dim, "类型": "基础库" if dim in BASE_DIMENSIONS else "自主学习", "关键词数": len(kw) if isinstance(kw, list) else 1}
                for dim, kw in all_dims.items()
            ]), use_container_width=True)
    
    with tabs[2]:
        st.markdown("## 🧠 自主学习发现的新维度")
        
        emerging = analysis_data.get("emerging_dimensions", {})
        new_summary = analysis_data.get("new_dimensions_summary", {})
        
        if emerging:
            st.success(f"✨ AI自动发现了 {len(emerging)} 个新维度")
            
            for dim, info in sorted(emerging.items(), key=lambda x: x[1]["mentions"], reverse=True):
                with st.expander(f"🔍 {dim} (出现 {info['mentions']} 次)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**关键词**：{', '.join(info.get('keywords', []))}")
                        st.markdown(f"**首次发现**：{info.get('first_seen', '未知')}")
                    with col2:
                        if info.get("sample_reviews"):
                            st.markdown(f"**示例评论**：{info['sample_reviews'][0][:100]}...")
                    
                    # 建议操作
                    if info["mentions"] >= 3:
                        st.info(f"💡 该维度已出现 {info['mentions']} 次，建议考虑加入基础维度库")
                        if st.button(f"➕ 将「{dim}」加入基础库", key=f"merge_{dim}"):
                            if learner.merge_to_base(dim):
                                st.success(f"已添加 {dim} 到基础维度库！")
                                st.rerun()
        else:
            st.info("🤖 继续分析更多评论，AI会自动发现新维度")
            st.markdown("""
            ### 如何触发新维度发现？
            1. 上传包含新概念的评论（如"环保材质"、"快充"等）
            2. AI会自动识别并记录
            3. 出现3次以上会建议加入维度库
            """)
        
        if new_summary:
            st.markdown("### 📊 新维度统计")
            st.dataframe(pd.DataFrame([
                {"维度": k, "出现次数": v} for k, v in list(new_summary.items())[:10]
            ]), use_container_width=True)
    
    with tabs[3]:
        st.plotly_chart(make_pie_chart(analysis_data.get("motivation_dist", {}), "购买动机分布"), use_container_width=True)
    
    with tabs[4]:
        st.plotly_chart(make_emotion_chart(analysis_data.get("emotion_dist", {})), use_container_width=True)
    
    with tabs[5]:
        st.plotly_chart(make_pie_chart(analysis_data.get("persona_dist", {}), "用户画像分布"), use_container_width=True)
    
    with tabs[6]:
        st.plotly_chart(make_bar_chart(analysis_data.get("scenario_dist", {}), "使用场景分布", "#3498db"), use_container_width=True)
    
    with tabs[7]:
        opportunities = analysis_data.get("opportunities", [])
        if opportunities:
            for opp in opportunities[:5]:
                with st.expander(f"🎯 {opp['dimension']} - 机会分数 {opp['score']}"):
                    st.write(f"提及次数：{opp['mentions']}")
                    st.write(f"差评率：{opp['complaint_rate']}%")
                    if opp['complaint_rate'] > 50:
                        st.warning("⚠️ 紧急改进项")
        else:
            st.info("暂无机会点数据")
    
    with tabs[8]:
        excel_data = export_all_data(df, analysis_data)
        st.download_button("📥 导出全部数据 (Excel)", excel_data, 
                          file_name=f"voc_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                          use_container_width=True)
        
        # 单独导出新维度
        if analysis_data.get("emerging_dimensions"):
            st.markdown("---")
            emerging_report = analysis_data.get("emerging_report", "")
            if emerging_report:
                st.download_button("📥 导出新维度报告", emerging_report, "emerging_dimensions.md")

if __name__ == "__main__":
    main()
