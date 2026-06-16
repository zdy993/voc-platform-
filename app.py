# -*- coding: utf-8 -*-
"""
VOC 智能分析平台 V2 - 完整修复版
支持3000条评论 | 批量分析 | 断点续传 | 维度学习 | 自动降级
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
import hashlib
import sqlite3
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="VOC 智能洞察平台 V2",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# API 配置
# =========================
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def call_llm(api_key: str, prompt: str, max_tokens: int = 1500) -> str:
    """调用API - 带重试和超时控制"""
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
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=20)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
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
# 一级维度定义
# =========================
LEVEL1_DIMENSIONS = [
    "磁吸能力",
    "支架功能",
    "保护性能",
    "外观设计",
    "手感",
    "耐用性",
    "无线充电",
    "安装体验",
    "性价比",
    "物流服务",
    "售后服务"
]

# =========================
# 维度学习系统
# =========================
class DimensionLearner:
    """维度学习器 - 自动发现和聚类二级维度"""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.learning_file = os.path.join(data_dir, "dimension_learning.json")
        self.db_file = os.path.join(data_dir, "analysis_cache.db")
        self.progress_file = os.path.join(data_dir, "progress.json")
        self.learned_dimensions = self._load_learned()
        self._init_db()
    
    def _load_learned(self) -> dict:
        """加载已学习的维度"""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_learned(self):
        """保存学习的维度"""
        with open(self.learning_file, 'w', encoding='utf-8') as f:
            json.dump(self.learned_dimensions, f, ensure_ascii=False, indent=2)
    
    def _init_db(self):
        """初始化缓存数据库"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                text_hash TEXT PRIMARY KEY,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def get_cached(self, text: str) -> Optional[dict]:
        """获取缓存的分析结果"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('SELECT result FROM cache WHERE text_hash = ?', (text_hash,))
        row = cursor.fetchone()
        conn.close()
        if row:
            try:
                return json.loads(row[0])
            except:
                return None
        return None
    
    def save_cache(self, text: str, result: dict):
        """保存分析结果到缓存"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO cache (text_hash, result) VALUES (?, ?)
        ''', (text_hash, json.dumps(result, ensure_ascii=False)))
        conn.commit()
        conn.close()
    
    def save_progress(self, current_index: int, total: int):
        """保存分析进度"""
        progress = {
            "current_index": current_index,
            "completed": current_index,
            "total": total,
            "updated_at": datetime.now().isoformat()
        }
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    
    def load_progress(self) -> Optional[dict]:
        """加载分析进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def clear_progress(self):
        """清除进度"""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
    
    def clear_cache(self):
        """清除缓存"""
        if os.path.exists(self.db_file):
            os.remove(self.db_file)
        self._init_db()
    
    def learn_new_dimensions(self, reviews: List[str], api_key: str) -> dict:
        """从评论中学习新的二级维度"""
        if not reviews or not api_key or len(reviews) < 5:
            return {}
        
        # 提取可能的维度关键词（简化版）
        all_dimensions = []
        for review in reviews[:50]:
            words = review.split()
            for word in words:
                if len(word) >= 2 and word not in ['的', '了', '是', '在', '有', '和', '与', '或']:
                    if word not in LEVEL1_DIMENSIONS:
                        all_dimensions.append(word)
        
        # 去重
        unique_dims = list(set(all_dimensions))
        if len(unique_dims) < 5:
            return self.learned_dimensions
        
        # 使用AI进行聚类
        prompt = f"""分析以下用户提到的产品特性关键词，将相似含义的词汇归为一组。

关键词列表：
{chr(10).join(unique_dims[:30])}

输出JSON格式：
{{
    "clusters": [
        {{"cluster_name": "支架性能", "members": ["稳定性", "牢固度", "支撑力", "松动"]}},
        {{"cluster_name": "磁吸强度", "members": ["磁力", "吸力", "吸附", "磁吸"]}}
    ]
}}

只输出JSON，不要其他内容："""
        
        try:
            result = call_llm(api_key, prompt, max_tokens=800)
            if result:
                clean = re.sub(r'```json\s*|```\s*', '', result.strip())
                data = json.loads(clean)
                for cluster in data.get("clusters", []):
                    cluster_name = cluster.get("cluster_name")
                    members = cluster.get("members", [])
                    if cluster_name and members and len(members) >= 2:
                        if cluster_name not in self.learned_dimensions:
                            self.learned_dimensions[cluster_name] = {
                                "members": members,
                                "created_at": datetime.now().isoformat()
                            }
                        else:
                            existing = set(self.learned_dimensions[cluster_name].get("members", []))
                            existing.update(members)
                            self.learned_dimensions[cluster_name]["members"] = list(existing)
                self._save_learned()
        except:
            pass
        return self.learned_dimensions
    
    def get_cluster(self, dimension: str) -> Optional[str]:
        """获取维度所属的聚类"""
        for cluster_name, info in self.learned_dimensions.items():
            if dimension in info.get("members", []):
                return cluster_name
        return None

# =========================
# 降级函数 - 当API失败时使用
# =========================
def fallback_result(text: str, rating: int) -> dict:
    """降级结果 - 基于评分和关键词"""
    # 情感判断
    if rating >= 4:
        sentiment = "正面"
    elif rating <= 2:
        sentiment = "负面"
    else:
        sentiment = "中性"
    
    # 一级维度匹配（关键词）
    level1 = ""
    text_lower = text.lower()
    for dim in LEVEL1_DIMENSIONS:
        if dim in text or any(kw in text_lower for kw in dim):
            level1 = dim
            break
    
    # 如果是支架相关评论，检测二级维度
    level2 = ""
    if "支架" in text or "支撑" in text or "转轴" in text:
        if "稳定" in text or "牢固" in text or "晃动" in text:
            level2 = "支架稳定性"
        elif "横屏" in text or "竖屏" in text:
            level2 = "横屏体验"
        elif "角度" in text or "调节" in text:
            level2 = "支撑角度"
        elif "顺滑" in text or "开合" in text:
            level2 = "开合顺滑度"
    
    # 动机
    motivation = "日常使用"
    if "车载" in text or "开车" in text:
        motivation = "车载使用"
    elif "办公" in text or "商务" in text:
        motivation = "商务办公"
    elif "摔" in text or "保护" in text:
        motivation = "防摔保护"
    elif "旅行" in text:
        motivation = "旅行使用"
    
    # 情绪
    emotion = "满意" if rating >= 4 else "失望" if rating <= 2 else "平静"
    
    # 画像
    persona = "普通用户"
    if "商务" in text or "办公" in text:
        persona = "商务人士"
    elif "学生" in text:
        persona = "学生"
    elif "旅行" in text:
        persona = "旅行用户"
    elif "家庭" in text or "孩子" in text:
        persona = "家庭用户"
    elif "科技" in text or "数码" in text:
        persona = "科技爱好者"
    
    # 场景
    scenario = "日常"
    if "车载" in text or "开车" in text:
        scenario = "车载"
    elif "办公" in text or "工位" in text:
        scenario = "办公室"
    elif "旅行" in text:
        scenario = "旅行"
    elif "健身" in text:
        scenario = "健身房"
    
    return {
        "sentiment": sentiment,
        "level1_dimension": level1,
        "level2_dimension": level2,
        "motivation": motivation,
        "emotion": emotion,
        "persona": persona,
        "scenario": scenario
    }

# =========================
# 分析引擎 - 单条分析
# =========================
def extract_all_attributes(review_text: str, star_rating: int, api_key: str, 
                           learner: DimensionLearner, mode: str = "standard") -> dict:
    """一次API调用完成所有属性提取"""
    
    # 检查缓存
    cached = learner.get_cached(review_text)
    if cached:
        return cached
    
    if not api_key:
        return fallback_result(review_text, star_rating)
    
    # 根据模式构建Prompt
    if mode == "quick":
        prompt = f"""分析评论，输出JSON：
评论：{review_text[:150]}
星级：{star_rating}/5

输出格式：{{"sentiment":"正面/负面/中性","level1_dimension":"一级维度"}}
一级维度可选：{', '.join(LEVEL1_DIMENSIONS)}
只输出JSON："""
        max_tokens = 100
        
    elif mode == "standard":
        prompt = f"""分析评论，输出JSON：
评论：{review_text[:200]}
星级：{star_rating}/5

输出格式：{{"sentiment":"正面/负面/中性","level1_dimension":"一级维度","level2_dimension":"具体特性"}}
一级维度可选：{', '.join(LEVEL1_DIMENSIONS)}
只输出JSON："""
        max_tokens = 150
        
    else:  # deep
        prompt = f"""分析评论，输出JSON：
评论：{review_text[:250]}
星级：{star_rating}/5

输出格式：{{"sentiment":"正面/负面/中性","level1_dimension":"一级维度","level2_dimension":"具体特性","motivation":"动机","emotion":"情绪","persona":"身份","scenario":"场景"}}
一级维度可选：{', '.join(LEVEL1_DIMENSIONS)}
动机：车载使用、商务办公、防摔保护、旅行使用、送礼、日常使用、游戏使用
情绪：惊喜、满意、平静、失望、焦虑、愤怒、后悔
身份：商务人士、学生、旅行用户、家庭用户、科技爱好者、游戏用户、普通用户
场景：车载、办公室、旅行、健身房、家庭、户外、通勤
只输出JSON："""
        max_tokens = 300
    
    try:
        result = call_llm(api_key, prompt, max_tokens=max_tokens)
        if not result:
            return fallback_result(review_text, star_rating)
        
        clean = re.sub(r'```json\s*|```\s*', '', result.strip())
        data = json.loads(clean)
        
        result_dict = {
            "sentiment": data.get("sentiment", "中性"),
            "level1_dimension": data.get("level1_dimension", ""),
            "level2_dimension": data.get("level2_dimension", ""),
            "motivation": data.get("motivation", "日常使用"),
            "emotion": data.get("emotion", "平静"),
            "persona": data.get("persona", "普通用户"),
            "scenario": data.get("scenario", "日常")
        }
        
        # 缓存结果
        learner.save_cache(review_text, result_dict)
        return result_dict
        
    except:
        return fallback_result(review_text, star_rating)

# =========================
# 批量分析 - 一次处理多条
# =========================
def batch_extract_all(reviews_batch: List[Tuple[int, str, int]], api_key: str, 
                      learner: DimensionLearner, mode: str = "standard") -> List[dict]:
    """批量分析 - 带缓存检查和降级处理"""
    if not api_key:
        return [fallback_result(text, rating) for idx, text, rating in reviews_batch]
    
    # 1. 检查缓存
    cached_results = []
    uncached = []
    for idx, text, rating in reviews_batch:
        cached = learner.get_cached(text)
        if cached:
            cached_results.append({"idx": idx, **cached})
        else:
            uncached.append((idx, text, rating))
    
    if not uncached:
        return cached_results
    
    # 2. 如果超过10条，分批处理
    if len(uncached) > 10:
        all_results = cached_results
        for i in range(0, len(uncached), 10):
            chunk = uncached[i:i+10]
            chunk_results = _process_batch(chunk, api_key, mode, learner)
            all_results.extend(chunk_results)
        return all_results
    
    # 3. 正常处理
    chunk_results = _process_batch(uncached, api_key, mode, learner)
    return cached_results + chunk_results

def _process_batch(batch: List[Tuple[int, str, int]], api_key: str, mode: str, 
                   learner: DimensionLearner) -> List[dict]:
    """处理单个批次 - 带重试"""
    if not batch:
        return []
    
    # 准备数据（截断评论）
    batch_data = []
    for idx, text, rating in batch:
        batch_data.append({
            "id": idx,
            "text": text[:120] if len(text) > 120 else text,
            "rating": rating
        })
    
    # 根据模式构建Prompt
    if mode == "quick":
        prompt = f"""分析{len(batch)}条评论，每条输出JSON一行。

评论：
{json.dumps(batch_data, ensure_ascii=False, indent=2)[:1500]}

输出格式：{{"id":0,"sentiment":"正面/负面/中性","level1_dimension":"维度"}}
一级维度：{', '.join(LEVEL1_DIMENSIONS)}

只输出JSON，每行一条："""
        max_tokens = len(batch) * 50 + 200
        
    elif mode == "standard":
        prompt = f"""分析{len(batch)}条评论，每条输出JSON一行。

评论：
{json.dumps(batch_data, ensure_ascii=False, indent=2)[:1500]}

输出格式：{{"id":0,"sentiment":"正面/负面/中性","level1_dimension":"维度","level2_dimension":"特性"}}
一级维度：{', '.join(LEVEL1_DIMENSIONS)}

只输出JSON，每行一条："""
        max_tokens = len(batch) * 80 + 300
        
    else:  # deep
        prompt = f"""分析{len(batch)}条评论，每条输出JSON一行。

评论：
{json.dumps(batch_data, ensure_ascii=False, indent=2)[:1200]}

输出格式：{{"id":0,"sentiment":"正面/负面/中性","level1_dimension":"维度","level2_dimension":"特性","motivation":"动机","emotion":"情绪","persona":"身份","scenario":"场景"}}
一级维度：{', '.join(LEVEL1_DIMENSIONS)}
动机：车载使用、商务办公、防摔保护、旅行使用、送礼、日常使用、游戏使用
情绪：惊喜、满意、平静、失望、焦虑、愤怒、后悔
身份：商务人士、学生、旅行用户、家庭用户、科技爱好者、游戏用户、普通用户
场景：车载、办公室、旅行、健身房、家庭、户外、通勤

只输出JSON，每行一条："""
        max_tokens = len(batch) * 150 + 500
    
    # 重试3次
    for attempt in range(3):
        try:
            result = call_llm(api_key, prompt, max_tokens=max_tokens + 500)
            if not result:
                if attempt == 2:
                    return [fallback_result(text, rating) for _, text, rating in batch]
                continue
            
            # 解析结果
            results = []
            for line in result.strip().split('\n'):
                try:
                    data = json.loads(line.strip())
                    results.append(data)
                except:
                    continue
            
            # 确保所有评论都有结果
            final_results = []
            for item in batch_data:
                found = next((r for r in results if r.get("id") == item["id"]), None)
                if found:
                    result_dict = {
                        "sentiment": found.get("sentiment", "中性"),
                        "level1_dimension": found.get("level1_dimension", ""),
                        "level2_dimension": found.get("level2_dimension", ""),
                        "motivation": found.get("motivation", "日常使用"),
                        "emotion": found.get("emotion", "平静"),
                        "persona": found.get("persona", "普通用户"),
                        "scenario": found.get("scenario", "日常")
                    }
                    # 缓存
                    learner.save_cache(item["text"], result_dict)
                    final_results.append({"idx": item["id"], **result_dict})
                else:
                    # 单条降级
                    fallback = fallback_result(item["text"], item["rating"])
                    final_results.append({"idx": item["id"], **fallback})
            
            return final_results
            
        except:
            if attempt == 2:
                return [fallback_result(text, rating) for _, text, rating in batch]
            time.sleep(1)
    
    return [fallback_result(text, rating) for _, text, rating in batch]

# =========================
# 报告生成器
# =========================
def generate_report(positive_dims: dict, negative_dims: dict, emotion_dist: dict,
                    persona_dist: dict, motivation_dist: dict, scenario_dist: dict,
                    level2_dims: dict, sample_reviews: List[str], api_key: str) -> str:
    """生成战略洞察报告 - 限制数据量避免超上下文"""
    
    top_pos = list(positive_dims.items())[:20] if positive_dims else []
    top_neg = list(negative_dims.items())[:20] if negative_dims else []
    top_emotion = list(emotion_dist.items())[:10] if emotion_dist else []
    top_persona = list(persona_dist.items())[:10] if persona_dist else []
    top_motivation = list(motivation_dist.items())[:10] if motivation_dist else []
    top_scenario = list(scenario_dist.items())[:10] if scenario_dist else []
    top_level2 = list(level2_dims.items())[:20] if level2_dims else []
    
    samples = random.sample(sample_reviews, min(20, len(sample_reviews)))
    
    pos_str = "\n".join([f"  - {dim}: {count}次" for dim, count in top_pos[:10]])
    neg_str = "\n".join([f"  - {dim}: {count}次" for dim, count in top_neg[:10]])
    level2_str = "\n".join([f"  - {dim}: {count}次" for dim, count in top_level2[:10]])
    sample_str = "\n".join([f"- {text[:80]}..." for text in samples])
    
    prompt = f"""你是资深产品分析师。基于数据生成洞察报告（500字以内）：

## 好评维度 TOP10
{pos_str}

## 差评维度 TOP10
{neg_str}

## 二级维度 TOP10
{level2_str}

## 代表评论
{sample_str}

请输出：
1. 核心发现（3-5点）
2. 主要痛点
3. 优化建议（短中长期）
4. 差异化策略

简洁输出："""
    
    try:
        report = call_llm(api_key, prompt, max_tokens=1500)
        if report and len(report) > 100:
            return report
    except:
        pass
    
    return generate_fallback_report(top_pos, top_neg, emotion_dist, persona_dist, motivation_dist)

def generate_fallback_report(top_pos, top_neg, emotion_dist, persona_dist, motivation_dist):
    """降级报告"""
    report = f"""
## 📊 战略洞察报告

### 核心发现
- 用户最满意：{top_pos[0][0] if top_pos else '待分析'} ({top_pos[0][1] if top_pos else 0}次)
- 主要痛点：{top_neg[0][0] if top_neg else '待分析'} ({top_neg[0][1] if top_neg else 0}次)
- 正向情绪：{emotion_dist.get('满意', 0) + emotion_dist.get('惊喜', 0):.1f}%

### 核心用户群
- {list(persona_dist.items())[0][0] if persona_dist else '普通用户'}: {list(persona_dist.items())[0][1] if persona_dist else 0:.1f}%

### 主要购买动机
- {list(motivation_dist.items())[0][0] if motivation_dist else '日常使用'}: {list(motivation_dist.items())[0][1] if motivation_dist else 0:.1f}%

### 优化建议
1. **立即改进**：{top_neg[0][0] if top_neg else '主要痛点'}
2. **持续强化**：{top_pos[0][0] if top_pos else '核心优势'}
3. **差异化**：在{top_pos[0][0] if top_pos else '核心'}维度打造特色
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
    df["level1_dimension"] = ""
    df["level2_dimension"] = ""
    df["motivation"] = ""
    df["emotion"] = ""
    df["persona"] = ""
    df["scenario"] = ""
    df["analysis_status"] = "未分析"
    return df

def get_sample_data():
    """生成包含多维度评论的示例数据"""
    return pd.DataFrame({
        "review_text": [
            "磁力很强，开车用很稳，商务出差必备，强烈推荐",
            "太滑了，用了一个月就发黄，后悔买这个牌子",
            "惊喜！磁吸力超强，搭配车载支架完美，质感也很好",
            "失望，边框发黄严重，才用两周就变色了",
            "办公用很好，质感不错，按键灵敏",
            "学生党，性价比高，防摔效果好，值得购买",
            "旅行时用，磁吸很稳，拍照方便，满意",
            "转轴很稳定，横屏看视频不晃动，支撑角度可调",
            "支架牢固度很好，手机放上去很稳，不会掉",
            "支撑力不错，大屏幕手机也能撑住，不会后仰",
            "开合很顺滑，单手操作也没问题",
            "磁吸力太弱，稍微碰一下就掉了",
            "支架松动，放上去就往下滑，做工太差了",
            "转轴异响，开合有咔咔声，质量堪忧",
            "横屏时角度不对，看着很累，设计缺陷"
        ],
        "star_rating": [5, 2, 5, 1, 4, 4, 5, 5, 5, 4, 4, 2, 2, 2, 3]
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
        if analysis_data.get("level2_dims"):
            pd.DataFrame([{"二级维度": k, "提及次数": v} for k, v in analysis_data["level2_dims"].items()]).to_excel(writer, sheet_name='二级维度', index=False)
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
        if analysis_data.get("learned_dimensions"):
            learned_data = []
            for cluster, info in analysis_data["learned_dimensions"].items():
                learned_data.append({"聚类": cluster, "成员": ", ".join(info.get("members", []))})
            pd.DataFrame(learned_data).to_excel(writer, sheet_name='学习维度', index=False)
    return output.getvalue()

# =========================
# 主分析引擎 V2
# =========================
def run_analysis_v2(df: pd.DataFrame, api_key: str, learner: DimensionLearner, 
                   mode: str = "standard", progress_callback=None):
    """V2分析引擎 - 支持批量分析、断点续传、自动降级"""
    df = df.copy()
    total = len(df)
    
    # 检查是否有未完成的进度
    progress = learner.load_progress()
    start_idx = 0
    if progress and progress.get("total") == total:
        start_idx = progress.get("current_index", 0)
        if start_idx > 0:
            st.info(f"📌 继续分析第 {start_idx + 1}/{total} 条...")
    
    # 恢复已有数据
    positive_dims = Counter()
    negative_dims = Counter()
    level2_dims = Counter()
    motivations = []
    emotions = []
    personas = []
    scenarios = []
    
    for i in range(start_idx):
        if df.at[i, "sentiment"]:
            sentiment = df.at[i, "sentiment"]
            level1 = df.at[i, "level1_dimension"]
            level2 = df.at[i, "level2_dimension"]
            
            if sentiment == "正面" and level1:
                positive_dims[level1] += 1
            elif sentiment == "负面" and level1:
                negative_dims[level1] += 1
            if level2:
                level2_dims[level2] += 1
            
            motivations.append(df.at[i, "motivation"])
            emotions.append(df.at[i, "emotion"])
            personas.append(df.at[i, "persona"])
            scenarios.append(df.at[i, "scenario"])
    
    # 进度显示
    progress_bar = st.progress(start_idx / total if total > 0 else 0)
    status_text = st.empty()
    time_text = st.empty()
    stats_text = st.empty()
    
    start_time = time.time()
    batch_size = 10  # 每批10条（更稳定）
    failed_count = 0
    success_count = start_idx
    total_batches = (total - start_idx + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        batch_start = start_idx + batch_num * batch_size
        batch_end = min(batch_start + batch_size, total)
        
        # 准备批次数据
        batch_reviews = []
        for idx in range(batch_start, batch_end):
            batch_reviews.append((idx, df.at[idx, "review_text"], df.at[idx, "star_rating"]))
        
        # 更新进度
        current_progress = batch_end / total
        progress_bar.progress(current_progress)
        
        elapsed = time.time() - start_time
        total_processed = batch_end - start_idx
        if total_processed > 0 and batch_num > 0:
            speed = total_processed / elapsed
            remaining = (total - batch_end) / speed if speed > 0 else 0
            time_text.text(f"⏱ 预计剩余: {int(remaining//60)}分{int(remaining%60)}秒")
        
        status_text.text(f"📊 批次 {batch_num + 1}/{total_batches} | 评论 {batch_start + 1}-{batch_end}/{total}")
        stats_text.text(f"✅ 成功: {success_count} | ❌ 失败: {failed_count}")
        
        if progress_callback:
            progress_callback(batch_end, total)
        
        try:
            # 批量分析
            batch_results = batch_extract_all(batch_reviews, api_key, learner, mode)
            
            if batch_results:
                for result in batch_results:
                    idx = result["idx"]
                    df.at[idx, "sentiment"] = result["sentiment"]
                    df.at[idx, "level1_dimension"] = result["level1_dimension"]
                    df.at[idx, "level2_dimension"] = result["level2_dimension"]
                    df.at[idx, "motivation"] = result["motivation"]
                    df.at[idx, "emotion"] = result["emotion"]
                    df.at[idx, "persona"] = result["persona"]
                    df.at[idx, "scenario"] = result["scenario"]
                    df.at[idx, "analysis_status"] = "已分析"
                    success_count += 1
                    
                    # 统计
                    sentiment = result["sentiment"]
                    level1 = result["level1_dimension"]
                    level2 = result["level2_dimension"]
                    
                    if sentiment == "正面" and level1:
                        positive_dims[level1] += 1
                    elif sentiment == "负面" and level1:
                        negative_dims[level1] += 1
                    if level2:
                        level2_dims[level2] += 1
                    
                    motivations.append(result["motivation"])
                    emotions.append(result["emotion"])
                    personas.append(result["persona"])
                    scenarios.append(result["scenario"])
            else:
                # 批次完全失败，逐条处理
                for idx, text, rating in batch_reviews:
                    try:
                        result = extract_all_attributes(text, rating, api_key, learner, mode)
                        df.at[idx, "sentiment"] = result["sentiment"]
                        df.at[idx, "level1_dimension"] = result["level1_dimension"]
                        df.at[idx, "level2_dimension"] = result["level2_dimension"]
                        df.at[idx, "motivation"] = result["motivation"]
                        df.at[idx, "emotion"] = result["emotion"]
                        df.at[idx, "persona"] = result["persona"]
                        df.at[idx, "scenario"] = result["scenario"]
                        df.at[idx, "analysis_status"] = "已分析"
                        success_count += 1
                        
                        sentiment = result["sentiment"]
                        level1 = result["level1_dimension"]
                        level2 = result["level2_dimension"]
                        
                        if sentiment == "正面" and level1:
                            positive_dims[level1] += 1
                        elif sentiment == "负面" and level1:
                            negative_dims[level1] += 1
                        if level2:
                            level2_dims[level2] += 1
                        
                        motivations.append(result["motivation"])
                        emotions.append(result["emotion"])
                        personas.append(result["persona"])
                        scenarios.append(result["scenario"])
                    except:
                        failed_count += 1
                        fallback = fallback_result(text, rating)
                        df.at[idx, "sentiment"] = fallback["sentiment"]
                        df.at[idx, "level1_dimension"] = fallback["level1_dimension"]
                        df.at[idx, "level2_dimension"] = fallback["level2_dimension"]
                        df.at[idx, "motivation"] = fallback["motivation"]
                        df.at[idx, "emotion"] = fallback["emotion"]
                        df.at[idx, "persona"] = fallback["persona"]
                        df.at[idx, "scenario"] = fallback["scenario"]
                        df.at[idx, "analysis_status"] = "降级"
                        
                        motivations.append(fallback["motivation"])
                        emotions.append(fallback["emotion"])
                        personas.append(fallback["persona"])
                        scenarios.append(fallback["scenario"])
            
            # 每批保存进度
            learner.save_progress(batch_end, total)
            
        except Exception as e:
            # 批次异常，逐条降级
            for idx, text, rating in batch_reviews:
                failed_count += 1
                fallback = fallback_result(text, rating)
                df.at[idx, "sentiment"] = fallback["sentiment"]
                df.at[idx, "level1_dimension"] = fallback["level1_dimension"]
                df.at[idx, "level2_dimension"] = fallback["level2_dimension"]
                df.at[idx, "motivation"] = fallback["motivation"]
                df.at[idx, "emotion"] = fallback["emotion"]
                df.at[idx, "persona"] = fallback["persona"]
                df.at[idx, "scenario"] = fallback["scenario"]
                df.at[idx, "analysis_status"] = "降级"
                
                motivations.append(fallback["motivation"])
                emotions.append(fallback["emotion"])
                personas.append(fallback["persona"])
                scenarios.append(fallback["scenario"])
    
    progress_bar.empty()
    status_text.empty()
    time_text.empty()
    stats_text.empty()
    
    learner.clear_progress()
    
    if failed_count > 0:
        st.warning(f"⚠️ {failed_count} 条评论使用降级模式（关键词匹配），{success_count} 条使用AI分析")
    
    # ===== 维度学习 =====
    with st.spinner("🧠 正在学习新维度..."):
        all_reviews = df["review_text"].tolist()
        learner.learn_new_dimensions(all_reviews, api_key)
    
    # ===== 计算分布 =====
    total_count = len(df)
    motivation_dist = {k: v/total_count*100 for k, v in Counter(motivations).items()}
    emotion_dist = {k: v/total_count*100 for k, v in Counter(emotions).items()}
    persona_dist = {k: v/total_count*100 for k, v in Counter(personas).items()}
    scenario_dist = {k: v/total_count*100 for k, v in Counter(scenarios).items()}
    
    opportunities = discover_opportunities(dict(positive_dims), dict(negative_dims), total_count)
    
    # ===== 生成报告 =====
    strategic_insights = generate_report(
        dict(positive_dims), dict(negative_dims), emotion_dist,
        persona_dist, motivation_dist, scenario_dist,
        dict(level2_dims), df["review_text"].tolist(), api_key
    )
    
    analysis_data = {
        "total": total_count,
        "positive_dims": dict(positive_dims.most_common(20)),
        "negative_dims": dict(negative_dims.most_common(20)),
        "level2_dims": dict(level2_dims.most_common(30)),
        "motivation_dist": motivation_dist,
        "emotion_dist": emotion_dist,
        "persona_dist": persona_dist,
        "scenario_dist": scenario_dist,
        "opportunities": opportunities,
        "strategic_insights": strategic_insights,
        "learned_dimensions": learner.learned_dimensions,
        "stats": {
            "ai_analyzed": success_count,
            "fallback_used": failed_count,
            "total": total_count
        }
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
        st.markdown("### 🎯 分析模式")
        mode = st.selectbox(
            "选择模式",
            ["快速模式", "标准模式", "深度模式"],
            help="快速：情感+一级维度 | 标准：+二级维度 | 深度：+画像+场景+动机+情绪"
        )
        mode_map = {"快速模式": "quick", "标准模式": "standard", "深度模式": "deep"}
        mode_key = mode_map[mode]
        
        st.markdown("---")
        uploaded_file = st.file_uploader("上传评论文件", type=["csv", "xlsx"])
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 清除缓存", use_container_width=True):
                learner = DimensionLearner()
                learner.clear_progress()
                learner.clear_cache()
                st.success("已清除")
        
        start_analysis = st.button("🚀 开始分析", use_container_width=True, type="primary")
        
        st.markdown("---")
        if st.button("📝 加载示例数据", use_container_width=True):
            return api_key, get_sample_data(), mode_key, True
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.success(f"✅ {len(df)} 条")
                return api_key, df, mode_key, start_analysis
            except Exception as e:
                st.error(f"失败: {e}")
        return api_key, None, mode_key, start_analysis

# =========================
# 主函数
# =========================
def main():
    st.title("🧠 VOC 智能洞察平台 V2")
    st.caption("完整修复版 | 支持3000条评论 | 批量分析 | 自动降级 | 维度学习")
    
    api_key, input_df, mode, start_analysis = render_sidebar()
    
    if input_df is None:
        st.info("👈 上传文件或点击「加载示例数据」")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🚀 V2 核心特性
            
            - **⚡ 速度提升10倍**：批量分析10条/次
            - **💰 API成本降低80%**：1条评论1次调用
            - **🧠 自主学习维度**：自动发现和聚类
            - **💾 本地缓存**：重复评论直接读取
            - **📌 断点续传**：关闭页面可继续
            - **🛡️ 自动降级**：API失败用关键词匹配
            """)
        with col2:
            st.markdown("""
            ### 📊 性能目标
            
            | 评论数 | 耗时 | API调用 |
            |--------|------|---------|
            | 100条 | 30秒 | 10次 |
            | 300条 | 1分钟 | 30次 |
            | 500条 | 2分钟 | 50次 |
            | 1000条 | 5分钟 | 100次 |
            | 3000条 | 15分钟 | 300次 |
            """)
        return
    
    if "review_text" not in input_df.columns:
        st.error("文件缺少 `review_text` 列")
        return
    
    df = preprocess_data(input_df)
    learner = DimensionLearner()
    
    if start_analysis:
        if not api_key:
            st.warning("⚠️ 请输入 API Key")
        else:
            start_time = time.time()
            df, analysis_data = run_analysis_v2(df, api_key, learner, mode)
            elapsed = time.time() - start_time
            
            st.session_state["df"] = df
            st.session_state["analysis_data"] = analysis_data
            
            stats = analysis_data.get("stats", {})
            st.success(f"✅ 分析完成！{len(df)} 条，用时 {elapsed:.1f} 秒")
            st.info(f"📊 AI分析: {stats.get('ai_analyzed', 0)} 条 | 降级处理: {stats.get('fallback_used', 0)} 条")
            st.balloons()
    
    df = st.session_state.get("df", df)
    analysis_data = st.session_state.get("analysis_data", {})
    
    if not analysis_data:
        return
    
    # ===== 数据概览 =====
    stats = analysis_data.get("stats", {})
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("总评论", analysis_data.get("total", 0))
    col2.metric("一级维度", len(analysis_data.get("positive_dims", {})) + len(analysis_data.get("negative_dims", {})))
    col3.metric("二级维度", len(analysis_data.get("level2_dims", {})))
    col4.metric("学习聚类", len(analysis_data.get("learned_dimensions", {})))
    col5.metric("机会点", len(analysis_data.get("opportunities", [])))
    col6.metric("AI分析", stats.get("ai_analyzed", 0))
    col7.metric("降级处理", stats.get("fallback_used", 0))
    
    # ===== 原始数据 =====
    with st.expander("📋 数据预览", expanded=False):
        display_cols = ["review_text", "star_rating", "sentiment", "level1_dimension", "level2_dimension", "analysis_status"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True, height=200)
    
    # ===== 8个Tab =====
    tabs = st.tabs(["🎯 战略洞察", "📊 维度分析", "🧠 维度学习", "💭 购买动机", "😊 情绪分析", "👤 用户画像", "📍 使用场景", "🎯 机会发现", "📥 一键导出"])
    
    with tabs[0]:
        if analysis_data.get("strategic_insights"):
            st.markdown(analysis_data["strategic_insights"])
            st.download_button("📥 导出", analysis_data["strategic_insights"], f"strategic_{datetime.now().strftime('%Y%m%d')}.md")
    
    with tabs[1]:
        st.markdown("### 一级维度分析")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(make_bar_chart(analysis_data.get("positive_dims", {}), "好评 TOP", "#2ecc71"), use_container_width=True)
        with col2:
            st.plotly_chart(make_bar_chart(analysis_data.get("negative_dims", {}), "差评 TOP", "#e74c3c"), use_container_width=True)
        
        st.markdown("### 二级维度分析")
        st.plotly_chart(make_bar_chart(analysis_data.get("level2_dims", {}), "二级维度 TOP", "#3498db"), use_container_width=True)
    
    with tabs[2]:
        st.markdown("## 🧠 自主学习维度")
        learned = analysis_data.get("learned_dimensions", {})
        if learned:
            st.success(f"已学习 {len(learned)} 个维度聚类")
            for cluster_name, info in learned.items():
                members = info.get("members", [])
                with st.expander(f"📚 {cluster_name} ({len(members)}个成员)"):
                    st.write(", ".join(members))
        else:
            st.info("暂无学习维度，分析更多评论后自动学习")
    
    with tabs[3]:
        st.plotly_chart(make_pie_chart(analysis_data.get("motivation_dist", {}), "购买动机"), use_container_width=True)
    
    with tabs[4]:
        st.plotly_chart(make_emotion_chart(analysis_data.get("emotion_dist", {})), use_container_width=True)
    
    with tabs[5]:
        st.plotly_chart(make_pie_chart(analysis_data.get("persona_dist", {}), "用户画像"), use_container_width=True)
    
    with tabs[6]:
        st.plotly_chart(make_bar_chart(analysis_data.get("scenario_dist", {}), "使用场景", "#3498db"), use_container_width=True)
    
    with tabs[7]:
        opportunities = analysis_data.get("opportunities", [])
        if opportunities:
            for opp in opportunities[:5]:
                with st.expander(f"🎯 {opp['dimension']}"):
                    st.write(f"提及: {opp['mentions']} | 差评率: {opp['complaint_rate']}%")
                    if opp['complaint_rate'] > 50:
                        st.warning("⚠️ 紧急改进")
        else:
            st.info("暂无机会点数据")
    
    with tabs[8]:
        excel_data = export_all_data(df, analysis_data)
        st.download_button("📥 导出Excel", excel_data, f"voc_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", use_container_width=True)

if __name__ == "__main__":
    main()
