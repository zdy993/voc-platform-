# -*- coding: utf-8 -*-
"""
VOC 智能分析平台 - 智能加速版
保留API深度分析 + 速度提升5倍
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import re
import requests
from collections import Counter
from typing import List, Tuple
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="VOC 智能洞察平台",
    page_icon="🎯",
    layout="wide"
)

# =========================
# API 配置
# =========================
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def call_llm(api_key: str, prompt: str, max_tokens: int = 800) -> str:
    """优化的API调用"""
    if not api_key:
        return ""
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }
    
    for attempt in range(2):  # 只重试2次
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=20)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                time.sleep(1)
                continue
            else:
                if attempt == 1:
                    return ""
                time.sleep(0.5)
        except:
            if attempt == 1:
                return ""
            time.sleep(0.5)
    return ""

# =========================
# 批量处理函数（核心优化）
# =========================
def batch_extract_dimensions(reviews_batch: List[Tuple[int, str, int]], api_key: str) -> List[dict]:
    """批量提取维度和情感（一次API调用处理多条评论）"""
    if not api_key:
        return []
    
    # 构建批量请求
    batch_data = []
    for idx, text, rating in reviews_batch:
        batch_data.append({
            "id": idx,
            "text": text[:200],  # 限制长度
            "rating": rating
        })
    
    prompt = f"""分析以下{len(reviews_batch)}条评论，为每条评论输出JSON格式结果。

评论列表：
{json.dumps(batch_data, ensure_ascii=False, indent=2)}

要求输出格式（每条评论一行JSON）：
{{"id": 0, "sentiment": "正面/负面/中性", "dimensions": ["维度1", "维度2"]}}

可选维度：磁吸能力、手感、防滑性、保护性、耐用性、外观设计、清洁度、安装体验、性价比

只输出JSON，每行一条，不要其他说明："""

    try:
        result = call_llm(api_key, prompt, max_tokens=800)
        if not result:
            return []
        
        # 解析批量结果
        results = []
        for line in result.strip().split('\n'):
            try:
                data = json.loads(line.strip())
                results.append(data)
            except:
                continue
        
        # 确保返回所有评论的结果
        final_results = []
        for item in batch_data:
            found = next((r for r in results if r.get("id") == item["id"]), None)
            if found:
                final_results.append({
                    "idx": item["id"],
                    "sentiment": found.get("sentiment", "中性"),
                    "dimensions": found.get("dimensions", [])[:3]
                })
            else:
                # 默认值
                final_results.append({
                    "idx": item["id"],
                    "sentiment": "正面" if item["rating"] >= 4 else "负面" if item["rating"] <= 2 else "中性",
                    "dimensions": []
                })
        return final_results
    except:
        return []

def extract_single_attributes(text: str, rating: int, api_key: str) -> dict:
    """提取单个评论的其他属性（快速版）"""
    # 使用简化的API调用
    prompt = f"""分析评论，输出JSON：
评论：{text[:150]}
星级：{rating}

输出格式：{{"motivation":"动机","emotion":"情绪","persona":"身份","scenario":"场景"}}
动机选项：车载使用、商务办公、防摔保护、旅行使用、送礼、日常使用、游戏使用
情绪选项：惊喜、满意、平静、失望、焦虑、愤怒、后悔
身份选项：商务人士、学生、旅行用户、家庭用户、科技爱好者、游戏用户、普通用户
场景选项：车载、办公室、旅行、健身房、家庭、户外、通勤

只输出JSON："""
    
    try:
        result = call_llm(api_key, prompt, max_tokens=200)
        if result:
            clean = re.sub(r'```json\s*|```\s*', '', result.strip())
            data = json.loads(clean)
            return {
                "motivation": data.get("motivation", "日常使用"),
                "emotion": data.get("emotion", "满意" if rating >= 4 else "失望"),
                "persona": data.get("persona", "普通用户"),
                "scenario": data.get("scenario", "日常")
            }
    except:
        pass
    
    # 默认值
    return {
        "motivation": "日常使用",
        "emotion": "满意" if rating >= 4 else "失望" if rating <= 2 else "平静",
        "persona": "普通用户",
        "scenario": "日常"
    }

# =========================
# 报告生成函数
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
    """生成AI战略洞察报告"""
    pos_total = sum(positive_dims.values()) if positive_dims else 1
    neg_total = sum(negative_dims.values()) if negative_dims else 1
    
    top_pos = list(positive_dims.items())[:5] if positive_dims else []
    top_neg = list(negative_dims.items())[:5] if negative_dims else []
    
    pos_str = "\n".join([f"  - {dim}: {count}次 ({count/pos_total*100:.1f}%)" for dim, count in top_pos])
    neg_str = "\n".join([f"  - {dim}: {count}次 ({count/neg_total*100:.1f}%)" for dim, count in top_neg])
    
    sample_str = "\n".join([f"- {text[:100]}..." for text in sample_reviews[:5]])
    
    prompt = f"""你是资深产品分析师。基于以下数据生成洞察报告（300字以内）：

用户好评TOP3：
{pos_str}

用户差评TOP3：
{neg_str}

代表评论：
{sample_str}

请输出：
1. 核心发现（3点）
2. 最紧急的改进项
3. 建议的差异化策略

简洁输出，不要表格："""
    
    try:
        report = call_llm(api_key, prompt, max_tokens=800)
        if report and not report.startswith("API错误"):
            return f"## 🎯 AI战略洞察\n\n{report}"
    except:
        pass
    
    # 降级报告
    return f"""## 🎯 战略洞察

### 核心发现
- 用户最认可：{top_pos[0][0] if top_pos else '产品优势'} ({top_pos[0][1]/pos_total*100:.1f}%)
- 主要痛点：{top_neg[0][0] if top_neg else '待改进'} ({top_neg[0][1]/neg_total*100:.1f}%)
- 正向情绪占比：{emotion_dist.get('惊喜',0)+emotion_dist.get('满意',0):.1f}%

### 改进优先级
1. 立即改进：{top_neg[0][0] if top_neg else '主要痛点'}
2. 持续强化：{top_pos[0][0] if top_pos else '核心优势'}"""

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

| 用户类型 | 占比 |
|---------|------|
"""
    for persona, pct in list(persona_dist.items())[:10]:
        report += f"| {persona} | {pct:.1f}% |\n"
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
        ],
        "star_rating": [5, 2, 5, 1, 4, 4]
    })

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

def export_all_data(df: pd.DataFrame, analysis_data: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='原始数据', index=False)
        
        if analysis_data.get("positive_dims"):
            pos_df = pd.DataFrame([{"维度": k, "提及次数": v} for k, v in analysis_data["positive_dims"].items()])
            pos_df.to_excel(writer, sheet_name='好评维度', index=False)
        
        if analysis_data.get("negative_dims"):
            neg_df = pd.DataFrame([{"维度": k, "提及次数": v} for k, v in analysis_data["negative_dims"].items()])
            neg_df.to_excel(writer, sheet_name='差评维度', index=False)
        
        if analysis_data.get("persona_dist"):
            persona_df = pd.DataFrame([{"用户类型": k, "占比": f"{v:.1f}%"} for k, v in analysis_data["persona_dist"].items()])
            persona_df.to_excel(writer, sheet_name='用户画像', index=False)
        
        if analysis_data.get("opportunities"):
            opp_df = pd.DataFrame(analysis_data["opportunities"])
            opp_df.to_excel(writer, sheet_name='机会点', index=False)
    
    return output.getvalue()

# =========================
# 主分析函数（批量并行版）
# =========================
def run_analysis(df: pd.DataFrame, api_key: str, progress_callback=None):
    """批量并行分析 - 速度提升5倍"""
    df = df.copy()
    total = len(df)
    
    positive_dims = Counter()
    negative_dims = Counter()
    motivations = []
    emotions = []
    personas = []
    scenarios = []
    
    # 批量大小：每批5条评论
    batch_size = 5
    
    # 准备所有评论
    reviews_list = [(idx, row["review_text"], row["star_rating"]) 
                    for idx, row in df.iterrows()]
    
    # 分批处理
    batches = [reviews_list[i:i+batch_size] for i in range(0, len(reviews_list), batch_size)]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = {}
    
    for batch_idx, batch in enumerate(batches):
        # 更新进度
        progress = batch_idx / len(batches)
        progress_bar.progress(progress)
        status_text.text(f"正在分析第 {batch_idx * batch_size + 1} - {min((batch_idx + 1) * batch_size, total)} 条评论...")
        
        # 批量提取维度和情感
        batch_results = batch_extract_dimensions(batch, api_key)
        
        for result in batch_results:
            idx = result["idx"]
            all_results[idx] = {
                "sentiment": result["sentiment"],
                "dimensions": result["dimensions"]
            }
            
            # 统计
            for dim in result["dimensions"]:
                if result["sentiment"] == "正面":
                    positive_dims[dim] += 1
                elif result["sentiment"] == "负面":
                    negative_dims[dim] += 1
        
        # 每个batch之间休息0.5秒
        time.sleep(0.5)
        
        if progress_callback:
            progress_callback(batch_idx + 1, len(batches))
    
    status_text.text("正在分析属性和生成报告...")
    
    # 并行提取其他属性
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for idx, row in df.iterrows():
            future = executor.submit(extract_single_attributes, row["review_text"], row["star_rating"], api_key)
            futures[future] = idx
        
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            df.at[idx, "motivation"] = result["motivation"]
            df.at[idx, "emotion"] = result["emotion"]
            df.at[idx, "persona"] = result["persona"]
            df.at[idx, "scenario"] = result["scenario"]
            
            motivations.append(result["motivation"])
            emotions.append(result["emotion"])
            personas.append(result["persona"])
            scenarios.append(result["scenario"])
    
    # 更新DataFrame
    for idx, result in all_results.items():
        df.at[idx, "sentiment"] = result["sentiment"]
        df.at[idx, "dimensions"] = ", ".join(result["dimensions"])
    
    progress_bar.empty()
    status_text.empty()
    
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
        persona_dist, motivation_dist, df["review_text"].tolist()[:20], api_key
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
        st.info("💡 使用 DeepSeek API（智能分析）\n注册：https://platform.deepseek.com")
        api_key = st.text_input("API Key", type="password", placeholder="sk-...", help="需要API Key才能进行智能分析")
        
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
    st.title("🎯 VOC 智能洞察平台")
    st.caption("AI深度分析 | 批量处理加速 | 500条评论 < 2分钟")
    
    api_key, input_df, start_analysis = render_sidebar()
    
    if input_df is None:
        st.info("👈 左侧上传文件或点击「加载示例数据」")
        st.markdown("""
        ### 🎯 平台功能
        
        - **🤖 AI智能分析**：基于DeepSeek大模型深度理解
        - **⚡ 批量加速**：一次API调用分析5条评论
        - **📊 8大维度**：全面洞察用户声音
        - **💾 一键导出**：Excel格式完整数据
        
        ### 速度说明
        - 100条评论：约30秒
        - 500条评论：约2-3分钟
        - 1000条评论：约5-6分钟
        """)
        return
    
    if "review_text" not in input_df.columns:
        st.error("文件缺少 `review_text` 列")
        st.write("当前列名:", input_df.columns.tolist())
        return
    
    df = preprocess_data(input_df)
    
    if start_analysis:
        if not api_key:
            st.warning("⚠️ 请输入 DeepSeek API Key")
            st.info("💡 注册地址：https://platform.deepseek.com（新用户免费500万tokens）")
        else:
            with st.spinner(f"🤖 AI正在分析 {len(df)} 条评论，请稍候..."):
                start_time = time.time()
                df, analysis_data = run_analysis(df, api_key)
                elapsed = time.time() - start_time
                
                st.session_state["df"] = df
                st.session_state["analysis_data"] = analysis_data
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
    col4.metric("用户画像", len(analysis_data.get("persona_dist", {})))
    col5.metric("识别情绪", len(analysis_data.get("emotion_dist", {})))
    
    # 8个Tab
    tabs = st.tabs(["🎯 战略洞察", "📊 维度分析", "💭 购买动机", "😊 情绪分析", "👤 用户画像", "📍 使用场景", "🎯 机会发现", "📥 一键导出"])
    
    with tabs[0]:
        if analysis_data.get("strategic_insights"):
            st.markdown(analysis_data["strategic_insights"])
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(make_bar_chart(analysis_data.get("positive_dims", {}), "好评 TOP 维度", "#2ecc71"), use_container_width=True)
            if analysis_data.get("dimension_report"):
                with st.expander("📄 查看详细维度报告"):
                    st.markdown(analysis_data["dimension_report"])
        with col2:
            st.plotly_chart(make_bar_chart(analysis_data.get("negative_dims", {}), "差评 TOP 维度", "#e74c3c"), use_container_width=True)
    
    with tabs[2]:
        st.plotly_chart(make_pie_chart(analysis_data.get("motivation_dist", {}), "购买动机分布"), use_container_width=True)
        if analysis_data.get("motivation_report"):
            with st.expander("📄 查看详细动机报告"):
                st.markdown(analysis_data["motivation_report"])
    
    with tabs[3]:
        st.plotly_chart(make_emotion_chart(analysis_data.get("emotion_dist", {})), use_container_width=True)
        if analysis_data.get("emotion_report"):
            with st.expander("📄 查看详细情绪报告"):
                st.markdown(analysis_data["emotion_report"])
    
    with tabs[4]:
        st.plotly_chart(make_pie_chart(analysis_data.get("persona_dist", {}), "用户画像分布"), use_container_width=True)
        if analysis_data.get("persona_report"):
            with st.expander("📄 查看详细画像报告"):
                st.markdown(analysis_data["persona_report"])
    
    with tabs[5]:
        st.plotly_chart(make_bar_chart(analysis_data.get("scenario_dist", {}), "使用场景分布", "#3498db"), use_container_width=True)
    
    with tabs[6]:
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
                with st.expander("📄 查看详细机会报告"):
                    st.markdown(analysis_data["opportunity_report"])
        else:
            st.info("暂无机会点数据")
    
    with tabs[7]:
        excel_data = export_all_data(df, analysis_data)
        st.download_button("📥 导出全部数据 (Excel)", excel_data, 
                          file_name=f"voc_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                          use_container_width=True)

if __name__ == "__main__":
    main()
