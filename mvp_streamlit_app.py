import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from transformers import pipeline
import time

# 设置 matplotlib 字体（避免标题显示问题）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用英文标题，无需中文字体
plt.rcParams['axes.unicode_minus'] = False

# 页面设置
st.set_page_config(page_title="暴力舆情检测MVP", layout="wide")
st.title("🔍 网络暴力舆情检测最小可行系统")
st.markdown("---")

# 侧边栏：控制面板
with st.sidebar:
    st.header("⚙️ 控制面板")
    
    # 数据选择
    data_option = st.radio(
        "选择数据源:",
        ["使用示例数据", "上传CSV文件"]
    )
    
    # 文件上传组件（始终显示，放在按钮外）
    uploaded_file = None
    if data_option == "上传CSV文件":
        uploaded_file = st.file_uploader("上传CSV文件", type=['csv'], key="csv_uploader")
        if uploaded_file is not None:
            # 缓存到 session_state，避免重复读取
            st.session_state['uploaded_df'] = pd.read_csv(uploaded_file)
    
    # 置信度阈值
    confidence_threshold = st.slider(
        "暴力检测置信度阈值 (%):",
        min_value=0,
        max_value=100,
        value=50,
        help="只有当模型置信度超过此阈值时，才标记为暴力言论"
    )
    
    sample_size = st.slider("分析样本数量:", 10, 200, 50)
    
    analyze_button = st.button("🚀 开始分析", type="primary")
    
    st.markdown("---")
    st.caption("系统状态: 最小可行版本")

# 主界面
if analyze_button:
    # 1. 进度显示
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 2. 加载数据
    status_text.text("步骤1/4: 加载数据中...")
    progress_bar.progress(25)
    
    if data_option == "使用示例数据":
        # 动态时间戳：以当前时间为终点，向前生成10个小时
        example_data = {
            'text': [
                "You are such an idiot, I can't believe how stupid you are!",
                "Beautiful weather today, perfect for a walk in the park.",
                "Why don't you just disappear? No one wants you here.",
                "The new movie release looks promising, great reviews so far.",
                "Everyone hates you, you should know that by now.",
                "Thanks for your help with the project, really appreciate it!",
                "I hope you fail miserably, you deserve nothing but pain.",
                "Great discussion in today's meeting, lots of good ideas.",
                "You're a worthless piece of trash, nobody cares about you.",
                "Looking forward to the holiday season, time with family."
            ],
            'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=10, freq='h')
        }
        df = pd.DataFrame(example_data)
    else:
        # 使用上传的文件（从 session_state 或直接读取）
        if uploaded_file is not None:
            df = st.session_state.get('uploaded_df')
            if df is None:
                df = pd.read_csv(uploaded_file)
        else:
            st.warning("请先上传CSV文件")
            st.stop()
        
        # 检查必要的列
        if 'text' not in df.columns:
            st.error("CSV文件必须包含 'text' 列")
            st.stop()
    
    texts = df['text'].tolist()[:sample_size]
    
    # 显示数据预览
    st.subheader("📊 数据预览")
    st.dataframe(df.head(5), use_container_width=True)
    
    # 3. 主题建模
    status_text.text("步骤2/4: 主题建模分析中...")
    progress_bar.progress(50)
    time.sleep(1)  # 模拟处理时间
    
    with st.spinner("正在分析文本主题..."):
        topic_model = BERTopic(language="english", verbose=False)
        topics, _ = topic_model.fit_transform(texts)
    
    # 4. 暴力检测
    status_text.text("步骤3/4: 检测暴力言论...")
    progress_bar.progress(75)
    
    with st.spinner("正在检测暴力言论..."):
        classifier = pipeline("text-classification", 
                            model="unitary/toxic-bert",
                            truncation=True)
        
        results = []
        toxic_labels = ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        for i, text in enumerate(texts):
            pred = classifier(text[:512])[0]
            confidence = pred['score'] * 100
            
            is_toxic = (pred['label'] in toxic_labels) and (confidence >= confidence_threshold)
            
            result_text = text[:100] + "..." if len(text) > 100 else text
            
            results.append({
                '文本': result_text,
                '原始文本': text,
                '主题': int(topics[i]),
                '预测标签': pred['label'],
                '置信度(%)': f"{confidence:.2f}",
                '是否暴力': "是" if is_toxic else "否",
                '超过阈值': "是" if confidence >= confidence_threshold else "否"
            })
    
    # 5. 结果显示
    status_text.text("步骤4/4: 生成报告...")
    progress_bar.progress(100)
    
    result_df = pd.DataFrame(results)
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Topic Distribution")
        
        topic_counts = pd.Series(topics).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        topic_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Number of Texts')
        ax.set_title('Topic Distribution of Texts')
        st.pyplot(fig)
        
        st.subheader("🔑 Topic Keywords")
        try:
            topic_info = topic_model.get_topic_info()
            for i, row in topic_info.head(5).iterrows():
                st.write(f"**Topic {row['Topic']}**: {row['Name']}")
        except:
            st.info("Failed to get topic information")
    
    with col2:
        st.subheader("⚠️ Toxic Speech Statistics")
        
        toxic_count = result_df['是否暴力'].value_counts().get('是', 0)
        total_count = len(result_df)
        toxic_ratio = toxic_count / total_count if total_count > 0 else 0
        
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.metric("Toxic Count", f"{toxic_count}")
        with col2_2:
            st.metric("Toxic Ratio", f"{toxic_ratio:.1%}")
        with col2_3:
            st.metric("Threshold", f"{confidence_threshold}%")
        
        st.subheader("📊 Toxicity by Topic")
        if '主题' in result_df.columns:
            topic_stats = []
            for topic in sorted(result_df['主题'].unique()):
                topic_data = result_df[result_df['主题'] == topic]
                toxic_in_topic = (topic_data['是否暴力'] == '是').sum()
                total_in_topic = len(topic_data)
                ratio = toxic_in_topic / total_in_topic if total_in_topic > 0 else 0
                topic_stats.append({
                    'Topic': topic,
                    'Total Texts': total_in_topic,
                    'Toxic Texts': toxic_in_topic,
                    'Toxic Ratio': ratio
                })
            
            if topic_stats:
                stats_df = pd.DataFrame(topic_stats)
                st.dataframe(stats_df.style.format({'Toxic Ratio': '{:.1%}'}), 
                            use_container_width=True)
    
    # 6. 详细结果表格
    st.subheader("📋 Detailed Results")
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        filter_option = st.selectbox(
            "Filter results:",
            ["All", "Only Toxic", "Only Non-Toxic"]
        )
    
    display_df = result_df.copy()
    if filter_option == "Only Toxic":
        display_df = display_df[display_df['是否暴力'] == '是']
    elif filter_option == "Only Non-Toxic":
        display_df = display_df[display_df['是否暴力'] == '否']
    
    st.dataframe(display_df, use_container_width=True)
    
    # 7. 置信度分布分析
    st.subheader("📊 Confidence Score Analysis")
    
    try:
        confidence_values = pd.to_numeric(result_df['置信度(%)'].str.rstrip('%'))
        
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.hist(confidence_values, bins=20, color='lightcoral', edgecolor='black')
        ax1.axvline(x=confidence_threshold, color='red', linestyle='--', label=f'Threshold ({confidence_threshold}%)')
        ax1.set_xlabel('Confidence (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Score Distribution')
        ax1.legend()
        
        toxic_conf = confidence_values[result_df['是否暴力'] == '是']
        non_toxic_conf = confidence_values[result_df['是否暴力'] == '否']
        
        ax2.boxplot([non_toxic_conf.dropna(), toxic_conf.dropna()], 
                   labels=['Non-Toxic', 'Toxic'])
        ax2.set_ylabel('Confidence (%)')
        ax2.set_title('Confidence Distribution by Toxicity')
        
        plt.tight_layout()
        st.pyplot(fig2)
    except:
        st.warning("Could not generate confidence distribution plot")
    
    # 8. 导出功能
    st.subheader("💾 Export Results")
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"violence_detection_results_threshold{confidence_threshold}.csv",
        mime="text/csv"
    )
    
    status_text.text("✅ Analysis complete!")
    progress_bar.empty()

else:
    # 初始界面：使用说明
    st.info("👈 Please select data source in the left panel and click 'Start Analysis'")
    
    st.markdown("""
    ### 📌 System Functions
    
    This MVP demonstrates the core process of online toxic speech detection:
    
    1. **Data Input**: Use sample data or upload a CSV file
    2. **Topic Analysis**: Automatically identify discussion topics
    3. **Toxicity Detection**: Classify each text as toxic or not
    4. **Result Visualization**: Interactive charts and detailed data
    
    ### 🚀 Quick Start
    
    1. Select "Use Sample Data" on the left
    2. Adjust the toxicity threshold (recommended 50%-70%)
    3. Adjust sample size (recommended 50-100)
    4. Click "Start Analysis"
    5. Wait 10-30 seconds for results
    
    ### 📁 CSV Format Requirements
    
    If you upload a CSV file, it must contain at least a column named `text`.
    """)
    
    # 显示示例数据格式
    example_df = pd.DataFrame({
        'text': ['Example text 1', 'Example text 2', 'Example text 3'],
        'timestamp': ['2024-01-01 10:00', '2024-01-01 11:00', '2024-01-01 12:00']
    })
    with st.expander("View example CSV format"):
        st.dataframe(example_df)

st.markdown("---")
st.caption("Cyberbullying Detection MVP | Based on BERTopic and Toxic-BERT | Version 1.2 (Fixed upload, charts, timestamp)")
