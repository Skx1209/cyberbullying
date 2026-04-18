import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from transformers import pipeline
import time

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
    
    # 添加置信度阈值设置
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
        # 修正后的示例数据
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
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='H')
        }
        df = pd.DataFrame(example_data)
    else:
        # 文件上传
        uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            # 检查必要的列
            if 'text' not in df.columns:
                st.error("CSV文件必须包含 'text' 列")
                st.stop()
        else:
            st.warning("请上传CSV文件")
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
    
    # 4. 暴力检测 - 修正逻辑
    status_text.text("步骤3/4: 检测暴力言论...")
    progress_bar.progress(75)
    
    with st.spinner("正在检测暴力言论..."):
        # 使用预训练的毒性检测模型
        classifier = pipeline("text-classification", 
                            model="unitary/toxic-bert",
                            truncation=True)
        
        results = []
        toxic_labels = ['toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        for i, text in enumerate(texts):
            # 只取前512字符（模型限制）
            pred = classifier(text[:512])[0]
            confidence = pred['score'] * 100  # 转换为百分比
            
            # 使用阈值判断：只有置信度超过阈值且标签为毒性时才标记为暴力
            is_toxic = (pred['label'] in toxic_labels) and (confidence >= confidence_threshold)
            
            # 获取最相关的毒性标签（如果有多个预测，这里简化处理）
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
        st.subheader("📈 主题分布")
        
        # 主题统计
        topic_counts = pd.Series(topics).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        topic_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_xlabel('主题编号')
        ax.set_ylabel('文本数量')
        ax.set_title('各主题文本分布')
        st.pyplot(fig)
        
        # 显示主题关键词
        st.subheader("🔑 主题关键词")
        try:
            topic_info = topic_model.get_topic_info()
            for i, row in topic_info.head(5).iterrows():
                st.write(f"**主题 {row['Topic']}**: {row['Name']}")
        except:
            st.info("主题信息获取失败")
    
    with col2:
        st.subheader("⚠️ 暴力言论统计")
        
        # 暴力言论比例
        toxic_count = result_df['是否暴力'].value_counts().get('是', 0)
        total_count = len(result_df)
        toxic_ratio = toxic_count / total_count if total_count > 0 else 0
        
        # 显示指标
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.metric("暴力言论数", f"{toxic_count} 条")
        with col2_2:
            st.metric("暴力比例", f"{toxic_ratio:.1%}")
        with col2_3:
            st.metric("当前阈值", f"{confidence_threshold}%")
        
        # 各主题暴力比例
        st.subheader("📊 各主题暴力情况")
        if '主题' in result_df.columns:
            topic_stats = []
            for topic in sorted(result_df['主题'].unique()):
                topic_data = result_df[result_df['主题'] == topic]
                toxic_in_topic = (topic_data['是否暴力'] == '是').sum()
                total_in_topic = len(topic_data)
                ratio = toxic_in_topic / total_in_topic if total_in_topic > 0 else 0
                topic_stats.append({
                    '主题': topic,
                    '总文本': total_in_topic,
                    '暴力文本': toxic_in_topic,
                    '暴力比例': ratio
                })
            
            if topic_stats:
                stats_df = pd.DataFrame(topic_stats)
                st.dataframe(stats_df.style.format({'暴力比例': '{:.1%}'}), 
                            use_container_width=True)
    
    # 6. 详细结果表格
    st.subheader("📋 详细分析结果")
    
    # 添加过滤选项
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        filter_option = st.selectbox(
            "筛选结果显示:",
            ["全部", "仅暴力言论", "仅非暴力言论"]
        )
    
    # 应用筛选
    display_df = result_df.copy()
    if filter_option == "仅暴力言论":
        display_df = display_df[display_df['是否暴力'] == '是']
    elif filter_option == "仅非暴力言论":
        display_df = display_df[display_df['是否暴力'] == '否']
    
    st.dataframe(display_df, use_container_width=True)
    
    # 7. 置信度分布分析
    st.subheader("📊 置信度分布分析")
    
    try:
        # 提取置信度数值用于分析
        confidence_values = pd.to_numeric(result_df['置信度(%)'].str.rstrip('%'))
        
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 置信度分布直方图
        ax1.hist(confidence_values, bins=20, color='lightcoral', edgecolor='black')
        ax1.axvline(x=confidence_threshold, color='red', linestyle='--', label=f'阈值 ({confidence_threshold}%)')
        ax1.set_xlabel('置信度 (%)')
        ax1.set_ylabel('频数')
        ax1.set_title('置信度分布')
        ax1.legend()
        
        # 箱型图：暴力 vs 非暴力的置信度分布
        toxic_conf = confidence_values[result_df['是否暴力'] == '是']
        non_toxic_conf = confidence_values[result_df['是否暴力'] == '否']
        
        ax2.boxplot([non_toxic_conf.dropna(), toxic_conf.dropna()], 
                   labels=['非暴力', '暴力'])
        ax2.set_ylabel('置信度 (%)')
        ax2.set_title('暴力/非暴力言论的置信度分布')
        
        plt.tight_layout()
        st.pyplot(fig2)
    except:
        st.warning("无法生成置信度分布图")
    
    # 8. 导出功能
    st.subheader("💾 导出结果")
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="下载分析结果(CSV)",
        data=csv,
        file_name=f"violence_detection_results_threshold{confidence_threshold}.csv",
        mime="text/csv"
    )
    
    status_text.text("✅ 分析完成！")
    progress_bar.empty()

else:
    # 初始界面：使用说明
    st.info("👈 请先在左侧控制面板选择数据源，然后点击'开始分析'按钮")
    
    st.markdown("""
    ### 📌 系统功能说明
    
    这个最小系统演示了网络暴力舆情检测的核心流程：
    
    1. **数据输入**：使用示例数据或上传CSV文件
    2. **主题分析**：自动识别文本中的讨论主题
    3. **暴力检测**：判断每条文本是否包含暴力/毒性内容
    4. **结果展示**：可视化分析和详细数据
    
    ### 🔧 重要修复说明
    
    针对之前版本中"置信度低但判断为暴力"的问题，本版本进行了以下修复：
    
    1. **添加置信度阈值控制**：可在侧边栏设置阈值（默认50%）
    2. **双重验证逻辑**：只有当预测标签为毒性类别 **且** 置信度超过阈值时才标记为暴力
    3. **增强的可视化**：新增置信度分布分析图表
    4. **更详细的结果**：显示预测标签和是否超过阈值
    
    ### 🚀 快速开始
    
    1. 在左侧选择"使用示例数据"
    2. 调整暴力检测阈值（建议50%-70%）
    3. 调整样本数量（建议50-100）
    4. 点击"开始分析"按钮
    5. 等待约10-30秒查看结果
    
    ### 📁 数据格式要求
    
    如需上传CSV文件，请确保包含至少一列名为 `text` 的列。
    """)
    
    # 显示示例数据格式
    example_df = pd.DataFrame({
        'text': ['示例文本1', '示例文本2', '示例文本3'],
        'timestamp': ['2024-01-01 10:00', '2024-01-01 11:00', '2024-01-01 12:00']
    })
    with st.expander("查看示例数据格式"):
        st.dataframe(example_df)

st.markdown("---")
st.caption("网络暴力舆情检测MVP系统 | 基于BERTopic和Toxic-BERT | 版本 1.1（已修复置信度问题）")
