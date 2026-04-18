import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from transformers import pipeline
import time

# Set matplotlib font (avoid display issues with titles)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Page config
st.set_page_config(page_title="Cyberbullying Detection MVP", layout="wide")
st.title("🔍 Cyberbullying Detection MVP System")
st.markdown("---")

# Sidebar: Control Panel
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Data source selection
    data_option = st.radio(
        "Select data source:",
        ["Use sample data", "Upload CSV file"]
    )
    
    # File uploader (always visible, outside button)
    uploaded_file = None
    if data_option == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], key="csv_uploader")
        if uploaded_file is not None:
            # Cache to session_state to avoid re-reading
            st.session_state['uploaded_df'] = pd.read_csv(uploaded_file)
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Toxicity detection confidence threshold (%):",
        min_value=0,
        max_value=100,
        value=50,
        help="Only mark text as toxic if model confidence exceeds this threshold"
    )
    
    sample_size = st.slider("Sample size for analysis:", 10, 200, 50)
    
    analyze_button = st.button("🚀 Start Analysis", type="primary")
    
    st.markdown("---")
    st.caption("System status: MVP version")

# Main interface
if analyze_button:
    # 1. Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 2. Load data
    status_text.text("Step 1/4: Loading data...")
    progress_bar.progress(25)
    
    if data_option == "Use sample data":
        # Dynamic timestamp: ends at current time, go back 10 hours
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
        # Use uploaded file (from session_state or direct read)
        if uploaded_file is not None:
            df = st.session_state.get('uploaded_df')
            if df is None:
                df = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file first")
            st.stop()
        
        # Check required columns
        if 'text' not in df.columns:
            st.error("CSV file must contain a column named 'text'")
            st.stop()
    
    texts = df['text'].tolist()[:sample_size]
    
    # Show data preview
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(5), use_container_width=True)
    
    # 3. Topic modeling
    status_text.text("Step 2/4: Performing topic modeling...")
    progress_bar.progress(50)
    time.sleep(1)  # Simulate processing time
    
    with st.spinner("Analyzing text topics..."):
        topic_model = BERTopic(language="english", verbose=False)
        topics, _ = topic_model.fit_transform(texts)
    
    # 4. Toxicity detection
    status_text.text("Step 3/4: Detecting toxic speech...")
    progress_bar.progress(75)
    
    with st.spinner("Detecting toxic content..."):
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
                'Text (short)': result_text,
                'Original Text': text,
                'Topic': int(topics[i]),
                'Predicted Label': pred['label'],
                'Confidence (%)': f"{confidence:.2f}",
                'Is Toxic': "Yes" if is_toxic else "No",
                'Above Threshold': "Yes" if confidence >= confidence_threshold else "No"
            })
    
    # 5. Display results
    status_text.text("Step 4/4: Generating report...")
    progress_bar.progress(100)
    
    result_df = pd.DataFrame(results)
    
    # Two-column layout
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
        
        toxic_count = result_df['Is Toxic'].value_counts().get('Yes', 0)
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
        if 'Topic' in result_df.columns:
            topic_stats = []
            for topic in sorted(result_df['Topic'].unique()):
                topic_data = result_df[result_df['Topic'] == topic]
                toxic_in_topic = (topic_data['Is Toxic'] == 'Yes').sum()
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
    
    # 6. Detailed results table
    st.subheader("📋 Detailed Results")
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        filter_option = st.selectbox(
            "Filter results:",
            ["All", "Only Toxic", "Only Non-Toxic"]
        )
    
    display_df = result_df.copy()
    if filter_option == "Only Toxic":
        display_df = display_df[display_df['Is Toxic'] == 'Yes']
    elif filter_option == "Only Non-Toxic":
        display_df = display_df[display_df['Is Toxic'] == 'No']
    
    st.dataframe(display_df, use_container_width=True)
    
    # 7. Confidence distribution analysis
    st.subheader("📊 Confidence Score Analysis")
    
    try:
        confidence_values = pd.to_numeric(result_df['Confidence (%)'].str.rstrip('%'))
        
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.hist(confidence_values, bins=20, color='lightcoral', edgecolor='black')
        ax1.axvline(x=confidence_threshold, color='red', linestyle='--', label=f'Threshold ({confidence_threshold}%)')
        ax1.set_xlabel('Confidence (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Score Distribution')
        ax1.legend()
        
        toxic_conf = confidence_values[result_df['Is Toxic'] == 'Yes']
        non_toxic_conf = confidence_values[result_df['Is Toxic'] == 'No']
        
        ax2.boxplot([non_toxic_conf.dropna(), toxic_conf.dropna()], 
                   labels=['Non-Toxic', 'Toxic'])
        ax2.set_ylabel('Confidence (%)')
        ax2.set_title('Confidence Distribution by Toxicity')
        
        plt.tight_layout()
        st.pyplot(fig2)
    except:
        st.warning("Could not generate confidence distribution plot")
    
    # 8. Export functionality
    st.subheader("💾 Export Results")
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"toxicity_detection_results_threshold{confidence_threshold}.csv",
        mime="text/csv"
    )
    
    status_text.text("✅ Analysis complete!")
    progress_bar.empty()

else:
    # Initial interface: instructions
    st.info("👈 Please select a data source in the left panel and click 'Start Analysis'")
    
    st.markdown("""
    ### 📌 System Functions
    
    This MVP demonstrates the core process of online toxic speech detection:
    
    1. **Data Input**: Use sample data or upload a CSV file
    2. **Topic Analysis**: Automatically identify discussion topics
    3. **Toxicity Detection**: Classify each text as toxic or not
    4. **Result Visualization**: Interactive charts and detailed data
    
    ### 🚀 Quick Start
    
    1. Select "Use sample data" on the left
    2. Adjust the toxicity threshold (recommended 50%-70%)
    3. Adjust sample size (recommended 50-100)
    4. Click "Start Analysis"
    5. Wait 10-30 seconds for results
    
    ### 📁 CSV Format Requirements
    
    If you upload a CSV file, it must contain at least a column named `text`.
    """)
    
    # Show example data format
    example_df = pd.DataFrame({
        'text': ['Example text 1', 'Example text 2', 'Example text 3'],
        'timestamp': ['2024-01-01 10:00', '2024-01-01 11:00', '2024-01-01 12:00']
    })
    with st.expander("View example CSV format"):
        st.dataframe(example_df)

st.markdown("---")
st.caption("Cyberbullying Detection MVP | Based on BERTopic and Toxic-BERT | Version 1.2 (English UI)")
