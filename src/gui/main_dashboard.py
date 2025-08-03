"""
Main GUI Dashboard for Data Science Project
Comprehensive interface for all 8 experiments with multiple activities
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import experiment modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.experiment_1 import Experiment1
from experiments.experiment_2 import Experiment2
from experiments.experiment_3 import Experiment3
from experiments.experiment_4 import Experiment4
from experiments.experiment_5 import Experiment5
from experiments.experiment_6 import Experiment6
from experiments.experiment_7 import Experiment7
from experiments.experiment_8 import Experiment8
from experiments.statistical_analysis import StatisticalAnalysis

# Configure Streamlit page
st.set_page_config(
    page_title="Zudio E-commerce Analytics Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .experiment-card {
        background: teal;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .activity-card {
        background: red;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196f3;
    }
    .metric-card {
        background: blue;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #8bc34a;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛍️ Zudio E-commerce Analytics Platform</h1>
        <p>Comprehensive Data Science Project with 8 Experiments & GUI-based Activities</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    # Project overview in sidebar
    with st.sidebar.expander("📊 Project Overview"):
        st.write("""
        **Domain**: Zudio E-commerce Analytics
        
        **Datasets**:
        - 🏷️ Product Catalog (1,242 items)
        - 💬 Customer Reviews (23,578 reviews)
        - 💰 Sales Data (7,901 transactions)
        
        **Goals**:
        - Customer Experience Optimization
        - Sales Pattern Analysis
        - Predictive Modeling
        - Responsible AI Implementation
        """)

    # Experiment selection
    experiment_options = {
        "🎯 Experiment 1: Case Study Framing & Dataset Preparation": "exp1",
        "🔍 Experiment 2: Data Profiling, Cleaning & Feature Engineering": "exp2", 
        "📊 Experiment 3: EDA & Statistical Analysis": "exp3",
        "🤖 Experiment 4: ML Modeling & Experiment Tracking": "exp4",
        "🔬 Experiment 5: Explainable AI & Fairness Evaluation": "exp5",
        "🐳 Experiment 6: Containerization & API Deployment": "exp6",
        "🔄 Experiment 7: CI/CD Pipeline with Open Source Tools": "exp7",
        "📈 Experiment 8: Dashboard, Responsible AI Reporting & Portfolio": "exp8",
        "📊 Statistical Analysis: Mathematical Properties & Analysis": "stats"
    }
    
    selected_experiment = st.sidebar.selectbox(
        "Select Experiment",
        list(experiment_options.keys())
    )
    
    exp_key = experiment_options[selected_experiment]
    
    # Main content area
    if exp_key == "exp1":
        render_experiment_1()
    elif exp_key == "exp2":
        render_experiment_2()
    elif exp_key == "exp3":
        render_experiment_3()
    elif exp_key == "exp4":
        render_experiment_4()
    elif exp_key == "exp5":
        render_experiment_5()
    elif exp_key == "exp6":
        render_experiment_6()
    elif exp_key == "exp7":
        render_experiment_7()
    elif exp_key == "exp8":
        render_experiment_8()
    elif exp_key == "stats":
        render_statistical_analysis()

def render_experiment_1():
    """Render Experiment 1: Case Study Framing & Dataset Preparation"""
    st.markdown("""
    <div class="experiment-card">
        <h2>🎯 Experiment 1: Case Study Framing & Dataset Preparation</h2>
        <p><strong>Objective</strong>: Define a real-world domain problem, benchmark existing solutions, acquire data (API/scraping/open portals). Document versioning plan with DVC.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize experiment
    exp1 = Experiment1()
    
    # Activity selection
    activities = [
        "📝 Problem Statement Development",
        "🔍 Literature Review & Benchmarking", 
        "📊 Success Metrics Definition",
        "🌐 Data Source Identification",
        "📥 Data Acquisition & Validation",
        "🗂️ Dataset Documentation",
        "🔧 DVC Setup & Versioning",
        "📋 Data Schema Definition",
        "🎯 Baseline Model Planning"
    ]
    
    selected_activities = st.multiselect(
        "Select Activities to Execute (Experiment 1):",
        activities,
        default=activities[:3]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Execute Selected Activities", key="exp1_execute"):
            for activity in selected_activities:
                with st.expander(f"✅ {activity}", expanded=True):
                    if activity == "📝 Problem Statement Development":
                        exp1.develop_problem_statement()
                    elif activity == "🔍 Literature Review & Benchmarking":
                        exp1.literature_review()
                    elif activity == "📊 Success Metrics Definition":
                        exp1.define_success_metrics()
                    elif activity == "🌐 Data Source Identification":
                        exp1.identify_data_sources()
                    elif activity == "📥 Data Acquisition & Validation":
                        exp1.data_acquisition()
                    elif activity == "🗂️ Dataset Documentation":
                        exp1.document_dataset()
                    elif activity == "🔧 DVC Setup & Versioning":
                        exp1.setup_dvc()
                    elif activity == "📋 Data Schema Definition":
                        exp1.define_schema()
                    elif activity == "🎯 Baseline Model Planning":
                        exp1.baseline_planning()
    
    # Add Enhanced Dataset Functionality at the bottom
    exp1.add_enhanced_dataset_section()
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📈 Progress Tracker</h4>
            <p><strong>Activities</strong>: 9 total</p>
            <p><strong>Selected</strong>: {}</p>
            <p><strong>Tools</strong>: DVC, Git, Python</p>
        </div>
        """.format(len(selected_activities)), unsafe_allow_html=True)

def render_experiment_2():
    """Render Experiment 2: Data Profiling, Cleaning & Feature Engineering"""
    st.markdown("""
    <div class="experiment-card">
        <h2>🔍 Experiment 2: Data Profiling, Cleaning & Feature Engineering</h2>
        <p><strong>Objective</strong>: Perform profiling, clean issues, engineer new features, validate, and version cleaned dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp2 = Experiment2()
    
    activities = [
        "📊 Comprehensive Data Profiling",
        "🔍 Data Quality Assessment",
        "🧹 Missing Value Analysis & Treatment",
        "🔗 Duplicate Detection & Removal",
        "🏷️ Categorical Variable Encoding",
        "📈 Numerical Feature Scaling",
        "🆕 Feature Engineering & Creation",
        "✅ Data Validation with Great Expectations",
        "📦 Cleaned Dataset Versioning"
    ]
    
    selected_activities = st.multiselect(
        "Select Activities to Execute (Experiment 2):",
        activities,
        default=activities[:4]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Execute Selected Activities", key="exp2_execute"):
            for activity in selected_activities:
                with st.expander(f"✅ {activity}", expanded=True):
                    if activity == "📊 Comprehensive Data Profiling":
                        exp2.comprehensive_profiling()
                    elif activity == "🔍 Data Quality Assessment":
                        exp2.quality_assessment()
                    elif activity == "🧹 Missing Value Analysis & Treatment":
                        exp2.handle_missing_values()
                    elif activity == "🔗 Duplicate Detection & Removal":
                        exp2.handle_duplicates()
                    elif activity == "🏷️ Categorical Variable Encoding":
                        exp2.encode_categorical()
                    elif activity == "📈 Numerical Feature Scaling":
                        exp2.scale_numerical()
                    elif activity == "🆕 Feature Engineering & Creation":
                        exp2.engineer_features()
                    elif activity == "✅ Data Validation with Great Expectations":
                        exp2.validate_data()
                    elif activity == "📦 Cleaned Dataset Versioning":
                        exp2.version_cleaned_data()
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🔧 Data Quality Metrics</h4>
            <p><strong>Activities</strong>: 9 total</p>
            <p><strong>Selected</strong>: {}</p>
            <p><strong>Tools</strong>: Pandas, PyJanitor, Great Expectations</p>
        </div>
        """.format(len(selected_activities)), unsafe_allow_html=True)

def render_experiment_3():
    """Render Experiment 3: EDA & Statistical Analysis"""
    st.markdown("""
    <div class="experiment-card">
        <h2>📊 Experiment 3: EDA & Statistical Analysis</h2>
        <p><strong>Objective</strong>: Explore data patterns, customer behavior, and statistical relationships.</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp3 = Experiment3()
    exp3.render_eda_dashboard()

def render_experiment_4():
    """Render Experiment 4: ML Modeling & Experiment Tracking"""
    st.markdown("""
    <div class="experiment-card">
        <h2>🤖 Experiment 4: ML Modeling & Experiment Tracking</h2>
        <p><strong>Objective</strong>: Build predictive models with proper experiment tracking and performance monitoring.</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp4 = Experiment4()
    exp4.render_modeling_dashboard()

def render_experiment_5():
    """Render Experiment 5: Explainable AI & Fairness Evaluation"""
    st.markdown("""
    <div class="experiment-card">
        <h2>🔬 Experiment 5: Explainable AI & Fairness Evaluation</h2>
        <p><strong>Objective</strong>: Ensure model interpretability and fairness through comprehensive evaluation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp5 = Experiment5()
    exp5.render_explainable_ai_dashboard()

def render_experiment_6():
    """Render Experiment 6: Containerization & API Deployment"""
    st.markdown("""
    <div class="experiment-card">
        <h2>🐳 Experiment 6: Containerization & API Deployment</h2>
        <p><strong>Objective</strong>: Package models in containers and deploy as scalable APIs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp6 = Experiment6()
    exp6.render_containerization_dashboard()

def render_experiment_7():
    """Render Experiment 7: CI/CD Pipeline"""
    st.markdown("""
    <div class="experiment-card">
        <h2>🔄 Experiment 7: CI/CD Pipeline with Open Source Tools</h2>
        <p><strong>Objective</strong>: Implement automated testing, validation, and deployment pipeline.</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp7 = Experiment7()
    exp7.render_cicd_dashboard()

def render_statistical_analysis():
    """Render Statistical Analysis Dashboard"""
    # Load datasets first
    datasets = {}
    data_path = r"C:\Users\Omkar\SEM_7\DS_Project\datasets\DS-2-8-25"
    
    try:
        # Product catalog
        product_file = os.path.join(data_path, "comprehensive_clothing_dataset_expanded_20250802_210102.csv")
        if os.path.exists(product_file):
            datasets['products'] = pd.read_csv(product_file)
        
        # Reviews dataset
        reviews_file = os.path.join(data_path, "zudio_reviews_dataset.csv")
        if os.path.exists(reviews_file):
            datasets['reviews'] = pd.read_csv(reviews_file)
        
        # Sales dataset
        sales_file = os.path.join(data_path, "Zudio_sales_data.csv")
        if os.path.exists(sales_file):
            datasets['sales'] = pd.read_csv(sales_file)
        
        if datasets:
            stats_analyzer = StatisticalAnalysis(datasets)
            stats_analyzer.render_statistical_dashboard()
        else:
            st.error("❌ No datasets found for statistical analysis")
            
    except Exception as e:
        st.error(f"❌ Error loading datasets: {str(e)}")

def render_experiment_8():
    """Render Experiment 8: Dashboard & Portfolio"""
    st.markdown("""
    <div class="experiment-card">
        <h2>📈 Experiment 8: Advanced Analytics & Business Intelligence</h2>
        <p><strong>Objective</strong>: Create comprehensive business analytics and intelligence dashboard.</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp8 = Experiment8()
    exp8.render_advanced_analytics_dashboard()

if __name__ == "__main__":
    main()
