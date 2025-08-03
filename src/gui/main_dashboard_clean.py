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
    page_title="Fashion E-commerce Analytics Platform",
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
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .experiment-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .activity-card {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196f3;
        transition: transform 0.2s;
    }
    .activity-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛍️ Fashion E-commerce Analytics Platform</h1>
        <p>Comprehensive Data Science Project with 8 Experiments & GUI-based Activities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for experiment selection
    st.sidebar.title("🧪 Select Experiment")
    
    experiments = {
        "exp1": "🎯 Experiment 1: Case Study Framing & Dataset Preparation",
        "exp2": "🔍 Experiment 2: Data Profiling, Cleaning & Feature Engineering", 
        "exp3": "📊 Experiment 3: EDA & Statistical Analysis",
        "exp4": "🤖 Experiment 4: ML Modeling & Experiment Tracking",
        "exp5": "🔬 Experiment 5: Explainable AI & Fairness Evaluation",
        "exp6": "🐳 Experiment 6: Containerization & API Deployment",
        "exp7": "🔄 Experiment 7: CI/CD Pipeline with Open Source Tools",
        "exp8": "📈 Experiment 8: Dashboard, Responsible AI Reporting & Portfolio"
    }
    
    selected_exp = st.sidebar.selectbox(
        "Choose an experiment:",
        list(experiments.keys()),
        format_func=lambda x: experiments[x]
    )
    
    # Display project metrics in sidebar
    st.sidebar.markdown("### 📊 Project Overview")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Experiments", "8")
        st.metric("Activities", "73+")
    with col2:
        st.metric("Datasets", "3")
        st.metric("Models", "15+")
    
    # Route to appropriate experiment
    if selected_exp == "exp1":
        render_experiment_1()
    elif selected_exp == "exp2":
        render_experiment_2()
    elif selected_exp == "exp3":
        render_experiment_3()
    elif selected_exp == "exp4":
        render_experiment_4()
    elif selected_exp == "exp5":
        render_experiment_5()
    elif selected_exp == "exp6":
        render_experiment_6()
    elif selected_exp == "exp7":
        render_experiment_7()
    elif selected_exp == "exp8":
        render_experiment_8()

def render_experiment_1():
    """Render Experiment 1: Case Study Framing & Dataset Preparation"""
    st.markdown("""
    <div class="experiment-card">
        <h2>🎯 Experiment 1: Case Study Framing & Dataset Preparation</h2>
        <p>Define problem statement, benchmark solutions, and establish data foundation</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp1 = Experiment1()
    exp1.render_case_study_dashboard()

def render_experiment_2():
    """Render Experiment 2: Data Profiling, Cleaning & Feature Engineering"""
    st.markdown("""
    <div class="experiment-card">
        <h2>🔍 Experiment 2: Data Profiling, Cleaning & Feature Engineering</h2>
        <p>Comprehensive data analysis, quality assessment, and feature creation</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp2 = Experiment2()
    exp2.render_data_profiling_dashboard()

def render_experiment_3():
    """Render Experiment 3: EDA & Statistical Analysis"""
    st.markdown("""
    <div class="experiment-card">
        <h2>📊 Experiment 3: EDA & Statistical Analysis</h2>
        <p>Deep exploratory analysis with advanced visualizations and statistical insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp3 = Experiment3()
    exp3.render_eda_dashboard()

def render_experiment_4():
    """Render Experiment 4: ML Modeling & Experiment Tracking"""
    st.markdown("""
    <div class="experiment-card">
        <h2>🤖 Experiment 4: ML Modeling & Experiment Tracking</h2>
        <p>Build predictive models with comprehensive tracking and evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp4 = Experiment4()
    exp4.render_ml_modeling_dashboard()

def render_experiment_5():
    """Render Experiment 5: Explainable AI & Fairness Evaluation"""
    st.markdown("""
    <div class="experiment-card">
        <h2>🔬 Experiment 5: Explainable AI & Fairness Evaluation</h2>
        <p>Model interpretability, bias detection, and responsible AI practices</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp5 = Experiment5()
    exp5.render_explainable_ai_dashboard()

def render_experiment_6():
    """Render Experiment 6: Containerization & API Deployment"""
    st.markdown("""
    <div class="experiment-card">
        <h2>🐳 Experiment 6: Containerization & API Deployment</h2>
        <p>Package models for production with Docker and API endpoints</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp6 = Experiment6()
    exp6.render_containerization_dashboard()

def render_experiment_7():
    """Render Experiment 7: CI/CD Pipeline & Automation"""
    st.markdown("""
    <div class="experiment-card">
        <h2>🔄 Experiment 7: CI/CD Pipeline & Automation</h2>
        <p>Automated testing, validation, and deployment pipelines</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp7 = Experiment7()
    exp7.render_cicd_dashboard()

def render_experiment_8():
    """Render Experiment 8: Advanced Analytics & Business Intelligence"""
    st.markdown("""
    <div class="experiment-card">
        <h2>📈 Experiment 8: Advanced Analytics & Business Intelligence</h2>
        <p>Interactive dashboards, reporting, and business insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    exp8 = Experiment8()
    exp8.render_advanced_analytics_dashboard()

if __name__ == "__main__":
    main()
