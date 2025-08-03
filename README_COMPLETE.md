# 🛍️ Fashion E-commerce Analytics Platform

A comprehensive Data Science project featuring 8 experiments with GUI-based activities for fashion e-commerce analytics, covering the complete ML lifecycle from data collection to deployment.

## 📋 Project Overview

**Domain**: Fashion E-commerce Analytics  
**Objective**: Build a complete data science platform with predictive modeling, explainable AI, and business intelligence capabilities.

### 🎯 Business Problem
Optimize customer experience and business operations for a fashion e-commerce platform through data-driven insights and machine learning solutions.

### 📊 Datasets
- **Product Catalog**: 1,242 fashion items with detailed specifications
- **Customer Reviews**: 23,578 authentic customer reviews and ratings  
- **Sales Data**: 7,901 transaction records with purchase patterns

## 🏗️ Project Structure

```
DS_Project/
├── datasets/                          # Data storage and DVC tracked files
│   ├── DS-2-8-25/                    # Main dataset folder
│   ├── datasets.dvc                   # DVC data tracking
│   └── DS-20250802T175924Z-1-001.zip  # Compressed datasets
├── src/                               # Source code
│   ├── experiments/                   # 8 comprehensive experiments
│   │   ├── experiment_1.py           # Case Study Framing & Data Prep
│   │   ├── experiment_2.py           # Data Profiling & Cleaning
│   │   ├── experiment_3.py           # EDA & Statistical Analysis
│   │   ├── experiment_4.py           # ML Modeling & Performance
│   │   ├── experiment_5.py           # Explainable AI & Interpretability
│   │   ├── experiment_6.py           # Containerization & DevOps
│   │   ├── experiment_7.py           # CI/CD Pipeline & Automation
│   │   ├── experiment_8.py           # Advanced Analytics & BI
│   │   └── statistical_analysis.py   # Mathematical Properties
│   └── gui/                          # Streamlit web interface
│       └── main_dashboard.py         # Main GUI dashboard
├── dvc_env/                          # Python virtual environment
├── requirements.txt                  # Project dependencies
├── client_secrets.json              # Google Drive API credentials
├── dvc_project_secrets.json         # DVC configuration
└── README.md                        # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ 
- Git
- Google Account (for DVC remote storage)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd DS_Project
```

2. **Create virtual environment**
```bash
python -m venv dvc_env
# Windows
dvc_env\Scripts\activate
# Linux/Mac
source dvc_env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup DVC and pull data**
```bash
dvc init
dvc remote add -d gdrive gdrive://your-google-drive-folder-id
dvc pull
```

5. **Launch the GUI dashboard**
```bash
streamlit run src/gui/main_dashboard.py
```

## 🧪 Experiments Overview

### 🎯 Experiment 1: Case Study Framing & Dataset Preparation
- **Objective**: Define problem statement and acquire data
- **Activities**: 9 comprehensive activities including problem definition, literature review, data acquisition, and DVC setup
- **Tools**: DVC, Git, Python, Google Drive API
- **Outputs**: Problem statement, data acquisition pipeline, versioned datasets

### 🔍 Experiment 2: Data Profiling, Cleaning & Feature Engineering  
- **Objective**: Profile, clean, and engineer features from raw data
- **Activities**: 9 activities covering data profiling, quality assessment, missing value treatment, and feature engineering
- **Tools**: Pandas, PyJanitor, Great Expectations, Pandas Profiling
- **Outputs**: Cleaned datasets, feature engineering pipeline, data quality reports

### 📊 Experiment 3: EDA & Statistical Analysis
- **Objective**: Explore data patterns and statistical relationships
- **Activities**: 10 activities including univariate/bivariate analysis, customer behavior, sales patterns, sentiment analysis
- **Tools**: Matplotlib, Seaborn, Plotly, SciPy, Statistical Analysis module
- **Outputs**: Interactive visualizations, statistical insights, correlation analysis

### 🤖 Experiment 4: ML Modeling & Performance Evaluation
- **Objective**: Build and evaluate predictive models
- **Activities**: 9 activities covering problem definition, preprocessing, model training, evaluation, hyperparameter tuning
- **Tools**: Scikit-learn, XGBoost, LightGBM, Optuna, MLflow
- **Outputs**: Trained models, performance metrics, hyperparameter optimization results

### 🔬 Experiment 5: Explainable AI & Model Interpretability
- **Objective**: Ensure model interpretability and fairness
- **Activities**: 9 activities including SHAP analysis, LIME explanations, feature importance, fairness assessment
- **Tools**: SHAP, LIME, Fairlearn, Custom interpretation methods
- **Outputs**: Model explanations, interpretability reports, fairness assessments

### 🐳 Experiment 6: Containerization & DevOps Integration
- **Objective**: Package models and implement DevOps practices
- **Activities**: 9 activities covering Docker containerization, Kubernetes deployment, CI/CD pipelines
- **Tools**: Docker, Kubernetes, GitHub Actions, FastAPI, Cloud platforms
- **Outputs**: Containerized applications, deployment manifests, DevOps workflows

### 🔄 Experiment 7: CI/CD Pipeline & Automation
- **Objective**: Implement comprehensive CI/CD automation
- **Activities**: 9 activities covering CI/CD strategy, testing frameworks, pipeline implementation, monitoring
- **Tools**: GitHub Actions, GitLab CI, Azure DevOps, pytest, SonarQube
- **Outputs**: Automated pipelines, testing frameworks, deployment automation

### 📈 Experiment 8: Advanced Analytics & Business Intelligence
- **Objective**: Create business intelligence dashboards and advanced analytics
- **Activities**: 9 activities covering executive dashboards, forecasting, customer segmentation, revenue analytics
- **Tools**: Plotly, Business intelligence tools, Forecasting algorithms
- **Outputs**: Executive dashboards, business insights, predictive analytics

## 🎨 GUI Features

### Interactive Dashboard
- **Multi-experiment Navigation**: Seamless switching between all 8 experiments
- **Activity Selection**: Choose specific activities within each experiment
- **Real-time Visualization**: Interactive charts and graphs with Plotly
- **Progress Tracking**: Monitor completion status across experiments
- **Data Integration**: Unified access to all datasets and models

### Key Features
- 📊 **Dynamic Visualizations**: Interactive charts with zoom, filter, and drill-down capabilities
- 🎯 **Model Comparison**: Side-by-side performance analysis
- 🔍 **Exploratory Interface**: User-friendly data exploration tools
- 📈 **Business Metrics**: KPI dashboards and executive summaries
- 🎨 **Responsive Design**: Mobile-friendly interface with custom CSS styling

## 🛠️ Technical Stack

### Core Technologies
- **Language**: Python 3.9+
- **Framework**: Streamlit for web interface
- **Data Processing**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Model Interpretability**: SHAP, LIME

### MLOps & DevOps
- **Version Control**: Git, DVC for data versioning
- **Containerization**: Docker, Kubernetes
- **CI/CD**: GitHub Actions, GitLab CI, Azure DevOps
- **API Development**: FastAPI, Uvicorn
- **Monitoring**: Prometheus, Grafana
- **Cloud Platforms**: AWS, GCP, Azure support

### Data & Analytics
- **Data Storage**: Google Drive (DVC remote)
- **Data Quality**: Great Expectations, PyJanitor
- **Statistical Analysis**: SciPy, Statsmodels
- **Business Intelligence**: Custom dashboards, KPI tracking

## 📋 Usage Guide

### Running Individual Experiments
Each experiment can be run independently through the GUI:

1. Start the dashboard: `streamlit run src/gui/main_dashboard.py`
2. Select desired experiment from sidebar
3. Choose specific activities to execute
4. View results in interactive interface

### Command Line Usage
```bash
# Run specific experiment modules
python src/experiments/experiment_1.py
python src/experiments/experiment_2.py
# ... etc

# Statistical analysis
python src/experiments/statistical_analysis.py
```

### API Deployment (Experiment 6)
```bash
# Build Docker container
docker build -t ml-fashion-api .

# Run container
docker run -p 8000:8000 ml-fashion-api

# Access API documentation
curl http://localhost:8000/docs
```

## 📊 Key Outputs & Deliverables

### Business Intelligence
- **Executive Dashboard**: KPI tracking and business metrics
- **Customer Segmentation**: RFM analysis and behavioral segmentation  
- **Sales Forecasting**: Predictive analytics for revenue planning
- **Market Basket Analysis**: Product recommendation insights

### Technical Deliverables
- **Trained Models**: Multiple ML models with performance benchmarks
- **Model Explanations**: SHAP and LIME interpretability reports
- **API Endpoints**: RESTful APIs for model serving
- **Container Images**: Production-ready Docker containers
- **CI/CD Pipelines**: Automated testing and deployment workflows

### Documentation
- **Technical Reports**: Comprehensive analysis for each experiment
- **Model Cards**: Standardized model documentation
- **API Documentation**: OpenAPI/Swagger specifications
- **Deployment Guides**: Step-by-step deployment instructions

## 🔒 Security & Compliance

### Data Privacy
- Anonymized customer data handling
- GDPR compliance considerations
- Secure credential management

### Model Governance
- Model fairness assessments
- Bias detection and mitigation
- Explainability requirements
- Performance monitoring

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-experiment`)
3. Commit changes (`git commit -am 'Add new experiment'`)
4. Push to branch (`git push origin feature/new-experiment`)
5. Create Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Fashion e-commerce datasets from various open sources
- Streamlit for the excellent web framework
- Scikit-learn and ML community for robust tools
- DVC team for data versioning capabilities

## 📞 Support

For questions, issues, or contributions:
- Create GitHub Issues for bug reports
- Submit Pull Requests for enhancements
- Contact: [Your Contact Information]

---

**Last Updated**: August 3, 2025  
**Version**: 2.0.0  
**Status**: Production Ready ✅
