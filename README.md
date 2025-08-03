# 🛍️ Fashion E-commerce Analytics Platform

A comprehensive Data Science project implementing 8 experiments with GUI-based activities using Streamlit. This project demonstrates a complete Machine Learning lifecycle for Fashion E-commerce analytics.

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Setup Guide](#step-by-step-setup-guide)
4. [DVC Setup & Configuration](#dvc-setup--configuration)
5. [Running the Project](#running-the-project)
6. [Project Structure](#project-structure)
7. [Troubleshooting](#troubleshooting)
8. [Features](#features)

## 🎯 Project Overview

This project covers the complete ML lifecycle with 8 comprehensive experiments:

| **Experiment** | **Focus Area** | **Activities** |
|----------------|----------------|----------------|
| **Experiment 1** | Case Study Framing & Dataset Preparation | 9 activities |
| **Experiment 2** | Data Profiling, Cleaning & Feature Engineering | 9 activities |
| **Experiment 3** | Exploratory Data Analysis & Statistical Analysis | 10 activities |
| **Experiment 4** | ML Modeling & Experiment Tracking | 9 activities |
| **Experiment 5** | Explainable AI & Fairness Evaluation | 9 activities |
| **Experiment 6** | Containerization & API Deployment | 9 activities |
| **Experiment 7** | CI/CD Pipeline Implementation | 9 activities |
| **Experiment 8** | Advanced Analytics & Business Intelligence | 9 activities |

**Total: 73+ Interactive Activities with Full GUI Interface**

## ⚡ Prerequisites

Before starting, ensure you have:

### Required Software:
- **Python 3.9 or higher** ([Download Python](https://www.python.org/downloads/))
- **Git** ([Download Git](https://git-scm.com/downloads))
- **Google Account** (for Google Drive integration with DVC)

### System Requirements:
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 5GB free space
- **Internet**: Stable connection for downloading dependencies

## 🚀 Step-by-Step Setup Guide

### Step 1: Clone the Repository

```bash
# Open Command Prompt or PowerShell as Administrator
# Navigate to your desired directory (e.g., Desktop, Documents)
cd C:\Users\YourUsername\Desktop

# Clone the repository
git clone https://github.com/HiOmkarrr/DS_SEM_7.git

# Navigate to project directory
cd DS_SEM_7
```

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python -m venv dvc_env

# Activate virtual environment
# On Windows:
dvc_env\Scripts\activate

# On macOS/Linux:
source dvc_env/bin/activate

# Verify activation (you should see (dvc_env) in your prompt)
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list | findstr streamlit
pip list | findstr dvc
```

**If you encounter any installation errors, try:**
```bash
# For Windows users with SSL issues:
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# For dependency conflicts:
pip install --force-reinstall -r requirements.txt
```

## 🗄️ DVC Setup & Configuration

### Step 4: Initialize DVC

```bash
# Initialize DVC in your project
dvc init

# Add Google Drive as remote storage
dvc remote add -d myremote gdrive://1BxT4_aIuqEJE-7d-Ty3_FrOyqlNvyM8V

# Configure DVC settings
dvc config core.autostage true
dvc config core.analytics false
```

### Step 5: Google Drive Authentication

**IMPORTANT: Follow these steps carefully**

1. **Run DVC Pull Command:**
   ```bash
   dvc pull
   ```

2. **Browser Authentication:**
   - A browser window will open automatically
   - Log in with your Google account
   - Click "Allow" to grant DVC access to Google Drive
   - Copy the authorization code shown

3. **Paste Authorization Code:**
   - Return to your terminal
   - Paste the authorization code when prompted
   - Press Enter

4. **Verify Data Download:**
   ```bash
   # Check if datasets folder exists and contains data
   dir datasets
   # You should see DS-20250802T175924Z-1-001.zip and DS-2-8-25 folder
   ```

### Step 6: Extract Dataset (if needed)

```bash
# If datasets are in ZIP format, extract them
cd datasets
# Extract using Windows built-in or 7-zip
# The extracted folder should be named "DS-2-8-25"
```

## 🎮 Running the Project

### Option 1: Using Batch Scripts (Recommended)

```bash
# Method 1: Automatic setup and run
setup.bat

# Then start the dashboard
start_dashboard.bat
```

### Option 2: Manual Commands

```bash
# Activate environment
dvc_env\Scripts\activate

# Start Streamlit application
streamlit run src/gui/main_dashboard.py
```

### Step 7: Access the Dashboard

1. **Open your web browser**
2. **Navigate to:** `http://localhost:8501`
3. **You should see:** Fashion E-commerce Analytics Platform dashboard

**🎉 Congratulations! Your setup is complete!**

## 📁 Project Structure

```
DS_Project/
├── 📊 datasets/                          # Data files managed by DVC
│   ├── DS-20250802T175924Z-1-001.zip    # Raw dataset archive
│   └── DS-2-8-25/                       # Extracted datasets
│       ├── comprehensive_clothing_dataset_expanded_20250802_210102.csv (1,242 products)
│       ├── zudio_reviews_dataset.csv    # Customer reviews (23,578 reviews)
│       └── Zudio_sales_data.csv         # Sales transactions (7,901 records)
├── 🐍 src/                              # Source code
│   ├── experiments/                     # 8 experiment modules
│   │   ├── experiment_1.py             # Case Study Framing (840+ lines)
│   │   ├── experiment_2.py             # Data Processing (1,200+ lines)
│   │   ├── experiment_3.py             # EDA & Statistics (801 lines)
│   │   ├── experiment_4.py             # ML Modeling (1,400+ lines)
│   │   ├── experiment_5.py             # Explainable AI (1,300+ lines)
│   │   ├── experiment_6.py             # Containerization (1,500+ lines)
│   │   ├── experiment_7.py             # CI/CD Pipeline (1,600+ lines)
│   │   ├── experiment_8.py             # Advanced Analytics (1,500+ lines)
│   │   └── statistical_analysis.py     # Mathematical Analysis (1,014 lines)
│   └── gui/                            # Streamlit interface
│       └── main_dashboard.py           # Main GUI (396 lines)
├── 🔧 dvc_env/                         # Python virtual environment
├── ⚙️ requirements.txt                 # Python dependencies (81 packages)
├── 🚀 setup.bat                        # Automated setup script
├── ▶️ start_dashboard.bat              # Dashboard launcher
├── 📖 README.md                        # This documentation
└── 🗂️ DVC configuration files         # .dvc, .dvcignore, etc.
```

## 🛠️ Troubleshooting

### Common Issues and Solutions:

#### 1. **"Python not found" Error**
```bash
# Solution: Add Python to PATH or use full path
C:\Python39\python.exe -m venv dvc_env
```

#### 2. **DVC Authentication Failed**
```bash
# Solution: Clear DVC cache and retry
dvc cache dir
rmdir /s .dvc\cache
dvc pull
```

#### 3. **Streamlit Port Already in Use**
```bash
# Solution: Use different port
streamlit run src/gui/main_dashboard.py --server.port 8502
```

#### 4. **Module Import Errors**
```bash
# Solution: Verify virtual environment is activated
# You should see (dvc_env) in your prompt
dvc_env\Scripts\activate
```

#### 5. **Dataset Files Missing**
```bash
# Solution: Re-run DVC pull
dvc pull --force
```

#### 6. **Permission Errors on Windows**
```bash
# Solution: Run Command Prompt as Administrator
# Right-click Command Prompt → "Run as administrator"
```

### Performance Tips:

1. **Close unnecessary applications** to free up RAM
2. **Use SSD storage** for better performance
3. **Ensure stable internet** for DVC operations
4. **Keep antivirus exclusions** for project folder

## ✨ Features

### 🎮 Interactive GUI Dashboard
- **Streamlit-based interface** with responsive design
- **Activity selection** for each experiment
- **Real-time visualization** and results
- **Progress tracking** and metrics display

### 📊 Comprehensive Analytics (73+ Activities)
- **Data Profiling & Quality Assessment**
- **Statistical Analysis & Hypothesis Testing**
- **Machine Learning Model Development**
- **Explainable AI & Fairness Evaluation**
- **Advanced Business Intelligence**

### 🔄 Data Version Control
- **DVC integration** for dataset management
- **Google Drive remote storage**
- **Automatic data synchronization**
- **Version tracking and rollback**

### 🚀 Production-Ready Features
- **Containerization with Docker**
- **API deployment capabilities**
- **CI/CD pipeline implementation**
- **Scalable architecture design**

## 🛡️ Technologies Used

| **Category** | **Technologies** |
|--------------|------------------|
| **Frontend** | Streamlit, HTML/CSS, JavaScript |
| **Data Science** | Pandas, NumPy, Scikit-learn, XGBoost, LightGBM |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **ML Operations** | DVC, MLflow, Docker, Kubernetes |
| **DevOps** | Git, GitHub Actions, CI/CD pipelines |
| **Cloud** | Google Drive, AWS, Azure integration |

## 📞 Need Help?

If you encounter any issues:

1. **Check this README** for troubleshooting steps
2. **Verify all prerequisites** are installed correctly
3. **Ensure virtual environment** is activated
4. **Check internet connection** for DVC operations
5. **Contact project maintainer** for additional support

## 🎓 Learning Outcomes

After completing this project, you will have hands-on experience with:

- ✅ **Complete ML Pipeline Development**
- ✅ **Data Version Control with DVC**
- ✅ **Interactive Dashboard Creation**
- ✅ **Statistical Analysis & Hypothesis Testing**
- ✅ **Model Deployment & Containerization**
- ✅ **CI/CD Pipeline Implementation**
- ✅ **Responsible AI & Fairness Evaluation**
- ✅ **Business Intelligence & Analytics**

---

**🎯 Ready to Start? Follow the setup guide above and dive into the world of Fashion E-commerce Analytics!**
