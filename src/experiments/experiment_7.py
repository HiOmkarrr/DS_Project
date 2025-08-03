"""
Experiment 7: CI/CD Pipeline & Automation
Complete CI/CD implementation with testing, deployment, and monitoring
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

class Experiment7:
    def __init__(self):
        self.project_path = r"C:\Users\Omkar\SEM_7\DS_Project"
        
    def render_cicd_dashboard(self):
        """Main CI/CD dashboard"""
        st.markdown("""
        <div class="experiment-card">
            <h2>🔄 CI/CD Pipeline & Automation</h2>
            <p>Complete CI/CD implementation with automated testing, deployment, and monitoring</p>
        </div>
        """, unsafe_allow_html=True)
        
        # CI/CD Options
        cicd_options = [
            "📋 CI/CD Strategy & Planning",
            "🧪 Automated Testing Framework",
            "🔄 GitHub Actions Workflow",
            "🐙 GitLab CI Implementation", 
            "🔵 Azure DevOps Pipeline",
            "📊 Quality Gates & Code Analysis",
            "🚀 Deployment Automation",
            "📈 Pipeline Monitoring & Metrics",
            "🔒 Security & Compliance Integration"
        ]
        
        selected_activity = st.selectbox(
            "Select CI/CD Activity:",
            cicd_options
        )
        
        # Route to appropriate activity
        if selected_activity == "📋 CI/CD Strategy & Planning":
            self.cicd_strategy()
        elif selected_activity == "🧪 Automated Testing Framework":
            self.testing_framework()
        elif selected_activity == "🔄 GitHub Actions Workflow":
            self.github_actions()
        elif selected_activity == "🐙 GitLab CI Implementation":
            self.gitlab_ci()
        elif selected_activity == "🔵 Azure DevOps Pipeline":
            self.azure_devops()
        elif selected_activity == "📊 Quality Gates & Code Analysis":
            self.quality_gates()
        elif selected_activity == "🚀 Deployment Automation":
            self.deployment_automation()
        elif selected_activity == "📈 Pipeline Monitoring & Metrics":
            self.pipeline_monitoring()
        elif selected_activity == "🔒 Security & Compliance Integration":
            self.security_compliance()

    def cicd_strategy(self):
        """CI/CD strategy and planning"""
        st.subheader("📋 CI/CD Strategy & Planning")
        
        # CI/CD Maturity Assessment
        st.subheader("📊 CI/CD Maturity Assessment")
        
        maturity_areas = {
            "Version Control": 90,
            "Automated Testing": 75,
            "Continuous Integration": 80,
            "Deployment Automation": 65,
            "Monitoring & Feedback": 70,
            "Infrastructure as Code": 60,
            "Security Integration": 55,
            "Documentation": 80
        }
        
        # Create radar chart
        categories = list(maturity_areas.keys())
        values = list(maturity_areas.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Maturity'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="CI/CD Maturity Assessment"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy roadmap
        st.subheader("🗺️ CI/CD Implementation Roadmap")
        
        roadmap_phases = {
            "Phase 1: Foundation (Weeks 1-2)": [
                "Set up version control with Git",
                "Implement basic unit testing",
                "Create simple build automation",
                "Establish code review process"
            ],
            "Phase 2: Integration (Weeks 3-4)": [
                "Implement continuous integration",
                "Add automated testing pipeline",
                "Set up artifact management",
                "Create staging environment"
            ],
            "Phase 3: Deployment (Weeks 5-6)": [
                "Automate deployment process",
                "Implement blue-green deployment",
                "Add rollback mechanisms",
                "Set up production monitoring"
            ],
            "Phase 4: Optimization (Weeks 7-8)": [
                "Add security scanning",
                "Implement advanced testing",
                "Optimize pipeline performance",
                "Add compliance checks"
            ]
        }
        
        for phase, tasks in roadmap_phases.items():
            with st.expander(f"📅 {phase}"):
                for task in tasks:
                    st.write(f"• {task}")
        
        # Success metrics
        st.subheader("📈 Success Metrics")
        
        metrics_data = {
            "Metric": [
                "Deployment Frequency",
                "Lead Time for Changes", 
                "Mean Time to Recovery",
                "Change Failure Rate",
                "Build Success Rate",
                "Test Coverage"
            ],
            "Current": [
                "Weekly",
                "3 days",
                "4 hours", 
                "15%",
                "85%",
                "70%"
            ],
            "Target": [
                "Daily",
                "< 1 day",
                "< 1 hour",
                "< 5%",
                "> 95%",
                "> 90%"
            ],
            "Priority": [
                "High",
                "High",
                "Medium",
                "High",
                "Medium",
                "High"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

    def testing_framework(self):
        """Automated testing framework"""
        st.subheader("🧪 Automated Testing Framework")
        
        # Testing pyramid
        st.subheader("🔺 Testing Pyramid")
        
        testing_levels = {
            "Unit Tests": {
                "coverage": 80,
                "count": 150,
                "description": "Test individual functions and methods"
            },
            "Integration Tests": {
                "coverage": 60,
                "count": 45,
                "description": "Test component interactions"
            },
            "End-to-End Tests": {
                "coverage": 40,
                "count": 15,
                "description": "Test complete user workflows"
            }
        }
        
        # Pyramid visualization
        fig = go.Figure()
        
        levels = list(testing_levels.keys())
        counts = [testing_levels[level]["count"] for level in levels]
        
        for i, (level, count) in enumerate(zip(levels, counts)):
            fig.add_trace(go.Bar(
                name=level,
                x=[level],
                y=[count],
                text=[f"{count} tests"],
                textposition='inside'
            ))
        
        fig.update_layout(
            title="Testing Pyramid - Test Distribution",
            xaxis_title="Test Level",
            yaxis_title="Number of Tests"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Test configuration files
        st.subheader("⚙️ Test Configuration")
        
        tab1, tab2, tab3, tab4 = st.tabs(["pytest.ini", "Unit Tests", "Integration", "E2E Tests"])
        
        with tab1:
            pytest_config = """[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
"""
            st.code(pytest_config, language='ini')
        
        with tab2:
            unit_test_example = """import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.experiments.experiment_4 import Experiment4

class TestExperiment4:
    def setup_method(self):
        \"\"\"Setup test fixtures\"\"\"
        self.experiment = Experiment4()
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_load_datasets(self):
        \"\"\"Test dataset loading functionality\"\"\"
        with patch('os.path.exists', return_value=True):
            with patch('pandas.read_csv', return_value=self.sample_data):
                self.experiment._load_datasets()
                assert len(self.experiment.datasets) > 0
    
    def test_data_preprocessing(self):
        \"\"\"Test data preprocessing\"\"\"
        # Mock session state
        with patch('streamlit.session_state') as mock_state:
            mock_state.ml_data = {
                'df': self.sample_data,
                'target': 'target'
            }
            
            # Test preprocessing logic
            assert self.sample_data.shape[0] == 5
            assert 'target' in self.sample_data.columns
    
    @pytest.mark.parametrize("model_type,expected", [
        ("classification", "Classification"),
        ("regression", "Regression"),
        ("clustering", "Clustering")
    ])
    def test_problem_types(self, model_type, expected):
        \"\"\"Test different problem types\"\"\"
        assert model_type in expected.lower()
"""
            st.code(unit_test_example, language='python')
        
        with tab3:
            integration_test = """import pytest
import requests
from fastapi.testclient import TestClient
from src.api.main import app

class TestAPIIntegration:
    def setup_method(self):
        \"\"\"Setup test client\"\"\"
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        \"\"\"Test API health check\"\"\"
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_prediction_endpoint(self):
        \"\"\"Test prediction endpoint\"\"\"
        test_data = {
            "feature1": 1.0,
            "feature2": 2.0,
            "feature3": 3.0
        }
        
        response = self.client.post("/predict", json=test_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "prediction" in result
        assert "confidence" in result
        assert isinstance(result["prediction"], (int, float))
    
    def test_batch_prediction(self):
        \"\"\"Test batch prediction\"\"\"
        test_batch = [
            {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0},
            {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0}
        ]
        
        response = self.client.post("/batch_predict", json=test_batch)
        assert response.status_code == 200
        
        result = response.json()
        assert "predictions" in result
        assert len(result["predictions"]) == 2
"""
            st.code(integration_test, language='python')
        
        with tab4:
            e2e_test = """import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class TestStreamlitE2E:
    def setup_method(self):
        \"\"\"Setup Selenium WebDriver\"\"\"
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=options)
        self.driver.get("http://localhost:8501")
        
        # Wait for Streamlit to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "stApp"))
        )
    
    def teardown_method(self):
        \"\"\"Cleanup WebDriver\"\"\"
        self.driver.quit()
    
    def test_dashboard_loads(self):
        \"\"\"Test that dashboard loads successfully\"\"\"
        title_element = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )
        assert "Data Science Project" in title_element.text
    
    def test_experiment_navigation(self):
        \"\"\"Test navigation between experiments\"\"\"
        # Find and click experiment selector
        selectbox = self.driver.find_element(By.CLASS_NAME, "stSelectbox")
        selectbox.click()
        
        # Select Experiment 1
        options = self.driver.find_elements(By.CLASS_NAME, "option")
        for option in options:
            if "Experiment 1" in option.text:
                option.click()
                break
        
        # Wait for content to load
        time.sleep(2)
        
        # Check if experiment content is displayed
        content = self.driver.find_element(By.CLASS_NAME, "stApp")
        assert "Case Study Framing" in content.text
    
    def test_model_training_workflow(self):
        \"\"\"Test complete model training workflow\"\"\"
        # Navigate to Experiment 4
        # ... navigation code ...
        
        # Select dataset
        # ... interaction code ...
        
        # Train model
        # ... training code ...
        
        # Verify results
        # ... verification code ...
        pass
"""
            st.code(e2e_test, language='python')
        
        # Test execution commands
        st.subheader("🚀 Test Execution Commands")
        
        test_commands = """# Run all tests
pytest

# Run specific test types
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only  
pytest -m e2e                    # End-to-end tests only

# Run with coverage
pytest --cov=src --cov-report=html

# Run tests in parallel
pytest -n auto

# Run tests with specific output
pytest -v --tb=short

# Generate test report
pytest --html=report.html --self-contained-html
"""
        
        st.code(test_commands, language='bash')

    def github_actions(self):
        """GitHub Actions workflow implementation"""
        st.subheader("🔄 GitHub Actions Workflow")
        
        # Workflow overview
        st.subheader("📋 Workflow Overview")
        
        workflow_stages = [
            "Code Checkout",
            "Environment Setup", 
            "Dependency Installation",
            "Code Quality Check",
            "Unit Testing",
            "Integration Testing",
            "Security Scanning",
            "Build Artifacts",
            "Deploy to Staging",
            "E2E Testing",
            "Deploy to Production"
        ]
        
        # Create workflow diagram
        fig = go.Figure()
        
        for i, stage in enumerate(workflow_stages):
            fig.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                text=[stage],
                textposition='top center',
                marker=dict(
                    size=20,
                    color='lightblue' if i % 2 == 0 else 'lightgreen'
                ),
                showlegend=False
            ))
            
            if i < len(workflow_stages) - 1:
                fig.add_annotation(
                    x=i + 0.5, y=0,
                    ax=i, ay=0,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2
                )
        
        fig.update_layout(
            title="GitHub Actions Workflow Stages",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Complete workflow file
        st.subheader("📄 Complete Workflow File")
        
        github_workflow = """name: ML Model CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
    paths-ignore:
      - 'README.md'
      - 'docs/**'
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM

env:
  PYTHON_VERSION: 3.11
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Job 1: Code Quality and Testing
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Fetch full history for SonarQube
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov flake8 black isort bandit safety
    
    - name: Code formatting check
      run: |
        black --check src/ tests/
        isort --check-only src/ tests/
    
    - name: Lint with flake8
      run: |
        flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Security check with bandit
      run: |
        bandit -r src/ -f json -o bandit-report.json
    
    - name: Dependency vulnerability check
      run: |
        safety check --json --output safety-report.json
      continue-on-error: true
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=html
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
    
    - name: Upload test artifacts
      uses: actions/upload-artifact@v3
      if: failure()
      with:
        name: test-reports-${{ matrix.python-version }}
        path: |
          htmlcov/
          bandit-report.json
          safety-report.json

  # Job 2: Security Scanning
  security:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

  # Job 3: Build and Push Docker Image
  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Image vulnerability scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        format: 'table'

  # Job 4: Deploy to Staging
  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add deployment commands here
        
    - name: Run smoke tests
      run: |
        echo "Running smoke tests"
        # Add smoke test commands
    
    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        text: 'Staging deployment completed'
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}

  # Job 5: Deploy to Production
  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: 
      name: production
      url: https://your-production-url.com
    
    steps:
    - name: Deploy to production
      run: |
        echo "Deploying to production environment"
        # Add production deployment commands
    
    - name: Run E2E tests
      run: |
        echo "Running end-to-end tests"
        # Add E2E test commands
    
    - name: Create release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: v${{ github.run_number }}
        release_name: Release v${{ github.run_number }}
        draft: false
        prerelease: false
"""
        
        st.code(github_workflow, language='yaml')
        
        # Workflow status dashboard
        st.subheader("📊 Workflow Status Dashboard")
        
        workflow_metrics = {
            "Build Success Rate": 95,
            "Average Build Time": "8 min",
            "Test Coverage": "87%",
            "Security Issues": 0,
            "Deployment Frequency": "5/day"
        }
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics_list = list(workflow_metrics.items())
        for i, (metric, value) in enumerate(metrics_list):
            with [col1, col2, col3, col4, col5][i]:
                st.metric(metric, value)

    def gitlab_ci(self):
        """GitLab CI implementation"""
        st.subheader("🐙 GitLab CI Implementation")
        
        gitlab_yaml = """.gitlab-ci.yml

# Global variables
variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  PYTHON_VERSION: "3.11"

# Cache configuration
cache:
  paths:
    - .cache/pip
    - venv/

# Stages definition
stages:
  - validate
  - test
  - security
  - build
  - deploy-staging
  - deploy-production

# Before script - runs before each job
before_script:
  - python --version
  - pip install virtualenv
  - virtualenv venv
  - source venv/bin/activate
  - pip install --upgrade pip

# Validate stage
validate:
  stage: validate
  image: python:$PYTHON_VERSION
  script:
    - pip install black flake8 isort
    - black --check src/ tests/
    - flake8 src/ tests/
    - isort --check-only src/ tests/
  only:
    - branches

# Unit tests
unit-tests:
  stage: test
  image: python:$PYTHON_VERSION
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - pytest tests/unit/ --cov=src --cov-report=xml --cov-report=html
  coverage: '/TOTAL.+ ([0-9]{1,3}%)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - htmlcov/
    expire_in: 1 week
  only:
    - branches

# Integration tests
integration-tests:
  stage: test
  image: python:$PYTHON_VERSION
  services:
    - postgres:13
    - redis:6
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: test_user
    POSTGRES_PASSWORD: test_pass
  script:
    - pip install -r requirements.txt
    - pip install pytest
    - pytest tests/integration/ -v
  only:
    - branches

# Security scanning
security-scan:
  stage: security
  image: python:$PYTHON_VERSION
  script:
    - pip install bandit safety
    - bandit -r src/ -f json -o bandit-report.json
    - safety check --json --output safety-report.json
  artifacts:
    reports:
      sast: bandit-report.json
    paths:
      - bandit-report.json
      - safety-report.json
    expire_in: 1 week
  allow_failure: true
  only:
    - branches

# Docker build
build-image:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - |
      if [ "$CI_COMMIT_BRANCH" == "main" ]; then
        docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest
        docker push $CI_REGISTRY_IMAGE:latest
      fi
  only:
    - main
    - develop

# Deploy to staging
deploy-staging:
  stage: deploy-staging
  image: alpine:latest
  before_script:
    - apk add --update curl
  script:
    - echo "Deploying to staging environment"
    - curl -X POST "$STAGING_WEBHOOK_URL" -H "Authorization: Bearer $STAGING_TOKEN"
  environment:
    name: staging
    url: https://staging.your-app.com
  only:
    - develop
  when: manual

# Deploy to production
deploy-production:
  stage: deploy-production
  image: alpine:latest
  before_script:
    - apk add --update curl
  script:
    - echo "Deploying to production environment"
    - curl -X POST "$PRODUCTION_WEBHOOK_URL" -H "Authorization: Bearer $PRODUCTION_TOKEN"
  environment:
    name: production
    url: https://your-app.com
  only:
    - main
  when: manual
"""
        
        st.code(gitlab_yaml, language='yaml')

    def azure_devops(self):
        """Azure DevOps pipeline"""
        st.subheader("🔵 Azure DevOps Pipeline")
        
        azure_yaml = """# Azure DevOps Pipeline
trigger:
  branches:
    include:
    - main
    - develop
  paths:
    exclude:
    - README.md
    - docs/*

pr:
  branches:
    include:
    - main
  paths:
    exclude:
    - README.md
    - docs/*

variables:
  pythonVersion: '3.11'
  vmImageName: 'ubuntu-latest'
  dockerRegistryServiceConnection: 'myDockerRegistry'
  imageRepository: 'ml-app'
  containerRegistry: 'myregistry.azurecr.io'
  dockerfilePath: '**/Dockerfile'
  tag: '$(Build.BuildId)'

stages:
- stage: Validate
  displayName: Code validation
  jobs:
  - job: Validate
    displayName: Validate code quality
    pool:
      vmImage: $(vmImageName)
    
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(pythonVersion)'
      displayName: 'Use Python $(pythonVersion)'
    
    - script: |
        python -m pip install --upgrade pip
        pip install black flake8 isort bandit safety
      displayName: 'Install linting tools'
    
    - script: |
        black --check src/ tests/
        flake8 src/ tests/
        isort --check-only src/ tests/
      displayName: 'Code quality checks'
    
    - script: |
        bandit -r src/ -f json -o $(Agent.TempDirectory)/bandit-report.json
        safety check --json --output $(Agent.TempDirectory)/safety-report.json
      displayName: 'Security scanning'
      continueOnError: true
    
    - task: PublishTestResults@2
      condition: succeededOrFailed()
      inputs:
        testResultsFiles: '$(Agent.TempDirectory)/bandit-report.json'
        testRunTitle: 'Security Scan Results'

- stage: Test
  displayName: Testing
  dependsOn: Validate
  jobs:
  - job: UnitTests
    displayName: Unit tests
    pool:
      vmImage: $(vmImageName)
    
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(pythonVersion)'
    
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-html
      displayName: 'Install dependencies'
    
    - script: |
        pytest tests/unit/ --cov=src --cov-report=xml --cov-report=html --junitxml=junit/test-results.xml
      displayName: 'Run unit tests'
    
    - task: PublishTestResults@2
      condition: succeededOrFailed()
      inputs:
        testResultsFiles: 'junit/test-*.xml'
        testRunTitle: 'Unit Tests'
    
    - task: PublishCodeCoverageResults@1
      inputs:
        codeCoverageTool: Cobertura
        summaryFileLocation: 'coverage.xml'
        reportDirectory: 'htmlcov'

  - job: IntegrationTests
    displayName: Integration tests
    pool:
      vmImage: $(vmImageName)
    services:
      postgres: postgres:13
    
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '$(pythonVersion)'
    
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest
      displayName: 'Install dependencies'
    
    - script: |
        pytest tests/integration/ -v --junitxml=junit/integration-results.xml
      displayName: 'Run integration tests'
    
    - task: PublishTestResults@2
      condition: succeededOrFailed()
      inputs:
        testResultsFiles: 'junit/integration-*.xml'
        testRunTitle: 'Integration Tests'

- stage: Build
  displayName: Build and push stage
  dependsOn: Test
  jobs:
  - job: Build
    displayName: Build Docker image
    pool:
      vmImage: $(vmImageName)
    
    steps:
    - task: Docker@2
      displayName: Build and push image
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: $(dockerfilePath)
        containerRegistry: $(dockerRegistryServiceConnection)
        tags: |
          $(tag)
          latest

- stage: DeployStaging
  displayName: Deploy to staging
  dependsOn: Build
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/develop'))
  jobs:
  - deployment: DeployStaging
    displayName: Deploy to staging
    pool:
      vmImage: $(vmImageName)
    environment: 'staging'
    strategy:
      runOnce:
        deploy:
          steps:
          - script: echo "Deploying to staging"
            displayName: 'Deploy to staging environment'

- stage: DeployProduction
  displayName: Deploy to production
  dependsOn: Build
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
  jobs:
  - deployment: DeployProduction
    displayName: Deploy to production
    pool:
      vmImage: $(vmImageName)
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - script: echo "Deploying to production"
            displayName: 'Deploy to production environment'
"""
        
        st.code(azure_yaml, language='yaml')

    def quality_gates(self):
        """Quality gates and code analysis"""
        st.subheader("📊 Quality Gates & Code Analysis")
        
        # Quality gate configuration
        st.subheader("🚪 Quality Gate Configuration")
        
        quality_metrics = {
            "Code Coverage": {"threshold": 80, "current": 87, "status": "✅"},
            "Duplication": {"threshold": 3, "current": 1.2, "status": "✅"}, 
            "Maintainability": {"threshold": "A", "current": "A", "status": "✅"},
            "Reliability": {"threshold": "A", "current": "B", "status": "⚠️"},
            "Security": {"threshold": "A", "current": "A", "status": "✅"},
            "Technical Debt": {"threshold": "< 1d", "current": "45m", "status": "✅"}
        }
        
        # Create quality gate status table
        gate_data = []
        for metric, info in quality_metrics.items():
            gate_data.append({
                "Metric": metric,
                "Threshold": info["threshold"],
                "Current": info["current"],
                "Status": info["status"]
            })
        
        gate_df = pd.DataFrame(gate_data)
        st.dataframe(gate_df, use_container_width=True)
        
        # SonarQube configuration
        st.subheader("🔍 SonarQube Configuration")
        
        sonar_config = """# sonar-project.properties
sonar.projectKey=ml-project
sonar.projectName=ML Data Science Project
sonar.projectVersion=1.0

# Source code
sonar.sources=src/
sonar.tests=tests/
sonar.python.coverage.reportPaths=coverage.xml

# Exclusions
sonar.exclusions=**/*.ipynb,**/datasets/**,**/__pycache__/**

# Quality gate conditions
sonar.qualitygate.wait=true

# Analysis parameters
sonar.python.xunit.reportPath=junit/test-*.xml
sonar.python.bandit.reportPaths=bandit-report.json
"""
        
        st.code(sonar_config, language='properties')
        
        # Code analysis tools
        st.subheader("🛠️ Code Analysis Tools")
        
        analysis_tools = {
            "Code Quality": {
                "flake8": "Python linting and style guide enforcement",
                "black": "Code formatting and style consistency", 
                "isort": "Import statement sorting and organization",
                "mypy": "Static type checking for Python"
            },
            "Security": {
                "bandit": "Security vulnerability detection",
                "safety": "Dependency vulnerability scanning",
                "semgrep": "Static analysis for security patterns"
            },
            "Testing": {
                "pytest": "Unit and integration testing framework",
                "pytest-cov": "Code coverage measurement",
                "pytest-html": "HTML test reporting"
            },
            "Documentation": {
                "pydocstyle": "Docstring style checking",
                "sphinx": "Documentation generation",
                "mkdocs": "Documentation site generation"
            }
        }
        
        for category, tools in analysis_tools.items():
            st.write(f"**{category}:**")
            for tool, description in tools.items():
                st.write(f"• **{tool}**: {description}")
            st.write("---")

    def deployment_automation(self):
        """Deployment automation strategies"""
        st.subheader("🚀 Deployment Automation")
        
        # Deployment strategies
        st.subheader("🎯 Deployment Strategies")
        
        strategies = {
            "Blue-Green": {
                "description": "Two identical production environments",
                "pros": ["Zero downtime", "Easy rollback", "Full testing"],
                "cons": ["Double resources", "Complex data sync"],
                "use_case": "Critical applications"
            },
            "Canary": {
                "description": "Gradual rollout to subset of users",
                "pros": ["Risk mitigation", "Real user feedback", "Gradual validation"],
                "cons": ["Complex monitoring", "Longer deployment time"],
                "use_case": "Feature validation"
            },
            "Rolling": {
                "description": "Sequential update of instances",
                "pros": ["Resource efficient", "Simple implementation"],
                "cons": ["Temporary inconsistency", "Slower rollback"],
                "use_case": "Standard updates"
            },
            "Feature Flags": {
                "description": "Toggle features without deployment",
                "pros": ["Instant rollback", "A/B testing", "Gradual rollout"],
                "cons": ["Code complexity", "Flag management"],
                "use_case": "Feature experimentation"
            }
        }
        
        for strategy, info in strategies.items():
            with st.expander(f"📋 {strategy} Deployment"):
                st.write(f"**Description:** {info['description']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Pros:**")
                    for pro in info['pros']:
                        st.write(f"• {pro}")
                
                with col2:
                    st.write("**Cons:**")
                    for con in info['cons']:
                        st.write(f"• {con}")
                
                st.write(f"**Best Use Case:** {info['use_case']}")
        
        # Deployment automation script
        st.subheader("🤖 Deployment Automation Script")
        
        deploy_script = """#!/bin/bash
# Automated deployment script

set -e  # Exit on any error

# Configuration
APP_NAME="ml-app"
ENVIRONMENT=${1:-staging}
IMAGE_TAG=${2:-latest}
REGISTRY="ghcr.io/company"

echo "🚀 Starting deployment of $APP_NAME to $ENVIRONMENT"
echo "📦 Image: $REGISTRY/$APP_NAME:$IMAGE_TAG"

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check if image exists
if ! docker manifest inspect $REGISTRY/$APP_NAME:$IMAGE_TAG > /dev/null 2>&1; then
    echo "❌ Image not found: $REGISTRY/$APP_NAME:$IMAGE_TAG"
    exit 1
fi

# Health check function
health_check() {
    local url=$1
    local retries=30
    local wait_time=10
    
    echo "🏥 Performing health check on $url"
    
    for i in $(seq 1 $retries); do
        if curl -f -s $url/health > /dev/null; then
            echo "✅ Health check passed"
            return 0
        fi
        echo "⏳ Waiting for service to be ready... ($i/$retries)"
        sleep $wait_time
    done
    
    echo "❌ Health check failed after $retries attempts"
    return 1
}

# Backup current deployment
echo "💾 Creating backup of current deployment..."
kubectl get deployment $APP_NAME -o yaml > backup-$APP_NAME-$(date +%Y%m%d-%H%M%S).yaml

# Deploy new version
echo "🔄 Deploying new version..."
kubectl set image deployment/$APP_NAME $APP_NAME=$REGISTRY/$APP_NAME:$IMAGE_TAG

# Wait for rollout to complete
echo "⏳ Waiting for rollout to complete..."
kubectl rollout status deployment/$APP_NAME --timeout=300s

# Get service URL
SERVICE_URL=$(kubectl get service $APP_NAME -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
if [ -z "$SERVICE_URL" ]; then
    SERVICE_URL="localhost:8080"  # Fallback for local testing
fi

# Perform health check
if health_check "http://$SERVICE_URL"; then
    echo "✅ Deployment successful!"
    
    # Run smoke tests
    echo "🧪 Running smoke tests..."
    python scripts/smoke_tests.py --url "http://$SERVICE_URL"
    
    # Notify success
    echo "📢 Notifying deployment success..."
    curl -X POST "$SLACK_WEBHOOK" -H 'Content-type: application/json' \\
         --data "{\\"text\\":\\"✅ $APP_NAME deployed successfully to $ENVIRONMENT\\"}"
    
else
    echo "❌ Health check failed, initiating rollback..."
    
    # Rollback
    kubectl rollout undo deployment/$APP_NAME
    kubectl rollout status deployment/$APP_NAME --timeout=300s
    
    # Notify failure
    curl -X POST "$SLACK_WEBHOOK" -H 'Content-type: application/json' \\
         --data "{\\"text\\":\\"❌ $APP_NAME deployment to $ENVIRONMENT failed and was rolled back\\"}"
    
    exit 1
fi

echo "🎉 Deployment complete!"
"""
        
        st.code(deploy_script, language='bash')

    def pipeline_monitoring(self):
        """Pipeline monitoring and metrics"""
        st.subheader("📈 Pipeline Monitoring & Metrics")
        
        # DORA metrics
        st.subheader("📊 DORA Metrics Dashboard")
        
        dora_metrics = {
            "Deployment Frequency": {
                "current": "5 per day",
                "target": "10 per day", 
                "trend": "↗️",
                "status": "good"
            },
            "Lead Time for Changes": {
                "current": "2.5 hours",
                "target": "< 1 hour",
                "trend": "↗️", 
                "status": "needs_improvement"
            },
            "Change Failure Rate": {
                "current": "8%",
                "target": "< 5%",
                "trend": "↘️",
                "status": "needs_improvement"
            },
            "Mean Time to Recovery": {
                "current": "45 minutes",
                "target": "< 30 minutes",
                "trend": "↘️",
                "status": "good"
            }
        }
        
        # Display DORA metrics
        col1, col2, col3, col4 = st.columns(4)
        
        cols = [col1, col2, col3, col4]
        for i, (metric, data) in enumerate(dora_metrics.items()):
            with cols[i]:
                status_color = "green" if data["status"] == "good" else "orange"
                st.metric(
                    label=metric,
                    value=data["current"],
                    delta=data["trend"]
                )
                st.caption(f"Target: {data['target']}")
        
        # Pipeline performance metrics
        st.subheader("⚡ Pipeline Performance")
        
        performance_data = {
            "Build Time": [8, 7, 9, 6, 8, 7, 5],
            "Test Time": [12, 11, 13, 10, 12, 11, 9], 
            "Deploy Time": [3, 4, 3, 2, 3, 4, 3],
            "Total Time": [23, 22, 25, 18, 23, 22, 17]
        }
        
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        fig = go.Figure()
        
        for metric, values in performance_data.items():
            fig.add_trace(go.Scatter(
                x=days,
                y=values,
                mode='lines+markers',
                name=metric,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Pipeline Performance Trends (Last 7 Days)",
            xaxis_title="Day",
            yaxis_title="Time (minutes)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Alerting configuration
        st.subheader("🚨 Alert Configuration")
        
        alert_rules = """# Prometheus alerting rules
groups:
- name: ci-cd-alerts
  rules:
  - alert: BuildTimeHigh
    expr: ci_build_duration_seconds > 600  # 10 minutes
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "CI/CD build time is too high"
      description: "Build time has been above 10 minutes for 5 minutes"

  - alert: BuildFailureRate
    expr: rate(ci_build_failures_total[1h]) > 0.2
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High build failure rate"
      description: "Build failure rate is above 20% for the last hour"

  - alert: DeploymentFailed
    expr: ci_deployment_status == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Deployment failed"
      description: "Latest deployment has failed"

  - alert: TestCoverageLow
    expr: ci_test_coverage_percent < 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Test coverage below threshold"
      description: "Test coverage is below 80%"
"""
        
        st.code(alert_rules, language='yaml')

    def security_compliance(self):
        """Security and compliance integration"""
        st.subheader("🔒 Security & Compliance Integration")
        
        # Security scanning tools
        st.subheader("🛡️ Security Scanning Tools")
        
        security_tools = {
            "Static Analysis": {
                "Bandit": "Python security vulnerability scanner",
                "Semgrep": "Multi-language static analysis",
                "SonarQube": "Code quality and security analysis",
                "CodeQL": "Semantic code analysis"
            },
            "Dependency Scanning": {
                "Safety": "Python dependency vulnerability check",
                "npm audit": "Node.js dependency vulnerabilities", 
                "Snyk": "Multi-language dependency scanning",
                "OWASP Dependency-Check": "Dependency vulnerability detection"
            },
            "Container Security": {
                "Trivy": "Container image vulnerability scanner",
                "Clair": "Container vulnerability analysis",
                "Docker Scout": "Docker image security analysis",
                "Aqua": "Container security platform"
            },
            "Infrastructure": {
                "Terraform Security": "Infrastructure as Code security",
                "Kubernetes Security": "K8s configuration security",
                "Cloud Security": "Cloud resource security scanning"
            }
        }
        
        for category, tools in security_tools.items():
            st.write(f"**{category}:**")
            for tool, description in tools.items():
                st.write(f"• **{tool}**: {description}")
            st.write("---")
        
        # Compliance framework
        st.subheader("📋 Compliance Framework")
        
        compliance_requirements = {
            "GDPR": [
                "Data encryption at rest and in transit",
                "Audit logging for data access",
                "Data retention policies",
                "Privacy by design implementation"
            ],
            "SOC 2": [
                "Access control management",
                "System monitoring and logging",
                "Incident response procedures",
                "Change management process"
            ],
            "ISO 27001": [
                "Information security management",
                "Risk assessment procedures",
                "Security awareness training",
                "Continuous improvement process"
            ]
        }
        
        for standard, requirements in compliance_requirements.items():
            with st.expander(f"📜 {standard} Requirements"):
                for req in requirements:
                    st.write(f"• {req}")
        
        # Security pipeline integration
        st.subheader("🔐 Security Pipeline Integration")
        
        security_pipeline = """# Security-focused CI/CD pipeline steps

# 1. Pre-commit hooks
pre-commit:
  repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
    - id: black
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.4
    hooks:
    - id: bandit
  - repo: https://github.com/gitguardian/ggshield
    rev: v1.18.0
    hooks:
    - id: ggshield

# 2. Security scanning in CI
security-scan:
  script:
    # Secret scanning
    - ggshield secret scan ci
    
    # Dependency vulnerabilities
    - safety check --json --output safety-report.json
    
    # Static code analysis
    - bandit -r src/ -f json -o bandit-report.json
    
    # Container image scanning
    - trivy image --exit-code 0 --severity HIGH,CRITICAL $IMAGE_NAME
    
    # Infrastructure scanning
    - checkov -f Dockerfile --framework dockerfile
    - terraform plan -out=tfplan
    - checkov -f tfplan --framework terraform_plan

# 3. Runtime security monitoring
runtime-security:
  monitoring:
    - Falco for runtime threat detection
    - OWASP ZAP for dynamic security testing
    - Prometheus for security metrics
    - Grafana for security dashboards
"""
        
        st.code(security_pipeline, language='yaml')
        
        # Security metrics dashboard
        st.subheader("📊 Security Metrics Dashboard")
        
        security_metrics = {
            "Vulnerabilities Found": 3,
            "Secrets Detected": 0,
            "Security Score": 95,
            "Compliance Rate": 98
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics_list = list(security_metrics.items())
        for i, (metric, value) in enumerate(metrics_list):
            with [col1, col2, col3, col4][i]:
                st.metric(metric, value)
