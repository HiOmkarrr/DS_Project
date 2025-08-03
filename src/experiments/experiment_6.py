"""
Experiment 6: Containerization & DevOps Integration
Docker, Kubernetes, and CI/CD pipeline implementation
"""

import streamlit as st
import pandas as pd
import json
import os
import subprocess
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

class Experiment6:
    def __init__(self):
        self.project_path = r"C:\Users\Omkar\SEM_7\DS_Project"
        
    def render_containerization_dashboard(self):
        """Main containerization dashboard"""
        st.markdown("""
        <div class="experiment-card">
            <h2>🐳 Containerization & Docker Deployment</h2>
            <p>Containerize ML models and applications for scalable deployment</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Containerization Options
        container_options = [
            "📋 Docker Fundamentals & Setup",
            "🐳 Dockerfile Generation",
            "📦 Requirements & Dependencies Management",
            "🚀 Model API Containerization", 
            "🌐 Streamlit App Containerization",
            "🔧 Docker Compose Configuration",
            "☁️ Cloud Deployment Preparation",
            "📊 Container Monitoring & Logging",
            "🔄 CI/CD Pipeline Integration"
        ]
        
        selected_activity = st.selectbox(
            "Select Containerization Activity:",
            container_options
        )
        
        # Route to appropriate activity
        if selected_activity == "📋 Docker Fundamentals & Setup":
            self.docker_fundamentals()
        elif selected_activity == "🐳 Dockerfile Generation":
            self.dockerfile_generation()
        elif selected_activity == "📦 Requirements & Dependencies Management":
            self.requirements_management()
        elif selected_activity == "🚀 Model API Containerization":
            self.model_api_containerization()
        elif selected_activity == "🌐 Streamlit App Containerization":
            self.streamlit_containerization()
        elif selected_activity == "🔧 Docker Compose Configuration":
            self.docker_compose_config()
        elif selected_activity == "☁️ Cloud Deployment Preparation":
            self.cloud_deployment_prep()
        elif selected_activity == "📊 Container Monitoring & Logging":
            self.container_monitoring()
        elif selected_activity == "🔄 CI/CD Pipeline Integration":
            self.cicd_integration()

    def docker_fundamentals(self):
        """Docker fundamentals and setup guide"""
        st.subheader("📋 Docker Fundamentals & Setup")
        
        st.markdown("""
        ### 🐳 What is Docker?
        
        Docker is a containerization platform that packages applications with their dependencies into lightweight, portable containers.
        
        **Key Benefits:**
        - **Consistency:** Same environment across development, testing, and production
        - **Portability:** Run anywhere Docker is supported
        - **Scalability:** Easy to scale applications horizontally
        - **Isolation:** Applications run in isolated environments
        """)
        
        # Docker installation check
        st.subheader("🔧 Docker Installation Status")
        
        if st.button("🔍 Check Docker Installation"):
            try:
                result = subprocess.run(['docker', '--version'], 
                                      capture_output=True, text=True, shell=True)
                if result.returncode == 0:
                    st.success(f"✅ Docker is installed: {result.stdout.strip()}")
                    
                    # Check Docker daemon
                    daemon_result = subprocess.run(['docker', 'info'], 
                                                 capture_output=True, text=True, shell=True)
                    if daemon_result.returncode == 0:
                        st.success("✅ Docker daemon is running")
                    else:
                        st.warning("⚠️ Docker daemon is not running. Please start Docker Desktop.")
                else:
                    st.error("❌ Docker is not installed")
            except Exception as e:
                st.error(f"❌ Error checking Docker: {str(e)}")
        
        # Docker installation guide
        st.subheader("📖 Docker Installation Guide")
        
        tab1, tab2, tab3 = st.tabs(["Windows", "macOS", "Linux"])
        
        with tab1:
            st.markdown("""
            **Windows Installation:**
            1. Download Docker Desktop from [docker.com](https://docker.com)
            2. Run the installer
            3. Enable WSL 2 features
            4. Restart your computer
            5. Start Docker Desktop
            """)
            
        with tab2:
            st.markdown("""
            **macOS Installation:**
            1. Download Docker Desktop for Mac
            2. Drag Docker to Applications folder
            3. Launch Docker Desktop
            4. Follow setup instructions
            """)
            
        with tab3:
            st.markdown("""
            **Linux Installation:**
            ```bash
            # Ubuntu/Debian
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            ```
            """)
        
        # Docker basic commands
        st.subheader("🛠️ Essential Docker Commands")
        
        docker_commands = {
            "Build an image": "docker build -t myapp .",
            "Run a container": "docker run -p 8000:8000 myapp",
            "List running containers": "docker ps",
            "List all containers": "docker ps -a",
            "Stop a container": "docker stop <container_id>",
            "Remove a container": "docker rm <container_id>",
            "List images": "docker images",
            "Remove an image": "docker rmi <image_id>",
            "View container logs": "docker logs <container_id>",
            "Execute command in container": "docker exec -it <container_id> bash"
        }
        
        for description, command in docker_commands.items():
            st.code(f"# {description}\n{command}", language='bash')

    def dockerfile_generation(self):
        """Generate Dockerfiles for different applications"""
        st.subheader("🐳 Dockerfile Generation")
        
        # Application type selection
        app_types = [
            "🤖 ML Model API (FastAPI)",
            "🎨 Streamlit Dashboard",
            "📊 Jupyter Notebook Server",
            "🐍 Python Script",
            "🌐 Full-Stack Application"
        ]
        
        selected_app = st.selectbox("Select Application Type:", app_types)
        
        # Python version selection
        python_version = st.selectbox("Select Python Version:", 
                                    ["3.11", "3.10", "3.9", "3.8"])
        
        # Generate Dockerfile based on selection
        if selected_app == "🤖 ML Model API (FastAPI)":
            self._generate_fastapi_dockerfile(python_version)
        elif selected_app == "🎨 Streamlit Dashboard":
            self._generate_streamlit_dockerfile(python_version)
        elif selected_app == "📊 Jupyter Notebook Server":
            self._generate_jupyter_dockerfile(python_version)
        elif selected_app == "🐍 Python Script":
            self._generate_python_dockerfile(python_version)
        elif selected_app == "🌐 Full-Stack Application":
            self._generate_fullstack_dockerfile(python_version)

    def _generate_fastapi_dockerfile(self, python_version):
        """Generate FastAPI Dockerfile"""
        st.subheader("🤖 FastAPI Model API Dockerfile")
        
        dockerfile_content = f"""# Use official Python runtime as base image
FROM python:{python_version}-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        build-essential \\
        curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        st.code(dockerfile_content, language='dockerfile')
        
        # Generate FastAPI requirements
        fastapi_requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
scikit-learn==1.3.2
numpy==1.25.2
pydantic==2.5.0
joblib==1.3.2
python-multipart==0.0.6"""
        
        st.subheader("📦 FastAPI Requirements")
        st.code(fastapi_requirements, language='text')
        
        # Build and run commands
        st.subheader("🚀 Build and Run Commands")
        build_commands = """# Build the Docker image
docker build -t ml-api .

# Run the container
docker run -p 8000:8000 ml-api

# Run with environment variables
docker run -p 8000:8000 -e MODEL_NAME=my_model ml-api

# Run in detached mode
docker run -d -p 8000:8000 --name ml-api-container ml-api"""
        
        st.code(build_commands, language='bash')

    def _generate_streamlit_dockerfile(self, python_version):
        """Generate Streamlit Dockerfile"""
        st.subheader("🎨 Streamlit Dashboard Dockerfile")
        
        dockerfile_content = f"""# Use official Python runtime as base image
FROM python:{python_version}-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        build-essential \\
        curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create streamlit config directory
RUN mkdir -p /app/.streamlit

# Copy streamlit config
COPY .streamlit/config.toml /app/.streamlit/config.toml

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "main_dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
"""
        
        st.code(dockerfile_content, language='dockerfile')
        
        # Streamlit config
        st.subheader("⚙️ Streamlit Configuration")
        streamlit_config = """[server]
headless = true
port = 8501
address = "0.0.0.0"

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
        
        st.code(streamlit_config, language='toml')

    def _generate_jupyter_dockerfile(self, python_version):
        """Generate Jupyter Dockerfile"""
        st.subheader("📊 Jupyter Notebook Server Dockerfile")
        
        dockerfile_content = f"""# Use official Jupyter image as base
FROM jupyter/scipy-notebook:python-{python_version}

# Switch to root to install packages
USER root

# Install additional system packages
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    vim \\
    && rm -rf /var/lib/apt/lists/*

# Switch back to jovyan user
USER ${{NB_UID}}

# Copy requirements
COPY requirements.txt /tmp/

# Install Python packages
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy notebooks and data
COPY notebooks/ /home/jovyan/work/notebooks/
COPY datasets/ /home/jovyan/work/datasets/

# Set working directory
WORKDIR /home/jovyan/work

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]
"""
        
        st.code(dockerfile_content, language='dockerfile')

    def _generate_python_dockerfile(self, python_version):
        """Generate Python script Dockerfile"""
        st.subheader("🐍 Python Script Dockerfile")
        
        dockerfile_content = f"""# Use official Python runtime as base image
FROM python:{python_version}-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Run the Python script
CMD ["python", "main.py"]
"""
        
        st.code(dockerfile_content, language='dockerfile')

    def _generate_fullstack_dockerfile(self, python_version):
        """Generate full-stack application Dockerfile"""
        st.subheader("🌐 Full-Stack Application Dockerfile")
        
        dockerfile_content = f"""# Multi-stage build for full-stack app

# Stage 1: Build frontend
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Python backend
FROM python:{python_version}-slim AS backend

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        build-essential \\
        nginx \\
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy frontend build from previous stage
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app

# Expose ports
EXPOSE 80 8000

# Start script
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
"""
        
        st.code(dockerfile_content, language='dockerfile')

    def requirements_management(self):
        """Manage requirements and dependencies"""
        st.subheader("📦 Requirements & Dependencies Management")
        
        # Current requirements analysis
        st.subheader("📋 Current Project Requirements")
        
        requirements_file = os.path.join(self.project_path, "requirements.txt")
        
        if os.path.exists(requirements_file):
            with open(requirements_file, 'r') as f:
                current_requirements = f.read()
            
            st.code(current_requirements, language='text')
            
            # Analyze requirements
            lines = current_requirements.strip().split('\n')
            packages = [line.split('==')[0] if '==' in line else line.split('>=')[0] if '>=' in line else line 
                       for line in lines if line and not line.startswith('#')]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Packages", len(packages))
            with col2:
                versioned = len([line for line in lines if '==' in line or '>=' in line])
                st.metric("Versioned Packages", versioned)
            with col3:
                unversioned = len(packages) - versioned
                st.metric("Unversioned Packages", unversioned)
        else:
            st.warning("⚠️ requirements.txt not found")
        
        # Generate optimized requirements
        st.subheader("🚀 Optimized Requirements for Docker")
        
        optimization_type = st.selectbox(
            "Select Optimization:",
            ["🏃 Minimal (Core only)", "📊 Data Science", "🤖 ML Production", "🎨 Full Dashboard"]
        )
        
        if optimization_type == "🏃 Minimal (Core only)":
            optimized_reqs = """# Core dependencies only
pandas==2.1.3
numpy==1.25.2
scikit-learn==1.3.2
"""
        elif optimization_type == "📊 Data Science":
            optimized_reqs = """# Data Science essentials
pandas==2.1.3
numpy==1.25.2
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.17.0
jupyter==1.0.0
"""
        elif optimization_type == "🤖 ML Production":
            optimized_reqs = """# ML Production ready
pandas==2.1.3
numpy==1.25.2
scikit-learn==1.3.2
fastapi==0.104.1
uvicorn[standard]==0.24.0
joblib==1.3.2
pydantic==2.5.0
mlflow==2.8.1
"""
        elif optimization_type == "🎨 Full Dashboard":
            optimized_reqs = """# Full dashboard with all features
streamlit==1.28.2
pandas==2.1.3
numpy==1.25.2
plotly==5.17.0
scikit-learn==1.3.2
seaborn==0.13.0
matplotlib==3.8.2
scipy==1.11.4
statsmodels==0.14.0
"""
        
        st.code(optimized_reqs, language='text')
        
        # Docker-specific optimizations
        st.subheader("🐳 Docker-Specific Optimizations")
        
        optimizations = [
            "**Pin exact versions** - Use `==` instead of `>=` for reproducibility",
            "**Multi-stage builds** - Separate build and runtime dependencies", 
            "**Layer caching** - Copy requirements.txt before copying code",
            "**Minimal base images** - Use `-slim` or `-alpine` variants",
            "**Remove build tools** - Clean up after installation",
            "**Use .dockerignore** - Exclude unnecessary files"
        ]
        
        for opt in optimizations:
            st.write(f"• {opt}")
        
        # Generate .dockerignore
        st.subheader("📝 .dockerignore File")
        
        dockerignore_content = """# Version control
.git
.gitignore

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# Documentation
README.md
docs/
*.md

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data files (if not needed in container)
datasets/
*.csv
*.json
*.xlsx

# Notebooks (if not needed)
*.ipynb
notebooks/

# Logs
logs/
*.log
"""
        
        st.code(dockerignore_content, language='text')

    def model_api_containerization(self):
        """Containerize ML model API"""
        st.subheader("🚀 Model API Containerization")
        
        # Check if models are available
        if 'ml_data' not in st.session_state or 'trained_models' not in st.session_state.ml_data:
            st.warning("⚠️ Please train models in Experiment 4 first")
            return
        
        models = st.session_state.ml_data['trained_models']
        
        # Model selection for API
        selected_model = st.selectbox("Select Model for API:", list(models.keys()))
        
        # Generate FastAPI application code
        st.subheader("📝 Generated FastAPI Application")
        
        # Get feature names
        X_train = st.session_state.ml_data['X_train']
        feature_names = X_train.columns.tolist()
        
        # Generate Pydantic model
        pydantic_fields = []
        for feature in feature_names[:10]:  # Limit to first 10 features for readability
            if X_train[feature].dtype in ['int64', 'float64']:
                pydantic_fields.append(f"    {feature}: float")
            else:
                pydantic_fields.append(f"    {feature}: str")
        
        fastapi_code = f"""from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List

app = FastAPI(title="ML Model API", version="1.0.0")

# Load model at startup
model = joblib.load("model.pkl")

class PredictionInput(BaseModel):
{chr(10).join(pydantic_fields)}
    # Add remaining features...

class PredictionOutput(BaseModel):
    prediction: float
    confidence: float
    model_used: str

@app.get("/")
def root():
    return {{"message": "ML Model API", "model": "{selected_model}"}}

@app.get("/health")
def health_check():
    return {{"status": "healthy", "model_loaded": model is not None}}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    # Convert to DataFrame
    df = pd.DataFrame([input_data.dict()])
    
    # Ensure all features are present (add defaults if missing)
    for feature in {feature_names}:
        if feature not in df.columns:
            df[feature] = 0.0  # or appropriate default
    
    # Reorder columns to match training data
    df = df[{feature_names}]
    
    # Make prediction
    prediction = model.predict(df)[0]
    
    # Get confidence (if available)
    confidence = 0.95  # Default confidence
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(df)[0]
        confidence = float(max(probabilities))
    
    return PredictionOutput(
        prediction=float(prediction),
        confidence=confidence,
        model_used="{selected_model}"
    )

@app.post("/batch_predict")
def batch_predict(input_data: List[PredictionInput]):
    predictions = []
    for item in input_data:
        # Individual prediction logic here
        df = pd.DataFrame([item.dict()])
        # ... same prediction logic as above
        prediction = model.predict(df)[0]
        predictions.append(float(prediction))
    
    return {{"predictions": predictions, "count": len(predictions)}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        
        st.code(fastapi_code, language='python')
        
        # Docker build instructions
        st.subheader("🐳 Docker Build Instructions")
        
        build_instructions = f"""# 1. Save the model
import joblib
model = {{'your_trained_model'}}  # Replace with actual model
joblib.dump(model, 'model.pkl')

# 2. Create the FastAPI app (save as main.py)
# Use the code above

# 3. Create requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
scikit-learn==1.3.2
joblib==1.3.2
pydantic==2.5.0

# 4. Build Docker image
docker build -t ml-api-{selected_model.lower().replace(' ', '-')} .

# 5. Run container
docker run -p 8000:8000 ml-api-{selected_model.lower().replace(' ', '-')}

# 6. Test API
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"feature1": 1.0, "feature2": 2.0}}'
"""
        
        st.code(build_instructions, language='bash')
        
        # API testing interface
        st.subheader("🧪 API Testing Interface")
        
        if st.button("🚀 Generate Test Script"):
            test_script = f"""import requests
import json

# API endpoint
API_URL = "http://localhost:8000"

# Test health check
response = requests.get(f"{{API_URL}}/health")
print("Health check:", response.json())

# Test prediction
test_data = {{
    {', '.join([f'"{feature}": 1.0' for feature in feature_names[:5]])}
}}

response = requests.post(
    f"{{API_URL}}/predict",
    json=test_data
)

print("Prediction response:", response.json())

# Test batch prediction
batch_data = [test_data] * 3
response = requests.post(
    f"{{API_URL}}/batch_predict",
    json=batch_data
)

print("Batch prediction:", response.json())
"""
            
            st.code(test_script, language='python')

    def streamlit_containerization(self):
        """Containerize Streamlit application"""
        st.subheader("🌐 Streamlit App Containerization")
        
        # Current app analysis
        st.subheader("📊 Current Application Analysis")
        
        # List current files
        src_path = os.path.join(self.project_path, "src")
        if os.path.exists(src_path):
            files = []
            for root, dirs, filenames in os.walk(src_path):
                for filename in filenames:
                    if filename.endswith('.py'):
                        rel_path = os.path.relpath(os.path.join(root, filename), src_path)
                        files.append(rel_path)
            
            st.write("**Python files in src/:**")
            for file in files:
                st.write(f"• {file}")
        
        # Generate Streamlit Dockerfile
        st.subheader("🐳 Streamlit Dockerfile")
        
        streamlit_dockerfile = """# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
        build-essential \\
        curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY datasets/ ./datasets/

# Create streamlit config directory
RUN mkdir -p /app/.streamlit

# Copy streamlit config
COPY .streamlit/ /app/.streamlit/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "src/gui/main_dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
"""
        
        st.code(streamlit_dockerfile, language='dockerfile')
        
        # Streamlit configuration
        st.subheader("⚙️ Streamlit Configuration for Docker")
        
        config_toml = """[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
serverAddress = "0.0.0.0"
serverPort = 8501

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[logger]
level = "info"
"""
        
        st.code(config_toml, language='toml')
        
        # Docker Compose for development
        st.subheader("🔧 Docker Compose for Development")
        
        docker_compose = """version: '3.8'

services:
  streamlit-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./src:/app/src
      - ./datasets:/app/datasets
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
    
  # Optional: Add database service
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: mlapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
"""
        
        st.code(docker_compose, language='yaml')

    def docker_compose_config(self):
        """Generate Docker Compose configurations"""
        st.subheader("🔧 Docker Compose Configuration")
        
        # Service selection
        services = st.multiselect(
            "Select Services to Include:",
            ["🎨 Streamlit App", "🤖 FastAPI", "🗄️ PostgreSQL", "📊 Jupyter", "🔍 Elasticsearch", "📈 Grafana"],
            default=["🎨 Streamlit App", "🗄️ PostgreSQL"]
        )
        
        if st.button("🚀 Generate Docker Compose"):
            compose_content = self._generate_docker_compose(services)
            st.code(compose_content, language='yaml')

    def _generate_docker_compose(self, services):
        """Generate Docker Compose based on selected services"""
        compose = {
            "version": "3.8",
            "services": {},
            "volumes": {},
            "networks": {
                "ml-network": {"driver": "bridge"}
            }
        }
        
        if "🎨 Streamlit App" in services:
            compose["services"]["streamlit"] = {
                "build": ".",
                "ports": ["8501:8501"],
                "volumes": ["./src:/app/src", "./datasets:/app/datasets"],
                "environment": ["PYTHONPATH=/app"],
                "networks": ["ml-network"],
                "restart": "unless-stopped"
            }
        
        if "🤖 FastAPI" in services:
            compose["services"]["api"] = {
                "build": "./api",
                "ports": ["8000:8000"],
                "volumes": ["./models:/app/models"],
                "networks": ["ml-network"],
                "restart": "unless-stopped"
            }
        
        if "🗄️ PostgreSQL" in services:
            compose["services"]["postgres"] = {
                "image": "postgres:15",
                "environment": [
                    "POSTGRES_DB=mlapp",
                    "POSTGRES_USER=user", 
                    "POSTGRES_PASSWORD=password"
                ],
                "volumes": ["postgres_data:/var/lib/postgresql/data"],
                "ports": ["5432:5432"],
                "networks": ["ml-network"]
            }
            compose["volumes"]["postgres_data"] = None
        
        if "📊 Jupyter" in services:
            compose["services"]["jupyter"] = {
                "image": "jupyter/scipy-notebook",
                "ports": ["8888:8888"],
                "volumes": ["./notebooks:/home/jovyan/work"],
                "networks": ["ml-network"],
                "command": "start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''"
            }
        
        if "🔍 Elasticsearch" in services:
            compose["services"]["elasticsearch"] = {
                "image": "elasticsearch:8.8.0",
                "environment": [
                    "discovery.type=single-node",
                    "xpack.security.enabled=false"
                ],
                "ports": ["9200:9200"],
                "networks": ["ml-network"]
            }
        
        if "📈 Grafana" in services:
            compose["services"]["grafana"] = {
                "image": "grafana/grafana",
                "ports": ["3000:3000"],
                "environment": [
                    "GF_SECURITY_ADMIN_PASSWORD=admin"
                ],
                "networks": ["ml-network"]
            }
        
        import yaml
        return yaml.dump(compose, default_flow_style=False)

    def cloud_deployment_prep(self):
        """Prepare for cloud deployment"""
        st.subheader("☁️ Cloud Deployment Preparation")
        
        # Cloud platform selection
        platforms = ["🌐 AWS", "☁️ Google Cloud", "🔷 Azure", "🚀 Heroku", "🐙 Digital Ocean"]
        selected_platform = st.selectbox("Select Cloud Platform:", platforms)
        
        if selected_platform == "🌐 AWS":
            self._aws_deployment_guide()
        elif selected_platform == "☁️ Google Cloud":
            self._gcp_deployment_guide()
        elif selected_platform == "🔷 Azure":
            self._azure_deployment_guide()
        elif selected_platform == "🚀 Heroku":
            self._heroku_deployment_guide()
        elif selected_platform == "🐙 Digital Ocean":
            self._digitalocean_deployment_guide()

    def _aws_deployment_guide(self):
        """AWS deployment guide"""
        st.subheader("🌐 AWS Deployment Guide")
        
        # ECS deployment
        st.write("**🐳 Amazon ECS (Elastic Container Service)**")
        
        ecs_task_definition = """{
  "family": "ml-app-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "ml-app",
      "image": "your-ecr-repo/ml-app:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ml-app",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}"""
        
        st.code(ecs_task_definition, language='json')
        
        # Deployment commands
        st.write("**🚀 Deployment Commands**")
        
        aws_commands = """# 1. Build and tag image
docker build -t ml-app .
docker tag ml-app:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-app:latest

# 2. Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-app:latest

# 3. Create ECS service
aws ecs create-service \\
    --cluster ml-cluster \\
    --service-name ml-app-service \\
    --task-definition ml-app-task \\
    --desired-count 2 \\
    --launch-type FARGATE \\
    --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
"""
        
        st.code(aws_commands, language='bash')

    def _gcp_deployment_guide(self):
        """Google Cloud deployment guide"""
        st.subheader("☁️ Google Cloud Deployment Guide")
        
        # Cloud Run deployment
        st.write("**🏃 Google Cloud Run**")
        
        gcp_commands = """# 1. Build and push to Container Registry
docker build -t gcr.io/PROJECT_ID/ml-app .
docker push gcr.io/PROJECT_ID/ml-app

# 2. Deploy to Cloud Run
gcloud run deploy ml-app \\
    --image gcr.io/PROJECT_ID/ml-app \\
    --platform managed \\
    --region us-central1 \\
    --allow-unauthenticated \\
    --memory 2Gi \\
    --cpu 2 \\
    --max-instances 10

# 3. Custom domain (optional)
gcloud run domain-mappings create \\
    --service ml-app \\
    --domain your-domain.com \\
    --region us-central1
"""
        
        st.code(gcp_commands, language='bash')

    def _azure_deployment_guide(self):
        """Azure deployment guide"""
        st.subheader("🔷 Azure Deployment Guide")
        
        # Container Instances
        azure_commands = """# 1. Build and push to Azure Container Registry
docker build -t ml-app .
docker tag ml-app myregistry.azurecr.io/ml-app:latest

az acr login --name myregistry
docker push myregistry.azurecr.io/ml-app:latest

# 2. Deploy to Container Instances
az container create \\
    --resource-group myResourceGroup \\
    --name ml-app \\
    --image myregistry.azurecr.io/ml-app:latest \\
    --cpu 2 \\
    --memory 4 \\
    --registry-login-server myregistry.azurecr.io \\
    --registry-username myregistry \\
    --registry-password password \\
    --dns-name-label ml-app \\
    --ports 8501
"""
        
        st.code(azure_commands, language='bash')

    def _heroku_deployment_guide(self):
        """Heroku deployment guide"""
        st.subheader("🚀 Heroku Deployment Guide")
        
        # Heroku deployment
        heroku_commands = """# 1. Install Heroku CLI and login
heroku login

# 2. Create Heroku app
heroku create your-ml-app

# 3. Set stack to container
heroku stack:set container

# 4. Create heroku.yml
echo "build:
  docker:
    web: Dockerfile
run:
  web: streamlit run src/gui/main_dashboard.py --server.address 0.0.0.0 --server.port $PORT" > heroku.yml

# 5. Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
"""
        
        st.code(heroku_commands, language='bash')

    def _digitalocean_deployment_guide(self):
        """Digital Ocean deployment guide"""
        st.subheader("🐙 Digital Ocean Deployment Guide")
        
        # App Platform deployment
        do_spec = """{
  "name": "ml-app",
  "services": [
    {
      "name": "web",
      "source_dir": "/",
      "github": {
        "repo": "your-username/your-repo",
        "branch": "main"
      },
      "run_command": "streamlit run src/gui/main_dashboard.py --server.address 0.0.0.0 --server.port 8080",
      "environment_slug": "python",
      "instance_count": 1,
      "instance_size_slug": "basic-xxs",
      "http_port": 8080,
      "routes": [
        {
          "path": "/"
        }
      ]
    }
  ]
}"""
        
        st.code(do_spec, language='json')

    def container_monitoring(self):
        """Container monitoring and logging"""
        st.subheader("📊 Container Monitoring & Logging")
        
        st.markdown("""
        ### 📈 Monitoring Best Practices
        
        **Key Metrics to Monitor:**
        - CPU and Memory usage
        - Request/response times
        - Error rates
        - Container restart count
        - Model prediction accuracy drift
        """)
        
        # Docker stats
        st.subheader("📊 Docker Stats Monitoring")
        
        monitoring_script = """#!/bin/bash
# Container monitoring script

echo "Container Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\\t{{.CPUPerc}}\\t{{.MemUsage}}\\t{{.NetIO}}\\t{{.BlockIO}}"

echo "\\nContainer Health Status:"
docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"

echo "\\nContainer Logs (last 10 lines):"
for container in $(docker ps --format "{{.Names}}"); do
    echo "=== $container ==="
    docker logs --tail 10 $container
done
"""
        
        st.code(monitoring_script, language='bash')
        
        # Prometheus configuration
        st.subheader("📊 Prometheus + Grafana Setup")
        
        prometheus_config = """# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'docker'
    static_configs:
      - targets: ['host.docker.internal:9323']
  
  - job_name: 'ml-app'
    static_configs:
      - targets: ['ml-app:8000']
    metrics_path: '/metrics'
"""
        
        st.code(prometheus_config, language='yaml')

    def cicd_integration(self):
        """CI/CD pipeline integration"""
        st.subheader("🔄 CI/CD Pipeline Integration")
        
        # Platform selection
        cicd_platforms = ["🐙 GitHub Actions", "🦊 GitLab CI", "🔵 Azure DevOps", "🟠 Jenkins"]
        selected_cicd = st.selectbox("Select CI/CD Platform:", cicd_platforms)
        
        if selected_cicd == "🐙 GitHub Actions":
            self._github_actions_pipeline()
        elif selected_cicd == "🦊 GitLab CI":
            self._gitlab_ci_pipeline()
        elif selected_cicd == "🔵 Azure DevOps":
            self._azure_devops_pipeline()
        elif selected_cicd == "🟠 Jenkins":
            self._jenkins_pipeline()

    def _github_actions_pipeline(self):
        """GitHub Actions CI/CD pipeline"""
        st.subheader("🐙 GitHub Actions Pipeline")
        
        github_workflow = """name: ML Model CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest flake8
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Test with pytest
      run: |
        pytest tests/ -v

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - name: Checkout repository
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
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Deploy to production
      run: |
        echo "Deploy to production server"
        # Add your deployment commands here
"""
        
        st.code(github_workflow, language='yaml')

    def _gitlab_ci_pipeline(self):
        """GitLab CI pipeline"""
        st.subheader("🦊 GitLab CI Pipeline")
        
        gitlab_ci = """stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: python:3.11
  before_script:
    - pip install -r requirements.txt
    - pip install pytest flake8
  script:
    - flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    - pytest tests/ -v
  only:
    - branches

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --update curl
  script:
    - echo "Deploying to production"
    # Add deployment commands
  only:
    - main
  when: manual
"""
        
        st.code(gitlab_ci, language='yaml')

    def _azure_devops_pipeline(self):
        """Azure DevOps pipeline"""
        st.subheader("🔵 Azure DevOps Pipeline")
        
        azure_pipeline = """trigger:
- main

variables:
  dockerRegistryServiceConnection: 'myDockerRegistry'
  imageRepository: 'ml-app'
  containerRegistry: 'myregistry.azurecr.io'
  dockerfilePath: '**/Dockerfile'
  tag: '$(Build.BuildId)'

stages:
- stage: Test
  displayName: Test stage
  jobs:
  - job: Test
    displayName: Test
    pool:
      vmImage: ubuntu-latest
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.11'
    - script: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest flake8
      displayName: 'Install dependencies'
    - script: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        pytest tests/ -v
      displayName: 'Test with pytest'

- stage: Build
  displayName: Build and push stage
  dependsOn: Test
  jobs:
  - job: Build
    displayName: Build
    pool:
      vmImage: ubuntu-latest
    steps:
    - task: Docker@2
      displayName: Build and push an image to container registry
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: $(dockerfilePath)
        containerRegistry: $(dockerRegistryServiceConnection)
        tags: |
          $(tag)
"""
        
        st.code(azure_pipeline, language='yaml')

    def _jenkins_pipeline(self):
        """Jenkins pipeline"""
        st.subheader("🟠 Jenkins Pipeline")
        
        jenkins_pipeline = """pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'your-registry.com'
        IMAGE_NAME = 'ml-app'
        IMAGE_TAG = "${BUILD_NUMBER}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/your-repo/ml-app.git'
            }
        }
        
        stage('Test') {
            steps {
                sh '''
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install pytest flake8
                    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
                    pytest tests/ -v
                '''
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}")
                }
            }
        }
        
        stage('Push to Registry') {
            steps {
                script {
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-registry-credentials') {
                        docker.image("${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}").push()
                        docker.image("${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}").push('latest')
                    }
                }
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    echo "Deploying to production"
                    # Add deployment commands
                '''
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            echo 'Pipeline succeeded!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
"""
        
        st.code(jenkins_pipeline, language='groovy')
