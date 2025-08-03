@echo off
echo =========================================================
echo   🛍️ Fashion E-commerce Analytics Platform Setup
echo =========================================================
echo.
echo This script will set up your complete development environment.
echo Please follow the prompts carefully.
echo.

REM Change to project directory
cd /d "%~dp0"

echo 📁 Current directory: %CD%
echo.

REM Check if Python is installed
echo 🐍 Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version
echo ✅ Python is installed
echo.

REM Check if virtual environment exists
if exist "dvc_env" (
    echo 📦 Virtual environment already exists
    echo Do you want to recreate it? (This will delete all installed packages)
    set /p recreate="Enter Y to recreate, or any other key to continue: "
    if /i "%recreate%"=="Y" (
        echo 🗑️ Removing existing virtual environment...
        rmdir /s /q dvc_env
    )
)

REM Create virtual environment if it doesn't exist
if not exist "dvc_env" (
    echo � Creating Python virtual environment...
    python -m venv dvc_env
    if %errorlevel% neq 0 (
        echo ❌ ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created successfully
) else (
    echo ✅ Using existing virtual environment
)

echo.

REM Activate virtual environment
echo � Activating virtual environment...
call dvc_env\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment activated
echo.

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo 📚 Installing Python packages (this may take several minutes)...
echo Please be patient while all dependencies are downloaded and installed.
echo.

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ⚠️ Some packages failed to install. Trying alternative method...
    pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ ERROR: Failed to install required packages
        echo Please check your internet connection and try again
        pause
        exit /b 1
    )
)

echo ✅ All packages installed successfully
echo.

REM Initialize DVC if not already done
echo �️ Setting up Data Version Control (DVC)...
if not exist ".dvc" (
    echo Initializing DVC...
    dvc init
    if %errorlevel% neq 0 (
        echo ⚠️ Warning: DVC initialization failed, but continuing...
    ) else (
        echo ✅ DVC initialized successfully
    )
) else (
    echo ✅ DVC already initialized
)

REM Add Google Drive remote if not exists
echo 🌐 Configuring Google Drive remote storage...
dvc remote list | findstr "myremote" >nul
if %errorlevel% neq 0 (
    dvc remote add -d myremote gdrive://1BxT4_aIuqEJE-7d-Ty3_FrOyqlNvyM8V
    if %errorlevel% neq 0 (
        echo ⚠️ Warning: Failed to add Google Drive remote
    ) else (
        echo ✅ Google Drive remote added successfully
    )
) else (
    echo ✅ Google Drive remote already configured
)

REM Configure DVC settings
echo ⚙️ Configuring DVC settings...
dvc config core.autostage true
dvc config core.analytics false
echo ✅ DVC settings configured

echo.

REM Check for datasets
echo 📊 Checking datasets...
if exist "datasets" (
    echo ✅ Datasets folder exists
    if exist "datasets\DS-2-8-25" (
        echo ✅ Dataset files found
    ) else (
        echo ⚠️ Dataset files not found in datasets\DS-2-8-25
        echo You may need to run 'dvc pull' to download datasets
        echo.
        echo 📥 Attempting to download datasets...
        dvc pull
        if %errorlevel% neq 0 (
            echo ⚠️ Failed to download datasets automatically
            echo You will need to run 'dvc pull' manually after setup
        ) else (
            echo ✅ Datasets downloaded successfully
        )
    )
) else (
    echo ⚠️ Datasets folder not found
    echo You will need to run 'dvc pull' to download datasets
)

echo.

REM Test Streamlit installation
echo 🧪 Testing Streamlit installation...
python -c "import streamlit; print('✅ Streamlit imported successfully')"
if %errorlevel% neq 0 (
    echo ❌ ERROR: Streamlit test failed
    pause
    exit /b 1
)

echo.
echo =========================================================
echo                  🎉 SETUP COMPLETE! 🎉
echo =========================================================
echo.
echo Your Fashion E-commerce Analytics Platform is ready!
echo.
echo 📋 NEXT STEPS:
echo.
echo 1. 🔐 If datasets didn't download automatically, run:
echo    dvc pull
echo    (This will prompt for Google account authentication)
echo.
echo 2. 🚀 To start the application, run:
echo    start_dashboard.bat
echo.
echo 3. 🌐 Open your browser to:
echo    http://localhost:8501
echo.
echo 📚 For detailed instructions, see README.md
echo.
echo ⚡ Quick start: Double-click 'start_dashboard.bat'
echo.

pause
