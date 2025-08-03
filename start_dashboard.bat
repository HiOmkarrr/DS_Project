@echo off
echo =========================================================
echo   🛍️ Fashion E-commerce Analytics Platform
echo =========================================================
echo.

REM Change to project directory
cd /d "%~dp0"

echo 📁 Starting from: %CD%
echo.

REM Check if virtual environment exists
if not exist "dvc_env\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo.
    echo 🔧 Please run setup.bat first to create the environment.
    echo.
    echo 📋 Quick fix:
    echo 1. Double-click setup.bat
    echo 2. Wait for setup to complete
    echo 3. Run this script again
    echo.
    pause
    exit /b 1
)

echo 🔄 Activating virtual environment...
call dvc_env\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment
    echo Please run setup.bat to fix the environment
    pause
    exit /b 1
)

echo ✅ Virtual environment activated
echo.

REM Check if datasets exist
echo 📊 Checking datasets...
if not exist "datasets\DS-2-8-25" (
    echo ⚠️ Datasets not found!
    echo.
    echo 🔄 Attempting to download datasets with DVC...
    dvc pull
    if %errorlevel% neq 0 (
        echo.
        echo ⚠️ DVC pull failed. Generating sample data instead...
        python generate_sample_data.py
        if %errorlevel% neq 0 (
            echo ❌ Failed to generate sample data
            echo The application will still work with limited functionality
        )
    )
) else (
    echo ✅ Datasets found
)

echo.

REM Check if Streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Streamlit not installed!
    echo Installing Streamlit...
    pip install streamlit
    if %errorlevel% neq 0 (
        echo ❌ Failed to install Streamlit
        echo Please run setup.bat to fix dependencies
        pause
        exit /b 1
    )
)

echo 🚀 Starting Streamlit dashboard...
echo.
echo =========================================================
echo                   🎯 DASHBOARD STARTING
echo =========================================================
echo.
echo 🌐 Dashboard URL: http://localhost:8501
echo � Mobile view: http://localhost:8501/?embedded=true
echo.
echo 💡 Tips:
echo   • Use Ctrl+C to stop the server
echo   • Refresh browser if page doesn't load
echo   • Check firewall settings if connection fails
echo.
echo 📊 Loading experiments and GUI interface...
echo =========================================================
echo.

REM Start Streamlit with better configuration
streamlit run src\gui\main_dashboard.py --server.port 8501 --server.address localhost --server.enableCORS false --server.enableXsrfProtection false

REM Check if Streamlit started successfully
if %errorlevel% neq 0 (
    echo.
    echo ❌ Streamlit failed to start!
    echo.
    echo � Troubleshooting steps:
    echo 1. Check if port 8501 is already in use
    echo 2. Try: netstat -an ^| findstr 8501
    echo 3. Close other applications using the port
    echo 4. Run this script again
    echo.
    echo 🆘 Alternative: Try a different port
    echo streamlit run src\gui\main_dashboard.py --server.port 8502
    echo.
    pause
    exit /b 1
)

echo.
echo =========================================================
echo              �👋 DASHBOARD SESSION ENDED
echo =========================================================
echo.
echo Thank you for using Fashion E-commerce Analytics Platform!
echo.
echo 📊 Session completed successfully
echo 🔄 Run this script again anytime to restart
echo 📚 Check README.md for more information
echo.

pause
