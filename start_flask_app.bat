@echo off
echo ================================================================================
echo Image Captioning Web Application
echo ================================================================================
echo.
echo Starting Flask application with virtual environment...
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo Checking dependencies...
python -c "import flask; print('  Flask:', flask.__version__)"
python -c "import torch; print('  PyTorch:', torch.__version__)"
echo.

echo Starting server on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

python flask_app\run.py

pause
