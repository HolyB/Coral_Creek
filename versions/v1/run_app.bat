@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting Streamlit app...
python -m streamlit run app.py --server.port 8501
pause


