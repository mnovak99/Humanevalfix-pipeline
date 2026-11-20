@echo off
rem Double-click to open the pipeline web UI at http://127.0.0.1:8501
rem Close this window to stop the UI. Stop runs from the UI before closing it.
cd /d "%~dp0"
uv run python -m pipeline ui
pause
