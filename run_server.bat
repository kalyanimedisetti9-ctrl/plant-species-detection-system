@echo off
title Plant Species Detection System
color 0A

echo ================================================================
echo           PLANT SPECIES DETECTION SYSTEM
echo       AI Botanical Classification Web Application
echo ================================================================
echo.

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b
)

echo [2/3] Checking Database and Model Initialization...
if not exist "db.sqlite3" (
    echo Initializing database and deep learning model...
    python seed_database.py
)

echo [3/3] Starting Django Development Server on http://127.0.0.1:8000/
echo.
echo ================================================================
echo  Website:    http://127.0.0.1:8000/
echo  Admin:      http://127.0.0.1:8000/admin/     (admin / admin123)
echo  Demo User:  http://127.0.0.1:8000/login/     (student / student123)
echo ================================================================
echo.
echo Press Ctrl+C in this terminal to stop the server.
echo.

python manage.py runserver 127.0.0.1:8000
pause
