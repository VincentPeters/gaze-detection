@echo off
echo Starting Eye Contact Detection Application...
echo.
echo Press 'q' to quit the application when running
echo.

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo Warning: Virtual environment not found. Dependencies may not be available.
    echo Run 'python -m venv venv' and 'pip install -r requirements.txt' first.
    pause
    exit /b
)

:: Run the application
python main.py

:: Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo.
echo Application closed.
pause