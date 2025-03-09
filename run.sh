#!/bin/bash

echo "Starting Eye Contact Detection Application..."
echo ""
echo "Press 'q' to quit the application when running"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    # Activate virtual environment
    source venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found. Dependencies may not be available."
    echo "Run 'python3 -m venv venv' and 'pip install -r requirements.txt' first."
    read -p "Press Enter to exit..."
    exit 1
fi

# Run the application
python main.py

# Deactivate virtual environment
deactivate

echo ""
echo "Application closed."
read -p "Press Enter to exit..."