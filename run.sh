#!/bin/bash

# Simple script to run the SII Recommender Backend

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting the backend server..."
python app.py