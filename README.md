# Sentiment Analysis on Amazon Review

This project performs sentiment analysis on Amazon product reviews using Python.
It compares two models — VADER and RoBERTa — and visualizes their performance using various graphs.
The analysis is embedded in the code comments, and results appear as the script runs.

## Dataset

The main dataset consists of Amazon product reviews with sentiment labels.
It includes text reviews and their corresponding sentiment (positive/negative).
At the end, we briefly explore tweets as a secondary example.

## Setup

1. Clone this repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # .\venv\Scripts\activate  # Windows
   ```
3. Install dependencies
   pip install -r requirements.txt

## Usage

Run the main script to perform sentiment analysis and visualize results:
python main.py

## Models Compared

- **VADER**: A rule-based sentiment analysis tool optimized for social media text.
- **RoBERTa**: A transformer-based deep learning model fine-tuned for sentiment classification.

The script calculates metrics and generates graphs to show how each model performs on the dataset.

## Project Structure

sentiment-analysis/
├─ main.py # Script for preprocessing, analysis, and visualization
├─ reviews.csv # Dataset CSV file
├─ requirements.txt # Dependencies
└─ README.md
