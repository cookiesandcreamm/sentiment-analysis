# Twitter Sentiment Analysis

This project performs sentiment analysis on Twitter data using Python. It compares two popular models — **VADER** and **RoBERTa** — and visualizes their performance through various graphs and metrics. The analysis includes detailed commentary embedded in the code, with results displayed as the script executes.

## Dataset

The dataset consists of Twitter posts (tweets) with sentiment labels. Each entry includes the tweet text and its corresponding sentiment classification (positive, negative, or neutral).

## Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone this repository**
   ```bash
   git clone <repository-url>
   cd sentiment-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to perform sentiment analysis and generate visualizations:

```bash
python main.py
```

The script will:
- Load and preprocess the Twitter dataset
- Apply both VADER and RoBERTa models
- Calculate performance metrics
- Generate comparison visualizations
- Display analysis results

## Models Compared

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
A rule-based sentiment analysis tool specifically optimized for social media text. VADER is lightweight, fast, and doesn't require training, making it excellent for real-time analysis of tweets with emojis, slang, and informal language.

### RoBERTa (Robustly Optimized BERT Approach)
A transformer-based deep learning model fine-tuned for sentiment classification. RoBERTa uses contextual understanding and has been pre-trained on large text corpora, providing nuanced sentiment detection for complex language patterns.

## Project Structure

```
sentiment-analysis/
├── main.py              # Main script for preprocessing, analysis, and visualization
├── reviews.csv          # Twitter dataset CSV file
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Features

- **Dual Model Comparison**: Side-by-side evaluation of rule-based vs. deep learning approaches
- **Performance Metrics**: Accuracy, precision, recall, and F1-scores for both models
- **Visualizations**: Charts and graphs comparing model performance
- **Real-time Analysis**: Process and analyze tweets with instant results

## Requirements

Key dependencies include:
- pandas
- numpy
- matplotlib/seaborn (for visualizations)
- vaderSentiment
- transformers (for RoBERTa)
- torch/tensorflow (backend for RoBERTa)

See `requirements.txt` for the complete list.

## Results

The script generates comparative visualizations showing:
- Model accuracy comparison
- Confusion matrices for each model
- Sentiment distribution across the dataset
- Performance metrics breakdown

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
