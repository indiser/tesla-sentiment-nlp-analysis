# 📊 Tesla Stock & Elon Musk Tweet Sentiment Analysis

<div align="center">

**Uncovering the Impact of Social Media Sentiment on Stock Market Performance**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg)](#)

</div>

---

## 📌 Overview

This sophisticated data science project analyzes the **quantifiable relationship between Elon Musk's tweet sentiment and Tesla (TSLA) stock price movements**. Through advanced NLP-based sentiment analysis and statistical correlation studies, we uncover how social media activity influences market behavior—a critical insight for traders, researchers, and investors.

**Key Insight:** Does the tone of a tweet from one of the world's most influential figures move billions in market cap?

---

## 🎯 Project Objectives

✅ Extract and preprocess comprehensive tweet datasets from Elon Musk's Twitter history  
✅ Perform polarity-based sentiment analysis using TextBlob NLP  
✅ Correlate daily tweet sentiment scores with TSLA stock price movements  
✅ Identify patterns in extreme market days and their relationship to social media activity  
✅ Visualize findings with professional-grade charts and statistical insights  

---

## 📂 Project Structure

```
├── Stock_Sentiment.ipynb           # Main analysis notebook (Jupyter)
├── stock_generator.py              # Data fetching script (yfinance)
├── all_musk_posts.csv              # Complete tweet dataset (~thousands of tweets)
├── daily_tweets.csv                # Aggregated daily sentiment metrics
├── extreme_days_analysis.csv       # Deep-dive into volatile market days
├── musk_quote_tweets.csv           # Isolated quote tweet analysis
├── Tsla_stock_price.csv            # 20-year historical TSLA price data
└── Readme.md                       # This file
```

---

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Sentiment Analysis** | TextBlob (NLP) |
| **Visualization** | Matplotlib, Seaborn |
| **Market Data** | yFinance (Yahoo Finance API) |
| **Notebook** | Jupyter |
| **Analysis** | Statistical Correlation, Time-Series Analysis |

---

## 📊 Methodology

### 1️⃣ **Data Collection**
- **Tweet Data:** Extracted comprehensive Elon Musk Twitter dataset (all_musk_posts.csv)
- **Stock Data:** 20-year historical TSLA pricing via yFinance API
- **Temporal Alignment:** All data indexed by date for correlation analysis

### 2️⃣ **Data Preprocessing & Cleaning**
```python
# Removed irrelevant columns (retweet counts, view counts, etc.)
# Converted timestamps to standard date format
# Aggregated daily metrics for comparative analysis
```
- Stripped engagement metrics and duplicative fields
- Normalized temporal data across both datasets
- Created daily aggregation views for sentiment-vs-price correlation

### 3️⃣ **Sentiment Analysis**
- **Algorithm:** Polarity scoring (TextBlob) → -1 (negative) to +1 (positive)
- **Scope:** Full text analysis of each tweet
- **Output:** Continuous sentiment scores for statistical analysis
- **Daily Metrics:** Average daily sentiment polarity

### 4️⃣ **Correlation & Insights**
- Computed Pearson correlation between daily sentiment and:
  - Daily closing price
  - Trading volume
  - Price volatility (extreme days analysis)
- Statistical significance testing
- Time-lag analysis (does today's sentiment predict tomorrow's movement?)

### 5️⃣ **Visualization & Reporting**
- Time-series plots of sentiment vs. stock price
- Distribution analysis and heatmaps
- Extreme day pattern identification
- Statistical summary metrics

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/indiser/tesla-sentiment-nlp-analysis.git
cd tesla-sentiment-nlp-analysis
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas numpy matplotlib seaborn textblob yfinance jupyter
```

4. **Run sentiment analysis:**
```bash
jupyter notebook Stock_Sentiment.ipynb
```

---

## 📈 Key Findings & Analysis

The notebook performs comprehensive exploratory data analysis including:

- **Tweet Volume Trends:** When does Elon tweet most frequently?
- **Sentiment Distribution:** Are his tweets predominantly positive or negative?
- **Market Impact:** Quantifiable correlation between sentiment and price movements
- **Extreme Days:** Tweet activity on days with ±5%+ stock movement
- **Time-Lag Effects:** Do tweets predict future price movements?

---

## 📁 Dataset Descriptions

| File | Records | Purpose |
|------|---------|---------|
| `all_musk_posts.csv` | ~16,000+ | Complete tweet repository with metadata |
| `daily_tweets.csv` | ~3,000+ days | Daily aggregated sentiment metrics |
| `extreme_days_analysis.csv` | Volatile days | Deep analysis of high-movement days |
| `musk_quote_tweets.csv` | ~500+ | Isolated quote tweet analysis |
| `Tsla_stock_price.csv` | ~5,000+ days | 20 years of TSLA OHLCV data |

---

## 💡 Technologies Demonstrated

### Data Science
- Time-series analysis
- Sentiment analysis & NLP
- Statistical correlation analysis
- Data visualization & storytelling

### Python Skills
- Pandas data manipulation
- NumPy numerical operations
- Matplotlib/Seaborn visualization
- Jupyter notebook best practices

### Domain Knowledge
- Financial markets & technical analysis
- API integration (yFinance)
- ETL pipeline design
- Exploratory data analysis (EDA)

---

## 📊 Sample Analysis Code

```python
# Sentiment Analysis Pipeline
from textblob import TextBlob
import pandas as pd

# Load tweet data
df = pd.read_csv('all_musk_posts.csv')

# Calculate polarity score (ranges from -1 to 1)
df['Sentiment'] = df['fullText'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Aggregate by date
daily_sentiment = df.groupby('createdAt')['Sentiment'].mean()

# Correlate with stock prices
correlation = daily_sentiment.corr(stock_df.set_index('Date')['Close'])
print(f"Correlation Coefficient: {correlation:.4f}")
```

---

## 🔍 How to Interpret Results

- **Correlation > 0.5:** Strong positive relationship (sentiment ↑ → price ↑)
- **Correlation -0.5 to 0.5:** Weak relationship (noise, other factors dominant)
- **Correlation < -0.5:** Strong inverse relationship (sentiment ↓ → price ↑)

---

## 🎓 Learning Outcomes

By studying this project, you'll gain expertise in:

✨ **Data Engineering:** Handling real-world, messy datasets  
✨ **NLP:** Sentiment analysis with Python  
✨ **Statistical Analysis:** Correlation, regression, time-series  
✨ **Data Visualization:** Communicating insights effectively  
✨ **Finance:** Understanding market dynamics and APIs  

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Advanced sentiment models (VADER, Hugging Face Transformers)
- [ ] Machine learning predictions (LSTM, Prophet)
- [ ] Real-time sentiment streaming
- [ ] Interactive Plotly/Dash dashboards
- [ ] Statistical hypothesis testing

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact & Support

**Questions or suggestions?** Feel free to reach out or open an issue!

---

## 🙏 Acknowledgments

- **Data Source:** Twitter API, yFinance
- **Libraries:** Pandas, TextBlob, Matplotlib
- **Inspiration:** The undeniable influence of social media on modern markets

3.  **Generate the stock data**:
    Run the `stock_generator.py` script to fetch the latest TSLA stock data:
    ```bash
    python stock_generator.py
    ```

4.  **Run the analysis**:
    Open and run the `Stock_Sentiment.ipynb` Jupyter Notebook to perform the sentiment analysis and see the results.

## Key Findings

The analysis in the `Stock_Sentiment.ipynb` notebook reveals insights into the following:
- The daily volume of Elon Musk's tweets and their corresponding sentiment scores.
- The correlation between tweet sentiment and TSLA stock price changes.
- Identification of specific dates where high-sentiment tweets may have coincided with significant stock price movements.
- The `extreme_days_analysis.csv` file contains a summary of tweet activity on days with unusual stock market performance.

## Future Work

- **Advanced Sentiment Analysis**: Use more sophisticated sentiment analysis models (e.g., transformer-based models like BERT) for more accurate sentiment scoring.
- **Time-Lag Analysis**: Investigate the time lag between a tweet and its potential impact on the stock price.
- **Wider Range of Stocks**: Extend the analysis to other companies or assets that may be influenced by Elon Musk's tweets.

This project serves as a foundational analysis of the "Musk Effect" on the stock market. For any questions or contributions, please feel free to open an issue or submit a pull request.
