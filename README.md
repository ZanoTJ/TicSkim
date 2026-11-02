<<<<<<< HEAD
# Stock Price Predictor with Reddit Sentiment Analysis

A machine learning project that predicts a stock's closing price using historical market data and sentiment extracted from Reddit posts. The project combines **natural language processing (NLP)** with a **bidirectional Long Short-Term Memory (BiLSTM)** neural network to explore whether community sentiment can provide additional signals for stock-price prediction.

The implementation is provided as a Jupyter notebook designed primarily for **Google Colab**.

## Overview

The notebook builds an end-to-end experimental pipeline:

1. Load historical stock data from Yahoo Finance using a ticker symbol, or upload a CSV file.
2. Collect ticker-related Reddit posts using an external `historical` helper module.
3. Classify Reddit post titles with a pretrained RoBERTa sentiment model.
4. Aggregate sentiment by date and construct a sentiment feature weighted by post upvotes.
5. Combine daily sentiment with historical stock features.
6. Create five-day input sequences and train a BiLSTM regression model.
7. Evaluate closing-price predictions using root mean squared error (RMSE) and plot predicted versus actual values.
8. Generate a sample prediction using recent records from the processed dataset.

## Tech Stack

| Area | Tools / Libraries |
| --- | --- |
| Language and environment | Python, Jupyter Notebook, Google Colab |
| Market data | `yfinance`, CSV input |
| Reddit data | `praw`, external `historical` module |
| NLP | Hugging Face `transformers`, `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Deep learning | TensorFlow / Keras, bidirectional LSTM |
| Data processing | pandas, NumPy, scikit-learn |
| Visualization | Matplotlib, Seaborn |

## Model Architecture

The regression network uses the following layers:

```text
Input: 5 consecutive trading records × 5 features
    ↓
Bidirectional LSTM (64 units, ReLU, return_sequences=True)
    ↓
Bidirectional LSTM (32 units)
    ↓
Dense (32 units, ReLU)
    ↓
Dense (16 units, ReLU)
    ↓
Dense (1 unit: predicted closing price)
```

**Input features:** `Open`, `High`, `Low`, `Volume`, and `sentiment`.

**Target:** the next record's `Close` price.

**Training configuration in the notebook:**

- Sequence length: 5 trading records
- Train/test split: 75% / 25% (`random_state=42`)
- Optimizer: Adam (`learning_rate=0.0005`)
- Loss: mean squared error (MSE)
- Maximum epochs: 100
- Early stopping: validation loss, `patience=5`, `min_delta=0.001`
- Best-model checkpoint: `dnn/rnn.keras`
- Evaluation metric: RMSE

> **Important:** The notebook currently uses a shuffled random train/test split rather than a chronological split. For a realistic forecasting evaluation, split observations in time order and fit normalization parameters using training data only.

## Sentiment Analysis Pipeline

The notebook analyzes Reddit post **titles**, using the pretrained Hugging Face model:

[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

The model returns a sentiment label (`positive`, `neutral`, or `negative`) and confidence score. The notebook converts these outputs into a numerical sentiment value:

- **Positive:** `+score`
- **Negative:** `-score`
- **Neutral:** `score - 0.5`

For each post date, it calculates mean upvotes and mean sentiment. It then constructs the feature:

```python
log_weighted_sentiment = sentiment_value * np.log(Upvotes)
```

The daily feature is merged with the historical stock-price dataset. Trading days without a matching sentiment record are assigned a sentiment value of `0`.

> **Caution:** The logarithm requires positive upvote counts to avoid invalid or non-finite values. The sentiment feature is an experimental heuristic, not a calibrated financial sentiment index.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ZanoTJ/TicSkim.git
cd TicSkim
```

Replace the placeholder URL with your own repository URL.

### 2. Install dependencies

In Google Colab or a Jupyter environment, install:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn yfinance praw transformers torch tensorflow
```

The `historical` module imported by the notebook is **not included in the supplied notebook**. You must provide a compatible `historical.py` implementation or replace its Reddit-data collection step. Depending on that module, Reddit API credentials and additional setup may be required.

### 3. Create a `config.py` file with your API keys:
```python
# Alpha Vantage API
API_KEY = "your_alphavantage_api_key"

# Reddit API
CLIENT_ID = "your_reddit_client_id"
CLIENT_SECRET = "your_reddit_client_secret"
```

### 4. Open the notebook

Open **`Ticker_Stock_Predictor.ipynb`** in Google Colab or Jupyter. The CSV upload flow uses `google.colab.files`, so Google Colab is the easiest environment for that option.

### 5. Load stock data

Choose **one** of the two notebook data-loading paths:

**Option A — Yahoo Finance:**

```python
df = loadDFbyTicker()
```

Enter a ticker symbol when prompted (for example, `AAPL`). The function downloads the maximum available price history and requires at least 100 rows.

**Option B — CSV upload:**

```python
df = loadDFbyCSV()
```

Upload a CSV with compatible historical-price columns, including `Open`, `High`, `Low`, `Close`, and `Volume`, and a date-based index suitable for the later date merge. The supplied CSV loader does not automatically parse a date column or establish the index, so you may need to adapt it for your file.

> Both options appear as executable cells in the notebook. Running both will overwrite `df`; use only the method you want.

### 6. Prepare Reddit sentiment data

The notebook calls:

```python
import historical
historical.launch(global_ticker)
```

It expects this step to produce a file named:

```text
<TICKER>_reddit_data.csv
```

with at least the following columns:

| Column | Description |
| --- | --- |
| `Title` | Reddit post title to classify |
| `Post_Date` | Post date to join with historical prices |
| `Upvotes` | Upvote count for sentiment weighting |

If you already have a compatible Reddit CSV, you can skip the collection call and place the file where the notebook expects it. Ensure post dates align with trading dates and the date column is parseable.

### 7. Run preprocessing, training, and evaluation

Continue through the notebook cells to compute sentiment values, aggregate daily features, generate five-record sequences, train the neural network, and display RMSE and a prediction chart.

Create the model checkpoint directory before training:

```python
import os
os.makedirs('dnn', exist_ok=True)
```

## Known Issues and Limitations

The uploaded notebook is an **experimental prototype** and has a few points to address before claiming a fully reproducible result:

1. **Undefined variable:** In the sentiment aggregation cell, `combined_df.groupby('Post_Date')` references `combined_df`, which is not defined in the provided notebook. It appears intended to be `grouped_df.groupby('Post_Date')`.
2. **Missing Reddit helper:** `historical` is imported but its source code is not included in this notebook; the collection step cannot run without it or a compatible replacement CSV.
3. **CSV format assumptions:** The CSV loader does not explicitly parse dates or set a date index, although the feature-merging step expects date-indexed stock data.
4. **Time-series evaluation:** The random split mixes dates between training and testing. A chronological split is preferable to estimate performance on future prices.
5. **Data leakage risk:** Z-score statistics are computed across the full dataset before splitting. Fit scalers on the training partition and apply them to validation/test observations.
6. **Checkpoint directory:** `dnn/` must exist before saving `dnn/rnn.keras`.
7. **Validation and test overlap:** The test partition is used for validation during training and for the final RMSE; use a separate held-out test partition for unbiased final evaluation.
8. **Sentiment/date coverage:** Reddit post dates, missing records, and non-positive upvote counts require careful handling.
9. **Prediction scope:** The sample “real implementation” predicts a held-out position from the existing processed dataset rather than fetching live inputs and making an independently verified future-market forecast.

## Results

The notebook calculates RMSE as:

```python
rmse = np.sqrt(metrics.mean_squared_error(pred, y_test))
```

It also plots the model's predictions against actual closing prices. **No verified numerical RMSE is reported in this README**, since a reproducible benchmark, dataset/ticker specification, and finalized evaluation protocol have not been established from the supplied notebook alone.

## Future Improvements

- Use a chronological train/validation/test split and walk-forward evaluation.
- Compare stock-only inputs against stock-plus-sentiment inputs to quantify the sentiment feature's contribution.
- Add baselines such as a naive last-close predictor and standard regression models.
- Package the Reddit collection helper, dependency versions, and sample data schema for reproducibility.
- Handle market holidays, post timestamps, after-market information, missing sentiment, and upvote edge cases.
- Add error metrics such as MAE and MAPE (where appropriate), plus directional accuracy.
- Add repeatable inference scripts or a simple user interface.

## Disclaimer

This project is for **education and research**, not investment advice. Stock-price predictions are uncertain and should not be used as the sole basis for trading decisions.
=======
# TicSkim
A stock prediction RNN model with traditional way + sentiment analysis through HuggingFace
>>>>>>> 0c3658b85bb485729406dec411db22031f72872f
