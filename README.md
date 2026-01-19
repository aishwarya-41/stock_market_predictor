# Stock Market Predictor

A machine learning–based stock price prediction system that forecasts the **next day closing price** and predicts whether the stock will go **UP or DOWN** using multiple regression, classification, and deep learning models.

The project uses historical stock data, technical indicators, and comparative modeling to evaluate performance across traditional ML models and LSTM neural networks.

---

## Table of Contents

* Overview
* Features
* System Architecture
* Model Details
* Tech Stack
* Setup Instructions
* Usage
* Results Summary
* Limitations
* Contributing
* License

---

## Overview

The Stock Market Predictor is a data science project focused on analyzing historical stock prices and predicting future behavior using machine learning.

The system performs:

* **Regression** → Predicts next day closing price
* **Classification** → Predicts market direction (UP / DOWN)
* **Deep Learning** → Uses LSTM for time-series forecasting

The dataset is fetched using Yahoo Finance (`yfinance`) and feature engineering is applied to create lag features and technical indicators.

Multiple models are trained and compared to understand accuracy, generalization, and stability.

---

## Features

### 1. Data Collection

* Fetches historical stock price data using **Yahoo Finance API**.
* Example stock used: **BIOCON.NS**.

### 2. Feature Engineering

* Lag features (previous closing prices).
* Moving averages.
* Volatility.
* Daily returns.
* Target creation:

  * **Target_Next_Close** → Regression target.
  * **Target_Direction** → Binary classification target (UP / DOWN).

### 3. Regression Models (Price Prediction)

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* LSTM Neural Network

### 4. Classification Models (Direction Prediction)

* Logistic Regression
* Random Forest Classifier
* Feature scaling for performance improvement.

### 5. Evaluation

* R² Score
* Mean Squared Error (MSE)
* Accuracy for classification
* Error analysis on last N samples.

### 6. Sentiment Experiment (Exploratory)

* News sentiment extraction using:

  * NewsAPI
  * NLTK VADER sentiment analyzer
* Attempted to merge news sentiment with stock data.
* Abandoned due to free API date limitations.

---



## System Architecture

### 1. Data Layer

* Yahoo Finance (`yfinance`)
* Historical OHLCV data

### 2. Feature Engineering

* Lag features
* Technical indicators
* Target generation

### 3. Modeling Layer

* Regression models
* Classification models
* LSTM neural network

### 4. Evaluation Layer

* Metrics comparison
* Error analysis
* Last N sample validation

---

## Model Details

### 1. Regression Models

#### Linear Regression

* R² ≈ **0.98**
* Very strong baseline performance.
* Low prediction error on recent samples.

#### Random Forest Regressor

* R² ≈ **0.96**
* Stable predictions with slightly higher error than linear.

#### XGBoost Regressor

* R² ≈ **0.97**
* High accuracy and robust performance.

#### LSTM Neural Network

* Lookback window: **60 timesteps**
* Input shape: `(samples, timesteps, features)`
* Scaling applied using `StandardScaler`
* Final R² ≈ **0.72**
* Last 15 samples R² ≈ **0.32**
* Higher error compared to classical models.

---

### 2. Classification Models

#### Logistic Regression

* Accuracy before scaling ≈ **70%**
* Accuracy after feature scaling ≈ **80%**
* Last 20 sample accuracy ≈ **75%**

#### Random Forest Classifier

* Accuracy ≈ **62%**
* Underperformed compared to Logistic Regression.

---

### 3. Sentiment Integration (Experimental)

* News articles fetched using NewsAPI.
* Sentiment scores computed using VADER.
* Only ~30 days of news available in free tier.
* Stock dataset spans multiple years → **date mismatch**.
* All merged sentiment values became zero → not useful.
* This experiment was documented but excluded from modeling.

---

## Tech Stack

### 1. Data & ML

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

### 2. Deep Learning

* TensorFlow / Keras

### 3. APIs

* Yahoo Finance (yfinance)
* NewsAPI (experimental)


### 4. Environment

* Jupyter Notebook
* Conda virtual environments

---

## Setup Instructions

### Clone Repository

```bash
git clone https://github.com/<your-username>/stock-market-predictor.git
cd stock-market-predictor
```

---

### Create Virtual Environment

```bash
conda create -n stock_env python=3.10
conda activate stock_env
```

---

### Install Dependencies

```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn xgboost tensorflow
```

---

### Launch Notebook

```bash
jupyter notebook
```

Open the `.ipynb` files and run cells sequentially.

---

## Usage

1. Load stock data from Yahoo Finance.
2. Generate features and targets.
3. Split data into training and testing sets.
4. Train regression models for price prediction.
5. Train classification models for direction prediction.
6. Evaluate models using metrics.
7. Visualize prediction performance.
8. Run LSTM notebook for deep learning comparison.

---

## Results Summary

| Model                        | Task                 | Performance    |
| ---------------------------- | -------------------- | -------------- |
| Linear Regression            | Price Prediction     | R² ≈ 0.98      |
| Random Forest Regressor      | Price Prediction     | R² ≈ 0.96      |
| XGBoost Regressor            | Price Prediction     | R² ≈ 0.97      |
| LSTM                         | Price Prediction     | R² ≈ 0.72      |
| Logistic Regression (Scaled) | Direction Prediction | Accuracy ≈ 80% |
| Random Forest Classifier     | Direction Prediction | Accuracy ≈ 62% |

---

## Limitations

* Stock market is inherently unpredictable.
* Models rely only on historical price data.
* No macroeconomic or financial indicators included.
* News sentiment integration limited by API restrictions.
* LSTM performance sensitive to scaling and hyperparameters.

---


## Contributing

Contributions are welcome!

You can:

* Improve feature engineering.
* Add new models.
* Optimize performance.
* Add deployment support.

Fork the repo and submit a pull request.

---

## License

Distributed under the MIT License.

---

