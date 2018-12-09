# Stock-Prediction

Kaggle's Two Sigma: Using News to Predict Stock Movements

## Installation

### Installing dependencies

`pip3 install -r requirements.txt`

### Setting up Kaggle

```sh
export KAGGLE_USERNAME=datadinosaur
export KAGGLE_KEY=xxxxxxxxxxxxxx

kaggle config set -n competition -v two-sigma-financial-news
```

## Running Predictors

```sh
# Running Neural Network
python3 prediction/NN.py

## Running XGBoost
python3 prediction/XGBoost.py
```