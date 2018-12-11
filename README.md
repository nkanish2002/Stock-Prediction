# Stock-Prediction

Kaggle's Two Sigma: Using News to Predict Stock Movements

![Poster](/docs/poster.png?raw=true "Stock-Prediction")

## Installation

### Installing dependencies

`pip3 install -r requirements.txt`

### Setting up Kaggle

```sh
export KAGGLE_USERNAME=datadinosaur
export KAGGLE_KEY=xxxxxxxxxxxxxx

kaggle config set -n competition -v two-sigma-financial-news
```

## Running NN and XGBOOST

```sh
# Running Neural Network
python3 NN_and_XGBOOST/NN.py

## Running XGBoost
python3 NN_and_XGBOOST/XGBoost.py
```

## Running LightGBM and Random classifier

Run `jupyter notebook .` and got to respective folders to access Jupyter Notebooks.