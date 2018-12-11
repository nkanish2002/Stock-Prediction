'''
Author: Anish Gupta and Harshil Prajapati

This holds code for import data from kaggle and pre-process it for NN and 
XGBoost.
'''
import numpy as np  # linear algebra
import pandas as pd  # data processing
import matplotlib.pyplot as plt  # graphing
import os
from datetime import datetime, timedelta  # Used to subtract days from a date

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

# Import environment
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

# Import training dataset
(market_train, news_train_df) = env.get_training_data()


def process_news_date(news_train_df):
    '''Process news data function'''
    # Define which columns I don't want - I just used my intuition
    # to select these columns
    news_columns_to_drop = ['firstCreated', 'sourceId', 'headline',
                            'takeSequence', 'provider', 'subjects', 'audiences',
                            'bodySize', 'companyCount', 'headlineTag',
                            'sentenceCount', 'assetCodes',
                            'firstMentionSentence', 'noveltyCount12H',
                            'noveltyCount24H', 'noveltyCount3D',
                            'noveltyCount5D', 'noveltyCount7D',
                            'volumeCounts12H', 'volumeCounts24H',
                            'volumeCounts3D', 'volumeCounts5D',
                            'volumeCounts7D']
    # Drop the columns chosen from above
    news_train_df.drop(columns=news_columns_to_drop, inplace=True)
    # Create sentiment word ratio from
    # sentimentWordCount and wordCount <- i think this feature is helpful.
    news_train_df['sentimentWordRatio'] = news_train_df['sentimentWordCount'] / \
        news_train_df['wordCount']
    # Drop sentimentWordCount and wordCount since they are incorporated
    # into the new column sentimentWordRatio now
    news_columns_to_drop = ['wordCount', 'sentimentWordCount']
    news_train_df = news_train_df.drop(columns=news_columns_to_drop)
    # return the news dataframe
    return news_train_df


def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id


def combined_index(df):
    '''
    Separate 'date' into year,month, and day. Then, add year,month, and day 
    to the 'assetName'. Performing this will allow me to merge news & market 
    data with this new 'combined_index' column
    '''
    df['combined_index'] = (df['time'].dt.year).astype(
        str)+(df['time'].dt.month).astype(str)+(df['time'].dt.day).astype(str) \
        + (df['assetName']).astype(str)
    return df


def merge_market_news(market_df, news_df):
    '''Merge market & news data by combined_index'''
    # By having .mean(), it will take average of numeric values if there are
    # duplicate news for the same 'combined_index'
    news_df = news_df.groupby('combined_index').mean()
    # merge news data to market data using the 'combined_index' we created
    market_df = market_df.merge(news_df, how='left', on='combined_index')
    # since there are more items in market data, ther are lots of rows with
    # NaNs, and we fill them with 0 for training purposes.
    fill_na_columns = ['urgency', 'marketCommentary', 'relevance',
                       'sentimentClass', 'sentimentNegative',
                       'sentimentNeutral', 'sentimentPositive',
                       'sentimentWordRatio']
    market_df[fill_na_columns] = market_df[fill_na_columns].fillna(0)
    return market_df


def get_input(market_train, indices):
    X_num = market_train.loc[indices, num_cols].values
    X = {'num': X_num}
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat_cols].values
    y = (market_train.loc[indices, 'returnsOpenNextMktres10'] >= 0).values
    r = market_train.loc[indices, 'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time'].dt.date
    return X, y, r, u, d


cat_cols = ['assetCode']
num_cols = [
    # Raw returns
    'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevRaw10',
    'returnsOpenPrevRaw10',
    # Residualized returns
    'returnsOpenPrevMktres1', 'returnsClosePrevMktres1',
    'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
    'volume', 'close', 'open',
    # Other
    'close_to_open', 'std1', 'std10', 'std_res1', 'std_res10', 'res_raw1',
    # Custom Inputs
    'res_raw10', 'raw_res1', 'raw_res10', 'ma10', 'ma50', 'ma200'
]

# # Custom Inputs
returns = market_train['returnsClosePrevRaw1']
res_returns = market_train['returnsClosePrevMktres1']

df_close_to_open = np.abs(market_train['close'] / market_train['open'])
std1 = np.abs(returns) / np.mean(np.abs(market_train['returnsClosePrevRaw1']))
std10 = np.abs(returns) / \
    np.mean(np.abs(market_train['returnsClosePrevRaw10']))

std_res1 = np.abs(res_returns) / \
    np.mean(np.abs(market_train['returnsClosePrevMktres1']))
std_res10 = np.abs(res_returns) / \
    np.mean(np.abs(market_train['returnsClosePrevMktres10']))

res_raw1 = np.abs(res_returns) / \
    np.mean(np.abs(market_train['returnsClosePrevRaw1']))
res_raw10 = np.abs(res_returns) / \
    np.mean(np.abs(market_train['returnsClosePrevRaw10']))

raw_res1 = np.abs(returns) / \
    np.mean(np.abs(market_train['returnsClosePrevMktres1']))
raw_res10 = np.abs(returns) / \
    np.mean(np.abs(market_train['returnsClosePrevMktres10']))

N = 10
ma10 = np.convolve(returns, np.ones((N,))/N, mode='valid')
ma10_fix = [sum(ma10)/len(ma10)] * 9
ma10 = np.insert(ma10, 0, ma10_fix)

N = 50
ma50 = np.convolve(returns, np.ones((N,))/N, mode='valid')
ma50_fix = [sum(ma50)/len(ma50)] * 49
ma50 = np.insert(ma50, 0, ma50_fix)

N = 200
ma200 = np.convolve(returns, np.ones((N,))/N, mode='valid')
ma200_fix = [sum(ma200)/len(ma200)] * 199
ma200 = np.insert(ma200, 0, ma200_fix)


encoders = [{} for cat in cat_cols]


for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(
        market_train.loc[:, cat].astype(str).unique())}
    market_train[cat] = market_train[cat].astype(
        str).apply(lambda x: encode(encoders[i], x))
    print('Done')

# +1 for possible unknown assets
embed_sizes = [len(encoder) + 1 for encoder in encoders]

# Add our engineered features to dataset
market_train['close_to_open'] = df_close_to_open
market_train['std1'] = std1
market_train['std10'] = std10
market_train['std_res1'] = std_res1
market_train['std_res10'] = std_res10
market_train['res_raw1'] = res_raw1
market_train['res_raw10'] = res_raw10
market_train['raw_res1'] = raw_res1
market_train['raw_res10'] = raw_res1
market_train['ma10'] = ma10
market_train['ma50'] = ma50
market_train['ma200'] = ma200

# Clean Our Data:
market_train[num_cols] = market_train[num_cols].replace(
    [np.inf, -np.inf], np.nan)
market_train[num_cols] = market_train[num_cols].fillna(0)

market_train['std_res1'] = market_train['std_res1'].replace(
    0, np.mean(market_train['std_res1']))
market_train['std_res10'] = market_train['std_res10'].replace(
    0, np.mean(market_train['std_res10']))
market_train['res_raw1'] = market_train['res_raw1'].replace(
    0, np.mean(market_train['res_raw1']))
market_train['res_raw10'] = market_train['res_raw10'].replace(
    0, np.mean(market_train['res_raw10']))
market_train['raw_res1'] = market_train['raw_res1'].replace(
    0, np.mean(market_train['raw_res1']))
market_train['raw_res10'] = market_train['raw_res10'].replace(
    0, np.mean(market_train['raw_res10']))

market_train['returnsOpenPrevMktres1'] = market_train['returnsOpenPrevMktres1'].replace(
    0, np.mean(market_train['returnsOpenPrevMktres1']))
market_train['returnsClosePrevMktres1'] = market_train['returnsClosePrevMktres1'].replace(
    0, np.mean(market_train['returnsClosePrevMktres1']))
market_train['returnsClosePrevMktres10'] = market_train['returnsClosePrevMktres10'].replace(
    0, np.mean(market_train['returnsClosePrevMktres10']))
market_train['returnsOpenPrevMktres10'] = market_train['returnsOpenPrevMktres10'].replace(
    0, np.mean(market_train['returnsOpenPrevMktres10']))

print('scaling numerical columns')
scaler = StandardScaler()
# market_train[num_cols] = scaler.fit_transform(market_train[num_cols])
market_train[num_cols[:-3]] = scaler.fit_transform(market_train[num_cols[:-3]])

market_train_df = market_train


# Process news data
news_train_df = process_news_date(news_train_df)
# Create 'combined_index' for news dataframe
news_train_df = combined_index(news_train_df).copy()
# Create 'combined_index' for market dataframe
market_train_df = combined_index(market_train_df).copy()
# Merge market & news data
market_train_df = merge_market_news(market_train_df, news_train_df)


market_train = market_train_df
train_indices, val_indices = train_test_split(
    market_train.index.values, test_size=0.25, random_state=23)

# r, u and d are used to calculate the scoring metric
X_train, y_train, r_train, u_train, d_train = get_input(
    market_train, train_indices)
X_valid, y_valid, r_valid, u_valid, d_valid = get_input(
    market_train, val_indices)
