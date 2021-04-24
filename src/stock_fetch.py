"""
Credit: https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras
"""

# mypy: ignore-errors

from sklearn import preprocessing
from yahoo_fin import stock_info as si
from collections import deque

import os
import numpy as np
import pandas as pd
# import random


def ShuffleInUnison(a, b) -> None:
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


STOCK_LOCAL_CACHE_PATH = "/resources/local_stock_data"


def _InitStockCache(stock_path: str):
    if not os.path.exists(stock_path) or not os.path.isdir(stock_path):
        os.mkdir(stock_path)


def GetStockData(ticker: str, stock_dir=STOCK_LOCAL_CACHE_PATH):
    """
    Loads data from Yahoo finance or from local cache if stored
    Creates local cache path if not previously created.
    """
    _InitStockCache(stock_dir)
    data_path = f"{stock_dir}/{ticker}.csv"
    if not os.path.exists(data_path):
        df = si.get_data(ticker)
        # TODO check df is not empty or errored out
        df.to_csv(data_path)
    else:
        df = pd.read_csv(data_path)
    return df


def LoadData(ticker, n_steps=50,
             scale=True, shuffle=True,
             lookup_step=1, split_by_date=True,
             test_size=0.2, feature_columns=['adjclose', 'volume', "open", "high", "low"]):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
            to False will split datasets in a random way
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """

    if isinstance(ticker, str):
        # load it from yahoo_fin library
        # df = si.get_data(ticker)
        df = GetStockData(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError(
            "ticker can be either a str or a `pd.DataFrame` instances")
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    # add date as a column
    # made the data column the table index
    if "date" not in df.columns:
        df["date"] = df.index
    # If scaling, scale each column and store the column scalar transformer
    # in the column_scaler
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(
                np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    """
    Shifting detail: Given the true label of a price today is the
    price listed `lookup_steps` ahead into the future, we can shift
    the adjclose price by the number of days and create a label column
    from the prices ahead.

    This leaves certain prices at the end of the data set with values of 
    NAN or null since there are no future prices
    """
    df['future'] = df['adjclose'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before dropping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)

    """
    Next we gathera sequence of stock information data related to a 
    future label to be predicted. If we look ahead 10 days, we group
    sequences of 10 previous data to be used to predict the 11th
    """
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values,
                             df['future'].values):  # entry:Row, target: label
        # length of `sequence` never exceeeds n_steps set above
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    """
    At this point sequence deque will have last remaining elements in df (which if you remember are
    all the elements left after calling dropna). Now we extend that sequence with the captured NAN
    rows giving us len(sequence) + len(dropped rows) number of elements. This list comprehension
    essentially converts deque to a list.
    """
    last_sequence = list([s[:len(feature_columns)]
                          for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            ShuffleInUnison(result["X_train"], result["y_train"])
            ShuffleInUnison(result["X_test"], result["y_test"])
    else:
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=test_size, shuffle=shuffle)
    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(
        keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :,
                                          :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :,
                                        :len(feature_columns)].astype(np.float32)
    return result


if __name__ == "__main__":
    print("Fetching example stock ticker: TSLA")

    print(LoadData("TSLA"))
