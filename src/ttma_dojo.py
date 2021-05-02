"""
Credit: https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras
"""
# type: ignore
# mypy: ignore-errors
import src.stock_fetch

import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import os
import numpy as np
import pandas as pd
import random


def CreateModel(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False) -> Sequential:
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True),
                                        batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(
                    None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)
    return model


def plot_graph(test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def get_final_df(model, data):
    """
    This function takes the `model` and `data` dict to 
    construct a final dataframe that includes the features along 
    with true and predicted prices of the testing dataset
    """
    # if predicted future price is higher than the current,
    # then calculate the true future price minus the current price, to get the buy profit
    def buy_profit(current, true_future, pred_future): return true_future - \
        current if pred_future > current else 0
    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price

    def sell_profit(current, true_future, pred_future): return current - \
        true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(
            np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]
                            ["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    # add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit,
                                      final_df["adjclose"],
                                      final_df[f"adjclose_{LOOKUP_STEP}"],
                                      final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                  # since we don't have profit for last sequence, add 0's
                                  )
    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit,
                                       final_df["adjclose"],
                                       final_df[f"adjclose_{LOOKUP_STEP}"],
                                       final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                   # since we don't have profit for last sequence, add 0's
                                   )
    return final_df


def Predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[
            0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


if __name__ == "__main__":
    # Window size or the sequence length
    N_STEPS = 50
    # Lookup step, 1 is the next day
    LOOKUP_STEP = 15  # days
    # whether to scale feature columns & output price as well
    SCALE = True
    scale_str = f"sc-{int(SCALE)}"
    # whether to shuffle the dataset
    SHUFFLE = True
    shuffle_str = f"sh-{int(SHUFFLE)}"
    # whether to split the training/testing set by date
    SPLIT_BY_DATE = False
    split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
    # test ratio size, 0.2 is 20%
    TEST_SIZE = 0.2
    # features to use
    FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
    # date now
    date_now = time.strftime("%Y-%m-%d")
    # model parameters
    N_LAYERS = 2
    # LSTM cell
    CELL = LSTM
    # 256 LSTM neurons
    UNITS = 256
    # 40% dropout
    DROPOUT = 0.4
    # whether to use bidirectional RNNs
    BIDIRECTIONAL = False
    # training parameters
    # mean absolute error loss
    # LOSS = "mae"
    # huber loss
    LOSS = "huber_loss"
    OPTIMIZER = "adam"
    BATCH_SIZE = 64
    EPOCHS = 500
    # Amazon stock market
    ticker = "AMZN"
    ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
    # model name to save, making it as unique as possible based on parameters
    model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-" +\
        f"{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
    if BIDIRECTIONAL:
        model_name += "-b"

    if not os.path.isdir("results"):
        os.mkdir("results")
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    if not os.path.isdir("data"):
        os.mkdir("data")

    # load the data
    data = stock_fetch.LoadData(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                                feature_columns=FEATURE_COLUMNS)
    # save the dataframe
    data["df"].to_csv(ticker_data_filename)
    # construct the model
    model = CreateModel(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(os.path.join(
        "results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    # train the model and save the weights whenever we see
    # a new optimal model using ModelCheckpoint
    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[checkpointer, tensorboard],
                        verbose=1)

    # Load saved model weights
    # load optimal model weights from results folder
    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)

    # evaluate the model
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # calculate the mean absolute error (inverse scaling)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[
            0][0]
    else:
        mean_absolute_error = mae

    final_df = get_final_df(model, data)
    # predict the future price
    future_price = predict(model, data)
    # we calculate the accuracy by counting the number of positive profits
    accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) +
                      len(final_df[final_df['buy_profit'] > 0])) / len(final_df)
    # calculating total buy & sell profit
    total_buy_profit = final_df["buy_profit"].sum()
    total_sell_profit = final_df["sell_profit"].sum()
    # total profit by adding sell & buy together
    total_profit = total_buy_profit + total_sell_profit
    # dividing total profit by number of testing samples (number of trades)
    profit_per_trade = total_profit / len(final_df)

    # printing metrics
    print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
    print(f"{LOSS} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    print("Accuracy score:", accuracy_score)
    print("Total buy profit:", total_buy_profit)
    print("Total sell profit:", total_sell_profit)
    print("Total profit:", total_profit)
    print("Profit per trade:", profit_per_trade)

    print(final_df.tail(10))
    # save the final dataframe to csv-results folder
    csv_results_folder = "csv-results"
    if not os.path.isdir(csv_results_folder):
        os.mkdir(csv_results_folder)
    csv_filename = os.path.join(csv_results_folder, model_name + ".csv")
    final_df.to_csv(csv_filename)

DEFAULT_LOGS = os.path.join("logs")
DEFAULT_RESULTS = os.path.join("results")


class DefaultFinModel(object):

    def __init__(self, ticker):
        # Window size or the sequence length
        self.N_STEPS = 50
        # Lookup step, 1 is the next day
        self.LOOKUP_STEP = 15  # days
        # whether to scale feature columns & output price as well
        self.SCALE = True
        scale_str = f"sc-{int(self.SCALE)}"
        # whether to shuffle the dataset
        self.SHUFFLE = True
        shuffle_str = f"sh-{int(self.SHUFFLE)}"
        # whether to split the training/testing set by date
        self.SPLIT_BY_DATE = False
        split_by_date_str = f"sbd-{int(self.SPLIT_BY_DATE)}"
        # test ratio size, 0.2 is 20%
        self.TEST_SIZE = 0.2
        # features to use
        self.FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
        # date now
        date_now = time.strftime("%Y-%m-%d")
        # model parameters
        self.N_LAYERS = 2
        # LSTM cell
        self.CELL = LSTM
        # 256 LSTM neurons
        self.UNITS = 256
        # 40% dropout
        self.DROPOUT = 0.4
        # whether to use bidirectional RNNs
        self.BIDIRECTIONAL = False
        # training parameters
        # mean absolute error loss
        # LOSS = "mae"
        # huber loss
        self.LOSS = "huber_loss"
        self.OPTIMIZER = "adam"
        self.BATCH_SIZE = 64
        self.EPOCHS = 500
        self.ticker = ticker.upper()
        del ticker
        # ticker_data_filename = os.path.join(
        #     "data", f"{self.ticker}_{date_now}.csv")
        # model name to save, making it as unique as possible based on parameters
        self.model_name = f"{date_now}_{self.ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-" +\
            f"{self.LOSS}-{self.OPTIMIZER}-{self.CELL.__name__}-seq-{self.N_STEPS}-step-{self.LOOKUP_STEP}-layers-{self.N_LAYERS}-units-{self.UNITS}"
        if self.BIDIRECTIONAL:
            self.model_name += "-b"
        self.data = None
        self.model = None
        self.checkpointer = None
        self.tensorboard = None

    def GetCheckPointer(self):
        # some tensorflow callbacks
        if not self.checkpointer:
            checkpointer = ModelCheckpoint(os.path.join(
                "results", self.model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
        return self.checkpointer
        self.tensorboard = None

    def GetTensorBoard(self):
        if not self.tensorboard:
            tensorboard = TensorBoard(
                log_dir=os.path.join(DEFAULT_LOGS, self.model_name))
        return self.tensorboard

    def LoadData(self, data_df=pd.DataFrame()):
        if data_df.empty:
            # load the data
            self.data = stock_fetch.LoadData(ticker, self.N_STEPS, scale=self.SCALE, split_by_date=self.SPLIT_BY_DATE,
                                             shuffle=self.SHUFFLE, lookup_step=self.LOOKUP_STEP, test_size=self.EST_SIZE,
                                             feature_columns=self.FEATURE_COLUMNS)
        else:
            self.data = stock_fetch.LoadData(data_df, self.N_STEPS, scale=self.SCALE, split_by_date=self.SPLIT_BY_DATE,
                                             shuffle=self.SHUFFLE, lookup_step=self.LOOKUP_STEP, test_size=self.EST_SIZE,
                                             feature_columns=self.FEATURE_COLUMNS)

        # save the dataframe
        # data["df"].to_csv(ticker_data_filename)

    def GetModel(self):
        if not self.model:
            # construct the model
            self.model = CreateModel(self.N_STEPS, len(self.FEATURE_COLUMNS), loss=self.LOSS, units=self.UNITS, cell=self.CELL, n_layers=self.N_LAYERS,
                                     dropout=self.DROPOUT, optimizer=self.OPTIMIZER, bidirectional=self.BIDIRECTIONAL)
        return self.model

    def LoadModelWeights(self, model_weight_path):
        assert(self.model), "call GetModel to instantiate model object"
        self.model.load_weights(model_weight_path)
