#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os

from typing import Dict, List, Tuple, Any
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import pandas as pd

import src.stock_fetch as stock_fetch
import src.ttma_dojo as ttma_dojo

RESULTS_PATH = "results"
LOGS_PATH = "logs"
DATA_PATH = "data"


def load_smp_tickers() -> List[str]:
    with open("resources/snp_tickers.txt") as ticker_file:
        return list(map(lambda s: s.strip(), ticker_file.readlines()))


"""
TODO: Assume there is one model .h5 file for each ticker
scrap path, grab filenames and make a ticker:path dict. Then
load the weihghts on SNPModel construction with the new weights

TODO decided (we have ) whther to use getmodels assumed paths 
to data

TODO complete prediction function for one 'other' ticker which
creates a matrics of predictions then average across the matrics and 
return the value of the furthest prediction

TODO create function that iterates over all the portfolio tickers

test test test and use in streamlit
would be great to show testing for accuracy
"""


class SNPModel(object):

    def __init__(self, num_members: int = 500):
        self.size = num_members
        self.tickers = load_smp_tickers()[:self.size]

        def MakeModel(ticker: str):
            dmodel = ttma_dojo.DefaultFinModel(ticker)
            dmodel.GetModel()
            dmodel.LoadData()  # Load data using set ticker name
            dmodel.LoadModelWeights(  # TODO Complete loading data and run test on left to pass. Need to complete a usage of the SNPModel
                "2021-05-01_ABT-sh-1-sc-1-sbd-0-huber_loss-adam-LSTM-seq-50-step-92-layers-2-units-256.h5")
            return dmodel
        self.models = dict([(ticker, MakeModel(ticker))
                            for ticker in self.tickers])

    def Predict(self, ticker_to_predict: str, ticker_data: pd.DataFrame) -> float:
        """
            Given a tickers, it will run the ticker
            related model and return the respective predicted
            future price as a ticker:price pair
        """
        predictions: List[pd.Series] = []
        for ticker in self.tickers:
            model_member = self.models[ticker]
            prediction = model_member.Predict(ticker_data)
            predictions.append(prediction)

        assert(len(predictions)
               ), "Not all the listed models contributed to prediction."

        return 0


if __name__ == "__main__":
    print("Loading S&P500")
    snp500_tickers = load_smp_tickers()
    print(f"{len(snp500_tickers)} tickers loaded...")

    # Window size or the sequence length
    N_STEPS = 50
    # Lookup step, 1 is the next day
    LOOKUP_STEP = 92  # days
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
    EPOCHS = 5  # Low but used in testing

    # def ClearAndMkdir(path: str) -> bool:
    #     assert(os.path.exists(path))
    #     shutil.rmtree(path)
    #     assert(not os.path.exists(path))
    #     os.mkdir(path)
    #     return os.path.exists(path)

    def ChecKPathAndRemake(path: str) -> None:
        if not os.path.isdir(path):
            os.mkdir(path)

    ChecKPathAndRemake(RESULTS_PATH)
    ChecKPathAndRemake(LOGS_PATH)
    ChecKPathAndRemake(DATA_PATH)

    # resulting_models: Dict[str, Dict[str, Any]] = dict()
    # trained_models: Dict[str, Dict[str, Any]] = dict()
    for ticker in snp500_tickers[:2]:
        print("{}/nLearning {}/n{}".format("#"*50, ticker, "#"*50))
        ticker_data_filename = os.path.join(
            DATA_PATH, f"{ticker}_{date_now}.csv")
        # model name to save, making it as unique as possible based on parameters
        model_name: str = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-" +\
            f"{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
        if BIDIRECTIONAL:
            model_name += "-b"

        # load the data
        data = stock_fetch.LoadData(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                                    shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                                    feature_columns=FEATURE_COLUMNS)
        # save the dataframe
        data["df"].to_csv(ticker_data_filename)
        # construct the model
        model = ttma_dojo.CreateModel(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                                      dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
        # some tensorflow callbacks
        checkpointer = ModelCheckpoint(os.path.join(
            RESULTS_PATH, model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
        tensorboard = TensorBoard(
            log_dir=os.path.join(LOGS_PATH, model_name))
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
        model_path = os.path.join(RESULTS_PATH, model_name) + ".h5"
        model.load_weights(model_path)
