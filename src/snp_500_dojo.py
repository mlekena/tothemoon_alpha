#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os

from typing import Dict, List, Tuple, Any
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import src.stock_fetch
import src.ttma_dojo


def load_smp_tickers() -> List[str]:
    with open("resources/snp_tickers.txt") as ticker_file:
        return list(map(lambda s: s.strip(), ticker_file.readlines()))


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

    result_path = "results"
    logs_path = "logs"
    data_path = "data"
    ChecKPathAndRemake(result_path)
    ChecKPathAndRemake(logs_path)
    ChecKPathAndRemake(data_path)

    # resulting_models: Dict[str, Dict[str, Any]] = dict()
    # trained_models: Dict[str, Dict[str, Any]] = dict()
    for ticker in snp500_tickers[:2]:
        print("{}/nLearning {}/n{}".format("#"*50, ticker, "#"*50))
        ticker_data_filename = os.path.join(
            data_path, f"{ticker}_{date_now}.csv")
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
            result_path, model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
        tensorboard = TensorBoard(
            log_dir=os.path.join(logs_path, model_name))
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
        model_path = os.path.join(result_path, model_name) + ".h5"
        model.load_weights(model_path)
