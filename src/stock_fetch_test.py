#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import unittest
# import mock
from . stock_fetch import LoadData, GetStockData
from . ttma_dojo import CreateModel

import tempfile
import pandas as pd
import pytest
import pytest_benchmark
import os
import time
import shutil

from typing import Any, List
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


def GenTempDirPath(file: str) -> str:
    return os.path.join(tempfile.gettempdir(), file)


tempDir = GenTempDirPath("tmp-testfile")
# tempDir = tempfile.gettempdir()


def test_loading_new_data() -> None:
    df = GetStockData("AAPL", tempDir)
    stock_file = os.path.join(tempDir, "AAPL.csv")
    assert(os.path.exists(stock_file))
    assert(not df.empty)
    assert(not pd.read_csv(stock_file).empty)


def test_model_classification() -> None:
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

    test_tickers = ["AMZN", "C", "BOA."]

    test_result_path = GenTempDirPath("test_results")
    test_logs_path = GenTempDirPath("test_logs")
    test_data_path = GenTempDirPath("test_data")

    def ClearAndMkdir(path: str) -> bool:
        assert(not os.path.exists(path))
        shutil.rmtree(path)
        assert(not os.path.exists(path))
        os.mkdir(path)
        return os.path.exists(path)

    def ChecKPathAndRemake(path: str) -> bool:
        if not os.path.isdir(path):
            os.mkdir(path)
            return True
        else:
            return ClearAndMkdir(path)

    ChecKPathAndRemake(test_result_path)
    ChecKPathAndRemake(test_logs_path)
    ChecKPathAndRemake(test_data_path)
    resulting_models: List[Sequential] = []

    for ticker in test_tickers:
        ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
        # model name to save, making it as unique as possible based on parameters
        model_name: str = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
        {LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
        if BIDIRECTIONAL:
            model_name += "-b"

        # load the data
        data = LoadData(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
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
        # model.load_weights(model_path)
        made_model = Sequential()
        made_model.load_weights(model_path)
        resulting_models.append(made_model)
    # class RmTestCase(unittest.TestCase):

    #     @ mock.patch('mymodule.os.path')
    #     @ mock.patch('mymodule.os')
    #     def test_rm(self, mock_os, mock_path):
    #         # set up the mock
    #         mock_path.isfile.return_value = False

    #         rm("any path")

    #         # test that the remove call was NOT called.
    #         self.assertFalse(mock_os.remove.called,
    #                          "Failed to not remove the file if not present.")

    #         # make the file 'exist'
    #         mock_path.isfile.return_value = True

    #         rm("any path")

    #         mock_os.remove.assert_called_with("any path")
