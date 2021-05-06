#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import unittest
# import mock
from src.stock_fetch import LoadData, GetStockData
from src.ttma_dojo import CreateModel, DefaultFinModel, Predict, GetFilesInDirectory

import tempfile
import pandas as pd
import pytest
import pytest_benchmark
import os
import time
import shutil

from typing import Any, List, Dict, Callable, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


def GenTempDirPath(file: str) -> str:
    return os.path.join(tempfile.gettempdir(), file)


raw_data_tempDir = GenTempDirPath("tmp-testfile")
# raw_data_tempDir = tempfile.gettempdir()


def test_reading_directories() -> None:
    temp_dir = "test_reading_directories_"
    temp_dir_path = GenTempDirPath(temp_dir)
    files = ["file_1.txt", "file_2.txt", "file_3.txt"]
    if (not os.path.exists(temp_dir_path)):
        os.mkdir(temp_dir_path)
    test_paths = [GenTempDirPath(f) for f in map(
        lambda afile: os.path.join(temp_dir, afile), files)]
    print(test_paths)
    actual_files_in_dir = GetFilesInDirectory(
        GenTempDirPath("test_reading_directories_"), lambda x: x)
    assert(actual_files_in_dir != None)
    for actual_file in actual_files_in_dir:
        assert(actual_file in files)
    if (os.path.exists(temp_dir_path)):
        shutil.rmtree(temp_dir_path)
        assert(not os.path.exists(temp_dir_path)
               ), "Test create files cleaned correctly"


def test_loading_new_data() -> None:
    df = GetStockData("AAPL", raw_data_tempDir)
    stock_file = os.path.join(raw_data_tempDir, "AAPL.csv")
    assert(os.path.exists(stock_file))
    assert(not df.empty)
    assert(not pd.read_csv(stock_file).empty)

# remove 'x' to test training and loading


def xtest_model_classification(benchmark) -> None:  # type: ignore
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

    test_tickers = ["AMZN", "C", "GOOG"]

    test_result_path = GenTempDirPath("test_results")
    test_logs_path = GenTempDirPath("test_logs")
    test_data_path = GenTempDirPath("test_data")

    def ClearAndMkdir(path: str) -> bool:
        assert(os.path.exists(path))
        shutil.rmtree(path)
        assert(not os.path.exists(path))
        os.mkdir(path)
        return os.path.exists(path)

    def CheckPathAndRemake(path: str) -> bool:
        if not os.path.isdir(path):
            os.mkdir(path)
            return True
        else:
            return ClearAndMkdir(path)

    CheckPathAndRemake(test_result_path)
    CheckPathAndRemake(test_logs_path)
    CheckPathAndRemake(test_data_path)

    resulting_models: Dict[str, Dict[str, Any]] = dict()
    trained_models: Dict[str, Dict[str, Any]] = dict()
    for ticker in test_tickers:
        ticker_data_filename = os.path.join(
            test_data_path, f"{ticker}_{date_now}.csv")
        # model name to save, making it as unique as possible based on parameters
        model_name: str = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
        {LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
        if BIDIRECTIONAL:
            model_name += "-b"

        # load the data
        data = LoadData(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                        shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                        feature_columns=FEATURE_COLUMNS, cache_path=raw_data_tempDir)
        # save the dataframe
        data["df"].to_csv(ticker_data_filename)
        # construct the model
        model = CreateModel(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                            dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
        # some tensorflow callbacks
        checkpointer = ModelCheckpoint(os.path.join(
            test_result_path, model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
        tensorboard = TensorBoard(
            log_dir=os.path.join(test_logs_path, model_name))
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
        model_path = os.path.join(test_result_path, model_name) + ".h5"
        # model.load_weights(model_path)
        made_model = CreateModel(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                                 dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
        made_model.load_weights(model_path)
        model_loss, model_mae = model.evaluate(data["X_test"], data["y_test"])
        made_model_loss, made_model_mae = made_model.evaluate(
            data["X_test"], data["y_test"])

        def round_3(num: float) -> float:
            return round(num, 3)
        resulting_models[ticker] = {"model": made_model,
                                    "loss": round_3(made_model_loss), "mae": round_3(made_model_mae)}
        trained_models[ticker] = {"model": model,
                                  "loss": round_3(model_loss), "mae": round_3(model_mae)}
        # trained_models.append(model)

    for ticker in test_tickers:
        assert(resulting_models[ticker]["loss"] ==
               trained_models[ticker]["loss"]), "loss values for ticker {} not equal".format(ticker)
        assert(resulting_models[ticker]["mae"] ==
               trained_models[ticker]["mae"]), "mae values for ticker {} not equal".format(ticker)


def test_defaultmodel_is_empty_one_create() -> None:
    model: DefaultFinModel = DefaultFinModel("GOOG")
    assert model.model == None
    assert model.tensorboard == None
    assert model.data == None

# TODO add test for passing in dataframe with incorrect ticker row value


def test_defaultmodel_loads_data_from_dataframe() -> None:
    model = DefaultFinModel("ABT")
    model.GetModel()
    assert(model.data == None)
    data_df = pd.read_csv("src/test_data/ABT_2021-05-01.csv")
    model.LoadData(data_df)
    assert(model.data != None)


def test_defaultmodel_loads_latest_ticker_model() -> None:
    model = DefaultFinModel("ABT")
    model.LoadData(pd.read_csv("src/test_data/ABT_2021-05-01.csv"))
    assert(model.data), "Test Model data is empty or failed to load."
    assert model.model == None
    model.GetModel()
    assert model.model != None
    test_model_path = os.path.join("src",
                                   "test_models", "2021-05-01_ABT-sh-1-sc-1-sbd-0-huber_loss-adam-LSTM-seq-50-step-92-layers-2-units-256.h5")
    model.LoadModelWeights(test_model_path)
    # TODO Need a data instance variable
    assert model.Predict(model.data['df']) > 0.0
# def test_get_stock_at_price():
#     tickers_and_price = [("MSFT", 100), ("MMM", 200), ("ABT", 300)]
#     stock
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
