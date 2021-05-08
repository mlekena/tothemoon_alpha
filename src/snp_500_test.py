# type: ignore
# mypy: ignore-errors
import src.stock_fetch as stock_fetch
from src.snp_500_dojo import SNPModel

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

TEST_DATA_DIR = "src/test_data"


def test_summoning_fusion_model():
    model = SNPModel(5)
    assert model.size == 5
    assert len(model.tickers) == model.size


def test_fusion_model_single_ticker_prediction():
    model = SNPModel(2)
    threeM_ticker = ("MMM", 203.07000732421875)
    # threeM_data = stock_fetch.GetStockData(threeM_ticker[0], TEST_DATA_DIR)
    abt_ticker = ("ABT", 123.30)
    # abt_data = stock_fetch.GetStockData(abt_ticker[0], TEST_DATA_DIR)
    assert(model.models[threeM_ticker[0]] != None)
    assert(model.models[abt_ticker[0]] != None)

    prediction_result = model.Predict(
        threeM_ticker, stock_fetch.GetStockData(threeM_ticker[0], TEST_DATA_DIR))
    assert(prediction_result > 0)
    prediction_result_abt = model.Predict(
        threeM_ticker, stock_fetch.GetStockData(abt_ticker[0], TEST_DATA_DIR))
    assert(prediction_result_abt > 0)


def test_fusion_model_consistency():
    model = SNPModel(2)

    def call_model():
        return model.Predict("MMM", stock_fetch.GetStockData("MMM", TEST_DATA_DIR))
    assert(call_model() == call_model())


# def test_fusion_model_predict_with_one_member
"""
((stock:price), dollar_amount_bought => ratio_purchased ) -> presult(fprice) -> compare dollar value of fRatio_purchased vs raio_purchased
"""


def test_prediction_flow():
    model = SNPModel(2)
    #                ticker, adj_close,  dollar amount purchased
    threeM_ticker = ("MMM", 203.07000732421875, 500)
    threeMData = ("MMM", stock_fetch.GetStockData("MMM", TEST_DATA_DIR))
    abtData = ("ABT", stock_fetch.GetStockData("ABT", TEST_DATA_DIR))
    abt_ticker = ("ABT", 123.30, 500)
    results = model.PredictProfitFlow([threeMData, abtData])
    assert len(results) == 2
    assert results[0][1] > 0  # MMM
    assert results[1][1] > 0  # ABT
    threeM_stocks = threeM_ticker[2]/threeM_ticker[1]
    abt_stocks = abt_ticker[2]/abt_ticker[1]
    threeM_earning = threeM_stocks*results[0][1]
    abt_earning = abt_stocks*results[1][1]
    # assert results
