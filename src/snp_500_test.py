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
    threeM_ticker = ("MMM", 200)
    abt_ticker = ("ABT", 300)
    assert(model.models[threeM_ticker[0]] != None)
    assert(model.models[abt_ticker[0]] != None)

    prediction_result = model.Predict(
        threeM_ticker, stock_fetch.GetStockData(threeM_ticker[0], TEST_DATA_DIR))
    assert(prediction_result > 0)

# def test_fusion_model_predict_with_one_member
