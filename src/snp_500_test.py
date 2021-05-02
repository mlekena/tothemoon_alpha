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


def test_summoning_fusion_model():
    model = SNPModel(20)
    assert model.size == 20
    assert len(model.tickers) == model.size
