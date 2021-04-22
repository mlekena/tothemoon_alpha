#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import unittest
# import mock
from . import stock_fetch

import tempfile
import pandas as pd
import pytest
import pytest_benchmark
import os
# from sqlalchemy import (MetaData, Table, Column, Integer, Numeric, String,
#                         DateTime, ForeignKey, Boolean, create_engine, func,
#                         select, update)
from typing import Any

tempDir = os.path.join(tempfile.gettempdir(), "tmp-testfile")
# tempDir = tempfile.gettempdir()


@pytest.fixture
def clear_local_cache():
    pass
    # if os.path.exists(stock_fetch.STOCK_LOCAL_CACHE_PATH):


def test_loading_new_data() -> None:
    df = stock_fetch.GetStockData("AAPL", tempDir)
    stock_file = os.path.join(tempDir, "AAPL.csv")
    assert(os.path.exists(stock_file))
    assert(not df.empty)
    assert(not pd.read_csv(stock_file).empty)

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
