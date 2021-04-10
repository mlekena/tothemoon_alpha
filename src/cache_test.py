from . cache import Cache

import pandas as pd
import pytest
import pytest_benchmark
from sqlalchemy import (MetaData, Table, Column, Integer, Numeric, String,
                        DateTime, ForeignKey, Boolean, create_engine)


class TestCache:
    cache = Cache.get_instance("sqlite:///:memory:")
    sp_session_id = "cache_test_id"
    sp_table = Table(sp_session_id,
                     cache.metadata,
                     Column("ticker", String(255), primary_key=True),
                     Column("allocation", Numeric(3, 12)))
    # insert = sp_table.insert()

    def prep_fake_stock_picker_data(self) -> None:
        mock_data = pd.DataFrame({'ticker': [], "allocation": []})
        pass

    def test_stock_picker_data_flow(self) -> None:
        # self.prep_fake_stock_picker_data()
        data = pd.DataFrame({"ticker": ['AAPL', "TSLA", "COIN"],
                             "allocation": [1, 2, 3]})
        TestCache.cache.write_state_df(data, TestCache.sp_session_id)
        assert data.shape == TestCache.cache.read_state_df(
            TestCache.sp_session_id).shape
        assert all(actual == expected for actual, expected in zip(data.columns, TestCache.cache.read_state_df(
            TestCache.sp_session_id).columns))

    def test_stock_picker_data_allocation_setting(self) -> None:
        data = pd.DataFrame({"ticker": ['AAPL', "TSLA", "COIN"],
                             "allocation": [1, 2, 3]})
        # data = pd.DataFrame({"allocation": [0, 0, 0]}, index=[
        #                     'AAPL', "TSLA", "COIN"])
        TestCache.cache.write_state_df(data, TestCache.sp_session_id)
        print("HELLO WORLD")
        print("______{}".format(TestCache.cache.read_state(
            "allocation", TestCache.sp_session_id)))
        assert TestCache.cache.read_state(
            "allocation", TestCache.sp_session_id) == 1  # initial value of AAPL
        data_to_update = TestCache.cache.read_state_df(TestCache.sp_session_id)
        new_update = pd.DataFrame()
        data_to_update["allocation"] = [99, 99, 99]
        for row in data_to_update.itertuples():
            # new_row = row["allocation"]
            # new_update.append(new_row)
            print(row)
        TestCache.cache.write_state_df(data_to_update, TestCache.sp_session_id)
        assert TestCache.cache.read_state(
            "allocation", TestCache.sp_session_id) == 99  # initial value of AAPL
        assert TestCache.cache.read_state_df(
            TestCache.sp_session_id).shape[0] == 3  # initial value of AAPL
