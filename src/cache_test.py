from . cache import Cache

import pandas as pd
import pytest
import pytest_benchmark
from sqlalchemy import (MetaData, Table, Column, Integer, Numeric, String,
                        DateTime, ForeignKey, Boolean, create_engine, func,
                        select, update)
from typing import Any

IN_MEMORY = "sqlite:///:memory:"


def _GenTable(id: str, metadata: MetaData) -> Table:
    return Table(id, metadata,
                 Column("ticker", String(
                        255), primary_key=True),
                 Column("allocation", Numeric(3, 12)))


class TestCache:
    sp_session_id = "_id_cache_test_id"
    # sp_table = Table(sp_session_id,
    #                  Cache.metadata,
    #                  Column("ticker", String(255), primary_key=True),
    #                  Column("allocation", Numeric(3, 12)))

    # insert = sp_table.insert()

    cache = Cache.get_instance("sqlite:///:memory:")
    cache.RegisterCacheTable(sp_session_id, _GenTable)
    cache.CreateRegisteredTablesAndConnect()

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
            print(row[1])
        TestCache.cache.write_state_df(data_to_update, TestCache.sp_session_id)
        assert TestCache.cache.read_state(
            "allocation", TestCache.sp_session_id) == 99  # initial value of AAPL
        assert TestCache.cache.read_state_df(
            TestCache.sp_session_id).shape[0] == 3  # initial value of AAPL
        # assert False

    def test_registering_(self):
        ctx = Cache.get_instance(IN_MEMORY)
        table_id = "_id_temp_user_table"

        def t_func(table_id, metadata) -> Table:
            table = Table(table_id,
                          metadata,
                          Column("user_id", String(
                              320), primary_key=True),
                          Column("current_page", String(255)))
        table = ctx.RegisterCacheTable(table_id, t_func)
        result = ctx.connection(table.select(func.count(table.c.user_id)))
        assert result.scalar() == 0

    def test_setting_data_in_cache(self) -> None:
        ctx = Cache.get_instance(IN_MEMORY)
        table_id = "_id_temp_user_table"

        def t_func(table_id: str, metadata: MetaData) -> Table:
            table = Table(table_id,
                          metadata,
                          Column("user_id", String(
                              320), primary_key=True),
                          Column("current_page", String(255)))
        table = ctx.RegisterCacheTable(table_id, t_func)

        ctx.CacheDataUpdate(update(table).where(
            user_id="new_user").values(current_page=("new_page")))


class MockUnifiedContextCache:
    connection = None
    engine = None  # type: ignore
    unified_context_id = "_id_unified_context_cache_test_id"
    metadata = MetaData()
    uc_table = Table(unified_context_id,
                     metadata,
                     Column("user_id", String(320), primary_key=True),
                     Column("current_page", String(255)))

    def df_init(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        # https://stackoverflow.com/questions/45045147/sqlalchemy-exc-unboundexecutionerror-table-object-responsibles-is-not-bound-t
        self.metadata.bind = self.engine
        self.metadata.create_all(self.engine)
        self.connection = self.engine.connect()


class TestUnifiedContextCache:
    muc = MockUnifiedContextCache()
    # @classmethod

    def GetTable(self) -> Table:
        return self.muc.uc_table

    def GetConn(self) -> Any:
        return self.muc.connection

    def setup_method(self) -> None:
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        print("MUC init")
        self.muc.df_init()

    # @classmethod
    def teardown_method(self) -> None:
        """ teardown any state that was previously setup with a call to
        setup_class.
        """
        print("MUC Connection Closing")
        self.muc.connection.close()  # type: ignore

    def test_check_table_exists(self) -> None:
        assert self.GetTable().exists()

    def test_empty_unified_context_table_is_zero(self) -> None:
        result = self.GetConn().execute(select(func.count(self.GetTable().c.user_id)))
        assert result.scalar() == 0

    def test_setting_current_page(self) -> None:
        result = self.GetConn().execute(
            self.GetTable().insert(),
            user_id="test_id",
            current_page="test_page")
        assert result.inserted_primary_key[0] == "test_id"
        result2 = self.GetConn().execute(select(func.count(self.GetTable().c.user_id)))
        assert result2.scalar() == 1
        assert self.GetConn().execute(
            select([self.GetTable()])).first().current_page == "test_page"

        # HOW TO DO A TABLE UPDATE
        self.GetConn().execute(
            update(self.GetTable())
            .where(self.GetTable().c.user_id == "test_id")
            .values(current_page=("a_new_page")))
        assert self.GetConn().execute(
            select([self.GetTable()])).first().current_page == "a_new_page"

        # assert self.GetTable().c["current_page"] == "test_page"
        # need to understand accessing Column object data
