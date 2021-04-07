from __future__ import annotations

import time
import pathlib
import pandas as pd
import psycopg2
from decouple import config
from functools import reduce
from sqlalchemy import create_engine
from typing import Dict, List, Tuple, Any, Optional, Callable, Set, NewType


TEXT_T = "text"
INT_T = "int"
FLOAT_T = "float"

PGUSER = config("DEVPOSTGRESUSER")
PGPASS = config("DEVPOSTGRESPASSWORD")


class Cache(object):
    """ designed to be a lazy evaluate"""
    __instance: "Cache" = None  # type: ignore
    __db_engine = create_engine(
        'postgresql://%s:%s@localhost:5432/postgres' % (PGUSER, PGPASS))

    def __init__(self) -> None:
        print("Calling initi Cache babe")
        if Cache.__instance == None:
            Cache.__instance = self
        else:
            raise RuntimeError(
                "Attempting to create multiple Cache objects. Use get_instance(..) function.")

    @staticmethod
    def get_instance() -> "Cache":
        if not Cache.__instance:
            Cache()
        return Cache.__instance

    def read_state_df(self, session_id: str) -> pd.DataFrame:
        try:
            df = pd.read_sql_table(session_id, con=Cache.__db_engine)
        except:
            df = pd.DataFrame([])
        return df

    def write_state_df(self, df: pd.DataFrame, session_id: str) -> None:
        df.to_sql('%s' % (session_id), Cache.__db_engine, index=False,
                  if_exists='replace', chunksize=1000)

    def write_state(column, value, session_id) -> None:  # type: ignore
        Cache.__db_engine.execute("UPDATE %s SET %s='%s'" %
                                  (session_id, column, value))

    def read_state(column, session_id) -> Any:  # type: ignore
        state_var = Cache.__db_engine.execute(
            "SELECT %s FROM %s" % (column, session_id))
        state_var = state_var.first()[0]
        return state_var

    def InitCache(self, table_id: str, fields_and_types: List[Tuple[str, str]]) -> None:
        print("Creating table {} ({}) and returning early.".format(table_id, reduce(lambda lft, rht:
                                                                                    "{} {} {}".format(
                                                                                        lft, rht[0], rht[1]),
                                                                                    fields_and_types, "")))
        return
        Cache.__db_engine.execute(
            "CREATE TABLE IF NOT EXISTS %s (%s)" %
            (table_id, reduce(lambda lft, rht:
                              "{} {} {}".format(lft, rht[0], rht[1]),
                              fields_and_types, "")))
