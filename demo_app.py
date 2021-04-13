from __future__ import annotations

import time
from typing import Dict, List, Tuple, Any, Optional, Callable, Set, NewType
import json
import os
import sys
import pathlib
import itertools
from functools import reduce


import streamlit as st
import pandas as pd
from streamlit import write as wr
# import psycopg2
from decouple import config

from streamlit import sidebar as sbar
from streamlit.report_thread import get_report_ctx

from src.orchastation import LoadUnifiedContext, StoreUnifiedContext, PageManager, Page
from src.specializedpages import HomePage, StockPickerPage, StockAllocationPage
from src.errors import Status, Success, Failure
from src.stockmon import Stocks


@st.cache()  # type: ignore
def get_data() -> pd.DataFrame:
    return pd.read_csv("__data_file/C_hist_data.csv")


stock_data = Stocks()

st.title("ToTheMoon.alpha")
home_title = "Home"
sp_title = "Stock Picker"
social_title = "Social"

# Construction of the sidebar section


def GenerateSideBar() -> str:
    sbar.title("ToTheMoon.alpha")
    sbar_user_title = sbar.header("Khabane S. Lekena")
    sbar_user_title = sbar.subheader("Ranking: AA")
    selected_window: str = st.sidebar.radio(
        "", (home_title, sp_title, social_title))
    fb, tw, insta = sbar.beta_columns(3)
    with fb:
        st.write("Facebook link")
    with tw:
        st.write("Twitter link")
    with insta:
        st.write("Instagram link")
    return selected_window


# Main Construction
selected_window = GenerateSideBar()
ctx = LoadUnifiedContext()
home_pm = PageManager("HomeManager", ctx)
home_pm.RegisterPage(HomePage("home_page", home_pm))

stock_pick_pm = PageManager("StockPickerManager", ctx)
# TODO assert for duplicate page ids in pagemanager
stock_pick_pm.RegisterPages([StockPickerPage("stock_pick_page_one", stock_pick_pm),
                             StockAllocationPage(
                                 "stock_allocation_page", stock_pick_pm),
                             StockAllocationPage("stock_evaluation_page", stock_pick_pm)])

if selected_window == home_title:
    st.write("Home page")
    home_pm.RenderCurrentPage()
elif selected_window == sp_title:
    st.write("Stock Picking")
    stock_pick_pm.RenderCurrentPage()
elif selected_window == social_title:
    st.write("Social")
else:
    st.error("Unknown section selected!")

StoreUnifiedContext(ctx)
st.balloons()
