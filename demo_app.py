from __future__ import annotations

import time
from typing import Dict, List, Tuple, Any, Optional, Callable, Set, NewType
import uuid
import json
import os
import sys
import pathlib
import itertools
from functools import reduce


import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from streamlit import write as wr
import psycopg2
from decouple import config

from streamlit import sidebar as sbar
from streamlit.report_thread import get_report_ctx

from src.cache import Cache, TEXT_T, FLOAT_T
from src.orchastation import LoadUnifiedContext, StoreUnifiedContext, PageManager, Page
from src.specializedpages import HomePage, StockPickerPage, StockAllocationPage
from src.errors import Status, Success, Failure
from src.stockmon import Stocks
# Implement this system in a MVC fashion where the pages are the views and
# the controller is the unified context system.

# DONE When creating Pages, when passed into a Page manager, a reference to the pagemanager
# is passed into the pages so they can specifically dictate which page available in the page
# manager should the the controller (unifiedcontext) render next.

# Next!
# Need to create a page and test out the goingToNextPage and infomration passing`


# From there, its just a matter of extending the available pages
#     add caching
#     then we flesh out the stock data pipeline
#     analytics engine and integration testing.

# Cross page information passing is done within the context of a PageManager
# Cross pagemanager information passing perhaps done by direct storage of user data
#  in the DB??

# A PageManager knows what sequence it is currently in
# but the controller knows which pageManager it is intending to ask to
# perform rendering

# Create a user object that knows how to store its self and load itself as a json
# object. THen as UCxt is stored or loaded, user data that doesnt change that
# much can as well. THis will happen alongside user stock information in the table

# BUT, all the other users will be stored in the cache as well. So we can simply have a
# users table and place all the information there. We will still need to know the current
# context to which we are running such as who the current user is so this might be better
# served as a json maybe??


@st.cache()  # type: ignore
def get_data() -> pd.DataFrame:
    return pd.read_csv("__data_file/C_hist_data.csv")


def GetID() -> int:
    return uuid.uuid4().int


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
                             StockPickerPage(
                                 "stock_pick_page_two", stock_pick_pm),
                             StockAllocationPage("stock_allocation_page", stock_pick_pm)])
# stock_pick_pm.RegisterPages([stock_pick_page_one,
#                              stock_pick_page_two])
# TODO based on the if statement we should set and check the current manager in
# the context and if a page change happened we should clean data in pages that make sense
# like stock pick
if selected_window == home_title:
    st.write("Home page")
    home_pm.RenderCurrentPage()
elif selected_window == sp_title:
    st.write("Stock Picking")
    # RenderStockPageOne()
    # ctx.SetCurrentPageManager(stock_pick_pm)
    stock_pick_pm.RenderCurrentPage()
elif selected_window == social_title:
    st.write("Social")
else:
    st.error("Unknown section selected!")

StoreUnifiedContext(ctx)
st.balloons()
