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
from src.unified_context import UnifiedContext, LoadUnifiedContext, StoreUnifiedContext
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

# class UnifiedContext(object):

#     cache_schema = [("currentPage", TEXT_T)]

#     def __init__(self, user: str) -> None:
#         self.user = user
#         # self.page_cache = page_cache
#         self.cache = Cache.get_instance()
#         self.user_data: pd.DataFrame = pd.DataFrame([])
#         self.unified_context_id = "_id_UnifiedContextCoreCache"
#         self.cache.InitCache(self.unified_context_id,
#                              UnifiedContext.cache_schema)
#         self.app_state: pd.DataFrame = self.cache.read_state_df(
#             self.unified_context_id)
#     # String type used to get forward usage of type names
#     # https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class

#     def SetCurrentPageManager(self, page_manager: "PageManager") -> None:
#         # TODO Think about how to ensure this gets called before each render
#         self.current_page_manager = page_manager

#     def RestorePageState(self, page_manager_id: str) -> pd.DataFrame:
#         print("Restore State")
#         return {}

#     def StorePageState(self, state: pd.DataFrame, page_manager_id: str) -> None:
#         print("Store State")

#     def CheckCacheExists(self, page_manager_id: str) -> bool:
#         pass

class Page(object):

    def __init__(self, id: str, page_manager: "PageManager") -> None:
        self.id = id
        self.page_manager = page_manager
        # self.public_cache: CacheType = {}

    def RenderPage(self, ctx: UnifiedContext) -> None:
        raise NotImplementedError


class PageManager(object):
    __cache_schema = [("ticker", TEXT_T), ("allocation", FLOAT_T)]

    def __init__(self, page_manager_id: str, context: UnifiedContext):
        self.NO_PAGES = ""
        self.page_manager_id = page_manager_id
        self.cache_id = "_id_%s" % self.page_manager_id
        self.pages: Dict[str, Page] = {}
        self.current_page: str = self.NO_PAGES
        self.cache_df: pd.DataFrame = None
        # context.InitCache(self.cache_id, PageManager.cache_schema)
        self.context = context

    def RegisterPage(self, new_page_renderer: Page) -> None:
        assert(new_page_renderer.id not in self.pages
               ), "Attempting to register duplicated ID"
        self.pages[new_page_renderer.id] = new_page_renderer
        if self.current_page == self.NO_PAGES:
            self.current_page = new_page_renderer.id

    def RegisterPages(self, pages: List[Page]) -> None:
        """
            Page order dictated by sequence they were added.
        """
        if not pages:
            return None
        for page in pages:
            self.RegisterPage(page)

    def RenderCurrentPage(self) -> None:
        if self.current_page == self.NO_PAGES:
            st.error(
                "PageManagerError: Attempting to render empty pages sequence.")
        cache_df = self.context.RestorePageState(self.cache_id)
        self.pages[self.current_page].RenderPage(self.cache_df)
        self.context.StorePageState(self.cache_df, self.cache_id)

    def GotoPage(self, page_id: str) -> None:
        if page_id not in self.pages:
            raise Exception("Attempting to head to Page {} PageManager {}".format(
                page_id, self.page_manager_id))
        self.current_page = page_id

    def UpdateUser(self) -> None:
        raise NotImplementedError


# UNIFIED_CONTEXT_CACHE_LOCATION = "__ucc/"


# def LoadUnifiedContext() -> UnifiedContext:
#     return UnifiedContext(user="Theko")
#     # ucc = os.path.join(UNIFIED_CONTEXT_CACHE_LOCATION, "theko.json")
#     # if not os.path.exists(UNIFIED_CONTEXT_CACHE_LOCATION):
#     #     return UnifiedContext(user="theko")
#     #     # os.mkdir(UNIFIED_CONTEXT_CACHE_LOCATION)
#     # with open(ucc, 'r') as uccfile:
#     #     uccjson = json.load(uccfile)
#     #     return UnifiedContext(user="saved_user")


# def StoreUnifiedContext(ctx: UnifiedContext) -> bool:
#     pass


@st.cache()  # type: ignore
def get_data() -> pd.DataFrame:
    return pd.read_csv("__data_file/C_hist_data.csv")


def GetID() -> int:
    return uuid.uuid4().int


class Status:
    def __init__(self) -> None:
        self.message = ""

    # https://github.com/python/mypy/issues/2783#issuecomment-276596902
    def __eq__(self, other: Any) -> bool:
        return self.message == other.message  # type: ignore


class Success(Status):
    def __init__(self) -> None:
        self.message = "SUCCESS"


class Failure(Status):
    def __init__(self) -> None:
        self.message = "FAILURE"


class Stocks:

    def __init__(self) -> None:
        def GenPath(ticker: str) -> str:
            filepath = "__data_file/{}_hist_data.csv".format(ticker)
            # print(filepath)
            return "__data_file/{}_hist_data.csv".format(ticker)
        self.portfolio_ = dict([("AAPL", GenPath("AAPL")),
                                ("C", GenPath("C")),
                                ("TSLA", GenPath("TSLA"))])
        self.stock_cache_: Dict[str, pd.DataFrame] = {}

    def GetStock(self, ticker: str) -> Tuple[Status, pd.DataFrame]:
        if ticker not in self.portfolio_:
            return (Failure(), pd.DataFrame())
        stock = self.portfolio_[ticker]
        if ticker not in self.stock_cache_:
            self.stock_cache_[ticker] = pd.read_csv(stock)
        return (Success(), self.stock_cache_[ticker])


class ResourceHandler(object):
    def __init__(self, resource_name: str) -> None:
        self.filepath = "resources/{}.jpg".format(resource_name)
        self.resource: object = None

    def Materialize(self) -> None:
        raise NotImplementedError


class ImageResourceHandler(ResourceHandler):
    def __init__(self, resource_name: str) -> None:
        super().__init__(resource_name)
    # TODO(theko): may need to cache this

    def Materialize(self) -> st.image:
        if self.resource:
            return self.resource
        self.resource = st.image(self.filepath)
        return self.resource


class OnTriggerPresenter(object):
    def __init__(self, trigger_label: str, call: Callable[[], None]) -> None:
        self.id = GetID()
        self.trigger_label = trigger_label
        self.trigger_button: bool = False
        # Expecting function performing rendering
        self.to_trigger = call

    def RenderTrigger(self) -> None:
        self.trigger_button = st.button(self.trigger_label)
        if self.trigger_button:
            print("%s flipped" % self.trigger_label)

    def RenderIfTriggered(self) -> None:
        print("trigger check")
        if self.trigger_button == True:
            print("triggering %s" % self.trigger_label)
            self.to_trigger()


class StockCategoryHandler(object):
    def __init__(self, id: str, tickers: List[str] = [],
                 image_path: str = "",
                 title: str = "Ticker",
                 description: str = "Desciption."):
        self.id = id
        self.tickers = tickers
        self.image_path = image_path
        self.title = title
        self.description = description


def ReadCategoryFromJsonFile(filepath: pathlib.Path) -> StockCategoryHandler:
    with open(filepath, 'r') as json_file:
        category_data = json.load(json_file)
        return StockCategoryHandler(
            id=category_data["id"],
            tickers=category_data["tickers"],
            image_path=category_data["image_path"],
            title=category_data["title"],
            description=category_data["description"]
        )


CATEGORY_DATA_PATH = "resources/category_info/"
CATEGORIES = [ReadCategoryFromJsonFile(f)
              for f in pathlib.Path(CATEGORY_DATA_PATH).iterdir() if f.suffix == '.json']

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


class HomePage(Page):

    def __init__(self, id: str,
                 page_manager: PageManager):
        super().__init__(id, page_manager)
        # self.page_manager_ = page_manager

    def RenderPage(self, context: UnifiedContext) -> None:
        def __RenderStocksInPortfolioPicker(stocks: Stocks) -> List[Tuple[str, bool]]:
            stock_picker: List[Tuple[str, bool]] = []
            for ticker, _ in stocks.portfolio_.items():
                stock_picker.append((ticker, st.checkbox(ticker)))
            return stock_picker

        def __GatherSelectedData(selected_tickers: List[Tuple[str, bool]],
                                 stocks: Stocks) -> pd.DataFrame:
            rtn_df = pd.DataFrame()
            for ticker, selected in selected_tickers:
                if not selected:
                    continue
                status_and_data = stocks.GetStock(ticker)
                print(type(status_and_data[0]))
                if status_and_data[0] == Success():
                    data = status_and_data[1]
                else:
                    raise Exception("Failed to get ticker: {}".format(ticker))
                column = "{}_close".format(ticker)
                rtn_df[column] = pd.Series(data["Close"])
            return rtn_df

        # self.public_cache["val"] = 99

        chart_placeholder = st.empty()
        stocks_to_show: List[Tuple[str, bool]]
        _1, mc, _2 = st.beta_columns(3)
        with mc:
            stocks_to_show = __RenderStocksInPortfolioPicker(stock_data)

        charted_data = __GatherSelectedData(stocks_to_show, stock_data)
        # st.dataframe(charted_data)
        # prints a table of the closing prices.
        chart_placeholder.line_chart(charted_data)


class StockAllocationPage(Page):

    def __init__(self, id: str, page_manager: PageManager):
        super().__init__(id, page_manager)

    def RenderPage(self, context: UnifiedContext) -> None:
        st.header("Stock Allocations")


class StockPickerPage(Page):

    def __init__(self, id: str, page_manager: PageManager):
        Page.__init__(self, id, page_manager)

    def RenderPage(self, context: UnifiedContext) -> None:
        # def RenderStockPageOne(ctx: UnifiedContext) -> None:
        def __RenderCategory() -> None:
            image_panel, stock_panel = st.beta_columns(2)
            with image_panel:
                st.image("resources/animal_1.jpg")
            with stock_panel:
                st.multiselect("Select Category Stocks", [1, 2, 3, 4, 5, 6])
                st.title("sector 1")
                st.write("brief information about the sector")
            with st.beta_expander("Category in-depth"):
                st.write("Potentially display sector tracked performance??")

        def __InflateCategory(category_data: StockCategoryHandler) -> List[str]:
            selected_stocks: List[str] = []
            with st.beta_container():
                image_panel, stock_panel = st.beta_columns(2)
                with image_panel:
                    st.image(category_data.image_path)
                with stock_panel:
                    selected_stocks = st.multiselect("Select Category Stocks",
                                                     category_data.tickers, key=category_data.id)
                    st.title(category_data.title)
                    st.write(category_data.description)
                with st.beta_expander("Category in-depth"):
                    st.write("In-depth breackdown of %s" % category_data.title)
                st.write("")  # add padding between __InflateCategory's
            return selected_stocks

        def __RenderTickerSearcher() -> None:
            rtn: str = st.text_input("Enter Ticker")
            if len(rtn.strip()) == 0:
                return
            status, _ = stock_data.GetStock(rtn)
            if status == Failure():
                st.info("Unable to find ticker :(")
            else:
                st.success("Found ticker.")

        GoToStockAllocation = st.empty()
        __RenderTickerSearcher()

        def __RenderEditorsChoiceStockList() -> None:
            our_top_stocks_list: List[str] = ["MSFT", "GOOGL",
                                              "TSLA", "TSM", "C", "CSCO", "AAPL"]
            st.multiselect("Our Top Stocks", our_top_stocks_list +
                           our_top_stocks_list)

        __RenderEditorsChoiceStockList()

        gathered_stock_selection: Set[str] = set()
        for category in CATEGORIES:
            gathered_stock_selection = gathered_stock_selection | {
                s for s in __InflateCategory(category)}
        GoToStockAllocation.beta_container()
        if not gathered_stock_selection:
            with GoToStockAllocation:
                next = st.button("Next", key="NoStocksSelected")
                if next:
                    st.warning(
                        "Please find and select some stocks before allocation.")
        else:
            with GoToStockAllocation:
                if st.button("Next", key="StocksSelected"):
                    self.page_manager.GotoPage("stock_allocation_page")


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
