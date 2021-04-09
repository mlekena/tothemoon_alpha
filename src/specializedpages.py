from .orchastation import Page, PageManager, UnifiedContext
from .errors import *
from .stockmon import Stocks

import pathlib
import json
import pandas as pd
from typing import List, Dict, Tuple, Set

import streamlit as st
from streamlit import sidebar as sbar
from streamlit.report_thread import get_report_ctx


class HomePage(Page):

    def __init__(self, id: str,
                 page_manager: PageManager):
        super().__init__(id, page_manager)
        # self.page_manager_ = page_manager
        self.stock_data = Stocks()

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
            stocks_to_show = __RenderStocksInPortfolioPicker(self.stock_data)

        charted_data = __GatherSelectedData(stocks_to_show, self.stock_data)
        # st.dataframe(charted_data)
        # prints a table of the closing prices.
        chart_placeholder.line_chart(charted_data)


class StockAllocationPage(Page):

    def __init__(self, id: str, page_manager: PageManager):
        super().__init__(id, page_manager)

    def RenderPage(self, context: UnifiedContext) -> None:
        st.header("Stock Allocations")


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


class StockPickerPage(Page):

    def __init__(self, id: str, page_manager: PageManager):
        Page.__init__(self, id, page_manager)
        self.stock_data = Stocks()

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
            status, _ = self.stock_data.GetStock(rtn)
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
