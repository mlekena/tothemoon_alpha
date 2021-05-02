from src.orchastation import Page, PageManager, UnifiedContext
from src.errors import Success, Failure
from src.stockmon import Stocks


import pathlib
import json
import pandas as pd
from decimal import *
from typing import List, Dict, Tuple, Set

import streamlit as st
import plotly.graph_objects as go
from streamlit import sidebar as sbar
from streamlit.report_thread import get_report_ctx


class HomePage(Page):

    def __init__(self, id: str,
                 page_manager: PageManager):
        super().__init__(id, page_manager)
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

        chart_placeholder = st.empty()
        stocks_to_show: List[Tuple[str, bool]]
        _1, mc, _2 = st.beta_columns(3)
        with mc:
            stocks_to_show = __RenderStocksInPortfolioPicker(self.stock_data)

        charted_data = __GatherSelectedData(stocks_to_show, self.stock_data)
        # prints a table of the closing prices.
        chart_placeholder.line_chart(charted_data)


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


class StockEvaluationPage(Page):
    def __init(self, id: str, page_manager: PageManager) -> None:
        super().__init__(id, page_manager)

    def RenderPage(self, context: UnifiedContext) -> None:
        st.header("Evaluate position")


class StockAllocationPage(Page):

    def __init__(self, id: str, page_manager: PageManager):
        super().__init__(id, page_manager)

    def plot_allocation_chart(allocations: pd.DataFrame) -> go.Figure:
        pass

    def RenderPage(self, context: UnifiedContext) -> None:

        st.header("Stock Allocations")
        cache_df = context.cache.read_state_df(self.page_manager.cache_id)

        allocation_chart = st.empty()
        allocs: Dict[str, List[Decimal]] = {
            "tickers": list(), "allocations": list()}
        ticker_idx = 1

        for row in cache_df.sort_values(by=["tickers"]).itertuples():
            allocs["tickers"].append(row[ticker_idx])
            allocs["allocations"].append(st.number_input(
                row[ticker_idx], min_value=0))

        cache_df["tickers"] = allocs["tickers"]
        cache_df["allocations"] = allocs["allocations"]
        fig = go.Figure(data=[go.Pie(
            labels=cache_df["tickers"], values=cache_df["allocations"], hole=0.2)])
        fig.update_traces(hoverinfo="label+percent",
                          marker=dict(line=dict(color="#000000", width=2)))
        allocation_chart.write(fig)
        _1, center, _2 = st.beta_columns(3)
        with center:
            all_allocated = True
            for ticker, alloc in zip(allocs["tickers"], allocs["allocations"]):
                if alloc == 0:
                    all_allocated = False
                    break

            if st.button("Evaluate Position"):
                if all_allocated:
                    self.page_manager.GotoPage("stock_evaluation_page")
                else:
                    st.info(
                        "Need to provide an allocation to all listed positions before evaluation.")


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

        def __RenderEditorsChoiceStockList() -> List[str]:
            our_top_stocks_list: List[str] = ["MSFT", "GOOGL",
                                              "TSLA", "TSM", "C", "CSCO", "AAPL"]
            rtn: List[str] = st.multiselect("Our Top Stocks", our_top_stocks_list +
                                            our_top_stocks_list)
            return rtn

        cached_data_df = pd.DataFrame()
        GoToStockAllocation = st.empty()
        gathered_stock_selection: Set[str] = set()
        __RenderTickerSearcher()
        gathered_stock_selection = gathered_stock_selection | {
            s for s in __RenderEditorsChoiceStockList()}
        for category in CATEGORIES:
            gathered_stock_selection = gathered_stock_selection | {
                s for s in __InflateCategory(category)}

        # Cache picked stocks and set to zero
        print("LOAD NEW DATA")
        cached_data_df["tickers"] = list(gathered_stock_selection)
        # reset to zero
        # TODO consider perhaps loading previously add allocations
        cached_data_df["allocation"] = [
            0 for _ in range(len(gathered_stock_selection))]
        print(cached_data_df)
        print("___END___")
        # cache
        context.cache.write_state_df(
            cached_data_df, self.page_manager.cache_id)
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


"""TODO need to determine why a double click on next is neede!!!!"""
