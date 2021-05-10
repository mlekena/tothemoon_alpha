from src.orchastation import Page, PageManager, UnifiedContext
from src.errors import Success, Failure
from src.stockmon import Stocks
from src.snp_500_dojo import SNPModel
from src.stock_fetch import GetStockData, STOCK_LOCAL_CACHE_PATH

import os
import pathlib
import json
import pandas as pd
from decimal import *
from typing import List, Dict, Tuple, Set

import streamlit as st
import plotly.graph_objects as go
from streamlit import sidebar as sbar
from streamlit.report_thread import get_report_ctx
from yahoo_fin import stock_info as si

UserDataPath = os.path.join("user_data.csv")


def MakeUserStockDF(arr: list = []):
    return pd.DataFrame(data=arr, columns=["ticker"])


def GetUserStockListing():
    if os.path.exists(UserDataPath):
        return pd.read_csv(UserDataPath)
    else:
        return MakeUserStockDF()


def StoreUserStockListing(df):
    df.to_csv(UserDataPath, index=False)


class HomePage(Page):

    def __init__(self, id: str,
                 page_manager: PageManager):
        super().__init__(id, page_manager)
        self.stock_data = Stocks()

    def RenderPage(self, context: UnifiedContext) -> None:
        def __RenderStocksInPortfolioPicker(stocks: Stocks) -> List[Tuple[str, bool]]:
            stock_picker: List[Tuple[str, bool]] = []
            user_stocks_df = GetUserStockListing()
            # for ticker, _ in stocks.portfolio_.items():
            st.write(user_stocks_df)
            if user_stocks_df.empty:
                st.write(
                    "No Stocks in Portfolio. Select, 'Stock Picker' and build your portfolio.")
                return []
            for row in user_stocks_df.sort_values(by=['ticker']).itertuples():
                ticker = row[1]
                stock_picker.append((ticker, st.checkbox(ticker)))
            return stock_picker

        def __GatherSelectedData(selected_tickers: List[Tuple[str, bool]],
                                 stocks: Stocks) -> pd.DataFrame:
            rtn_df = pd.DataFrame()
            for ticker, selected in selected_tickers:
                if not selected:
                    continue
                status_and_data = stocks.GetStock(ticker)
                # print(type(status_and_data[0]))
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
TICKER_COL = "tickers"
TICKER_IDX = 1
ALLOC_COL = "allocation"
ALLOC_IDX = 2


def GetCurrentPrice(ticker: str) -> float:
    """Query Yfin for the current ticker price."""
    return si.get_live_price(ticker)


class StockEvaluationPage(Page):
    def __init(self, id: str, page_manager: PageManager) -> None:
        super().__init__(id, page_manager)

    def GetModel(self):
        """Serves to cache model creation using @st.cache"""
        return SNPModel(1)

    def RenderPage(self, context: UnifiedContext) -> None:
        def gain_loss_styler(val):
            color = "green" if val > 0 else "red"
            return "color: %s" % color
        st.header("Evaluate position")
        banner = st.empty()
        banner_detail = st.empty()
        cached_df = context.cache.read_state_df(self.page_manager.cache_id)
        model = self.GetModel()
        predictions = model.PredictProfitFlow(list(map(
            lambda row: (row[TICKER_IDX], GetStockData(
                row[TICKER_IDX], STOCK_LOCAL_CACHE_PATH)),
            cached_df.sort_values(by=[TICKER_COL]).itertuples())))
        # st.write(predictions)
        price_movements = list()
        idx = 0
        for ticker, prediction in predictions:
            alloc = cached_df[cached_df[TICKER_COL] == ticker][ALLOC_COL][idx]
            price_at_purchase = GetCurrentPrice(ticker)
            purchase_price = price_at_purchase * alloc
            price_change = (prediction -
                            price_at_purchase)/price_at_purchase
            price_change *= 100  # multiple into percentage
            new_asset_value = alloc * prediction
            price_movements.append(
                [ticker, price_at_purchase,
                    prediction, price_change, purchase_price, new_asset_value])
            idx += 1
        price_movement_df = pd.DataFrame(price_movements,
                                         columns=["Ticker", "Price",
                                                  "Future Price", r"%change",
                                                  "Org. Asset Value", "New Asset Value"])
        st.write(price_movement_df.style.applymap(
            gain_loss_styler, subset=[r"%change"]))
        total_org_asset_value = price_movement_df["Org. Asset Value"].sum()
        total_future_asset_value = price_movement_df["New Asset Value"].sum(
        )
        asset_difference = total_future_asset_value - total_org_asset_value
        if asset_difference > 0:
            banner.title("Congratulations!")
            banner_detail.text("Your Porfolio will set positive gains.")
        else:
            banner.title("Bad Position")
            banner_detail.text(
                "Seems like your position is predicted to have a negative returns.")

        if st.button("Add to portfolio."):
            portfolio = GetUserStockListing()
            newFolio = set()
            for row in portfolio.itertuples():
                newFolio |= {row[1]}
            for row in price_movement_df.itertuples():
                newFolio |= {row[1]}
            newFolio_df = pd.DataFrame(data=list(newFolio), columns=["ticker"])
            StoreUserStockListing(newFolio_df)
            st.write("Added to Portfolio")
        if st.button("Re-pick Stocks"):
            self.page_manager.GotoPage("stock_pick_page_one")


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
            TICKER_COL: list(), ALLOC_COL: list()}
        for row in cache_df.sort_values(by=[TICKER_COL]).itertuples():
            lhs, mid, rhs = st.beta_columns(3)
            with lhs:
                price = GetCurrentPrice(row[TICKER_IDX])
                assert(
                    price >= 0), "Retrieved zero or negative price in StockAllocationPage"
                st.write("Price %f" % price)
            with mid:
                allocs[TICKER_COL].append(row[TICKER_IDX])
                allocs[ALLOC_COL].append(st.number_input(
                    row[TICKER_IDX], min_value=0))
                purchased = allocs[ALLOC_COL][-1]
            with rhs:
                st.write("Final Price:{}".format(price * purchased))
        cache_df[TICKER_COL] = allocs[TICKER_COL]
        cache_df[ALLOC_COL] = allocs[ALLOC_COL]
        # st.write(cache_df[TICKER_COL])
        # st.write(cache_df[ALLOC_COL])

        fig = go.Figure(data=[go.Pie(
            labels=cache_df[TICKER_COL], values=cache_df[ALLOC_COL], hole=0.2)])
        fig.update_traces(hoverinfo="label+percent",
                          marker=dict(line=dict(color="#000000", width=2)))
        allocation_chart.write(fig)
        # STORE CACHE UPDATES TO THE DATABASE
        context.cache.write_state_df(
            cache_df, self.page_manager.cache_id)
        _1, center, _2 = st.beta_columns(3)
        with center:
            all_allocated = True
            for ticker, alloc in zip(allocs[TICKER_COL], allocs[ALLOC_COL]):
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
