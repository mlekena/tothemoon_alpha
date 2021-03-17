import time
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
from streamlit import write as wr
from streamlit import sidebar as sbar


@st.cache()  # type: ignore
def get_data() -> pd.DataFrame:
    return pd.read_csv("__data_file/C_hist_data.csv")


class Status:
    def __init__(self) -> None:
        self.message = ""


class Success(Status):
    def __init__(self) -> None:
        self.message = "SUCCESS"


class Failure(Status):
    def __init__(self) -> None:
        self.message = "FAILURE"


class Stocks:

    def __init__(self) -> None:
        def GenPath(ticker: str) -> str:
            return "__data_file/{}_hist_data.csv".format(ticker)
        self.portfolio_ = dict([("AAPL", GenPath("AAPL")),
                                ("C", GenPath("C")),
                                ("TSLA", GenPath("TSLA"))])
        self.stock_cache_: Dict[str, pd.DataFrame]

    def GetStock(self, ticker: str) -> Tuple[Status, pd.DataFrame]:
        if ticker not in self.portfolio_:
            return (Failure(), pd.Dataframe())
        stock = self.portfolio_[ticker]
        if ticker not in self.stock_cache_:
            self.stock_cache_[ticker] = pd.read_csv(stock[1])
        return (Success(), self.stock_cache_[ticker])


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


# data = get_data()

# time_cprice_data = data[["Close"]]#.set_index("Date")
# charted_data = time_cprice_data

# # pbar = st.progress(0)
# chart = st.line_chart(charted_data)
# lc, mc, rc = st.beta_columns(3)
# with lc:
#     use_lReg = st.checkbox("Use Linear Regression")
# with mc:
#     use_o = st.checkbox("Use Neural Network")
# with rc:
#     use_m = st.checkbox("Use Deep NN")


st.balloons()


def RenderHome() -> None:
    def RenderStocksInPortfolioPicker(stocks: Stocks) -> List[Tuple[str, bool]]:
        stock_picker: List[Tuple[str, bool]] = []
        for ticker, _ in stocks.portfolio_.items():
            stock_picker.append((ticker, st.checkbox(ticker)))
        return stock_picker

    def GatherSelectedData(selected_tickers: List[Tuple[str, bool]]) -> pd.DataFrame:
        pass

    data = get_data()
    time_cprice_data = data[["Close"]]  # .set_index("Date")
    charted_data = time_cprice_data
    chart_placeholder = st.empty()
    stocks_to_show: List[Tuple[str, bool]]
    _1, mc, _2 = st.beta_columns(3)
    with mc:
        # use_lReg = st.checkbox("Use Linear Regression")
        # use_o = st.checkbox("Use Neural Network")
        # use_m = st.checkbox("Use Deep NN")
        stocks_to_show = RenderStocksInPortfolioPicker(stock_data)

    GatherSelectedData(stocks_to_show)
    st.dataframe(charted_data)
    chart_placeholder.line_chart(charted_data)

    # pbar = st.progress(0)
# Main Construction


selected_window = GenerateSideBar()

if selected_window == home_title:
    st.write("Home page")
    RenderHome()
elif selected_window == sp_title:
    st.write("Stock Picking")
elif selected_window == social_title:
    st.write("Social")
else:
    st.error("Unknown section selected!")
