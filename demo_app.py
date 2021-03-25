import time
from typing import Dict, List, Tuple, Any, Optional
import uuid

import streamlit as st
import pandas as pd
from streamlit import write as wr
from streamlit import sidebar as sbar
import itertools


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
        self.filepath = "resources/{}".format(resource_name)
        self.resource: object = None

    def Materialize(self) -> None:
        raise NotImplementedError


class ImageResourceHandler(ResourceHandler):
    def __init__(self, resource_name: str) -> None:
        super().__init__(resource_name)
        # resource = super().resource
    # TODO(theko): may need to cache this

    def Materialize(self) -> st.image:
        if self.resource:
            return self.resource
        self.resource = st.image
        return self.resource


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


st.balloons()


def RenderHome() -> None:
    def RenderStocksInPortfolioPicker(stocks: Stocks) -> List[Tuple[str, bool]]:
        stock_picker: List[Tuple[str, bool]] = []
        for ticker, _ in stocks.portfolio_.items():
            stock_picker.append((ticker, st.checkbox(ticker)))
        return stock_picker

    def GatherSelectedData(selected_tickers: List[Tuple[str, bool]],
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
            # rtn_df.insert(rtn_df.shape[1], column, data[["Close"]])
            rtn_df[column] = pd.Series(data["Close"])
        return rtn_df

    # data = get_data()
    # time_cprice_data = data[["Close"]]  # .set_index("Date")
    # charted_data = time_cprice_data
    chart_placeholder = st.empty()
    stocks_to_show: List[Tuple[str, bool]]
    _1, mc, _2 = st.beta_columns(3)
    with mc:
        stocks_to_show = RenderStocksInPortfolioPicker(stock_data)

    charted_data = GatherSelectedData(stocks_to_show, stock_data)
    st.dataframe(charted_data)
    # prints a table of the closing prices.
    chart_placeholder.line_chart(charted_data)

    # pbar = st.progress(0)


def RenderStockPageOne() -> None:
    def RenderTickerSearcher() -> None:
        rtn: str = st.text_input("Enter Ticker")
        if len(rtn.strip()) == 0:
            return
        status, _ = stock_data.GetStock(rtn)
        if status == Failure():
            st.info("Unable to find ticker :(")
        else:
            st.success("Found ticker.")
    RenderTickerSearcher()

    def RenderEditorsChoiceStockList() -> None:
        our_top_stocks_list: List[str] = ["MSFT", "GOOGL",
                                          "TSLA", "TSM", "C", "CSCO", "AAPL"]
        st.multiselect("Our Top Stocks", our_top_stocks_list +
                       our_top_stocks_list)

    RenderEditorsChoiceStockList()

    def RenderStockThemeCarousel() -> None:
        def GetCarouselMembers() -> List[ImageResourceHandler]:
            # returns List[Tuple[ThemeImage, ThemeExploreButton]]
            return [ImageResourceHandler("animal_1"),
                    ImageResourceHandler("animal_2"),
                    ImageResourceHandler("animal_3"),
                    ImageResourceHandler("animal_4")]

        def GetCurrentCarouselWindow(num_on_display) -> int:  # type: ignore
            # returns number between 1 and 3
            windows = num_on_display//3
            # # if num_on_display doesnt divide cleanly by 3 then we need one more window
            # # for the remaining items
            # windows = windows + 1 if (num_on_display/3) * \
            #     2 != windows else windows

            rtn: int = st.slider(
                "Slide through our stock categories", min_value=0, max_value=windows)
            return rtn

        def RenderImgAndButton(c_members) -> None:
            for cm, layer in zip(c_members, st.beta_columns(3)):
                with layer:
                    cm.Materialize()
                    # TODO Materialize some images and complete the flow
                    # TODO then generate stock names for the category
                    # img_uri = cm[0]
                    # member_category = cm[1]
                    # # st.image(img_uri)
                    # img_uri.Get()

        carousel_members = GetCarouselMembers()
        if len(carousel_members) > 9:
            st.warning(
                "carousel has more than 9 members. Trimming down to first 9.")
            carousel_members = carousel_members[:9]
        c_window = GetCurrentCarouselWindow(len(carousel_members))
        # sliding_window = range(c_window * 3 - 3, c_window * 3)
        carousel_member_limit = 3
        RenderImgAndButton(
            carousel_members[c_window: c_window + carousel_member_limit])

    RenderStockThemeCarousel()
    # ls = list(map(lambda i: "./resources/animal_{}.jpg".format(i), range(0, 5, 1)))
    # print(ls)
    # img_keys = [(img, key) for img, key in zip(
    #     map(lambda i: "./resources/animal_{}.jpg".format(i), range(0, 5, 1)), range(4))]
    # images_to_render = []

    # section_button_rotators = st.beta_columns(4)
    # n = 0
    # with section_button_rotators[1]:
    #     if st.button("<"):
    #         n += 1
    # with section_button_rotators[2]:
    #     st.button(">")
    # rotator = 1
    # for _ in range(3):
    #     images_to_render.append(img_keys[rotator])
    #     rotator = (rotator + 1 % 4)
    # l, m, r = st.beta_columns(3)
    # with l:
    #     rk1 = images_to_render[0]
    #     st.image(rk1[0], key=rk1[1])
    #     st.button("Explore", key="b1")
    # with m:
    #     rk2 = images_to_render[1]
    #     st.image(rk2[0], key=rk2[1])
    #     st.button("Explore", key="m")
    # with r:
    #     rk3 = images_to_render[2]
    #     st.image(rk3[0], key=rk3[1])
    #     st.button("Explore", key="t")

    # st.write(n)
# st.text()

# Main Construction


selected_window = GenerateSideBar()

if selected_window == home_title:
    st.write("Home page")
    RenderHome()
elif selected_window == sp_title:
    st.write("Stock Picking")
    RenderStockPageOne()
elif selected_window == social_title:
    st.write("Social")
else:
    st.error("Unknown section selected!")
