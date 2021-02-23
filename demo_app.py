import streamlit as st
import pandas as pd
from streamlit import write as wr
import time

st.title("ToTheMoon.alpha")
@st.cache()
def get_data():
    return pd.read_csv("__data_file/C_hist_data.csv")
data = get_data()

time_cprice_data = data[["Close"]]#.set_index("Date")
charted_data = time_cprice_data

# pbar = st.progress(0)
chart = st.line_chart(charted_data)
lc, mc, rc = st.beta_columns(3)
with lc:
    use_lReg = st.checkbox("Use Linear Regression")
with mc:
    use_o = st.checkbox("Use Neural Network")
with rc:
    use_m = st.checkbox("Use Deep NN")


st.balloons()
