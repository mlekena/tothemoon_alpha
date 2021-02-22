import streamlit as st
import pandas as pd
from streamlit import write as wr

st.title("ToTheMoon.alpha")
@st.cache()
def get_data():
    return pd.read_csv("__data_file/C_hist_data.csv")
data = get_data()
st.write(data)

time_cprice_data = data[["Date", "Close"]].set_index("Date")
charted_data = time_cprice_data[0:1]
charted_data = charted_data#.set_index("Date")

st.write(charted_data)
wr(charted_data.shape)
pbar = st.progress(0)
status = st.empty()
chart = st.line_chart(charted_data)

progress_step = 100/time_cprice_data.shape[0]
current_progress = 0
row = 1
while current_progress <= 100:
    pbar.progress(current_progress + 1)#progress_step)
    current_progress += 1#progress_step
    new_row = time_cprice_data.iloc[row]

    # status.text("THe latest random number is %s:%s" % new_row[0], new_row[1])

    chart.add_rows(new_row)

    time.sleep(0.1)

status.text("DONE")
st.balloons()