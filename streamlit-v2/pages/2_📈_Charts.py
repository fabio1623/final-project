import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Charts", 
    page_icon="📈"
)

st.write("# 📈 Charts")

data = pd.read_csv("../data/data_cleaned.csv")

column1 = st.selectbox("Select a column to plot (x-axis)", data.columns)

filtered_columns = [col for col in data.columns if col != column1]

column2 = st.selectbox("Select a column to plot (y-axis)", filtered_columns)

st.bar_chart(data, x=column1, y=column2)