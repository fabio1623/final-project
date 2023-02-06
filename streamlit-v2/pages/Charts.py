import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def charts():
    st.title("Charts")

    df = pd.read_csv("../data/data_cleaned.csv")

    column1 = st.selectbox("Select a column to plot (x-axis)", df.columns)
    column2 = st.selectbox("Select a column to plot (y-axis)", df.columns)

    st.bar_chart(df, x=column1, y=column2)

if __name__ == "__main__":
    charts()
