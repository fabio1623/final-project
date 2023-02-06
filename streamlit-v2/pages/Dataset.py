import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_page():
    st.title("Dataset")

    data = st.cache(pd.read_csv)("../data/data_cleaned.csv")
    st.write(data)
    if st.checkbox("Show summary statistics"):
        st.write(data.describe())
    if st.checkbox("Show column names"):
        st.write(data.columns)
    if st.checkbox("Show data types"):
        st.write(data.dtypes)
    if st.checkbox("Show heatmap"):
        with st.spinner('Loading...'):
            corr=data.corr()

            mask=np.triu(np.ones_like(corr, dtype=bool))     # generate a mask for the upper triangle

            f, ax=plt.subplots(figsize=(11, 9))                 # set up the matplotlib figure

            cmap=sns.diverging_palette(220, 10, as_cmap=True)   # generate a custom diverging colormap

            sns.heatmap(corr, mask=mask, cmap=cmap,             # draw the heatmap with the mask and correct aspect ratio
                        vmax=.3, center=0, square=True,
                        linewidths=.5, cbar_kws={"shrink": .5})
            st.write(f)

if __name__ == "__main__":
    dataset_page()
