import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.markdown("# Merge Data :hammer_and_wrench:")

st.markdown("# Merged Data :hammer_and_wrench:")

data = pd.read_csv('../data/data.csv')
data_cleaned = pd.read_csv('../data/data_cleaned.csv')

show_cleaned_data = st.checkbox(':broom: Clean?')

if show_cleaned_data:
    st.text('Data cleaned.')
    st.balloons()
    st.dataframe(data_cleaned)

    st.markdown("## Correlation Matrix")
    with st.spinner('Wait for it...'):
        corr=data_cleaned.corr()

        mask=np.triu(np.ones_like(corr, dtype=bool))     # generate a mask for the upper triangle

        f, ax=plt.subplots(figsize=(11, 9))                 # set up the matplotlib figure

        cmap=sns.diverging_palette(220, 10, as_cmap=True)   # generate a custom diverging colormap

        sns.heatmap(corr, mask=mask, cmap=cmap,             # draw the heatmap with the mask and correct aspect ratio
                    vmax=.3, center=0, square=True,
                    linewidths=.5, cbar_kws={"shrink": .5})
        st.write(f)
else:
    st.text('Data not cleaned.')
    st.dataframe(data)