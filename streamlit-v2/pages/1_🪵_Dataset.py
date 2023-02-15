import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

def fill_na_with_mean(dataframe, column_name):
    mean = dataframe[column_name].mean()
    dataframe[column_name] = dataframe[column_name].fillna(mean)

    return dataframe

def display_heatmap(dataframe):
    with st.spinner('Loading...'):
            corr=dataframe.corr()

            mask=np.triu(np.ones_like(corr, dtype=bool))     # generate a mask for the upper triangle

            f, ax=plt.subplots(figsize=(11, 9))                 # set up the matplotlib figure

            cmap=sns.diverging_palette(220, 10, as_cmap=True)   # generate a custom diverging colormap

            sns.heatmap(corr, mask=mask, cmap=cmap,             # draw the heatmap with the mask and correct aspect ratio
                        vmax=.3, center=0, square=True,
                        linewidths=.5, cbar_kws={"shrink": .5})
            st.write(f)

@st.cache_data
def get_UN_data():
    data = pd.read_csv("../data/data.csv")
    cols = ['paper_pulp_prod_tonnes', 'paper_pulp_export_tonnes', 'paper_pulp_import_tonnes', 'wood_pulp_production_tonnes', 'wood_pulp_export_tonnes', 'wood_pulp_import_tonnes', 'paper_price']

    for col in cols:
        data = fill_na_with_mean(data, col)

    return data

def display_chart(dataframe, selected_feature, selected_countries):
    dataframe = dataframe[dataframe['country_code'].isin(selected_countries)]
    st.write(f"### {selected_feature}", dataframe.sort_values(by='year', ascending=False))

    chart = (
        alt.Chart(dataframe)
        .mark_area(opacity=0.3)
        .encode(
            x="year",
            y=f"{selected_feature}",
            color="country_code"
        )
    )
    st.altair_chart(chart, use_container_width=True)


st.set_page_config(
    page_title="Dataset", 
    page_icon="🪵")

st.write("# 🪵 Dataset")

data = get_UN_data()

tab1, tab2, tab3 = st.tabs(["Data", "Country Plot", "Structure"])

with tab1:
    st.write(data)

with tab2:
    selected_feature = st.selectbox(
        "Choose feature", [col for col in data.columns if col not in ['country_code', 'year']]
    )

    selected_countries = st.multiselect(
        "Choose countries", list(data['country_code'].unique())
    )

    if selected_countries:
        display_chart(data, selected_feature, selected_countries)

with tab3:
    if st.checkbox("Show summary statistics"):
        st.write(data.describe())
    if st.checkbox("Show column names"):
        st.write(data.columns)
    if st.checkbox("Show data types"):
        st.write(data.dtypes)
    if st.checkbox("Show heatmap"):
        display_heatmap(data)