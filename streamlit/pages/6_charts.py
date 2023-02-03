import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.markdown("# Charts :chart_with_upwards_trend:")
st.sidebar.markdown("# Charts :chart_with_upwards_trend:")

data = pd.read_csv('../data/data.csv')
data_cleaned = pd.read_csv('../data/data_cleaned.csv')

cols_to_show = [col for col in data.columns if col not in ['year', 'paper_price', 'wood_pulp_price', 'country_code']]
cols_to_show_cleaned = [col for col in data_cleaned.columns if col != 'year']

tab1, tab2 = st.tabs(["General", "Before / After Covid"])

with tab1:
   for col in cols_to_show_cleaned:
    st.markdown(f"## {col}")
    groups = data_cleaned.groupby('year')[col].mean().reset_index()
    st.line_chart(groups, x='year', y=col)


with tab2:
    col1, col2 = st.columns(2)

    with col1:
        # Before Covid
        st.header("Before Covid")
       
    with col2:
        # After Covid
        st.header("After Covid")

    for col in cols_to_show:

        with col1:
            # Before Covid
            data_before_covid = data[(data['year'] == 2018) | (data['year'] == 2019)]
            
            avg_values = data_before_covid.groupby(['country_code'], as_index=False).mean()
            avg_values.sort_values(by=col, ascending=False, inplace=True)
            top_values = avg_values.head(10)

            top_values[col] = top_values[col]
            st.bar_chart(top_values, x='country_code', y=col)
        
        with col2:
            # After Covid
            data_after_covid = data[data['year'] > 2019]

            avg_values = data_after_covid.groupby(['country_code'], as_index=False).mean()
            avg_values.sort_values(by=col, ascending=False, inplace=True)
            top_values = avg_values.head(10)

            top_values[col] = top_values[col]
            st.bar_chart(top_values, x='country_code', y=col)

#    col1, col2 = st.columns(2)
#    for col in cols_to_show:
      
#     with col1:
#         # Before Covid
#         st.header(col)
#         st.header("Before Covid")

#         data_before_covid = data[(data['year'] == 2018) | (data['year'] == 2019)]

#         avg_values = data_before_covid.groupby(['country_code'], as_index=False).mean()
#         avg_values.sort_values(by=col, ascending=False, inplace=True)
#         top_values = avg_values.head(10)

#         top_values[col] = np.log(top_values[col])
#         st.bar_chart(top_values, x='country_code', y=col)

#         with col2:
#             # After Covid
#             st.header("After Covid")

#             data_after_covid = data[data['year'] > 2019]

#             avg_values = data_after_covid.groupby(['country_code'], as_index=False).mean()
#             avg_values.sort_values(by=col, ascending=False, inplace=True)
#             top_values = avg_values.head(10)

#             top_values[col] = np.log(top_values[col])
#             st.bar_chart(top_values, x='country_code', y=col)


