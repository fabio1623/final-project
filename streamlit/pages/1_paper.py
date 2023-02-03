import streamlit as st
import pandas as pd

st.markdown("# Paper Data :roll_of_paper:")
st.sidebar.markdown("# Paper :roll_of_paper:")

export_data = pd.read_csv('../data/paper_pulp_export_data.csv')
import_data = pd.read_csv('../data/paper_pulp_import_data.csv')
production_data = pd.read_csv('../data/paper_pulp_production_data.csv')

# Create a dictionary of dataframes
df_dict = {
    '': None,
    'Exports': export_data,
    'Imports': import_data,
    'Production': production_data
}

# Create a selection box for the dataframes
df_select = st.selectbox("Which kind of data do you want to display?", df_dict)

# Show the selected dataframe
if df_select != '':
    st.dataframe(df_dict[df_select])