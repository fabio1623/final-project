import streamlit as st
import pandas as pd
import datetime as dt
import pickle

st.sidebar.markdown("# Predictions Tester")
st.markdown("# Prediction Tester")

selected_label = st.selectbox(
    'For which label do you want to make predictions?',
    ('', 'paper_price', 'wood_pulp_price'))

year = st.slider("For which year?", min_value=1900, max_value=2030, value=dt.datetime.now().year, key='year')

paper_prod = st.number_input('What is the paper production for this date? (tonnes)', min_value=0)
paper_export = st.number_input('What is the paper export for this date? (tonnes)', min_value=0)
paper_import = st.number_input('What is the paper import for this date? (tonnes)', min_value=0)

wood_pulp_prod = st.number_input('What is the wood pulp production for this date? (tonnes)', min_value=0)
wood_pulp_export = st.number_input('What is the wood pulp export for this date? (tonnes)', min_value=0)
wood_pulp_import = st.number_input('What is the wood pulp import for this date? (tonnes)', min_value=0)

data = pd.DataFrame({
    'year': [year],
    'paper_pulp_prod_tonnes': [paper_prod],
    'paper_pulp_export_tonnes': [paper_export],
    'paper_pulp_import_tonnes': [paper_import],
    'wood_pulp_prod_tonnes': [wood_pulp_prod],
    'wood_pulp_export_tonnes': [wood_pulp_export],
    'wood_pulp_import_tonnes': [wood_pulp_import]
})

data_cleaned = pd.read_csv('../data/data_cleaned.csv')
historical_paper_prices = data_cleaned.groupby('year')['paper_price'].mean().reset_index()
historical_wood_prices = data_cleaned.groupby('year')['wood_pulp_price'].mean().reset_index()

with open(f'../data/models/paper_price_model_with_scaler.pkl', "rb") as file:
    paper_pulp_model_with_scaler = pickle.load(file)

with open(f'../data/models/wood_pulp_price_model_with_scaler.pkl', "rb") as file:
    wood_pulp_model_with_scaler = pickle.load(file)


def predict_value(model, scaler):
    scaled_data = scaler.transform(data)
    predictions = model.predict(scaled_data)

    return predictions[0]


if selected_label == 'paper_price':
    value = predict_value(paper_pulp_model_with_scaler['model'], paper_pulp_model_with_scaler['min_max_scaler'])
    st.text(f"Based on provided information, paper price in {year} should be near '{value}$'")
    predicted_df = pd.DataFrame({
        'year': [year],
        selected_label: [value]
    })
    df = pd.concat([historical_paper_prices, predicted_df], axis=0)
    st.line_chart(df, x='year', y=selected_label)
elif selected_label == 'wood_pulp_price':
    value = predict_value(wood_pulp_model_with_scaler['model'], wood_pulp_model_with_scaler['min_max_scaler'])
    st.text(f"Based on provided information, wood pulp price in {year} should be near '{value}$'")
    predicted_df = pd.DataFrame({
        'year': [year],
        selected_label: [value]
    })
    df = pd.concat([historical_wood_prices, predicted_df], axis=0)
    st.line_chart(df, x='year', y=selected_label)