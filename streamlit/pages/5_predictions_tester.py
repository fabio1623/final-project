import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
# import pickle
import random
from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

st.sidebar.markdown("# Predictions Tester :crystal_ball:")
st.markdown("# Prediction Tester :crystal_ball:")

def generate_random_int():
    return random.randint(0, 100000000)

def predict_value(model, scaler):
    scaled_data = scaler.transform(data)
    predictions = model.predict(scaled_data)

    return predictions

def get_trained_model_with_scaler(model, dataframe, label):
    dataframe = dataframe.sample(frac=1)

    y = dataframe[label]
    X = dataframe.drop(label, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scaling data = X_train
    min_max_scaler = MinMaxScaler().fit(X_train)
    X_train_normalized = min_max_scaler.transform(X_train)
    X_train_normalized = pd.DataFrame(X_train_normalized)

    # Scaling data = X_test
    X_test_normalized = min_max_scaler.transform(X_test)
    X_test_normalized = pd.DataFrame(X_test_normalized)

    model.fit(X_train_normalized, y_train)

    return {
        'model': model,
        'min_max_scaler': min_max_scaler
    }


models = [
    '',
    LinearRegression(n_jobs=-1),
    Lasso(),
    Ridge(),
    ElasticNet(),
    XGBRegressor(),
    LGBMRegressor(n_jobs=-1),
    DecisionTreeRegressor(),
    KNeighborsRegressor(n_jobs=-1),
    MLPRegressor(),
    RandomForestRegressor(n_jobs=-1)
]

selected_label = st.selectbox(
    'For which label do you want to make predictions?',
    ('', 'paper_price', 'wood_pulp_price'))

selected_model = st.selectbox(
    'For which label do you want to make predictions?',
    models)

if selected_model != '':

    year = st.slider("For which year?", min_value=2023, max_value=2050, value=dt.datetime.now().year, key='year')

    paper_prod = st.number_input('What is the paper production for this date? (tonnes)', min_value=0, value=generate_random_int())
    paper_export = st.number_input('What is the paper export for this date? (tonnes)', min_value=0, value=generate_random_int())
    paper_import = st.number_input('What is the paper import for this date? (tonnes)', min_value=0, value=generate_random_int())

    wood_pulp_prod = st.number_input('What is the wood pulp production for this date? (tonnes)', min_value=0, value=generate_random_int())
    wood_pulp_export = st.number_input('What is the wood pulp export for this date? (tonnes)', min_value=0, value=generate_random_int())
    wood_pulp_import = st.number_input('What is the wood pulp import for this date? (tonnes)', min_value=0, value=generate_random_int())

    nb_years = year - 2022
    years = range(2023, year+1)

    data = pd.DataFrame({
        'year': years,
        'paper_pulp_prod_tonnes': np.full(nb_years, paper_prod),
        'paper_pulp_export_tonnes': np.full(nb_years, paper_export),
        'paper_pulp_import_tonnes': np.full(nb_years, paper_import),
        'wood_pulp_prod_tonnes': np.full(nb_years, wood_pulp_prod),
        'wood_pulp_export_tonnes': np.full(nb_years, wood_pulp_export),
        'wood_pulp_import_tonnes': np.full(nb_years, wood_pulp_import),
    })

    data_cleaned = pd.read_csv('../data/data_cleaned.csv')
    historical_paper_prices = data_cleaned.groupby('year')['paper_price'].mean().reset_index()
    historical_wood_prices = data_cleaned.groupby('year')['wood_pulp_price'].mean().reset_index()

    # with open(f'../data/models/paper_price_model_with_scaler.pkl', "rb") as file:
    #     paper_pulp_model_with_scaler = pickle.load(file)

    # with open(f'../data/models/wood_pulp_price_model_with_scaler.pkl', "rb") as file:
    #     wood_pulp_model_with_scaler = pickle.load(file)

    data_cleaned = pd.read_csv('../data/data_cleaned.csv')

    if selected_label == 'paper_price':
        with st.spinner('Wait for it...'):
            model_with_scaler = get_trained_model_with_scaler(selected_model, data_cleaned.drop('wood_pulp_price', axis=1), selected_label)
            values = predict_value(model_with_scaler['model'], model_with_scaler['min_max_scaler'])
            st.text(f"Based on provided information, paper price by {year} could be something like this:")
            predicted_df = pd.DataFrame({
                'year': years,
                selected_label: values
            })

            df = pd.concat([historical_paper_prices, predicted_df], axis=0)
            st.line_chart(df, x='year', y=selected_label)
            st.dataframe(predicted_df)
    elif selected_label == 'wood_pulp_price':
        with st.spinner('Wait for it...'):
            model_with_scaler = get_trained_model_with_scaler(selected_model, data_cleaned.drop('paper_price', axis=1), selected_label)
            values = predict_value(model_with_scaler['model'], model_with_scaler['min_max_scaler'])
            st.text(f"Based on provided information, wood pulp price by {year} could be something like this:")
            predicted_df = pd.DataFrame({
                'year': years,
                selected_label: values
            })

            df = pd.concat([historical_wood_prices, predicted_df], axis=0)
            st.line_chart(df, x='year', y=selected_label)
            st.dataframe(predicted_df)





    # if selected_label == 'paper_price':
    #     values = predict_value(paper_pulp_model_with_scaler['model'], paper_pulp_model_with_scaler['min_max_scaler'])
    #     st.text(f"Based on provided information, paper price by {year} could be something like this:")
    #     predicted_df = pd.DataFrame({
    #         'year': years,
    #         selected_label: values
    #     })

    #     df = pd.concat([historical_paper_prices, predicted_df], axis=0)
    #     st.line_chart(df, x='year', y=selected_label)
    #     st.dataframe(predicted_df)
    # elif selected_label == 'wood_pulp_price':
    #     values = predict_value(wood_pulp_model_with_scaler['model'], wood_pulp_model_with_scaler['min_max_scaler'])
    #     st.text(f"Based on provided information, wood pulp price by {year} could be something like this:")
    #     predicted_df = pd.DataFrame({
    #         'year': years,
    #         selected_label: values
    #     })

    #     df = pd.concat([historical_wood_prices, predicted_df], axis=0)
    #     st.line_chart(df, x='year', y=selected_label)
    #     st.dataframe(predicted_df)