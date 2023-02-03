import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import pickle


algorithms = [
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

def compare_models(models, dataframe, label):
    
    fitted_models_list = []
    min_max_scaler_list = []

    r2_list = []
    mse_list = []
    rmse_list = []
    mae_list = []

    for model in models:

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

        # Make predictions on the test data
        y_pred = model.predict(X_test_normalized)

        # R2 validation
        r2 = r2_score(y_test, y_pred)

        # MSE validation
        mse=mean_squared_error(y_test, y_pred)

        # RMSE validation
        rmse = np.sqrt(mse)

        # MAE validation
        mae=mean_absolute_error(y_test, y_pred)

        fitted_models_list.append({
            'model': model,
            'min_max_scaler' : min_max_scaler
        })
        min_max_scaler_list.append(min_max_scaler)

        r2_list.append(r2)
        mse_list.append(mse)
        rmse_list.append(rmse)
        mae_list.append(mae)


    summary = {
        'Algorithm': [item['model'] for item in fitted_models_list],
        'R2': r2_list,
        'MSE': mse_list,
        'RMSE': rmse_list,
        'MAE': mae_list
    }
    summary = pd.DataFrame(summary)

    return {
        'summary' : summary,
        'models' : fitted_models_list
    }


def display_ordered_rmse(dataframe_summary):
    st.write(dataframe_summary.sort_values(by='RMSE'))

def save_model(model_with_scaler, label):
    try:
        with open(f"../data/models/{label}_{model_with_scaler['model'][:3]}_with_scaler.pkl", 'wb') as file:
            pickle.dump(model_with_scaler, file)
    except:
        pass


st.sidebar.markdown("# Test Algorithms :brain:")

st.markdown("# Algorithms Testing :brain:")

data_cleaned = pd.read_csv('../data/data_cleaned.csv')

selected_algorithms = st.multiselect(
    'Which algorithms do you want to test?',
    algorithms
)

if len(selected_algorithms) > 0:
    selected_label = st.selectbox(
    'For which label do you want to test?',
    ('', 'paper_price', 'wood_pulp_price'))

    if selected_label != '':

        if st.button('Run Test'):

            with st.spinner('Wait for it...'):
                if selected_label == 'paper_price':
                    result = compare_models(selected_algorithms, data_cleaned.drop('wood_pulp_price', axis=1), selected_label)
                    for model_with_scaler in result['models']:
                        save_model(model_with_scaler, selected_label)
                else:
                    result = compare_models(selected_algorithms, data_cleaned.drop('paper_price', axis=1), selected_label)
                    for model_with_scaler in result['models']:
                        save_model(model_with_scaler, selected_label)

                display_ordered_rmse(result['summary'])