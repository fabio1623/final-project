import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

st.set_page_config(
    page_title="Model Tester",
    page_icon="🧠",
)

st.write("# 🧠 Model Tester")

data = st.cache_data(pd.read_csv)("../data/data_cleaned.csv")

label = st.selectbox("Select Label Column", ["paper_price", "wood_pulp_price"])
X = data.drop([label], axis=1)
y = data[label]

algorithms = [
    {
        "name": "Linear Regression",
        "model": LinearRegression()
    },
    {
        "name": "Ridge Regression",
        "model": Ridge()
    },
    {
        "name": "Lasso Regression",
        "model": Lasso()
    },
    {
        "name": "Decision Tree Regressor",
        "model": DecisionTreeRegressor()
    },
    {
        "name": "Elastic Net",
        "model": ElasticNet()
    },
    {
        "name": "XGB Regressor",
        "model": XGBRegressor()
    },
    {
        "name": "LGBM Regressor",
        "model": LGBMRegressor()
    },
    {
        "name": "KNeighbors",
        "model": KNeighborsRegressor()
    },
    {
        "name": "MLP Regressor",
        "model": MLPRegressor()
    },
    {
        "name": "RandomForest",
        "model": RandomForestRegressor()
    }
]

selected_algorithms = st.multiselect("Select Algorithms", [obj['name'] for obj in algorithms])
result_table = pd.DataFrame(columns=["Algorithm", "MSE", "RMSE", "MAE", "R2 Score"])

latest_iteration = st.empty()
bar = st.progress(0)

for idx, algorithm in zip(range(1, len(selected_algorithms)+1), selected_algorithms):
    latest_iteration.text(f'{idx} / {len(selected_algorithms)} models tested')
    bar.progress(round(100 / len(selected_algorithms)) * idx)

    model = [obj['model'] for obj in algorithms if obj['name'] == algorithm][0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    result_table = result_table.append({"Algorithm": algorithm, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2 Score": r2}, ignore_index=True)

st.dataframe(result_table, use_container_width=True)