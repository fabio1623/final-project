import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Predictions Tester",
    page_icon="🔮",
)

st.write("# 🔮 Predictions Tester")

data = st.cache_data(pd.read_csv)("../data/data_cleaned.csv")

label = st.selectbox("Select Label Column", ["paper_price", "wood_pulp_price"])
if label == "paper_price":
    label_to_drop = "wood_pulp_price"
else:
    label_to_drop = "paper_price"

X = data.drop([label, label_to_drop], axis=1)
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

selected_algorithm = st.selectbox("Select Algorithm", [obj['name'] for obj in algorithms])

model = [obj['model'] for obj in algorithms if obj['name'] == selected_algorithm][0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train)

year = st.slider("Year", min_value=data["year"].max()+1, max_value=2050, value=dt.datetime.now().year+1, step=1)

if st.button("Set random values for prediction inputs"):
    paper_pulp_prod_tonnes_value = np.random.randint(100000000)
    paper_pulp_export_tonnes_value = np.random.randint(100000000)
    paper_pulp_import_tonnes_value = np.random.randint(100000000)
    wood_pulp_production_tonnes_value = np.random.randint(100000000)
    wood_pulp_export_tonnes_value = np.random.randint(100000000)
    wood_pulp_import_tonnes_value = np.random.randint(100000000)
else:
    paper_pulp_prod_tonnes_value = 0
    paper_pulp_export_tonnes_value = 0
    paper_pulp_import_tonnes_value = 0
    wood_pulp_production_tonnes_value = 0
    wood_pulp_export_tonnes_value = 0
    wood_pulp_import_tonnes_value = 0

paper_pulp_prod_tonnes = st.number_input("Paper Pulp Production (tonnes)", value=paper_pulp_prod_tonnes_value, step=10000000)
paper_pulp_export_tonnes = st.number_input("Paper Pulp Export (tonnes)", value=paper_pulp_export_tonnes_value, step=10000000)
paper_pulp_import_tonnes = st.number_input("Paper Pulp Import (tonnes)", value=paper_pulp_import_tonnes_value, step=10000000)
wood_pulp_production_tonnes = st.number_input("Wood Pulp Production (tonnes)", value=wood_pulp_production_tonnes_value, step=10000000)
wood_pulp_export_tonnes = st.number_input("Wood Pulp Export (tonnes)", value=wood_pulp_export_tonnes_value, step=10000000)
wood_pulp_import_tonnes = st.number_input("Wood Pulp Import (tonnes)", value=wood_pulp_import_tonnes_value, step=10000000)

years = range(data["year"].max()+1, year+1)
nb_years = year - data["year"].max()

prediction_input = pd.DataFrame({
    "year": years,
    "paper_pulp_prod_tonnes": np.full(nb_years, paper_pulp_prod_tonnes),
    "paper_pulp_export_tonnes": np.full(nb_years, paper_pulp_export_tonnes),
    "paper_pulp_import_tonnes": np.full(nb_years, paper_pulp_import_tonnes),
    "wood_pulp_production_tonnes": np.full(nb_years, wood_pulp_production_tonnes),
    "wood_pulp_export_tonnes": np.full(nb_years, wood_pulp_export_tonnes),
    "wood_pulp_import_tonnes": np.full(nb_years, wood_pulp_import_tonnes),
})
st.write("Predictions by ", year, ": ")
prediction = model.predict(prediction_input)

prediction_output = pd.DataFrame({
    "year": years,
    label: prediction
})

label_with_years = data.groupby('year')[label].mean().reset_index()
label_with_predictions = pd.concat([label_with_years, prediction_output], axis=0)

st.line_chart(label_with_predictions, x='year', y=label)
st.dataframe(prediction_output, use_container_width=True)