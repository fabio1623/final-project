import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def models_tester_page():
    st.title("Models Tester")

    data = st.cache(pd.read_csv)("../data/data_cleaned.csv")

    label = st.selectbox("Select Label Column", ["paper_price", "wood_pulp_price"])
    X = data.drop([label], axis=1)
    y = data[label]

    algorithms = st.multiselect("Select Algorithms", ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree Regressor"])
    result_table = pd.DataFrame(columns=["Algorithm", "MSE", "RMSE", "MAE", "R2 Score"])
    for algorithm in algorithms:
        if algorithm == "Linear Regression":
            model = LinearRegression()
        elif algorithm == "Ridge Regression":
            model = Ridge()
        elif algorithm == "Lasso Regression":
            model = Lasso()
        else:
            model = DecisionTreeRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        result_table = result_table.append({"Algorithm": algorithm, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2 Score": r2}, ignore_index=True)

    st.dataframe(result_table, use_container_width=True)

if __name__ == "__main__":
    models_tester_page()
